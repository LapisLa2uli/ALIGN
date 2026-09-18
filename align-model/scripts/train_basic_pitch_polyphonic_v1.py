"""Train a zero-initialized polyphonic residual on ORN Basic Pitch maps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    BasicPitchFeatures,
    decode_basic_pitch_features,
)
from alignmodel.transcription.basic_pitch_polyphonic_v1 import (
    SCHEMA_VERSION,
    BasicPitchPolyphonicConfig,
    BasicPitchPolyphonicRefiner,
    BasicPitchRefinerDataset,
    basic_pitch_refiner_loss,
    refine_basic_pitch_features,
)


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)
DECODE_GRID = (
    BasicPitchDecodeConfig(),
    BasicPitchDecodeConfig(
        onset_threshold=0.40,
        frame_threshold=0.30,
        minimum_note_length_ms=30.0,
    ),
    BasicPitchDecodeConfig(
        onset_threshold=0.30,
        frame_threshold=0.20,
        minimum_note_length_ms=20.0,
    ),
    BasicPitchDecodeConfig(
        onset_threshold=0.60,
        frame_threshold=0.50,
        minimum_note_length_ms=40.0,
    ),
)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _lcs(left: Sequence[int], right: Sequence[int]) -> int:
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            row[column] = (
                previous + 1
                if item == other
                else max(row[column], row[column - 1])
            )
            previous = saved
    return row[-1]


def _load_features(path: Path) -> BasicPitchFeatures:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(np.asarray(saved["metadata"]).item()))
        return BasicPitchFeatures(
            np.asarray(saved["note"], np.float32),
            np.asarray(saved["onset"], np.float32),
            np.asarray(saved["contour"], np.float32),
            np.asarray(saved["frame_times"], np.float64),
            metadata,
        )


def _examples(
    release: Mapping[str, Any],
    cache_index: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    target_artifact = release["artifacts"]["development_targets"]
    target_path = Path(target_artifact["path"])
    if sha256_file(target_path) != target_artifact["sha256"]:
        raise ValueError("Development target archive mismatch")
    targets = {}
    with gzip.open(target_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] in {"train", "calibration"}:
                targets[row["sample"]] = tuple(
                    {
                        "pitch": int(event["pitch_midi_written"]),
                        "start_sec": float(event["start_sec"]),
                        "end_sec": float(event["end_sec"]),
                    }
                    for event in row["lineage"]["rendered_notes"]
                )
    cache_rows = {
        row["sample"]: row for row in cache_index["artifacts"]
    }
    output = {"train": [], "calibration": []}
    for split in output:
        manifest_rows = release["splits"]["development"][split]
        for row in manifest_rows:
            sample = row["sample"]
            cache = cache_rows[sample]
            path = Path(cache["cache"])
            if (
                sha256_file(path) != cache["cache_sha256"]
                or cache["audio_sha256"]
                != row["source_hashes"]["performance_audio.wav"]
            ):
                raise ValueError(f"Basic Pitch cache mismatch: {sample}")
            output[split].append(
                {
                    "sample": sample,
                    "cache": str(path),
                    "target": targets[sample],
                }
            )
    return output


@torch.inference_mode()
def _calibrate(
    model: BasicPitchPolyphonicRefiner,
    examples: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[BasicPitchDecodeConfig, dict[str, Any]]:
    refined = [
        (
            refine_basic_pitch_features(
                model, _load_features(Path(example["cache"])), device
            ),
            example,
        )
        for example in examples
    ]
    variants = []
    for index, decode in enumerate(DECODE_GRID):
        matched = predicted = gold = 0
        for features, example in refined:
            notes = decode_basic_pitch_features(features, decode)
            target_pitch = [int(row["pitch"]) for row in example["target"]]
            correct = _lcs(
                [int(value.pitch) for value in notes], target_pitch
            )
            matched += correct
            predicted += len(notes)
            gold += len(target_pitch)
        precision = matched / max(predicted, 1)
        recall = matched / max(gold, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ratio = predicted / max(gold, 1)
        variants.append(
            {
                "variant": index,
                "decode": asdict(decode),
                "matched": matched,
                "predicted": predicted,
                "gold": gold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count_ratio": ratio,
                "rank": f1 - 0.03 * abs(math.log(max(ratio, 1e-5))),
            }
        )
    selected = max(
        variants,
        key=lambda row: (
            row["rank"],
            row["f1"],
            -abs(row["count_ratio"] - 1.0),
            -row["variant"],
        ),
    )
    return BasicPitchDecodeConfig(**selected["decode"]), {
        "selection_metric": "score_agnostic_pitch_sequence_lcs",
        "timestamps_used_for_selection": False,
        "selected": selected,
        "variants": variants,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def _payload(
    *,
    model: BasicPitchPolyphonicRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    decode: BasicPitchDecodeConfig,
    args: argparse.Namespace,
    progress: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "decode_config": asdict(decode),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": {
            "release_manifest_sha256": EXPECTED_RELEASE_SHA256,
            "cache_index_sha256": sha256_file(args.cache_index),
            "train_rows": 379,
            "calibration_rows": 53,
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
        "progress": dict(progress),
        "history": list(history),
        "rng_state": _rng_state(),
    }


def train(args: argparse.Namespace) -> Path:
    if sha256_file(args.release_manifest) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
    cache_index = json.loads(args.cache_index.read_text(encoding="utf-8"))
    examples = _examples(release, cache_index)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = BasicPitchPolyphonicConfig()
    model = BasicPitchPolyphonicRefiner(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    dataset = BasicPitchRefinerDataset(
        examples["train"],
        crop_frames=args.crop_frames,
        crops_per_clip=args.crops_per_clip,
        epoch=1,
        seed=args.seed,
    )
    steps = math.ceil(len(dataset) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    decode = DECODE_GRID[0]
    history: list[dict[str, Any]] = []
    epoch_start = 1
    position_start = 0
    global_step = 0
    best_score = -1.0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (
            saved.get("schema_version") != SCHEMA_VERSION
            or saved["data"]["release_manifest_sha256"]
            != EXPECTED_RELEASE_SHA256
            or saved["data"]["cache_index_sha256"]
            != sha256_file(args.cache_index)
        ):
            raise ValueError("Basic Pitch refiner resume mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        decode = BasicPitchDecodeConfig(**saved["decode_config"])
        progress = saved["progress"]
        epoch_start = int(progress["epoch"])
        position_start = int(progress["position"])
        global_step = int(progress["global_step"])
        history = list(saved.get("history") or [])
        best_score = max(
            (
                float(row["calibration"]["selected"]["rank"])
                for row in history
            ),
            default=-1.0,
        )
        _restore_rng(saved["rng_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    for epoch in range(epoch_start, args.epochs + 1):
        dataset.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        position = position_start if epoch == epoch_start else 0
        loader = DataLoader(
            Subset(dataset, order[position:]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        model.train()
        running: dict[str, float] = {}
        started = time.perf_counter()
        rows = 0
        for batch in loader:
            batch_device = {
                key: (
                    value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch_device["note_map"],
                    batch_device["onset_map"],
                    batch_device["contour_map"],
                )
                loss, parts = basic_pitch_refiner_loss(
                    output, batch_device
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            batch_rows = int(batch["note_map"].shape[0])
            position += batch_rows
            rows += batch_rows
            global_step += 1
            for name, value in parts.items():
                running[name] = running.get(name, 0.0) + float(value)
            if (
                global_step % args.checkpoint_every_steps == 0
                and position < len(dataset)
            ):
                _atomic_torch(
                    args.output_dir / "mid_epoch_checkpoint.pt",
                    _payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        decode=decode,
                        args=args,
                        progress={
                            "epoch": epoch,
                            "position": position,
                            "global_step": global_step,
                            "optimizer_boundary": True,
                        },
                        history=history,
                    ),
                )
            if global_step == 1 or global_step % 20 == 0:
                print(
                    f"epoch={epoch} rows={position}/{len(dataset)} "
                    f"rows_per_sec={rows/max(time.perf_counter()-started,1e-9):.2f}",
                    flush=True,
                )
        decode, calibration = _calibrate(
            model, examples["calibration"], device
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": {
                name: value / max(len(loader), 1)
                for name, value in running.items()
            },
            "calibration": calibration,
        }
        history.append(row)
        saved = _payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            decode=decode,
            args=args,
            progress={
                "epoch": epoch + 1,
                "position": 0,
                "global_step": global_step,
                "optimizer_boundary": True,
            },
            history=history,
        )
        _atomic_torch(args.output_dir / "last.pt", saved)
        _atomic_torch(
            args.output_dir / f"candidate-epoch-{epoch:03d}.pt", saved
        )
        score = float(calibration["selected"]["rank"])
        if score > best_score:
            best_score = score
            _atomic_torch(best_path, saved)
        _atomic_json(args.output_dir / "history.json", {"history": history})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "calibration": calibration["selected"],
                    "best_score": best_score,
                }
            ),
            flush=True,
        )
        position_start = 0
    return best_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-frames", type=int, default=1024)
    parser.add_argument("--crops-per-clip", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args(argv)
    with resource_lease(
        args.resource_status,
        "gpu",
        track="basic-pitch-polyphonic-v1-training",
        command=[sys.executable, *sys.argv],
        metadata={
            "train_rows": 379,
            "calibration_rows": 53,
            "locked_test": False,
        },
    ):
        best = train(args)
    print(
        json.dumps(
            {"best": str(best), "best_sha256": sha256_file(best)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
