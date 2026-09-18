"""Train the polyphonic ORN transcriber with balanced OutputRaw replay."""

from __future__ import annotations

import argparse
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import load_mel_checkpoint
from alignmodel.transcription.mel_v1_data import (
    MelPackedCache,
    augment_mel_batch,
)
from alignmodel.transcription.ornament_multipitch_v1 import (
    SCHEMA_VERSION,
    MultiPitchConfig,
    MultiPitchCropDataset,
    MultiPitchDecodeConfig,
    OrnamentMultiPitchTranscriber,
    decode_multipitch_notes,
    infer_multipitch_probabilities,
    multipitch_loss,
)


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)
EXPECTED_TRACK_B_SHA256 = (
    "3d8f93732a810c3f8470a88316debb9f92b4680b2333c2187866e6090a730a4e"
)
DECODE_GRID = (
    MultiPitchDecodeConfig(),
    MultiPitchDecodeConfig(
        activity_on=0.55,
        activity_off=0.30,
        onset_threshold=0.50,
        offset_threshold=0.50,
        min_confidence=0.25,
    ),
    MultiPitchDecodeConfig(
        activity_on=0.65,
        activity_off=0.35,
        onset_threshold=0.60,
        offset_threshold=0.55,
        min_confidence=0.30,
    ),
    MultiPitchDecodeConfig(
        activity_on=0.75,
        activity_off=0.40,
        onset_threshold=0.70,
        offset_threshold=0.60,
        min_confidence=0.35,
    ),
    MultiPitchDecodeConfig(
        activity_on=0.40,
        activity_off=0.22,
        onset_threshold=0.35,
        offset_threshold=0.40,
        min_note_sec=0.020,
        min_confidence=0.18,
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


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in value["cuda"]]
        )


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


@torch.inference_mode()
def _calibrate(
    model: OrnamentMultiPitchTranscriber,
    cache: MelPackedCache,
    records: Sequence[Any],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[MultiPitchDecodeConfig, dict[str, Any]]:
    cached = [
        (
            infer_multipitch_probabilities(
                model,
                np.asarray(cache.mel(record), np.float32),
                device,
                batch_size=batch_size,
            ),
            record,
        )
        for record in records
    ]
    variants = []
    for index, decode in enumerate(DECODE_GRID):
        matched = predicted = gold = 0
        clip_f1 = []
        for probability, record in cached:
            notes = decode_multipitch_notes(
                probability,
                midi_min=model.midi_min,
                hop_sec=cache.frontend.hop_sec,
                config=decode,
            )
            pred_pitch = [value.pitch for value in notes]
            gold_pitch = [int(value["pitch"]) for value in record.target]
            correct = _lcs(pred_pitch, gold_pitch)
            matched += correct
            predicted += len(pred_pitch)
            gold += len(gold_pitch)
            clip_f1.append(
                2 * correct / max(len(pred_pitch) + len(gold_pitch), 1)
            )
        precision = matched / max(predicted, 1)
        recall = matched / max(gold, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ratio = predicted / max(gold, 1)
        variants.append(
            {
                "variant": index,
                "decode": decode.to_dict(),
                "matched": matched,
                "predicted": predicted,
                "gold": gold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count_ratio": ratio,
                "clip_macro_f1": float(np.mean(clip_f1)),
                "rank": f1 - 0.03 * abs(math.log(max(ratio, 1e-5))),
            }
        )
    selected = max(
        variants,
        key=lambda value: (
            value["rank"],
            value["f1"],
            -abs(value["count_ratio"] - 1.0),
            -value["variant"],
        ),
    )
    return MultiPitchDecodeConfig.from_dict(selected["decode"]), {
        "selection_metric": "score_agnostic_pitch_sequence_lcs",
        "timestamps_used_for_selection": False,
        "predeclared_variants": len(DECODE_GRID),
        "selected_variant": selected["variant"],
        "selected": selected,
        "variants": variants,
    }


def _checkpoint(
    *,
    model: OrnamentMultiPitchTranscriber,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    decode: MultiPitchDecodeConfig,
    args: argparse.Namespace,
    orn_cache: MelPackedCache,
    replay_cache: MelPackedCache,
    progress: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "frontend_config": orn_cache.frontend.to_dict(),
        "decode_config": decode.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": {
            "release_manifest_sha256": EXPECTED_RELEASE_SHA256,
            "orn_cache_pack_id": orn_cache.pack_id,
            "replay_cache_pack_id": replay_cache.pack_id,
            "orn_train_rows": len(orn_cache.records("train")),
            "calibration_rows": len(orn_cache.records("val")),
            "outputraw_replay_rows_per_epoch": len(
                orn_cache.records("train")
            ),
            "open_validation_materialized": False,
            "lockbox_materialized": False,
        },
        "initialization": {
            "track_b_checkpoint": str(args.track_b.resolve()),
            "track_b_checkpoint_sha256": sha256_file(args.track_b),
        },
        "progress": dict(progress),
        "history": list(history),
        "rng_state": _rng_state(),
    }


def train(args: argparse.Namespace) -> Path:
    if sha256_file(args.release_manifest) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release manifest mismatch")
    if sha256_file(args.track_b) != EXPECTED_TRACK_B_SHA256:
        raise ValueError("Track B initialization mismatch")
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
    target = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    orn_cache = MelPackedCache(args.orn_cache, deep=False)
    replay_cache = MelPackedCache(args.replay_cache, deep=False)
    if orn_cache.frontend != replay_cache.frontend:
        raise ValueError("ORN and OutputRaw replay frontends differ")
    orn_records = orn_cache.records("train", include_targets=True)
    calibration_records = orn_cache.records("val", include_targets=True)
    replay_records = replay_cache.records("train", include_targets=True)
    track_b, _frontend, _old_decode, _payload = load_mel_checkpoint(
        args.track_b, target
    )
    config = MultiPitchConfig(backbone=track_b.config)
    model = OrnamentMultiPitchTranscriber(config).to(target)
    model.initialize_track_b(track_b)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": args.backbone_learning_rate,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("backbone.")
                ],
                "lr": args.learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
        fused=(
            target.type == "cuda"
            and "fused"
            in __import__("inspect").signature(
                torch.optim.AdamW
            ).parameters
        ),
    )
    rows_per_epoch = 2 * len(orn_records) * args.crops_per_clip
    steps_per_epoch = math.ceil(rows_per_epoch / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, steps_per_epoch * args.epochs),
        eta_min=1e-6,
    )
    use_bf16 = (
        target.type == "cuda"
        and args.amp == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = target.type == "cuda" and args.amp == "fp16"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    epoch_start = 1
    position_start = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    decode = MultiPitchDecodeConfig()
    best_score = -1.0
    if args.resume is not None:
        payload = torch.load(
            args.resume, map_location=target, weights_only=False
        )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported multi-pitch resume checkpoint")
        if (
            payload["data"]["release_manifest_sha256"]
            != EXPECTED_RELEASE_SHA256
            or payload["data"]["orn_cache_pack_id"] != orn_cache.pack_id
            or payload["data"]["replay_cache_pack_id"] != replay_cache.pack_id
        ):
            raise ValueError("Resume data fingerprint mismatch")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        progress = payload["progress"]
        epoch_start = int(progress["epoch"])
        position_start = int(progress["position"])
        global_step = int(progress["global_step"])
        history = list(payload.get("history") or [])
        decode = MultiPitchDecodeConfig.from_dict(
            payload.get("decode_config")
        )
        best_score = max(
            (
                float(row.get("calibration", {}).get("selected", {}).get("rank", -1.0))
                for row in history
            ),
            default=-1.0,
        )
        _restore_rng(payload["rng_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mid_path = args.output_dir / "mid_epoch_checkpoint.pt"
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    for epoch in range(epoch_start, args.epochs + 1):
        replay_generator = torch.Generator().manual_seed(
            args.seed + epoch * 7919
        )
        replay_order = torch.randperm(
            len(replay_records), generator=replay_generator
        ).tolist()[: len(orn_records)]
        selected_replay = [replay_records[index] for index in replay_order]
        orn_dataset = MultiPitchCropDataset(
            orn_cache.root,
            orn_records,
            crop_frames=args.crop_frames,
            epoch=epoch,
            seed=args.seed,
            crops_per_clip=args.crops_per_clip,
        )
        replay_dataset = MultiPitchCropDataset(
            replay_cache.root,
            selected_replay,
            crop_frames=args.crop_frames,
            epoch=epoch,
            seed=args.seed + 1,
            crops_per_clip=args.crops_per_clip,
        )
        dataset = ConcatDataset((orn_dataset, replay_dataset))
        order_generator = torch.Generator().manual_seed(
            args.seed + epoch * 1009
        )
        order = torch.randperm(
            len(dataset), generator=order_generator
        ).tolist()
        position = position_start if epoch == epoch_start else 0
        loader = DataLoader(
            Subset(dataset, order[position:]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=target.type == "cuda",
            drop_last=False,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        rows = 0
        started = time.perf_counter()
        for batch in loader:
            batch_rows = int(batch["mel"].shape[0])
            batch_device = {
                key: (
                    value.to(target, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in batch.items()
            }
            batch_device["mel"] = augment_mel_batch(
                batch_device["mel"],
                probability=args.augmentation_probability,
            )
            with torch.autocast(
                device_type=target.type,
                dtype=amp_dtype,
                enabled=target.type == "cuda" and (use_bf16 or use_fp16),
            ):
                output = model(batch_device["mel"])
                loss, parts = multipitch_loss(output, batch_device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
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
                    mid_path,
                    _checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        decode=decode,
                        args=args,
                        orn_cache=orn_cache,
                        replay_cache=replay_cache,
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
                rate = rows / max(time.perf_counter() - started, 1e-9)
                print(
                    f"epoch={epoch} rows={position}/{len(dataset)} "
                    f"rows_per_sec={rate:.2f} "
                    f"vram_mb={torch.cuda.max_memory_allocated()/1024**2:.0f}",
                    flush=True,
                )
        decode, calibration = _calibrate(
            model,
            orn_cache,
            calibration_records,
            target,
            batch_size=args.inference_batch_size,
        )
        row = {
            "epoch": epoch,
            "train_rows": rows,
            "global_step": global_step,
            "loss": {
                name: value / max(len(loader), 1)
                for name, value in running.items()
            },
            "calibration": calibration,
            "learning_rates": [
                group["lr"] for group in optimizer.param_groups
            ],
        }
        history.append(row)
        payload = _checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            decode=decode,
            args=args,
            orn_cache=orn_cache,
            replay_cache=replay_cache,
            progress={
                "epoch": epoch + 1,
                "position": 0,
                "global_step": global_step,
                "optimizer_boundary": True,
            },
            history=history,
        )
        epoch_path = args.output_dir / f"candidate-epoch-{epoch:03d}.pt"
        _atomic_torch(epoch_path, payload)
        _atomic_torch(last_path, payload)
        score = float(calibration["selected"]["rank"])
        if score > best_score:
            best_score = score
            _atomic_torch(best_path, payload)
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
    orn_cache.close()
    replay_cache.close()
    return best_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orn-cache", type=Path, required=True)
    parser.add_argument("--replay-cache", type=Path, required=True)
    parser.add_argument("--track-b", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--crop-frames", type=int, default=1024)
    parser.add_argument("--crops-per-clip", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--backbone-learning-rate", type=float, default=5e-5
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    parser.add_argument("--augmentation-probability", type=float, default=0.35)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args(argv)
    with resource_lease(
        args.resource_status,
        "gpu",
        track="ornament-multipitch-v1-training",
        command=[sys.executable, *sys.argv],
        metadata={
            "orn_train_rows": 379,
            "calibration_rows": 53,
            "outputraw_replay": True,
            "locked_test": False,
        },
    ):
        best = train(args)
    print(
        json.dumps(
            {"best": str(best), "best_sha256": sha256_file(best)},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
