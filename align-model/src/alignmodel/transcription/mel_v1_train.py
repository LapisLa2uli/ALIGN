"""Training loop with deterministic train-fold calibration and atomic resume."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .mel_v1 import (
    SCHEMA_VERSION,
    MelDecodeConfig,
    MelNoteTranscriber,
    MelTranscriberConfig,
    decode_mel_notes,
    infer_mel_probabilities,
    mel_transcriber_loss,
)
from .mel_v1_data import (
    MelCacheRecord,
    MelCropDataset,
    MelPackedCache,
    augment_mel_batch,
)


@dataclass
class MelTrainConfig:
    output_dir: Path
    epochs: int = 12
    batch_size: int = 8
    crop_frames: int = 1024
    crops_per_clip: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: float = 2.0
    accumulation_steps: int = 1
    workers: int = 4
    prefetch_factor: int = 4
    seed: int = 20260916
    calibration_fraction: float = 0.05
    calibration_max_clips: int = 64
    calibration_every_epochs: int = 2
    early_stop_patience: int = 3
    checkpoint_every_steps: int = 100
    amp: str = "bf16"
    compile_model: bool = False
    max_train_rows: int | None = None
    augmentation_probability: float = 0.55
    model: MelTranscriberConfig = field(default_factory=MelTranscriberConfig)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with source.open("rb") as source_stream, temporary.open("wb") as target:
            while chunk := source_stream.read(4 * 1024 * 1024):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def _checkpoint_payload(
    *,
    model: MelNoteTranscriber,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    config: MelTrainConfig,
    cache: MelPackedCache,
    progress: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    decode_config: MelDecodeConfig,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "frontend_config": cache.frontend.to_dict(),
        "decode_config": decode_config.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "model": config.model.to_dict(),
        },
        "data": {
            "cache_pack_id": cache.pack_id,
            "release_pack_id": cache.metadata["source"]["pack_id"],
            "train_rows": cache.metadata["split_counts"]["train"],
            "val_rows": cache.metadata["split_counts"]["val"],
            "locked_test_materialized": False,
            "intonation_masked": True,
        },
        "progress": dict(progress),
        "history": list(history),
        "rng_state": _rng_state(),
    }


def _lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            if item == other:
                row[column] = previous + 1
            else:
                row[column] = max(row[column], row[column - 1])
            previous = saved
    return row[-1]


@torch.inference_mode()
def calibrate_train_fold(
    model: MelNoteTranscriber,
    cache: MelPackedCache,
    records: Sequence[MelCacheRecord],
    device: torch.device,
    *,
    max_clips: int,
) -> tuple[MelDecodeConfig, dict[str, Any]]:
    """Select among a predeclared decoder grid using pitch-sequence LCS only."""

    selected = list(records[:max_clips or None])
    cached = [
        (
            infer_mel_probabilities(
                model, np.asarray(cache.mel(record), np.float32), device,
                window_frames=2048, overlap_frames=512,
                batch_size=4,
            ),
            record,
        )
        for record in selected
    ]
    candidates = (
        MelDecodeConfig(),
        MelDecodeConfig(voice_on=0.46, voice_off=0.30, onset_threshold=0.42),
        MelDecodeConfig(voice_on=0.58, voice_off=0.40, onset_threshold=0.54),
        MelDecodeConfig(
            boundary_threshold=0.52, strong_rearticulation=0.70,
            merge_gap_sec=0.055,
        ),
        MelDecodeConfig(
            boundary_threshold=0.40, strong_rearticulation=0.58,
            min_note_sec=0.035,
        ),
    )
    best_config = candidates[0]
    best_metrics: dict[str, Any] = {}
    best_rank = (-1.0, -float("inf"))
    for decode in candidates:
        matched = predicted = target = 0
        per_clip = []
        for probability, record in cached:
            notes = decode_mel_notes(
                probability,
                midi_min=model.config.midi_min,
                hop_sec=cache.frontend.hop_sec,
                config=decode,
            )
            pred_pitch = [item.pitch for item in notes]
            gold_pitch = [int(item["pitch"]) for item in record.target]
            correct = _lcs_length(pred_pitch, gold_pitch)
            matched += correct
            predicted += len(pred_pitch)
            target += len(gold_pitch)
            per_clip.append(
                2.0 * correct / max(len(pred_pitch) + len(gold_pitch), 1)
            )
        precision = matched / max(predicted, 1)
        recall = matched / max(target, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        ratio = predicted / max(target, 1)
        metrics = {
            "metric": "score_agnostic_pitch_sequence_lcs",
            "selection_uses_timestamps": False,
            "clips": len(selected),
            "matched": matched,
            "predicted": predicted,
            "target": target,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count_ratio": ratio,
            "clip_mean_f1": float(np.mean(per_clip)) if per_clip else 0.0,
        }
        rank = (f1 - 0.04 * abs(math.log(max(ratio, 1e-4))), -abs(ratio - 1.0))
        if rank > best_rank:
            best_config, best_metrics, best_rank = decode, metrics, rank
    return best_config, best_metrics


def _split_train_fold(
    records: Sequence[MelCacheRecord], fraction: float, seed: int
) -> tuple[list[MelCacheRecord], list[MelCacheRecord]]:
    if fraction <= 0.0:
        return list(records), []
    if fraction >= 1.0:
        raise ValueError("calibration_fraction must be smaller than one")
    ranked = sorted(
        records,
        key=lambda item: __import__("hashlib").sha256(
            f"{seed}:{item.sample}".encode("utf-8")
        ).digest(),
    )
    count = max(1, int(round(len(ranked) * fraction)))
    calibration = ranked[:count]
    calibration_ids = {item.sample for item in calibration}
    training = [item for item in records if item.sample not in calibration_ids]
    return training, calibration


def train_mel_transcriber(
    cache_root: Path | str,
    config: MelTrainConfig,
    *,
    device: torch.device | str = "cuda",
    resume: Path | str | None = None,
) -> Path:
    """Train from scratch or resume exactly at an optimizer-step boundary."""

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    cache = MelPackedCache(cache_root, deep=False)
    if cache.frontend.n_mels != config.model.n_mels:
        raise ValueError("Model and cache mel dimensions differ")
    all_train = cache.records("train", include_targets=True)
    if config.max_train_rows is not None:
        all_train = all_train[:max(2, int(config.max_train_rows))]
    training, calibration = _split_train_fold(
        all_train, config.calibration_fraction, config.seed
    )
    pitch_counts = torch.ones(config.model.n_pitches, dtype=torch.float64)
    for record in training:
        for note in record.target:
            pitch = int(note["pitch"]) - config.model.midi_min
            if 0 <= pitch < len(pitch_counts):
                pitch_counts[pitch] += 1.0
    nonzero = pitch_counts > 1.0
    reference = pitch_counts[nonzero].mean() if bool(nonzero.any()) else 1.0
    pitch_class_weight = (
        (reference / pitch_counts).sqrt().clamp(0.5, 3.0).float().to(target)
    )
    model = MelNoteTranscriber(config.model).to(target)
    fused = target.type == "cuda" and "fused" in __import__("inspect").signature(
        torch.optim.AdamW
    ).parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=fused,
    )
    total_optimizer_steps = math.ceil(
        len(training) * config.crops_per_clip / config.batch_size
    ) * config.epochs // max(1, config.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_optimizer_steps), eta_min=2e-5
    )
    use_bf16 = (
        target.type == "cuda"
        and config.amp == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = target.type == "cuda" and config.amp == "fp16"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    forward_model = torch.compile(model) if config.compile_model else model

    config.output_dir.mkdir(parents=True, exist_ok=True)
    mid_path = config.output_dir / "mid_epoch_checkpoint.pt"
    last_path = config.output_dir / "last.pt"
    best_path = config.output_dir / "best.pt"
    history_path = config.output_dir / "history.json"
    history: list[dict[str, Any]] = []
    epoch_start, position_start, global_step = 1, 0, 0
    decode_config = MelDecodeConfig()
    best_calibration = -1.0
    patience_best = -1.0
    stale = 0

    if resume is not None:
        payload = torch.load(Path(resume), map_location=target, weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported resume checkpoint")
        if payload["data"]["cache_pack_id"] != cache.pack_id:
            raise ValueError("Resume checkpoint cache fingerprint mismatch")
        if payload["model_config"] != config.model.to_dict():
            raise ValueError("Resume checkpoint model configuration mismatch")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload.get("scaler_state_dict"):
            scaler.load_state_dict(payload["scaler_state_dict"])
        progress = payload["progress"]
        epoch_start = int(progress["epoch"])
        position_start = int(progress["position"])
        global_step = int(progress["global_step"])
        history = list(payload.get("history") or [])
        decode_config = MelDecodeConfig.from_dict(payload.get("decode_config"))
        _restore_rng_state(payload["rng_state"])

    stopped_reason = "max_epochs"
    for epoch in range(epoch_start, config.epochs + 1):
        dataset = MelCropDataset(
            cache.root,
            training,
            crop_frames=config.crop_frames,
            epoch=epoch,
            seed=config.seed,
            crops_per_clip=config.crops_per_clip,
        )
        generator = torch.Generator().manual_seed(config.seed + epoch * 1009)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        position = position_start if epoch == epoch_start else 0
        subset = Subset(dataset, order[position:])
        loader_options: dict[str, Any] = {
            "batch_size": config.batch_size,
            "shuffle": False,
            "num_workers": config.workers,
            "pin_memory": target.type == "cuda",
            "drop_last": False,
        }
        if config.workers:
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=max(1, config.prefetch_factor),
            )
        loader = DataLoader(subset, **loader_options)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        rows = 0
        started = time.perf_counter()
        optimizer_steps_epoch = 0
        for batch_index, batch in enumerate(loader):
            batch_rows = int(batch["mel"].shape[0])
            batch_device = {
                key: (
                    value.to(target, non_blocking=True)
                    if torch.is_tensor(value) else value
                )
                for key, value in batch.items()
            }
            batch_device["mel"] = augment_mel_batch(
                batch_device["mel"],
                probability=config.augmentation_probability,
            )
            with torch.autocast(
                device_type=target.type,
                dtype=amp_dtype,
                enabled=target.type == "cuda" and (use_bf16 or use_fp16),
            ):
                output = forward_model(batch_device["mel"])
                loss, parts = mel_transcriber_loss(
                    output,
                    batch_device,
                    pitch_class_weight=pitch_class_weight,
                )
                scaled_loss = loss / max(1, config.accumulation_steps)
            scaler.scale(scaled_loss).backward()
            position += batch_rows
            rows += batch_rows
            for name, value in parts.items():
                running[name] = running.get(name, 0.0) + float(value)
            should_step = (
                (batch_index + 1) % max(1, config.accumulation_steps) == 0
                or position >= len(dataset)
            )
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            optimizer_steps_epoch += 1
            if (
                global_step % max(1, config.checkpoint_every_steps) == 0
                and position < len(dataset)
            ):
                _atomic_torch_save(
                    mid_path,
                    _checkpoint_payload(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, config=config, cache=cache,
                        progress={
                            "epoch": epoch, "position": position,
                            "global_step": global_step,
                            "optimizer_boundary": True,
                        },
                        history=history, decode_config=decode_config,
                    ),
                )
            if global_step == 1 or global_step % 25 == 0:
                elapsed = time.perf_counter() - started
                rate = rows / max(elapsed, 1e-9)
                remaining = len(dataset) - position
                peak = (
                    torch.cuda.max_memory_allocated() / 1024**2
                    if target.type == "cuda" else 0.0
                )
                print(
                    f"epoch={epoch} rows={position}/{len(dataset)} "
                    f"rows_per_sec={rate:.2f} vram_mb={peak:.0f} "
                    f"eta_sec={remaining/max(rate,1e-9):.1f}",
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_rows": rows,
            "rows_per_second": rows / max(elapsed, 1e-9),
            "optimizer_steps": optimizer_steps_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{
                f"train_{key}": value / max(len(loader), 1)
                for key, value in running.items()
            },
        }
        if (
            calibration
            and epoch % max(1, config.calibration_every_epochs) == 0
        ):
            decode_config, calibration_metrics = calibrate_train_fold(
                model, cache, calibration, target,
                max_clips=config.calibration_max_clips,
            )
            row["train_fold_calibration"] = calibration_metrics
            score = float(calibration_metrics["f1"])
            candidate_path = config.output_dir / f"candidate-epoch-{epoch:03d}.pt"
            _atomic_torch_save(
                candidate_path,
                _checkpoint_payload(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, config=config, cache=cache,
                    progress={
                        "epoch": epoch + 1, "position": 0,
                        "global_step": global_step,
                        "optimizer_boundary": True,
                    },
                    history=[*history, row], decode_config=decode_config,
                ),
            )
            if score > best_calibration:
                best_calibration = score
                _atomic_copy(candidate_path, best_path)
            if score > patience_best + 0.001:
                patience_best = score
                stale = 0
            else:
                stale += 1
        elif not calibration and epoch == config.epochs:
            candidate_path = (
                config.output_dir / f"candidate-epoch-{epoch:03d}.pt"
            )
            _atomic_torch_save(
                candidate_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    cache=cache,
                    progress={
                        "epoch": epoch + 1,
                        "position": 0,
                        "global_step": global_step,
                        "optimizer_boundary": True,
                    },
                    history=[*history, row],
                    decode_config=decode_config,
                ),
            )
            _atomic_copy(candidate_path, best_path)
        history.append(row)
        payload = _checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, config=config, cache=cache,
            progress={
                "epoch": epoch + 1, "position": 0,
                "global_step": global_step, "optimizer_boundary": True,
            },
            history=history, decode_config=decode_config,
        )
        _atomic_torch_save(last_path, payload)
        _atomic_json(history_path, {
            "schema_version": SCHEMA_VERSION,
            "selection_metric": "train-fold score-agnostic pitch-sequence LCS F1",
            "timestamp_metric_used_for_selection": False,
            "official_validation_selection_pending": True,
            "training_rows": len(training),
            "calibration_rows": len(calibration),
            "validation_rows": cache.metadata["split_counts"]["val"],
            "best_train_fold_calibration_f1": (
                best_calibration if calibration else None
            ),
            "history": history,
            "stopped_reason": None,
        })
        position_start = 0
        if (
            calibration
            and epoch >= max(4, config.calibration_every_epochs * 2)
            and stale >= config.early_stop_patience
        ):
            stopped_reason = "train_fold_calibration_patience"
            break

    _atomic_json(history_path, {
        "schema_version": SCHEMA_VERSION,
        "selection_metric": "train-fold score-agnostic pitch-sequence LCS F1",
        "timestamp_metric_used_for_selection": False,
        "official_validation_selection_pending": True,
        "training_rows": len(training),
        "calibration_rows": len(calibration),
        "validation_rows": cache.metadata["split_counts"]["val"],
        "best_train_fold_calibration_f1": (
            best_calibration if calibration else None
        ),
        "history": history,
        "stopped_reason": stopped_reason,
    })
    cache.close()
    return best_path if best_path.exists() else last_path

