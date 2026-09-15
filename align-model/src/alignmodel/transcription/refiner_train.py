"""Training loop for the cached Basic Pitch clarinet interval refiner."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .decode import TransNote
from .evaluate import evaluate_note_lists
from .refiner import (
    NoteRefiner,
    NoteRefinerConfig,
    decode_refined_notes,
    note_refiner_loss,
    save_note_refiner,
)
from .refiner_data import (
    RefinerAugmentConfig,
    RefinerCropDataset,
    collate_refiner_batch,
    load_cached_refiner_features,
    load_refiner_examples,
    load_rendered_target_notes,
)


@dataclass
class RefinerTrainConfig:
    manifest: Path
    basic_cache_root: Path
    pesto_cache_root: Path
    output_dir: Path
    epochs: int = 12
    batch_size: int = 8
    crop_frames: int = 384
    crops_per_clip: int = 2
    workers: int = 8
    prefetch_factor: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: float = 2.0
    device: str = "cuda"
    seed: int = 365
    resume_from: Path | None = None
    max_train_samples: int = 0
    max_val_samples: int = 80
    interval_weight: float = 0.20
    early_stop_patience: int = 4
    early_stop_min_epochs: int = 4
    min_f1_delta: float = 0.002
    augment: RefinerAugmentConfig = field(
        default_factory=RefinerAugmentConfig
    )
    model: NoteRefinerConfig = field(
        default_factory=lambda: NoteRefinerConfig(
            pesto_pitch_unit="midi",
            pesto_written_shift=0.0,
        )
    )


def _device(name: str) -> torch.device:
    resolved = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else name
    )
    result = torch.device(resolved)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _aggregate(rows: list[dict]) -> dict:
    n_pred = sum(int(row["n_pred"]) for row in rows)
    n_target = sum(int(row["n_target"]) for row in rows)
    n_matched = sum(int(row["n_matched"]) for row in rows)
    precision = n_matched / max(n_pred, 1)
    recall = n_matched / max(n_target, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    def weighted(key: str):
        values = [
            (float(row[key]), int(row["n_matched"]))
            for row in rows
            if row.get(key) is not None and int(row["n_matched"])
        ]
        if not values:
            return None
        return sum(value * count for value, count in values) / sum(
            count for _value, count in values
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": n_pred,
        "n_target": n_target,
        "n_matched": n_matched,
        "pred_target_ratio": n_pred / max(n_target, 1),
        "onset_mae_sec": weighted("onset_mae_sec"),
        "offset_mae_sec": weighted("offset_mae_sec"),
        "cents_mae": weighted("cents_mae"),
        "n_clips": len(rows),
    }


@torch.no_grad()
def evaluate_refiner(
    model: NoteRefiner,
    examples,
    config: RefinerTrainConfig,
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> tuple[dict, list[dict[str, torch.Tensor]]]:
    model.eval()
    rows = []
    cached_outputs = []
    selected = examples[:max_samples] if max_samples else examples
    for example in selected:
        basic, pesto = load_cached_refiner_features(
            example, config.basic_cache_root, config.pesto_cache_root
        )
        tensors = {
            "note": torch.from_numpy(basic.note)[None].to(device),
            "onset": torch.from_numpy(basic.onset)[None].to(device),
            "contour": torch.from_numpy(basic.contour)[None].to(device),
            "pesto": torch.from_numpy(pesto)[None].to(device),
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                tensors["note"],
                tensors["onset"],
                tensors["contour"],
                tensors["pesto"],
            )
        outputs_cpu = {key: value.float().cpu() for key, value in outputs.items()}
        predicted = decode_refined_notes(outputs_cpu, model.config)
        target = [
            TransNote(pitch, start, end, 1.0, cents=cents)
            for pitch, start, end, cents in load_rendered_target_notes(example)
        ]
        rows.append(evaluate_note_lists(predicted, target))
        cached_outputs.append(outputs_cpu)
    return _aggregate(rows), cached_outputs


def calibrate_refiner(
    model: NoteRefiner,
    examples,
    config: RefinerTrainConfig,
    device: torch.device,
) -> tuple[NoteRefinerConfig, dict]:
    _metrics, outputs = evaluate_refiner(
        model,
        examples,
        config,
        device,
        max_samples=min(config.max_val_samples or len(examples), 80),
    )
    targets = [
        [
            TransNote(pitch, start, end, 1.0, cents=cents)
            for pitch, start, end, cents in load_rendered_target_notes(example)
        ]
        for example in examples[: len(outputs)]
    ]
    best_config = NoteRefinerConfig.from_dict(model.config.to_dict())
    best_metrics: dict | None = None
    best_rank = -float("inf")
    for threshold in (0.15, 0.25, 0.35, 0.45, 0.55):
        candidate = NoteRefinerConfig.from_dict(model.config.to_dict())
        candidate.boundary_threshold = threshold
        rows = [
            evaluate_note_lists(
                decode_refined_notes(output, candidate), target
            )
            for output, target in zip(outputs, targets)
        ]
        metrics = _aggregate(rows)
        ratio = max(float(metrics["pred_target_ratio"]), 1e-3)
        rank = float(metrics["f1"]) - 0.08 * abs(float(np.log(ratio)))
        if rank > best_rank:
            best_config, best_metrics, best_rank = candidate, metrics, rank
    assert best_metrics is not None
    return best_config, best_metrics


def train_note_refiner(config: RefinerTrainConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _device(config.device)
    train_examples = load_refiner_examples(
        config.manifest, "train", procedural_only=True
    )
    val_examples = load_refiner_examples(
        config.manifest, "val", procedural_only=True
    )
    manifest_document = json.loads(
        Path(config.manifest).read_text(encoding="utf-8")
    )
    effective_transpose = int(
        (manifest_document.get("policy") or {}).get(
            "effective_audio_transpose",
            train_examples[0].effective_audio_transpose,
        )
    )
    if config.max_train_samples:
        train_examples = train_examples[: config.max_train_samples]
    train_data = RefinerCropDataset(
        train_examples,
        config.basic_cache_root,
        config.pesto_cache_root,
        crop_frames=config.crop_frames,
        midi_min=config.model.midi_min,
        midi_max=config.model.midi_max,
        training=True,
        crops_per_clip=config.crops_per_clip,
        augment_config=config.augment,
    )
    loader_args = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_refiner_batch,
    }
    if config.workers:
        loader_args.update(
            persistent_workers=True,
            prefetch_factor=max(1, config.prefetch_factor),
        )
    loader = DataLoader(train_data, **loader_args)
    model = NoteRefiner(config.model).to(device)
    if config.resume_from is not None:
        payload = torch.load(
            Path(config.resume_from),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / "best.pt"
    last_path = config.output_dir / "last.pt"
    history_path = config.output_dir / "history.json"
    history = []
    best_f1 = -1.0
    patience_best = -1.0
    stale = 0
    stopped_reason = "max_epochs"
    for epoch in range(1, config.epochs + 1):
        model.train()
        running: dict[str, float] = {}
        steps = 0
        for batch in loader:
            batch_device = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(
                    batch_device["note"],
                    batch_device["onset_map"],
                    batch_device["contour"],
                    batch_device["pesto"],
                )
                targets = {
                    key: batch_device[key]
                    for key in (
                        "voiced",
                        "onset",
                        "offset",
                        "pitch",
                        "cents",
                        "frame_mask",
                        "frame_weight",
                        "onset_weight",
                        "offset_weight",
                        "intervals",
                    )
                }
                loss, parts = note_refiner_loss(
                    outputs,
                    targets,
                    model.config,
                    interval_weight=config.interval_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + float(value)
            steps += 1

        metrics, _outputs = evaluate_refiner(
            model,
            val_examples,
            config,
            device,
            max_samples=config.max_val_samples or None,
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value / max(steps, 1)
                for key, value in running.items()
            },
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        extra = {
            "format_version": 3,
            "frontend": "basic-pitch-0.4.0",
            "fine_pitch": "pesto-2.0.1",
            "corpus": "procedural12k",
            "effective_audio_transpose": effective_transpose,
            "epoch": epoch,
            "metrics": metrics,
            "train_config": {
                **asdict(config),
                "manifest": str(config.manifest),
                "basic_cache_root": str(config.basic_cache_root),
                "pesto_cache_root": str(config.pesto_cache_root),
                "output_dir": str(config.output_dir),
                "model": config.model.to_dict(),
            },
        }
        save_note_refiner(last_path, model, extra=extra)
        if metrics["f1"] > best_f1 or not best_path.exists():
            best_f1 = float(metrics["f1"])
            save_note_refiner(best_path, model, extra=extra)
        if metrics["f1"] > patience_best + config.min_f1_delta:
            patience_best = float(metrics["f1"])
            stale = 0
        else:
            stale += 1
        history_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "best_f1": best_f1,
                    "stopped_reason": None,
                    "train_examples": len(train_examples),
                    "val_examples": len(val_examples),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_f1={metrics['f1']:.4f} "
            f"ratio={metrics['pred_target_ratio']:.3f}",
            flush=True,
        )
        if (
            epoch >= config.early_stop_min_epochs
            and stale >= config.early_stop_patience
        ):
            stopped_reason = "validation_f1_patience"
            break

    best_model, best_extra = __import__(
        "alignmodel.transcription.refiner", fromlist=["load_note_refiner"]
    ).load_note_refiner(best_path, device)
    calibrated_config, calibrated_metrics = calibrate_refiner(
        best_model, val_examples, config, device
    )
    best_model.config = calibrated_config
    best_extra["calibration"] = {
        "boundary_threshold": calibrated_config.boundary_threshold,
        "metrics": calibrated_metrics,
    }
    save_note_refiner(best_path, best_model, extra=best_extra)
    history_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_f1": best_f1,
                "stopped_reason": stopped_reason,
                "calibration": best_extra["calibration"],
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return best_path
