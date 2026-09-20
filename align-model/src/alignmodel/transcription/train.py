from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import (
    NoteCropDataset,
    TranscriptionExample,
    load_split,
    load_written_notes_with_cents,
)
from .decode import DecodeConfig, decode_notes, infer_probabilities
from .evaluate import evaluate_note_lists
from .model import NoteFrameNet, NoteFrameNetConfig, note_frame_loss


@dataclass
class NoteTrainConfig:
    output_dir: Path = field(
        default_factory=lambda: Path("align-model/runs/note-transcriber")
    )
    epochs: int = 30
    batch_size: int = 8
    crop_frames: int = 1024
    crops_per_clip: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: float = 2.0
    num_workers: int = 0
    prefetch_factor: int = 4
    gpu_augment: bool = True
    device: str = "auto"
    seed: int = 365
    resume_from: Path | None = None
    val_fraction: float = 0.1
    early_stop_patience: int = 5
    early_stop_min_epochs: int = 4
    min_f1_delta: float = 0.002
    max_val_clips: int = 80
    calibrate_val_clips: int = 0
    infer_window_frames: int = 2048
    infer_overlap_frames: int = 512
    infer_batch_size: int = 4
    decode_fusion: str = "neural"
    model: NoteFrameNetConfig = field(default_factory=NoteFrameNetConfig)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _aggregate(rows: list[dict]) -> dict:
    n_pred = sum(int(row["n_pred"]) for row in rows)
    n_target = sum(int(row["n_target"]) for row in rows)
    n_matched = sum(int(row["n_matched"]) for row in rows)
    precision = n_matched / max(n_pred, 1)
    recall = n_matched / max(n_target, 1)
    if n_pred == 0 and n_target == 0:
        precision = recall = 1.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    onset = [
        (float(row["onset_mae_sec"]), int(row["n_matched"]))
        for row in rows
        if row["onset_mae_sec"] is not None
    ]
    offset = [
        (float(row["offset_mae_sec"]), int(row["n_matched"]))
        for row in rows
        if row["offset_mae_sec"] is not None
    ]
    cents = [
        (float(row["cents_mae"]), int(row["n_matched"]))
        for row in rows
        if row.get("cents_mae") is not None
    ]

    def weighted(values):
        return (
            sum(value * count for value, count in values)
            / max(sum(count for _, count in values), 1)
            if values
            else None
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onset_mae_sec": weighted(onset),
        "offset_mae_sec": weighted(offset),
        "cents_mae": weighted(cents),
        "n_pred": n_pred,
        "n_target": n_target,
        "n_matched": n_matched,
        "n_semitone_errors": sum(int(row.get("n_semitone_errors") or 0) for row in rows),
        "n_octave_errors": sum(int(row.get("n_octave_errors") or 0) for row in rows),
        "n_plus_minus_2": sum(int(row.get("n_plus_minus_2") or 0) for row in rows),
        "pred_target_ratio": n_pred / max(n_target, 1),
        "n_clips": len(rows),
    }


def augment_mel_batch(mel: torch.Tensor, probability: float = 0.85) -> torch.Tensor:
    """Apply transfer and room/mic perturbations to a whole batch."""

    batch, n_mels, n_frames = mel.shape
    active = torch.rand(batch, 1, 1, device=mel.device) < probability
    gain = torch.empty(batch, 1, 1, device=mel.device).uniform_(-7.0, 4.0)
    knots = torch.randn(batch, 1, 8, device=mel.device) * 2.5
    equalization = F.interpolate(
        knots, size=n_mels, mode="linear", align_corners=True
    ).transpose(1, 2)
    noise_scale = torch.empty(batch, 1, 1, device=mel.device).uniform_(0.0, 1.8)
    breath = torch.randn(batch, 8, n_frames, device=mel.device)
    breath = F.interpolate(
        breath.transpose(1, 2), size=n_mels, mode="linear", align_corners=False
    ).transpose(1, 2)
    breath = breath * torch.empty(batch, 1, 1, device=mel.device).uniform_(0.0, 1.2)
    augmented = mel + gain + equalization + torch.randn_like(mel) * noise_scale + breath

    if n_frames > 2:
        smoothed = augmented.clone()
        smoothed[:, :, 1:-1] = (
            0.20 * augmented[:, :, :-2]
            + 0.60 * augmented[:, :, 1:-1]
            + 0.20 * augmented[:, :, 2:]
        )
        # Longer smear approximates room response.
        if n_frames > 6:
            decay = 0.35 * smoothed[:, :, :-2] + 0.65 * smoothed[:, :, 2:]
            smoothed[:, :, 2:] = 0.7 * smoothed[:, :, 2:] + 0.3 * decay
        smooth = torch.rand(batch, 1, 1, device=mel.device) < 0.45
        augmented = torch.where(smooth, smoothed, augmented)

    t = torch.arange(n_frames, device=mel.device).view(1, 1, n_frames)
    vibrato = 1.5 * torch.sin(
        2 * torch.pi * t * torch.empty(batch, 1, 1, device=mel.device).uniform_(4.0, 7.0)
        / max(n_frames, 1)
    )
    drift = torch.linspace(0, 1, n_frames, device=mel.device).view(1, 1, n_frames)
    drift = drift * torch.empty(batch, 1, 1, device=mel.device).uniform_(-2.0, 2.0)
    if torch.rand(1, device=mel.device) < 0.45:
        shift = int(torch.randint(-2, 3, ()).item())
        if shift:
            augmented = torch.roll(augmented, shifts=shift, dims=1)
    augmented = augmented + vibrato + drift
    augmented = torch.tanh(augmented / 40.0) * 40.0

    mel_positions = torch.arange(n_mels, device=mel.device).view(1, n_mels, 1)
    mel_width = torch.randint(
        1, max(2, n_mels // 16 + 1), (batch, 1, 1), device=mel.device
    )
    mel_start = (
        torch.rand(batch, 1, 1, device=mel.device) * (n_mels - mel_width)
    ).long()
    mask_mel = (
        active
        & (torch.rand(batch, 1, 1, device=mel.device) < 0.45)
        & (mel_positions >= mel_start)
        & (mel_positions < mel_start + mel_width)
    )
    augmented = augmented.masked_fill(mask_mel, -80.0)

    if n_frames > 8:
        frame_positions = torch.arange(n_frames, device=mel.device).view(
            1, 1, n_frames
        )
        frame_width = torch.randint(
            1, max(2, n_frames // 24 + 1), (batch, 1, 1), device=mel.device
        )
        frame_start = (
            torch.rand(batch, 1, 1, device=mel.device) * (n_frames - frame_width)
        ).long()
        mask_frame = (
            active
            & (torch.rand(batch, 1, 1, device=mel.device) < 0.35)
            & (frame_positions >= frame_start)
            & (frame_positions < frame_start + frame_width)
        )
        augmented = augmented.masked_fill(mask_frame, -80.0)

    return torch.where(active, augmented.clamp(-100.0, 20.0), mel)


def _warp_time_batch(batch: dict[str, torch.Tensor], max_ratio: float = 0.05) -> None:
    """Mild tempo change that warps both the spectrogram and timing labels."""

    mel = batch.get("mel")
    if mel is None or mel.size(-1) < 16:
        return
    if torch.rand(1, device=mel.device) > 0.35:
        return
    ratio = 1.0 + float((2 * torch.rand(()) - 1.0) * max_ratio)
    frames = mel.size(-1)
    new_frames = max(8, int(round(frames * ratio)))
    warped_mel = F.interpolate(mel, size=new_frames, mode="linear", align_corners=False)
    batch["mel"] = F.interpolate(
        warped_mel, size=frames, mode="linear", align_corners=False
    )
    if "f0" in batch:
        warped_f0 = F.interpolate(batch["f0"], size=new_frames, mode="nearest")
        batch["f0"] = F.interpolate(warped_f0, size=frames, mode="nearest")
    for key in ("voiced", "onset", "offset", "cents", "frame_mask"):
        if key not in batch:
            continue
        value = batch[key].float().unsqueeze(1)
        warped = F.interpolate(value, size=new_frames, mode="nearest")
        restored = F.interpolate(warped, size=frames, mode="nearest").squeeze(1)
        batch[key] = restored if key != "frame_mask" else restored.bool()
    if "pitch" in batch:
        pitch = batch["pitch"].float().unsqueeze(1)
        warped = F.interpolate(pitch, size=new_frames, mode="nearest")
        batch["pitch"] = (
            F.interpolate(warped, size=frames, mode="nearest").squeeze(1).long()
        )


@torch.no_grad()
def collect_validation_probabilities(
    model: NoteFrameNet,
    examples: list[TranscriptionExample],
    device: torch.device,
    cfg: NoteTrainConfig,
) -> list[tuple[dict[str, np.ndarray], list[tuple[int, float, float]]]]:
    model.eval()
    cached = []
    from .data import load_fine_pitch

    selected = examples[: cfg.max_val_clips or None]
    for example in selected:
        mel = np.load(example.sample_dir / "performance_mel.npy", mmap_mode="r")
        arr = np.asarray(mel)
        probs = infer_probabilities(
            model,
            arr,
            device,
            window_frames=cfg.infer_window_frames,
            overlap_frames=cfg.infer_overlap_frames,
            batch_size=cfg.infer_batch_size,
            f0=load_fine_pitch(example.sample_dir, int(arr.shape[-1])),
        )
        cached.append((probs, load_written_notes_with_cents(example.sample_dir)))
    return cached


def score_cached(
    cached: list[tuple[dict[str, np.ndarray], list[tuple[int, float, float]]]],
    model_cfg: NoteFrameNetConfig,
    decode_cfg: DecodeConfig,
) -> dict:
    rows = []
    for probabilities, target in cached:
        predicted = decode_notes(
            probabilities, midi_min=model_cfg.midi_min, config=decode_cfg
        )
        rows.append(evaluate_note_lists(predicted, target))
    return _aggregate(rows)


def calibrate_decoder(
    cached: list[tuple[dict[str, np.ndarray], list[tuple[int, float, float]]]],
    model_cfg: NoteFrameNetConfig,
) -> tuple[DecodeConfig, dict]:
    """Tune only decoding thresholds on held-out clips."""

    best_cfg = DecodeConfig()
    best_metrics = score_cached(cached, model_cfg, best_cfg)

    def _rank(metrics: dict) -> tuple:
        ratio = float(metrics.get("pred_target_ratio") or 0.0)
        offset = (
            metrics["offset_mae_sec"]
            if metrics["offset_mae_sec"] is not None
            else float("inf")
        )
        ratio_penalty = abs(float(np.log(max(ratio, 1e-3))))
        if ratio > 1.35 or ratio < 0.75:
            ratio_penalty += 0.25
        return (metrics["f1"] - 0.08 * ratio_penalty, -offset)

    best_rank = _rank(best_metrics)
    for voiced in (0.45, 0.55, 0.65):
        for onset in (0.40, 0.50, 0.60):
            for offset in (0.40, 0.50, 0.60):
                for pitch_change_frames in (6, 8, 12):
                    candidate = DecodeConfig(
                        voiced_threshold=voiced,
                        onset_threshold=onset,
                        offset_threshold=offset,
                        pitch_change_frames=pitch_change_frames,
                    )
                    metrics = score_cached(cached, model_cfg, candidate)
                    rank = _rank(metrics)
                    if rank > best_rank:
                        best_cfg, best_metrics, best_rank = candidate, metrics, rank
    return best_cfg, best_metrics


def _save_checkpoint(
    path: Path,
    model: NoteFrameNet,
    cfg: NoteTrainConfig,
    epoch: int,
    metrics: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    calibration: DecodeConfig | None = None,
) -> None:
    train_config = {
        **asdict(cfg),
        "output_dir": str(cfg.output_dir),
        "resume_from": str(cfg.resume_from) if cfg.resume_from else None,
        "model": cfg.model.to_dict(),
    }
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": cfg.model.to_dict(),
            "train_config": train_config,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "calibration": (calibration or DecodeConfig()).to_dict(),
        },
        path,
    )


def train_note_transcriber(
    roots: Iterable[Path | str],
    split_manifest: Path | str | None,
    config: NoteTrainConfig | None = None,
) -> Path:
    """Train from cached ALIGN mels and MIDI labels across multiple roots."""

    cfg = config or NoteTrainConfig()
    roots = list(roots)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = _device(cfg.device)

    train_examples = load_split(
        roots,
        split_manifest,
        "train",
        seed=cfg.seed,
        val_fraction=cfg.val_fraction,
    )
    val_examples = load_split(
        roots,
        split_manifest,
        "val",
        seed=cfg.seed,
        val_fraction=cfg.val_fraction,
    )
    if not train_examples:
        raise ValueError("Training split contains no valid examples")
    if not val_examples:
        raise ValueError("Validation split contains no valid examples")
    val_examples = sorted(val_examples, key=lambda item: item.sample_id)
    train_data = NoteCropDataset(
        train_examples,
        crop_frames=cfg.crop_frames,
        midi_min=cfg.model.midi_min,
        midi_max=cfg.model.midi_max,
        training=True,
        augment=not cfg.gpu_augment,
        crops_per_clip=cfg.crops_per_clip,
    )
    loader_options = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if cfg.num_workers > 0:
        loader_options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": max(1, cfg.prefetch_factor),
            }
        )
    loader = DataLoader(train_data, **loader_options)
    model = NoteFrameNet(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "best.pt"
    last_path = cfg.output_dir / "last.pt"
    history_path = cfg.output_dir / "history.json"
    history: list[dict] = []
    start_epoch = 1
    best_f1 = -1.0
    patience_best = -1.0
    stale = 0
    stopped_reason = "max_epochs"

    if cfg.resume_from is not None:
        resume_path = Path(cfg.resume_from)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        completed_epoch = int(payload.get("epoch", 0))
        start_epoch = completed_epoch + 1
        if history_path.exists():
            prior = json.loads(history_path.read_text(encoding="utf-8"))
            history = [
                row
                for row in prior.get("history", [])
                if int(row.get("epoch", 0)) <= completed_epoch
            ]
        for row in history:
            score = float(row.get("val_f1", -1.0))
            best_f1 = max(best_f1, score)
            if score > patience_best + cfg.min_f1_delta:
                patience_best = score
                stale = 0
            else:
                stale += 1
        print(
            f"resumed={resume_path} completed_epoch={completed_epoch} "
            f"next_epoch={start_epoch}",
            flush=True,
        )

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        running = {
            key: torch.zeros((), device=device)
            for key in ("loss", "voiced", "pitch", "onset", "offset", "cents")
        }
        steps = 0
        for batch in loader:
            batch_dev = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if cfg.gpu_augment:
                batch_dev["mel"] = augment_mel_batch(batch_dev["mel"])
                _warp_time_batch(batch_dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(batch_dev["mel"], batch_dev.get("f0"))
                loss, parts = note_frame_loss(outputs, batch_dev)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for key in running:
                running[key] += parts[key]
            steps += 1

        cached = collect_validation_probabilities(model, val_examples, device, cfg)
        val_metrics = score_cached(cached, cfg.model, DecodeConfig())
        row = {
            "epoch": epoch,
            **{
                f"train_{key}": float(value.item()) / max(steps, 1)
                for key, value in running.items()
            },
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        _save_checkpoint(
            last_path, model, cfg, epoch, val_metrics, optimizer, scaler
        )
        # Component diagnostic only: acoustic onset F1 cannot promote a model.
        if val_metrics["f1"] > best_f1 or not best_path.exists():
            best_f1 = float(val_metrics["f1"])
            _save_checkpoint(
                best_path, model, cfg, epoch, val_metrics, optimizer, scaler
            )
        if val_metrics["f1"] > patience_best + cfg.min_f1_delta:
            patience_best = float(val_metrics["f1"])
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"onset_mae={val_metrics['onset_mae_sec']} "
            f"offset_mae={val_metrics['offset_mae_sec']}",
            flush=True,
        )
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
            ),
            encoding="utf-8",
        )
        if epoch >= cfg.early_stop_min_epochs and stale >= cfg.early_stop_patience:
            stopped_reason = "validation_f1_patience"
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    calibrate_limit = cfg.max_val_clips
    cfg.max_val_clips = cfg.calibrate_val_clips or None
    cached = collect_validation_probabilities(model, val_examples, device, cfg)
    cfg.max_val_clips = calibrate_limit
    calibration, calibrated_metrics = calibrate_decoder(cached, cfg.model)
    best_payload["calibration"] = calibration.to_dict()
    best_payload["calibrated_metrics"] = calibrated_metrics
    torch.save(best_payload, best_path)
    history_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_f1": best_f1,
                "stopped_reason": stopped_reason,
                "calibration": calibration.to_dict(),
                "calibrated_metrics": calibrated_metrics,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "roots": [str(root) for root in roots],
                "manifest": str(split_manifest) if split_manifest else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_path
