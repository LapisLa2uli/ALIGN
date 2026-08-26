from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SCORE_CLASSES = ("match", "miss", "wrong", "rhythm", "intonation")

HOP_LENGTH = 512
SAMPLE_RATE = 22050
FRAME_HOP_SEC = HOP_LENGTH / SAMPLE_RATE


@dataclass
class ModelConfig:
    n_mels: int = 128
    d_model: int = 256
    nhead: int = 4
    audio_layers: int = 4
    score_layers: int = 4
    fusion_layers: int = 2
    dropout: float = 0.1
    audio_stride: int = 4
    max_audio_frames: int = 1920
    max_score_notes: int = 128
    num_score_classes: int = len(SCORE_CLASSES)
    error_loss_weight: float = 8.0
    extra_loss_weight: float = 4.0
    repeat_loss_weight: float = 2.0


@dataclass
class TrainConfig:
    data_root: Path = field(
        default_factory=lambda: Path("synth-pipeline/1000dataexport")
    )
    output_dir: Path = field(default_factory=lambda: Path("align-model/runs/rumaa-lite"))
    epochs: int = 20
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-2
    val_fraction: float = 0.1
    seed: int = 365
    num_workers: int = 0
    overfit: int = 0
    device: str = "cuda"
    grad_clip: float = 1.0
    log_every: int = 10
    model: ModelConfig = field(default_factory=ModelConfig)
