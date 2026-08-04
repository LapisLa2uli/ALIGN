"""
Train a clarinet sound quality classifier on Good-sounds data.

Two modes:
  1. good_sounds  – binary (good / bad) classification on single notes
  2. align_bundle – multi-label error detection on ALIGN pipeline output

Usage:
  python training/train_quality.py --mode good_sounds --data-root external_data/good_sounds
  python training/train_quality.py --mode align_bundle --data-root DataCreate/samples
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── hyperparameters ──────────────────────────────────────────────────
SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 30.0
MAX_DURATION_SEC = 4.0  # pad/truncate to this
MAX_FRAMES = int(MAX_DURATION_SEC * SR / HOP) + 1

BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 30
SEED = 42


# ── dataset: Good-sounds ────────────────────────────────────────────
def _label_is_good(klass: str) -> bool:
    return klass.startswith("good") or klass.startswith("scale-good")


def _load_good_sounds_manifest(data_root: Path) -> list[dict]:
    """Build (audio_path, label) pairs from Good-sounds metadata."""
    sounds_path = data_root / "sounds.json"
    if not sounds_path.exists():
        raise FileNotFoundError(f"sounds.json not found in {data_root}")

    with sounds_path.open() as f:
        sounds = json.load(f)

    # Find the extracted audio directory
    audio_root = data_root / "good-sounds"
    if not audio_root.exists():
        # Try alternative layout
        audio_root = data_root / "audio"
    if not audio_root.exists():
        # Maybe the zip hasn't been extracted yet; look for pack dirs directly
        audio_root = data_root

    manifest = []
    for sid, meta in sounds.items():
        instrument = (meta.get("instrument") or "").lower()
        if instrument != "clarinet":
            continue
        klass = meta.get("klass", "")
        if not klass:
            continue

        pack_id = meta.get("pack_id")
        filename = meta.get("pack_filename")
        if pack_id is None or filename is None:
            continue

        # Good-sounds packs are in directories named by pack_id
        wav_path = audio_root / str(pack_id) / filename
        if not wav_path.exists():
            # Try zero-padded pack name
            wav_path = audio_root / f"{pack_id:04d}" / filename
        if not wav_path.exists():
            continue

        label = 0 if _label_is_good(klass) else 1  # 0=good, 1=bad
        manifest.append({
            "path": str(wav_path),
            "label": label,
            "klass": klass,
            "sound_id": sid,
        })

    return manifest


# ── dataset: ALIGN bundles ──────────────────────────────────────────
ERROR_TYPES = [
    "wrong_note", "intonation_error", "rhythm_error",
    "missed_note", "extra_note",
]


def _load_align_manifest(data_root: Path) -> list[dict]:
    """Build (mel_path, multi-hot label) pairs from ALIGN bundle output."""
    manifest = []
    for sample_dir in sorted(data_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        mel_path = sample_dir / "performance_mel.npy"
        labels_path = sample_dir / "labels.json"
        if not mel_path.exists() or not labels_path.exists():
            continue

        with labels_path.open() as f:
            doc = json.load(f)

        # Multi-hot vector for error types present
        error_vec = [0] * len(ERROR_TYPES)
        for lbl in doc.get("labels", []):
            ltype = lbl.get("type", "")
            if ltype in ERROR_TYPES:
                error_vec[ERROR_TYPES.index(ltype)] = 1

        # Also provide a binary "has_error" label
        has_error = 1 if any(error_vec) else 0

        manifest.append({
            "path": str(mel_path),
            "label": has_error,
            "error_vec": error_vec,
            "sample_id": sample_dir.name,
        })

    return manifest


# ── mel extraction ──────────────────────────────────────────────────
def extract_mel(audio_path: str) -> np.ndarray:
    """Load audio and compute log-mel spectrogram, padded/truncated to MAX_FRAMES."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True, duration=MAX_DURATION_SEC)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to fixed width
    n_frames = log_mel.shape[1]
    if n_frames < MAX_FRAMES:
        pad = np.full((N_MELS, MAX_FRAMES - n_frames), log_mel.min(), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)
    else:
        log_mel = log_mel[:, :MAX_FRAMES]

    return log_mel.astype(np.float32)


# ── PyTorch dataset ─────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, manifest: list[dict], precomputed_mel: bool = False):
        self.manifest = manifest
        self.precomputed_mel = precomputed_mel

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        if self.precomputed_mel:
            mel = np.load(item["path"]).astype(np.float32)
            n_frames = mel.shape[1]
            if n_frames < MAX_FRAMES:
                pad = np.full((mel.shape[0], MAX_FRAMES - n_frames), mel.min(), dtype=np.float32)
                mel = np.concatenate([mel, pad], axis=1)
            else:
                mel = mel[:, :MAX_FRAMES]
        else:
            mel = extract_mel(item["path"])

        # Normalize to [0, 1] range roughly
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, n_frames)
        label = torch.tensor(item["label"], dtype=torch.long)
        return tensor, label


# ── model ───────────────────────────────────────────────────────────
class MelCNN(nn.Module):
    """Simple CNN for mel spectrogram classification."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── training loop ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(mel)
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += mel.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        logits = model(mel)
        loss = F.cross_entropy(logits, label)
        total_loss += loss.item() * mel.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(label.cpu().tolist())
    n = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n if n else 0
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / n, acc, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train clarinet quality classifier")
    parser.add_argument("--mode", choices=["good_sounds", "align_bundle"], default="good_sounds")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output", type=str, default="training/checkpoints")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    data_root = Path(args.data_root)
    precomputed = False

    if args.mode == "good_sounds":
        manifest = _load_good_sounds_manifest(data_root)
        log.info("Good-sounds clarinet: %d samples", len(manifest))
    else:
        manifest = _load_align_manifest(data_root)
        precomputed = True
        log.info("ALIGN bundles: %d samples", len(manifest))

    if len(manifest) < 10:
        log.error("Not enough data (%d samples). Need at least 10.", len(manifest))
        sys.exit(1)

    # Class distribution
    labels = [m["label"] for m in manifest]
    dist = Counter(labels)
    log.info("Class distribution: %s", dict(dist))

    # Stratified split: 70/15/15
    train_m, test_m = train_test_split(manifest, test_size=0.3, stratify=labels, random_state=SEED)
    test_labels = [m["label"] for m in test_m]
    val_m, test_m = train_test_split(test_m, test_size=0.5, stratify=test_labels, random_state=SEED)
    log.info("Split: train=%d, val=%d, test=%d", len(train_m), len(val_m), len(test_m))

    train_ds = AudioDataset(train_m, precomputed_mel=precomputed)
    val_ds = AudioDataset(val_m, precomputed_mel=precomputed)
    test_ds = AudioDataset(test_m, precomputed_mel=precomputed)

    # Weighted sampler for class imbalance
    train_labels = [m["label"] for m in train_m]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_classes = len(set(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s, Classes: %d", device, n_classes)

    model = MelCNN(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        log.info(
            "Epoch %02d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f val_acc=%.3f val_f1=%.3f",
            epoch, train_loss, train_acc, val_loss, val_acc, val_f1,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "n_classes": n_classes,
                "mode": args.mode,
            }, ckpt_path)
            log.info("  -> Saved best model (val_f1=%.3f)", val_f1)

    # Final test evaluation
    log.info("\n=== Test Set Evaluation ===")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_f1, preds, true_labels = evaluate(model, test_loader, device)
    log.info("Test loss=%.4f  acc=%.3f  f1=%.3f", test_loss, test_acc, test_f1)

    if args.mode == "good_sounds":
        target_names = ["good", "bad"]
    else:
        target_names = ["no_error", "has_error"]
    print("\n" + classification_report(true_labels, preds, target_names=target_names, zero_division=0))

    # Save results
    results = {
        "mode": args.mode,
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "test_loss": round(test_loss, 4),
        "n_train": len(train_m),
        "n_val": len(val_m),
        "n_test": len(test_m),
        "best_epoch": ckpt["epoch"],
        "class_distribution": dict(dist),
    }
    results_path = out_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
