"""
Quick proof-of-concept: train a clarinet quality classifier using TinySOL data.

- "Good" samples: original TinySOL clarinet recordings (clean, professional)
- "Bad" samples: synthetically corrupted versions (pitch-shift, noise, distortion)

This validates:
  1. The mel-CNN architecture works
  2. Synthetic corruption is distinguishable from clean performance
  3. The training pipeline end-to-end

Usage:
  python training/train_tinysol.py
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────
SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 30.0
MAX_SEC = 4.0
MAX_FRAMES = int(MAX_SEC * SR / HOP) + 1

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 40
SEED = 42

CLARINET_DIR = Path("external_data/tinysol/Winds/Clarinet_Bb/ordinario")

# Error types to simulate (maps to ALIGN taxonomy)
CORRUPTION_TYPES = {
    "clean": 0,           # good performance
    "wrong_note": 1,      # pitch shifted
    "intonation": 2,      # slight detuning
    "rhythm_error": 3,    # time-stretched
    "noisy": 4,           # added noise (bad tone)
}
N_CLASSES = len(CORRUPTION_TYPES)


# ── synthetic corruption functions ──────────────────────────────────
def corrupt_pitch_shift(y: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Shift pitch by 1-4 semitones (simulates wrong note)."""
    semitones = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)


def corrupt_intonation(y: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Slight detuning: 0.3-0.8 semitones (simulates intonation error)."""
    cents = rng.choice([-0.8, -0.5, -0.3, 0.3, 0.5, 0.8])
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=cents)


def corrupt_rhythm(y: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Time-stretch to simulate rhythm error."""
    rate = rng.choice([0.6, 0.7, 1.4, 1.6])
    return librosa.effects.time_stretch(y=y, rate=rate)


def corrupt_noise(y: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Add noise / distortion (simulates bad tone quality)."""
    noise_level = rng.uniform(0.05, 0.15)
    noise = np.random.RandomState(rng.randint(0, 99999)).randn(len(y)).astype(np.float32)
    return y + noise_level * noise


CORRUPTION_FNS = {
    "wrong_note": corrupt_pitch_shift,
    "intonation": corrupt_intonation,
    "rhythm_error": corrupt_rhythm,
    "noisy": corrupt_noise,
}


# ── dataset building ────────────────────────────────────────────────
def build_dataset() -> list[dict]:
    """Build manifest: for each clean file, create 1 clean + 4 corrupted versions."""
    wav_files = sorted(CLARINET_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {CLARINET_DIR}")

    log.info("Found %d clarinet WAV files", len(wav_files))

    manifest = []
    rng = random.Random(SEED)

    for wav_path in wav_files:
        # Load once
        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        if len(y) < SR * 0.3:  # skip very short files
            continue

        # Clean version
        mel = _audio_to_mel(y)
        manifest.append({"mel": mel, "label": 0, "type": "clean", "file": wav_path.name})

        # Corrupted versions (one per corruption type)
        for ctype, label in CORRUPTION_TYPES.items():
            if ctype == "clean":
                continue
            fn = CORRUPTION_FNS[ctype]
            corrupted = fn(y.copy(), SR, rng)
            mel_c = _audio_to_mel(corrupted)
            manifest.append({"mel": mel_c, "label": label, "type": ctype, "file": wav_path.name})

    log.info("Total samples: %d", len(manifest))
    return manifest


def _audio_to_mel(y: np.ndarray) -> np.ndarray:
    """Compute normalized log-mel spectrogram, padded to MAX_FRAMES."""
    if len(y) > int(MAX_SEC * SR):
        y = y[: int(MAX_SEC * SR)]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Pad to fixed width
    if log_mel.shape[1] < MAX_FRAMES:
        pad = np.full((N_MELS, MAX_FRAMES - log_mel.shape[1]), -80.0, dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)
    else:
        log_mel = log_mel[:, :MAX_FRAMES]

    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel


# ── PyTorch ─────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel = torch.from_numpy(item["mel"]).unsqueeze(0)  # (1, N_MELS, T)
        label = torch.tensor(item["label"], dtype=torch.long)
        return mel, label


class MelCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── training ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
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
    total_loss = 0.0
    preds, labels = [], []
    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        logits = model(mel)
        loss = F.cross_entropy(logits, label)
        total_loss += loss.item() * mel.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(label.cpu().tolist())
    n = len(labels)
    acc = sum(p == l for p, l in zip(preds, labels)) / n if n else 0
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return total_loss / n, acc, f1, preds, labels


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Build dataset
    log.info("Building dataset with synthetic corruptions...")
    manifest = build_dataset()

    # Split
    all_labels = [m["label"] for m in manifest]
    dist = Counter(all_labels)
    log.info("Class distribution: %s", {list(CORRUPTION_TYPES.keys())[k]: v for k, v in sorted(dist.items())})

    train_m, test_m = train_test_split(manifest, test_size=0.3, stratify=all_labels, random_state=SEED)
    test_labels = [m["label"] for m in test_m]
    val_m, test_m = train_test_split(test_m, test_size=0.5, stratify=test_labels, random_state=SEED)
    log.info("Split: train=%d, val=%d, test=%d", len(train_m), len(val_m), len(test_m))

    # Weighted sampler
    train_labels = [m["label"] for m in train_m]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(MelDataset(train_m), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(MelDataset(val_m), batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(MelDataset(test_m), batch_size=BATCH_SIZE, num_workers=0)

    # Model
    model = MelCNN(N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    best_f1 = 0.0
    out_dir = Path("training/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), out_dir / "best_tinysol.pt")
            marker = " *"

        if epoch % 5 == 0 or epoch == 1 or marker:
            log.info(
                "Epoch %02d  train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f f1=%.3f%s",
                epoch, train_loss, train_acc, val_loss, val_acc, val_f1, marker,
            )

    # Test
    log.info("\n===== Test Set Results =====")
    model.load_state_dict(torch.load(out_dir / "best_tinysol.pt", map_location=device))
    test_loss, test_acc, test_f1, preds, true_labels = evaluate(model, test_loader, device)

    target_names = list(CORRUPTION_TYPES.keys())
    print(f"\nTest Accuracy: {test_acc:.3f}  |  Test F1: {test_f1:.3f}\n")
    print(classification_report(true_labels, preds, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels, preds)
    # Pretty print
    header = "        " + " ".join(f"{n[:7]:>8}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{target_names[i][:8]:>8} " + " ".join(f"{v:>8}" for v in row))

    log.info("\nModel saved to %s", out_dir / "best_tinysol.pt")


if __name__ == "__main__":
    main()
