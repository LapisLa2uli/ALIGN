"""
Train clarinet quality classifier on Good-sounds real annotations.

1688 clarinet recordings with 40+ quality labels from professional annotators.
We map these to binary (good/bad) and multi-class (error type) tasks.

Usage:
  python training/train_good_sounds.py                    # binary (good vs bad)
  python training/train_good_sounds.py --task multiclass  # 6-class error type
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
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

SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 30.0
MAX_SEC = 4.0
MAX_FRAMES = int(MAX_SEC * SR / HOP) + 1
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
SEED = 42

DB_PATH = Path("external_data/good_sounds/good-sounds/database.sqlite")
AUDIO_ROOT = Path("external_data/good_sounds/good-sounds/sound_files")

# ── label mapping ───────────────────────────────────────────────────

# Multi-class: group 40+ fine-grained labels into 6 categories
MULTICLASS_MAP = {
    # 0 = good
    "good-sound": 0,
    "good-attack-accent>": 0,
    "good-attack-accent^": 0,
    "good-attack-apoyo": 0,
    # 1 = dynamics problem
    "bad-dynamics-stability-crescendo": 1,
    "bad-dynamics-stability-decrescendo": 1,
    "bad-dynamics-stability-tremolo": 1,
    "bad-dynamics-stability-errors": 1,
    "bad-dynamic-stability": 1,
    # 2 = pitch problem
    "bad-pitch-stability-vibrato": 2,
    "bad-pitch-stability-errors": 2,
    "bad-pitch-stability": 2,
    # 3 = timbre problem
    "bad-timbre-stability": 3,
    "bad-timbre-stability-errors": 3,
    "bad-richness-pato": 3,
    "bad-richness-gaita": 3,
    "bad-richness": 3,
    # 4 = attack problem
    "bad-attack-tongue-block": 4,
    "bad-attack-too-strong": 4,
    "bad-attack-multiphonic": 4,
    "bad-attack-gallo": 4,
    "bad-attack-air": 4,
    "bad-attack": 4,
    # 5 = air/noise
    "air-outside": 5,
    "air-inside": 5,
}

MULTICLASS_NAMES = ["good", "dynamics", "pitch", "timbre", "attack", "air"]

# Binary: good vs bad
def _binary_label(klass: str) -> int:
    return 0 if klass.startswith("good") else 1


# ── data loading ────────────────────────────────────────────────────
def load_manifest(task: str) -> list[dict]:
    db = sqlite3.connect(str(DB_PATH))
    cur = db.cursor()

    # Pack name map
    cur.execute("SELECT id, name FROM packs")
    pack_map = {row[0]: row[1] for row in cur.fetchall()}

    # Clarinet sounds (skip scales for now - they're multi-note)
    cur.execute(
        "SELECT id, pack_id, pack_filename, klass FROM sounds "
        "WHERE instrument='clarinet' AND klass NOT LIKE 'scale-%'"
    )
    rows = cur.fetchall()
    db.close()

    manifest = []
    skipped = 0

    for sid, pack_id, filename, klass in rows:
        if not klass:
            continue

        pack_name = pack_map.get(pack_id, str(pack_id))

        # Resolve audio path
        wav_path = None
        for subdir in ["", "iphone", "neumann", "earth"]:
            candidate = AUDIO_ROOT / pack_name / subdir / filename if subdir else AUDIO_ROOT / pack_name / filename
            if candidate.exists():
                wav_path = candidate
                break

        if wav_path is None:
            skipped += 1
            continue

        if task == "binary":
            label = _binary_label(klass)
        else:
            label = MULTICLASS_MAP.get(klass)
            if label is None:
                skipped += 1
                continue

        manifest.append({
            "path": str(wav_path),
            "label": label,
            "klass": klass,
            "player": pack_name.split("_")[1] if "_" in pack_name else "unknown",
        })

    if skipped:
        log.info("Skipped %d samples (unmapped klass or missing file)", skipped)
    return manifest


def _audio_to_mel(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_SEC)
    if len(y) < SR * 0.1:
        y = np.zeros(int(SR * 0.5), dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    if log_mel.shape[1] < MAX_FRAMES:
        pad = np.full((N_MELS, MAX_FRAMES - log_mel.shape[1]), -80.0, dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)
    else:
        log_mel = log_mel[:, :MAX_FRAMES]

    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel


# ── dataset ─────────────────────────────────────────────────────────
class GoodSoundsDataset(Dataset):
    def __init__(self, items: list[dict], cache: dict | None = None):
        self.items = items
        self.cache = cache if cache is not None else {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item["path"]

        if path not in self.cache:
            self.cache[path] = _audio_to_mel(path)

        mel = torch.from_numpy(self.cache[path]).unsqueeze(0)
        label = torch.tensor(item["label"], dtype=torch.long)
        return mel, label


# ── model ───────────────────────────────────────────────────────────
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
            nn.Linear(256 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


# ── training ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, lbl in loader:
        mel, lbl = mel.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = model(mel)
        loss = F.cross_entropy(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(1) == lbl).sum().item()
        total += mel.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    for mel, lbl in loader:
        mel, lbl = mel.to(device), lbl.to(device)
        logits = model(mel)
        total_loss += F.cross_entropy(logits, lbl).item() * mel.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbl.cpu().tolist())
    n = len(labels)
    acc = sum(p == l for p, l in zip(preds, labels)) / n if n else 0
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return total_loss / n, acc, f1, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | Task: %s", device, args.task)

    # Load data
    manifest = load_manifest(args.task)
    log.info("Loaded %d clarinet samples", len(manifest))

    if args.task == "binary":
        target_names = ["good", "bad"]
    else:
        target_names = MULTICLASS_NAMES

    labels = [m["label"] for m in manifest]
    dist = Counter(labels)
    log.info("Class distribution: %s", {target_names[k]: v for k, v in sorted(dist.items())})

    # Player-aware split: put one player entirely in test for generalization test
    players = set(m["player"] for m in manifest)
    log.info("Players: %s", players)

    # Stratified split (70/15/15)
    train_m, test_m = train_test_split(manifest, test_size=0.3, stratify=labels, random_state=SEED)
    test_labels = [m["label"] for m in test_m]
    val_m, test_m = train_test_split(test_m, test_size=0.5, stratify=test_labels, random_state=SEED)
    log.info("Split: train=%d val=%d test=%d", len(train_m), len(val_m), len(test_m))

    # Precompute mel cache for speed
    log.info("Extracting mel spectrograms (this may take a minute)...")
    cache = {}

    train_ds = GoodSoundsDataset(train_m, cache)
    val_ds = GoodSoundsDataset(val_m, cache)
    test_ds = GoodSoundsDataset(test_m, cache)

    # Weighted sampler
    train_labels = [m["label"] for m in train_m]
    cc = Counter(train_labels)
    sampler = WeightedRandomSampler([1.0 / cc[l] for l in train_labels], len(train_labels))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    n_classes = len(set(labels))
    model = MelCNN(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path("training/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for ep in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, device)
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        mk = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), out_dir / f"best_good_sounds_{args.task}.pt")
            mk = " *"

        if ep % 5 == 0 or ep == 1 or mk:
            log.info("Epoch %02d  t_loss=%.4f t_acc=%.3f | v_loss=%.4f v_acc=%.3f v_f1=%.3f%s",
                     ep, t_loss, t_acc, v_loss, v_acc, v_f1, mk)

    # Test
    log.info("\n===== Test Set Results (%s) =====", args.task)
    model.load_state_dict(torch.load(out_dir / f"best_good_sounds_{args.task}.pt", map_location=device))
    t_loss, t_acc, t_f1, preds, true = evaluate(model, test_loader, device)

    print(f"\nTest Accuracy: {t_acc:.3f}  |  Test F1: {t_f1:.3f}\n")
    print(classification_report(true, preds, target_names=target_names, zero_division=0))

    # Save results
    results = {
        "task": args.task,
        "test_accuracy": round(t_acc, 4),
        "test_f1": round(t_f1, 4),
        "n_train": len(train_m),
        "n_val": len(val_m),
        "n_test": len(test_m),
        "class_distribution": {target_names[k]: v for k, v in sorted(dist.items())},
    }
    with (out_dir / f"results_good_sounds_{args.task}.json").open("w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved")


if __name__ == "__main__":
    main()
