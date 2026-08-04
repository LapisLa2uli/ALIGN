"""
Binary classifier: clean vs corrupted clarinet sound.
Simpler task to establish upper bound on mel-CNN approach.
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
from sklearn.metrics import classification_report, f1_score
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
EPOCHS = 40
SEED = 42

CLARINET_DIR = Path("external_data/tinysol/Winds/Clarinet_Bb/ordinario")


def _audio_to_mel(y: np.ndarray) -> np.ndarray:
    if len(y) > int(MAX_SEC * SR):
        y = y[: int(MAX_SEC * SR)]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN)
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if log_mel.shape[1] < MAX_FRAMES:
        pad = np.full((N_MELS, MAX_FRAMES - log_mel.shape[1]), -80.0, dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)
    else:
        log_mel = log_mel[:, :MAX_FRAMES]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel


def build_dataset() -> list[dict]:
    wav_files = sorted(CLARINET_DIR.glob("*.wav"))
    log.info("Found %d clarinet WAV files", len(wav_files))
    manifest = []
    rng = random.Random(SEED)

    corruption_fns = [
        ("pitch_big", lambda y, r: librosa.effects.pitch_shift(y=y, sr=SR, n_steps=r.choice([-3, -2, 2, 3]))),
        ("pitch_small", lambda y, r: librosa.effects.pitch_shift(y=y, sr=SR, n_steps=r.choice([-0.5, 0.5]))),
        ("noise", lambda y, r: y + r.uniform(0.05, 0.2) * np.random.RandomState(r.randint(0, 99999)).randn(len(y)).astype(np.float32)),
        ("time_stretch", lambda y, r: librosa.effects.time_stretch(y=y, rate=r.choice([0.6, 0.7, 1.4, 1.6]))),
        ("clip", lambda y, r: np.clip(y * r.uniform(3.0, 6.0), -1.0, 1.0)),
    ]

    for wav_path in wav_files:
        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        if len(y) < SR * 0.3:
            continue

        # 1 clean
        manifest.append({"mel": _audio_to_mel(y), "label": 0})

        # 2 random corruptions per file -> more bad samples for balance
        for _ in range(2):
            name, fn = rng.choice(corruption_fns)
            corrupted = fn(y.copy(), rng)
            manifest.append({"mel": _audio_to_mel(corrupted), "label": 1})

    log.info("Total: %d (good=%d, bad=%d)",
             len(manifest),
             sum(1 for m in manifest if m["label"] == 0),
             sum(1 for m in manifest if m["label"] == 1))
    return manifest


class MelDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return torch.from_numpy(self.items[idx]["mel"]).unsqueeze(0), torch.tensor(self.items[idx]["label"])


class MelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(128 * 16, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    manifest = build_dataset()
    labels = [m["label"] for m in manifest]
    train_m, test_m = train_test_split(manifest, test_size=0.3, stratify=labels, random_state=SEED)
    val_m, test_m = train_test_split(test_m, test_size=0.5, stratify=[m["label"] for m in test_m], random_state=SEED)
    log.info("Split: train=%d val=%d test=%d", len(train_m), len(val_m), len(test_m))

    train_labels = [m["label"] for m in train_m]
    cc = Counter(train_labels)
    sampler = WeightedRandomSampler([1.0 / cc[l] for l in train_labels], len(train_labels))

    train_loader = DataLoader(MelDataset(train_m), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(MelDataset(val_m), batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(MelDataset(test_m), batch_size=BATCH_SIZE, num_workers=0)

    model = MelCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_f1 = 0.0
    out = Path("training/checkpoints")
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_corr, t_n = 0.0, 0, 0
        for mel, lbl in train_loader:
            mel, lbl = mel.to(device), lbl.to(device)
            opt.zero_grad()
            logits = model(mel)
            loss = F.cross_entropy(logits, lbl)
            loss.backward()
            opt.step()
            t_loss += loss.item() * mel.size(0)
            t_corr += (logits.argmax(1) == lbl).sum().item()
            t_n += mel.size(0)
        sched.step()

        model.eval()
        v_preds, v_labels = [], []
        v_loss = 0.0
        with torch.no_grad():
            for mel, lbl in val_loader:
                mel, lbl = mel.to(device), lbl.to(device)
                logits = model(mel)
                v_loss += F.cross_entropy(logits, lbl).item() * mel.size(0)
                v_preds.extend(logits.argmax(1).cpu().tolist())
                v_labels.extend(lbl.cpu().tolist())
        v_acc = sum(p == l for p, l in zip(v_preds, v_labels)) / len(v_labels)
        v_f1 = f1_score(v_labels, v_preds, average="weighted", zero_division=0)

        mk = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), out / "best_binary.pt")
            mk = " *"
        if ep % 5 == 0 or ep == 1 or mk:
            log.info("Epoch %02d  t_loss=%.4f t_acc=%.3f | v_loss=%.4f v_acc=%.3f v_f1=%.3f%s",
                     ep, t_loss / t_n, t_corr / t_n, v_loss / len(v_labels), v_acc, v_f1, mk)

    # Test
    model.load_state_dict(torch.load(out / "best_binary.pt", map_location=device))
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for mel, lbl in test_loader:
            mel, lbl = mel.to(device), lbl.to(device)
            logits = model(mel)
            preds.extend(logits.argmax(1).cpu().tolist())
            true.extend(lbl.cpu().tolist())

    print(f"\n===== Test Results =====")
    print(f"Accuracy: {sum(p==l for p,l in zip(preds,true))/len(true):.3f}")
    print(f"F1: {f1_score(true, preds, average='weighted'):.3f}\n")
    print(classification_report(true, preds, target_names=["good", "bad"], zero_division=0))


if __name__ == "__main__":
    main()
