from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from alignmodel.config import FRAME_HOP_SEC
from alignmodel.dataset import list_sample_dirs
from alignmodel.device import device_label, resolve_device
from alignmodel.stages.gold import load_labels, overlaps, repetition_labs
from alignmodel.stages.models import (
    EDIT_CLASSES,
    EditCropNet,
    RestartScorer,
    RhythmNet,
    cosine_pair_loss,
)

CROP_FRAMES = 64
CROP_FRAMES_RESTART = 96
MEL_HOP = FRAME_HOP_SEC
NEG_PER_POS = 2
COSINE_PAIR_WEIGHT = 0.08
EARLY_STOP_PATIENCE = 8


def _load_mel(sample_dir: Path) -> np.ndarray:
    mel = np.load(sample_dir / "performance_mel.npy").astype(np.float32)
    if mel.ndim != 2:
        raise ValueError(f"bad mel {mel.shape} in {sample_dir}")
    if mel.shape[0] > mel.shape[1] and mel.shape[0] != 128:
        mel = mel.T
    if mel.shape[0] != 128:
        if mel.shape[0] > 128:
            mel = mel[:128]
        else:
            pad = np.zeros((128 - mel.shape[0], mel.shape[1]), dtype=np.float32)
            mel = np.concatenate([mel, pad], axis=0)
    return mel


def _pool_span(mel: np.ndarray, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(t0 / MEL_HOP))
    i1 = min(mel.shape[1], max(i0 + 1, int(np.ceil(t1 / MEL_HOP))))
    crop = mel[:, i0:i1]
    if crop.size == 0:
        return np.zeros(mel.shape[0], dtype=np.float32)
    return crop.mean(axis=1)


def _mel_crop(mel: np.ndarray, t0: float, t1: float, width: int = CROP_FRAMES) -> np.ndarray:
    mid = 0.5 * (t0 + t1)
    center = int(mid / MEL_HOP)
    half = width // 2
    i0 = center - half
    i1 = i0 + width
    out = np.zeros((mel.shape[0], width), dtype=np.float32)
    src0 = max(0, i0)
    src1 = min(mel.shape[1], i1)
    dst0 = src0 - i0
    dst1 = dst0 + (src1 - src0)
    if src1 > src0:
        out[:, dst0:dst1] = mel[:, src0:src1]
    return out


def _span_indices(mel: np.ndarray, t0: float, t1: float) -> tuple[int, int]:
    i0 = max(0, int(t0 / MEL_HOP))
    i1 = min(mel.shape[1], max(i0 + 1, int(np.ceil(t1 / MEL_HOP))))
    return i0, i1


def _mel_span_resize(mel: np.ndarray, t0: float, t1: float, width: int = CROP_FRAMES) -> np.ndarray:
    """Pack the actual labeled span into `width` frames so duration texture remains."""
    i0, i1 = _span_indices(mel, t0, t1)
    crop = mel[:, i0:i1]
    if crop.shape[1] == 0:
        return np.zeros((mel.shape[0], width), dtype=np.float32)
    if crop.shape[1] == width:
        return crop.astype(np.float32, copy=True)
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0)
    out = torch.nn.functional.interpolate(
        tensor, size=width, mode="linear", align_corners=False
    )
    return out.squeeze(0).numpy().astype(np.float32)


def _rhythm_aux(mel: np.ndarray, t0: float, t1: float) -> np.ndarray:
    i0, i1 = _span_indices(mel, t0, t1)
    crop = mel[:, i0:i1]
    dur = max(float(t1 - t0), 1e-3)
    if crop.size == 0:
        return np.array([np.log(dur), 0.0, 0.0, 0.0], dtype=np.float32)
    rms = float(np.sqrt(np.mean(crop * crop)))
    if crop.shape[1] > 2:
        flux = float(np.mean(np.abs(np.diff(crop, axis=1))))
        mid = crop.shape[1] // 2
        e0 = float(np.sqrt(np.mean(crop[:, :mid] * crop[:, :mid])) + 1e-6)
        e1 = float(np.sqrt(np.mean(crop[:, mid:] * crop[:, mid:])) + 1e-6)
        split = float(np.log(e1 / e0))
    else:
        flux, split = 0.0, 0.0
    return np.array([np.log(dur), rms, flux, split], dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mel_duration(sample_dir: Path) -> float:
    arr = np.load(sample_dir / "performance_mel.npy", mmap_mode="r")
    if arr.shape[0] == 128:
        t = int(arr.shape[1])
    elif arr.shape[1] == 128:
        t = int(arr.shape[0])
    else:
        t = int(max(arr.shape))
    return float(t) * MEL_HOP


def _clip_duration(sample_dir: Path, mel: np.ndarray | None = None) -> float:
    if mel is not None:
        t = mel.shape[1] if mel.shape[0] == 128 else mel.shape[0]
        return t * MEL_HOP
    return _mel_duration(sample_dir)


def _restart_items(dirs: list[Path], rng: random.Random) -> list[dict]:
    """Balanced copy-detection pairs. Positives = gold repeat vs its source."""
    items = []
    for sample in dirs:
        labels = load_labels(sample)
        reps = repetition_labs(labels)
        if not reps:
            continue
        dur = _mel_duration(sample)
        error_spans = [
            (float(lab["start_time"]), float(lab["end_time"]))
            for lab in labels
            if lab.get("type") == "repetition"
        ]
        for lab in reps:
            src = lab.get("repeats_label_range") or {}
            s0 = float(src.get("start_time", 0.0))
            s1 = float(src.get("end_time", s0 + 0.5))
            t0 = float(lab["start_time"])
            t1 = float(lab["end_time"])
            items.append(
                {"dir": sample, "t0": t0, "t1": t1, "s0": s0, "s1": s1, "y": 1.0}
            )
            length = max(0.35, t1 - t0)
            # Hard negative: continuation after the repeat (same length, not a copy).
            after = t1
            if after + length < dur - 0.05:
                items.append(
                    {
                        "dir": sample,
                        "t0": after,
                        "t1": after + length,
                        "s0": t0,
                        "s1": t1,
                        "y": 0.0,
                    }
                )
            # Hard negative: the measure before the source vs the restated measure.
            before = s0 - length
            if before >= 0.0:
                items.append(
                    {
                        "dir": sample,
                        "t0": t0,
                        "t1": t1,
                        "s0": before,
                        "s1": s0,
                        "y": 0.0,
                    }
                )
            # Random non-overlapping pair of the same length.
            for _ in range(max(0, NEG_PER_POS - 1)):
                start = rng.uniform(0.0, max(0.05, dur - 2 * length - 0.05))
                other = start + length + rng.uniform(0.15, 0.6)
                if other + length > dur:
                    continue
                if any(overlaps(start, start + length, a, b) for a, b in error_spans):
                    continue
                items.append(
                    {
                        "dir": sample,
                        "t0": other,
                        "t1": other + length,
                        "s0": start,
                        "s1": start + length,
                        "y": 0.0,
                    }
                )
    # Cross-clip negatives: two clarinet spans that are not copies of each other.
    pos = [it for it in items if it["y"] > 0.5]
    if len(pos) >= 2:
        for it in pos:
            other = rng.choice(pos)
            if other["dir"] == it["dir"]:
                continue
            items.append(
                {
                    "dir": it["dir"],
                    "dir_b": other["dir"],
                    "t0": it["t0"],
                    "t1": it["t1"],
                    "s0": other["s0"],
                    "s1": other["s1"],
                    "y": 0.0,
                }
            )
    return items


def _edit_items(dirs: list[Path], rng: random.Random) -> list[dict]:
    error_types = set(EDIT_CLASSES) - {"match"}
    items = []
    for sample in dirs:
        labels = load_labels(sample)
        dur = None
        used = []
        for lab in labels:
            kind = lab.get("type")
            if kind not in error_types:
                continue
            t0 = float(lab["start_time"])
            t1 = float(lab["end_time"])
            items.append({"dir": sample, "t0": t0, "t1": t1, "y": EDIT_CLASSES.index(kind)})
            used.append((t0, t1))
        if dur is None:
            dur = _clip_duration(sample)
        n_neg = max(1, min(3, len(used)))
        for _ in range(n_neg):
            length = rng.uniform(0.25, 0.9)
            start = rng.uniform(0.0, max(0.05, dur - length))
            end = start + length
            if any(overlaps(start, end, a, b) for a, b in used):
                continue
            items.append(
                {"dir": sample, "t0": start, "t1": end, "y": EDIT_CLASSES.index("match")}
            )
    return items


def _rhythm_items(dirs: list[Path], rng: random.Random) -> list[dict]:
    """Span-level rhythm crops from gold labels, not linear-mapped score notes."""
    items: list[dict] = []
    for sample in dirs:
        labels = load_labels(sample)
        rhythm_spans = [
            (float(lab["start_time"]), float(lab["end_time"]))
            for lab in labels
            if lab.get("type") == "rhythm_error"
        ]
        if not rhythm_spans:
            continue
        used = [
            (float(lab["start_time"]), float(lab["end_time"]))
            for lab in labels
            if lab.get("type") not in {"stylistic_choice"}
        ]
        dur = _mel_duration(sample)
        mel = None
        for t0, t1 in rhythm_spans:
            items.append({"dir": sample, "t0": t0, "t1": t1, "y": 1.0})
            if mel is None:
                mel = _load_mel(sample)
            pos_rms = float(_rhythm_aux(mel, t0, t1)[1])
            length = max(0.35, t1 - t0)
            after = t1
            if after + length < dur - 0.05 and not any(
                overlaps(after, after + length, a, b) for a, b in used if (a, b) != (t0, t1)
            ):
                items.append({"dir": sample, "t0": after, "t1": after + length, "y": 0.0})
            added = 0
            tries = 0
            while added < 2 and tries < 16:
                tries += 1
                nlen = rng.uniform(0.35, 1.4)
                start = rng.uniform(0.0, max(0.05, dur - nlen))
                end = start + nlen
                if any(overlaps(start, end, a, b) for a, b in used):
                    continue
                if float(_rhythm_aux(mel, start, end)[1]) < 0.35 * max(pos_rms, 1e-4):
                    continue
                items.append({"dir": sample, "t0": start, "t1": end, "y": 0.0})
                added += 1
    return items


class RestartDataset(Dataset):
    def __init__(self, items: list[dict], cache: _MelCache | None = None, augment: bool = False):
        self.items = items
        self.cache = cache
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel_a = self.cache.get(it["dir"]) if self.cache is not None else _load_mel(it["dir"])
        dir_b = it.get("dir_b", it["dir"])
        mel_b = (
            mel_a
            if dir_b == it["dir"]
            else (self.cache.get(dir_b) if self.cache is not None else _load_mel(dir_b))
        )
        j = random.uniform(-0.05, 0.05) if self.augment else 0.0
        crop_a = _mel_span_resize(mel_a, it["t0"] + j, it["t1"] + j, width=CROP_FRAMES_RESTART)
        crop_b = _mel_span_resize(mel_b, it["s0"] + j, it["s1"] + j, width=CROP_FRAMES_RESTART)
        if self.augment and random.random() < 0.5:
            crop_a, crop_b = crop_b, crop_a
        return (
            torch.from_numpy(crop_a),
            torch.from_numpy(crop_b),
            torch.tensor(it["y"], dtype=torch.float32),
        )


class EditDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel = _load_mel(it["dir"])
        crop = _mel_crop(mel, it["t0"], it["t1"])
        return torch.from_numpy(crop), torch.tensor(it["y"], dtype=torch.long)


class RhythmDataset(Dataset):
    def __init__(self, items: list[dict], cache: _MelCache | None = None, augment: bool = False):
        self.items = items
        self.cache = cache
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel = self.cache.get(it["dir"]) if self.cache is not None else _load_mel(it["dir"])
        j = random.uniform(-0.05, 0.05) if self.augment else 0.0
        crop = _mel_span_resize(mel, it["t0"] + j, it["t1"] + j)
        aux = _rhythm_aux(mel, it["t0"] + j, it["t1"] + j)
        return (
            torch.from_numpy(crop),
            torch.from_numpy(aux),
            torch.tensor(it["y"], dtype=torch.float32),
        )


@dataclass
class StageTrainConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cuda"
    seed: int = 365
    val_fraction: float = 0.1
    stages: tuple[int, ...] = (1, 2, 3)
    max_samples: int = 0


class _MelCache:
    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._data: dict[str, np.ndarray] = {}
        self._order: list[str] = []

    def get(self, sample_dir: Path) -> np.ndarray:
        key = str(sample_dir)
        hit = self._data.get(key)
        if hit is not None:
            return hit
        mel = _load_mel(sample_dir)
        if len(self._order) >= self.maxsize:
            old = self._order.pop(0)
            self._data.pop(old, None)
        self._data[key] = mel
        self._order.append(key)
        return mel


def _split(ds: Dataset, frac: float, seed: int):
    n_val = max(1, int(len(ds) * frac))
    n_train = len(ds) - n_val
    if n_train < 1:
        return ds, ds
    return random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))


def _split_by_sample(items: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    dirs = sorted({it["dir"] for it in items}, key=str)
    rng = random.Random(seed)
    rng.shuffle(dirs)
    n_val = max(1, int(round(len(dirs) * frac)))
    if len(dirs) > 1:
        n_val = min(n_val, len(dirs) - 1)
    val_dirs = set(dirs[:n_val])
    train = [it for it in items if it["dir"] not in val_dirs]
    val = [it for it in items if it["dir"] in val_dirs]
    if not train:
        train = list(val)
    if not val:
        val = list(train)
    return train, val


def _time_freq_mask(x: torch.Tensor) -> torch.Tensor:
    """Light SpecAugment on a (B, n_mels, T) crop batch."""
    _b, n_mels, n_t = x.shape
    x = x.clone()
    f = int(torch.randint(0, 8, (1,)).item())
    if f > 0:
        f0 = int(torch.randint(0, max(1, n_mels - f + 1), (1,)).item())
        x[:, f0 : f0 + f, :] = 0
    w = int(torch.randint(0, 8, (1,)).item())
    if w > 0:
        t0 = int(torch.randint(0, max(1, n_t - w + 1), (1,)).item())
        x[:, :, t0 : t0 + w] = 0
    return x


def _f1_from_logits(logits: np.ndarray, y: np.ndarray, thresh: float) -> tuple[float, float, float]:
    pred = logits > thresh
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return f1, prec, rec


def _average_precision(logits: np.ndarray, y: np.ndarray) -> float:
    if y.size == 0 or float(y.sum()) < 1:
        return 0.0
    order = np.argsort(-logits)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    prec = tp / np.maximum(tp + fp, 1.0)
    return float((prec * ys).sum() / ys.sum())


def _best_logit_threshold(logits: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Prefer a threshold with usable precision and recall."""
    candidates: list[tuple[float, float, float, float]] = []
    for t in np.concatenate(([0.0], np.linspace(-1.5, 1.5, 31))):
        f1, prec, rec = _f1_from_logits(logits, y, float(t))
        candidates.append((f1, float(t), prec, rec))
    usable = [c for c in candidates if c[2] >= 0.45 and c[3] >= 0.30]
    if not usable:
        usable = [c for c in candidates if c[2] >= 0.40]
    pool = usable or candidates
    best = max(pool, key=lambda c: (c[0], c[2], -abs(c[1])))
    return best[1], best[0], best[2], best[3]


def _forward_binary(model: nn.Module, batch) -> torch.Tensor:
    if len(batch) == 3:
        a, b, _y = batch
        return model(a, b)
    x, _y = batch
    return model(x)


def _run_binary(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: StageTrainConfig,
    out_path: Path,
    name: str,
    *,
    siamese: bool = False,
) -> Path:
    device = resolve_device(cfg.device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )
    best_f1 = -1.0
    best_ap = -1.0
    stale = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            y = batch[-1]
            opt.zero_grad(set_to_none=True)
            if siamese:
                a, b = _time_freq_mask(batch[0]), _time_freq_mask(batch[1])
                ha = model.encode(a)
                hb = model.encode(b)
                feat = torch.cat([ha, hb, (ha - hb).abs(), ha * hb], dim=-1)
                logits = model.head(feat).squeeze(-1)
                loss = bce(logits, y) + COSINE_PAIR_WEIGHT * cosine_pair_loss(ha, hb, y)
            elif len(batch) == 3:
                x = _time_freq_mask(batch[0])
                logits = model(x, batch[1])
                loss = bce(logits, y)
            else:
                x = _time_freq_mask(batch[0])
                logits = model(x)
                loss = bce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            n += 1
        sched.step()
        model.eval()
        val_loss = 0.0
        all_logits = []
        all_y = []
        with torch.no_grad():
            for batch in val_loader:
                batch_d = [x.to(device) for x in batch]
                y = batch_d[-1]
                if siamese:
                    logits = model(batch_d[0], batch_d[1])
                elif len(batch_d) == 3:
                    logits = model(batch_d[0], batch_d[1])
                else:
                    logits = model(batch_d[0])
                val_loss += float(bce(logits, y).detach())
                all_logits.append(logits.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
        val_loss /= max(len(val_loader), 1)
        logits_np = np.concatenate(all_logits) if all_logits else np.zeros(1)
        y_np = np.concatenate(all_y) if all_y else np.zeros(1)
        thresh, f1, prec, rec = _best_logit_threshold(logits_np, y_np)
        f1_at_0, p0, r0 = _f1_from_logits(logits_np, y_np, 0.0)
        ap = _average_precision(logits_np, y_np)
        pred = logits_np > thresh
        acc = float((pred == y_np).mean()) if y_np.size else 0.0
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_loss": val_loss,
            "acc": acc,
            "f1": f1,
            "f1_at_0": f1_at_0,
            "ap": ap,
            "precision": prec,
            "recall": rec,
            "logit_threshold": thresh,
        }
        history.append(row)
        print(
            f"{name} epoch {epoch} train={row['train_loss']:.4f} val={val_loss:.4f} "
            f"acc={acc:.3f} f1={f1:.3f} f1@0={f1_at_0:.3f} ap={ap:.3f} "
            f"p={prec:.3f} r={rec:.3f} thr={thresh:.2f} device={device_label(device)}"
        )
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": row,
            "logit_threshold": thresh,
            "arch": "siamese" if siamese else "crop_aux",
        }
        torch.save(ckpt, out_path.parent / f"{out_path.stem}_last.pt")
        improved = ap > best_ap + 1e-4 or (abs(ap - best_ap) <= 1e-4 and f1 >= best_f1)
        if improved:
            best_ap = max(best_ap, ap)
            best_f1 = max(best_f1, f1)
            stale = 0
            torch.save(ckpt, out_path)
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                print(
                    f"{name} early stop at epoch {epoch} best_f1={best_f1:.3f} best_ap={best_ap:.3f}"
                )
                break
    (out_path.parent / f"{out_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} best_f1={best_f1:.3f}")
    return out_path


def _run_multiclass(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: StageTrainConfig,
    out_path: Path,
    weights: torch.Tensor,
    name: str,
) -> Path:
    device = resolve_device(cfg.device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss(weight=weights.to(device))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    best = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n += 1
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        err_correct = 0
        n_err = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += float(ce(logits, y).detach())
                pred = logits.argmax(-1)
                correct += int((pred == y).sum())
                total += int(y.numel())
                err = y != 0
                err_correct += int(((pred == y) & err).sum())
                n_err += int(err.sum())
        val_loss /= max(len(val_loader), 1)
        acc = correct / max(total, 1)
        err_acc = err_correct / max(n_err, 1)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_loss": val_loss,
            "acc": acc,
            "error_acc": err_acc,
        }
        history.append(row)
        print(
            f"{name} epoch {epoch} train={row['train_loss']:.4f} val={val_loss:.4f} "
            f"acc={acc:.3f} error_acc={err_acc:.3f} device={device_label(device)}"
        )
        ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": row}
        torch.save(ckpt, out_path.parent / f"{out_path.stem}_last.pt")
        if val_loss <= best:
            best = val_loss
            torch.save(ckpt, out_path)
    (out_path.parent / f"{out_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path}")
    return out_path


def train_stages(cfg: StageTrainConfig) -> dict[str, Path]:
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    dirs = list_sample_dirs(cfg.data_root)
    if cfg.max_samples:
        dirs = dirs[: cfg.max_samples]
    if not dirs:
        raise FileNotFoundError(f"No bundles under {cfg.data_root}")
    print(f"samples={len(dirs)} out={cfg.output_dir} stages={cfg.stages}")
    written: dict[str, Path] = {}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if 1 in cfg.stages:
        items = _restart_items(dirs, rng)
        n_pos = sum(1 for i in items if i["y"] > 0.5)
        train_items, val_items = _split_by_sample(items, cfg.val_fraction, cfg.seed)
        print(
            f"stage1 examples={len(items)} pos={n_pos} "
            f"train={len(train_items)} val={len(val_items)}"
        )
        cache = _MelCache(maxsize=768)
        written["stage1"] = _run_binary(
            RestartScorer(),
            RestartDataset(train_items, cache=cache, augment=True),
            RestartDataset(val_items, cache=cache, augment=False),
            cfg,
            cfg.output_dir / "stage1.pt",
            name="stage1",
            siamese=True,
        )

    if 2 in cfg.stages:
        items = _edit_items(dirs, rng)
        counts = [0] * len(EDIT_CLASSES)
        for it in items:
            counts[it["y"]] += 1
        print(f"stage2 examples={len(items)} counts={dict(zip(EDIT_CLASSES, counts))}")
        ds = EditDataset(items)
        train_ds, val_ds = _split(ds, cfg.val_fraction, cfg.seed)
        total = max(sum(counts), 1)
        weights = torch.tensor(
            [total / max(c, 1) for c in counts], dtype=torch.float32
        )
        weights = weights / weights.mean()
        written["stage2"] = _run_multiclass(
            EditCropNet(),
            train_ds,
            val_ds,
            cfg,
            cfg.output_dir / "stage2.pt",
            weights,
            name="stage2",
        )

    if 3 in cfg.stages:
        items = _rhythm_items(dirs, rng)
        n_pos = sum(1 for i in items if i["y"] > 0.5)
        train_items, val_items = _split_by_sample(items, cfg.val_fraction, cfg.seed)
        print(
            f"stage3 examples={len(items)} pos={n_pos} "
            f"train={len(train_items)} val={len(val_items)}"
        )
        cache = _MelCache(maxsize=768)
        written["stage3"] = _run_binary(
            RhythmNet(),
            RhythmDataset(train_items, cache=cache, augment=True),
            RhythmDataset(val_items, cache=cache, augment=False),
            cfg,
            cfg.output_dir / "stage3.pt",
            name="stage3",
            siamese=False,
        )
    return written
