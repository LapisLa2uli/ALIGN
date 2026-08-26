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
from alignmodel.dataset import _load_score_notes, list_sample_dirs
from alignmodel.device import device_label, resolve_device
from alignmodel.stages.gold import load_labels, map_notes_to_perf, overlaps, repetition_labs
from alignmodel.stages.models import EDIT_CLASSES, EditCropNet, RestartScorer, RhythmNet

CROP_FRAMES = 64
MEL_HOP = FRAME_HOP_SEC


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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _clip_duration(sample_dir: Path, mel: np.ndarray | None = None) -> float:
    if mel is None:
        mel = _load_mel(sample_dir)
    return mel.shape[1] * MEL_HOP


def _restart_items(dirs: list[Path], rng: random.Random) -> list[dict]:
    items = []
    for sample in dirs:
        labels = load_labels(sample)
        reps = repetition_labs(labels)
        dur = None
        for lab in reps:
            src = lab.get("repeats_label_range") or {}
            s0 = float(src.get("start_time", 0.0))
            s1 = float(src.get("end_time", s0 + 0.5))
            t0 = float(lab["start_time"])
            t1 = float(lab["end_time"])
            items.append(
                {"dir": sample, "t0": t0, "t1": t1, "s0": s0, "s1": s1, "y": 1.0}
            )
            if dur is None:
                dur = _clip_duration(sample)
            length = max(0.35, t1 - t0)
            for _ in range(2):
                start = rng.uniform(0.0, max(0.05, dur - length))
                if overlaps(start, start + length, t0, t1):
                    continue
                items.append(
                    {
                        "dir": sample,
                        "t0": start,
                        "t1": start + length,
                        "s0": max(0.0, start - length),
                        "s1": start,
                        "y": 0.0,
                    }
                )
        if not reps:
            if dur is None:
                dur = _clip_duration(sample)
            length = min(1.5, max(0.4, dur * 0.12))
            start = rng.uniform(0.0, max(0.05, dur - length))
            items.append(
                {
                    "dir": sample,
                    "t0": start,
                    "t1": start + length,
                    "s0": max(0.0, start - length),
                    "s1": start,
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


def _rhythm_items(dirs: list[Path]) -> list[tuple[np.ndarray, float]]:
    items: list[tuple[np.ndarray, float]] = []
    for sample in dirs:
        labels = load_labels(sample)
        notes = _load_score_notes(sample)
        if not notes:
            continue
        if not labels:
            dur = max(notes[-1].end, 1.0)
        else:
            dur = max(float(lab["end_time"]) for lab in labels) + 0.5
            dur = max(dur, notes[-1].end)
        mapped = map_notes_to_perf(notes, labels, dur)
        rhythm_spans = [
            (float(lab["start_time"]), float(lab["end_time"]))
            for lab in labels
            if lab.get("type") == "rhythm_error"
        ]
        ratios: list[float | None] = []
        feats_y: list[tuple[np.ndarray, float]] = []
        for note, (p0, p1) in zip(notes, mapped):
            ref = max(note.duration, 1e-3)
            perf = max(p1 - p0, 1e-3)
            ratio = perf / ref
            ratios.append(ratio)
            y = 1.0 if any(overlaps(p0, p1, a, b) for a, b in rhythm_spans) else 0.0
            feats_y.append((None, y, ratio, ref, perf, p0))  # type: ignore
        ewma = None
        for i, (_blank, y, ratio, ref, perf, _p0) in enumerate(feats_y):
            if ewma is None:
                ewma = ratio
                log_jump = 0.0
            else:
                log_jump = abs(float(np.log(ratio / max(ewma, 1e-6))))
                ewma = 0.3 * ratio + 0.7 * ewma
            far = ratios[max(0, i - 12) : max(0, i - 6)]
            far = [r for r in far if r is not None]
            log_far = 0.0
            if far:
                med = float(np.median(far))
                if med > 0:
                    log_far = abs(float(np.log(ratio / med)))
            prev = ratios[i - 1] if i else ratio
            feat = np.array(
                [
                    float(np.log(max(ratio, 1e-4))),
                    abs(float(np.log(max(ratio, 1e-4)))),
                    float(np.log(max(ref, 1e-4))),
                    float(np.log(max(perf, 1e-4))),
                    log_jump,
                    log_far,
                    i / max(len(notes) - 1, 1),
                    float(np.log(max(prev or ratio, 1e-4))),
                ],
                dtype=np.float32,
            )
            items.append((feat, y))
    return items


class RestartDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel = _load_mel(it["dir"])
        cur = _pool_span(mel, it["t0"], it["t1"])
        src = _pool_span(mel, it["s0"], it["s1"])
        dur = max(it["t1"] - it["t0"], 1e-3)
        sdur = max(it["s1"] - it["s0"], 1e-3)
        feat = np.concatenate(
            [
                cur,
                src,
                np.array(
                    [
                        np.log(dur),
                        np.log(sdur),
                        _cosine(cur, src),
                        it["t0"] / max(mel.shape[1] * MEL_HOP, 1e-3),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return torch.from_numpy(feat), torch.tensor(it["y"], dtype=torch.float32)


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
    def __init__(self, items: list[tuple[np.ndarray, float]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, y = self.items[idx]
        return torch.from_numpy(feat), torch.tensor(y, dtype=torch.float32)


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


def _split(ds: Dataset, frac: float, seed: int):
    n_val = max(1, int(len(ds) * frac))
    n_train = len(ds) - n_val
    if n_train < 1:
        return ds, ds
    return random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))


def _run_binary(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: StageTrainConfig,
    out_path: Path,
    pos_weight: float,
    name: str,
) -> Path:
    device = resolve_device(cfg.device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
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
            loss = bce(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n += 1
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += float(bce(logits, y).detach())
                pred = (logits > 0).float()
                correct += int((pred == y).sum())
                total += int(y.numel())
                tp += int(((pred == 1) & (y == 1)).sum())
                fp += int(((pred == 1) & (y == 0)).sum())
                fn += int(((pred == 0) & (y == 1)).sum())
        val_loss /= max(len(val_loader), 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        acc = correct / max(total, 1)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_loss": val_loss,
            "acc": acc,
            "f1": f1,
        }
        history.append(row)
        print(
            f"{name} epoch {epoch} train={row['train_loss']:.4f} val={val_loss:.4f} "
            f"acc={acc:.3f} f1={f1:.3f} device={device_label(device)}"
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
        print(f"stage1 examples={len(items)} pos={sum(1 for i in items if i['y']>0.5)}")
        ds = RestartDataset(items)
        train_ds, val_ds = _split(ds, cfg.val_fraction, cfg.seed)
        n_pos = sum(1 for i in items if i["y"] > 0.5)
        n_neg = max(len(items) - n_pos, 1)
        written["stage1"] = _run_binary(
            RestartScorer(),
            train_ds,
            val_ds,
            cfg,
            cfg.output_dir / "stage1.pt",
            pos_weight=n_neg / max(n_pos, 1),
            name="stage1",
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
        items = _rhythm_items(dirs)
        n_pos = sum(1 for _f, y in items if y > 0.5)
        print(f"stage3 examples={len(items)} pos={n_pos}")
        ds = RhythmDataset(items)
        train_ds, val_ds = _split(ds, cfg.val_fraction, cfg.seed)
        n_neg = max(len(items) - n_pos, 1)
        written["stage3"] = _run_binary(
            RhythmNet(),
            train_ds,
            val_ds,
            cfg,
            cfg.output_dir / "stage3.pt",
            pos_weight=n_neg / max(n_pos, 1),
            name="stage3",
        )
    return written
