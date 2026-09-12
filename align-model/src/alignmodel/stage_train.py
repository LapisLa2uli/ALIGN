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
from alignmodel.stages.gold import (
    extra_copies_of,
    gap_span,
    load_first_pass_labels,
    overlaps,
    replay_spans,
    repetition_labs,
)
from alignmodel.stages.models import (
    EDIT_CLASSES,
    MIN_EDIT_CROP_SEC,
    EditCropNet,
    RestartScorer,
    RhythmNet,
    cosine_pair_loss,
)

MATCH_CLASS_WEIGHT = 1.0
MATCH_NEGS_PER_ERROR = 1
CLEAN_MATCH_PER_CLIP = 1
EDIT_SOFTMAX_FLOOR = 0.35
EDIT_IOU_POSITIVE = 0.30

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
        labels = load_first_pass_labels(sample)
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
            copies = extra_copies_of(lab)
            for t0, t1 in replay_spans(lab):
                items.append(
                    {
                        "dir": sample,
                        "t0": t0,
                        "t1": t1,
                        "s0": s0,
                        "s1": s1,
                        "y": 1.0,
                        "extra_copies": copies,
                    }
                )
                length = max(0.35, t1 - t0)
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
                gap = gap_span(lab)
                if gap is not None:
                    g0, g1 = gap
                    items.append(
                        {
                            "dir": sample,
                            "t0": g0,
                            "t1": g1,
                            "s0": s0,
                            "s1": s1,
                            "y": 0.0,
                        }
                    )
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


def span_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0.0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / max(union, 1e-8)


def heuristic_edit_class(
    t0: float,
    t1: float,
    kind: str,
    gold_spans: list[tuple[str, float, float]],
    *,
    min_iou: float = EDIT_IOU_POSITIVE,
) -> int:
    """Map a heuristic crop to an EDIT_CLASSES index; unmatched proposals are match."""
    scored = [
        (span_iou(t0, t1, g0, g1), gold_type == kind, gold_type)
        for gold_type, g0, g1 in gold_spans
    ]
    if not scored:
        return EDIT_CLASSES.index("match")
    best_iou, _same_type, best_type = max(scored, key=lambda row: (row[0], row[1]))
    if best_iou >= min_iou and best_type in set(EDIT_CLASSES) - {"match"}:
        return EDIT_CLASSES.index(best_type)
    return EDIT_CLASSES.index("match")


def _gold_edit_spans(sample: Path) -> list[tuple[str, float, float]]:
    error_types = set(EDIT_CLASSES) - {"match"}
    spans = []
    for lab in load_first_pass_labels(sample):
        kind = lab.get("type")
        if kind not in error_types:
            continue
        spans.append((str(kind), float(lab["start_time"]), float(lab["end_time"])))
    return spans


def _heuristic_edit_cache_path(cache_dir: Path, sample: Path) -> Path:
    return cache_dir / sample.name / "heuristic_edits.json"


def _collect_heuristic_edit_spans(
    sample: Path,
    *,
    device: str,
    cache_dir: Path | None = None,
) -> list[dict]:
    if cache_dir is not None:
        path = _heuristic_edit_cache_path(cache_dir, sample)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    from alignmodel.pipeline import run_pipeline
    from alignmodel.types import PipelineConfig

    state = run_pipeline(
        sample,
        stages={1, 2},
        device=device,
        config=PipelineConfig(weights_dir=None),
    )
    error_types = set(EDIT_CLASSES) - {"match"}
    rows = [
        {
            "t0": float(lab.start_time),
            "t1": float(lab.end_time),
            "type": lab.type,
        }
        for lab in state.labels
        if lab.type in error_types
        and float(lab.end_time) - float(lab.start_time) >= MIN_EDIT_CROP_SEC
    ]
    if cache_dir is not None:
        path = _heuristic_edit_cache_path(cache_dir, sample)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _heuristic_edit_items(
    dirs: list[Path],
    *,
    device: str,
    cache_dir: Path | None = None,
    max_samples: int = 0,
) -> list[dict]:
    """Chroma-DTW proposals labeled by IoU against first-pass gold edits."""
    selected = list(dirs)
    if max_samples:
        selected = selected[: max_samples]
    items: list[dict] = []
    for i, sample in enumerate(selected, start=1):
        gold = _gold_edit_spans(sample)
        try:
            proposals = _collect_heuristic_edit_spans(
                sample, device=device, cache_dir=cache_dir
            )
        except Exception as exc:
            print(f"stage2 heuristic mine skip {sample.name}: {exc}", flush=True)
            continue
        for row in proposals:
            items.append(
                {
                    "dir": sample,
                    "t0": float(row["t0"]),
                    "t1": float(row["t1"]),
                    "y": heuristic_edit_class(
                        float(row["t0"]),
                        float(row["t1"]),
                        str(row["type"]),
                        gold,
                    ),
                }
            )
        if i == 1 or i % 50 == 0:
            print(
                f"stage2 heuristic mine {i}/{len(selected)} items={len(items)}",
                flush=True,
            )
    return items


def _edit_items(dirs: list[Path], rng: random.Random) -> list[dict]:
    """Gold first-pass errors plus extra match negatives (neighbors and clean spans)."""
    error_types = set(EDIT_CLASSES) - {"match"}
    match_i = EDIT_CLASSES.index("match")
    items = []
    for sample in dirs:
        labels = load_first_pass_labels(sample)
        dur = _clip_duration(sample)
        used: list[tuple[float, float]] = []
        errors: list[tuple[float, float]] = []

        def add_match(t0: float, t1: float) -> bool:
            if t1 - t0 < MIN_EDIT_CROP_SEC:
                return False
            if t0 < -1e-6 or t1 > dur + 1e-6:
                return False
            if any(overlaps(t0, t1, a, b) for a, b in used):
                return False
            items.append({"dir": sample, "t0": t0, "t1": t1, "y": match_i})
            used.append((t0, t1))
            return True

        for lab in labels:
            kind = lab.get("type")
            if kind not in error_types:
                continue
            t0 = float(lab["start_time"])
            t1 = float(lab["end_time"])
            if t1 - t0 < MIN_EDIT_CROP_SEC:
                continue
            items.append({"dir": sample, "t0": t0, "t1": t1, "y": EDIT_CLASSES.index(kind)})
            used.append((t0, t1))
            errors.append((t0, t1))
        for t0, t1 in errors:
            length = t1 - t0
            add_match(t1, t1 + length)
        n_neg = max(CLEAN_MATCH_PER_CLIP, MATCH_NEGS_PER_ERROR * max(len(errors), 1))
        added = 0
        tries = 0
        while added < n_neg and tries < n_neg * 8:
            tries += 1
            length = rng.uniform(0.25, 0.9)
            start = rng.uniform(0.0, max(0.05, dur - length))
            if add_match(start, start + length):
                added += 1
    return items


def _rhythm_items(dirs: list[Path], rng: random.Random) -> list[dict]:
    """Span-level rhythm crops from gold labels, not linear-mapped score notes."""
    items: list[dict] = []
    for sample in dirs:
        labels = load_first_pass_labels(sample)
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
            for g0, g1 in _phrase_gap_spans(mel, dur, used, t0, t1):
                items.append({"dir": sample, "t0": g0, "t1": g1, "y": 0.0})
                used.append((g0, g1))
    return items


def _phrase_gap_spans(
    mel: np.ndarray,
    dur: float,
    used: list[tuple[float, float]],
    t0: float,
    t1: float,
    *,
    min_width: float = 0.18,
    max_width: float = 0.85,
) -> list[tuple[float, float]]:
    """Low-energy breath / phrase-gap negatives around a gold rhythm span."""
    pos_rms = float(_rhythm_aux(mel, t0, t1)[1])
    quiet = max(0.12 * pos_rms, 1e-4)
    candidates = [
        (max(0.0, t0 - 0.65), t0),
        (t1, min(dur, t1 + 0.65)),
    ]
    out: list[tuple[float, float]] = []
    for a0, a1 in candidates:
        if a1 - a0 < min_width:
            continue
        if any(overlaps(a0, a1, u0, u1) for u0, u1 in used):
            continue
        rms = float(_rhythm_aux(mel, a0, a1)[1])
        if rms > quiet:
            continue
        width = min(max_width, a1 - a0)
        if a0 < t0:
            span = (a1 - width, a1)
        else:
            span = (a0, a0 + width)
        if span[1] - span[0] < min_width:
            continue
        out.append(span)
    return out


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
    def __init__(self, items: list[dict], cache: _MelCache | None = None, augment: bool = False):
        self.items = items
        self.cache = cache
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel = self.cache.get(it["dir"]) if self.cache is not None else _load_mel(it["dir"])
        j = random.uniform(-0.04, 0.04) if self.augment else 0.0
        crop = _mel_crop(mel, it["t0"] + j, it["t1"] + j)
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
    skip_holdout: bool = False
    mine_heuristic_edits: bool = False
    heuristic_mine_max: int = 0
    stage2_max_train_examples: int = 20_000


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


def _limit_edit_training_items(
    items: list[dict], max_examples: int, seed: int
) -> list[dict]:
    """Cap Stage 2 training while retaining every error class.

    Forty percent of the budget is reserved for match examples and the rest
    is divided across the four error classes, including intonation errors.
    Any unused class quota is filled from the remaining shuffled examples.
    """
    if max_examples <= 0 or len(items) <= max_examples:
        return list(items)
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = {i: [] for i in range(len(EDIT_CLASSES))}
    for item in items:
        buckets[int(item["y"])].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    match_quota = min(len(buckets[0]), int(round(max_examples * 0.40)))
    remaining_budget = max_examples - match_quota
    error_classes = list(range(1, len(EDIT_CLASSES)))
    base_error_quota = remaining_budget // len(error_classes)
    selected = buckets[0][:match_quota]
    used = {0: match_quota}
    for class_i in error_classes:
        count = min(len(buckets[class_i]), base_error_quota)
        selected.extend(buckets[class_i][:count])
        used[class_i] = count

    leftovers = [
        item
        for class_i, bucket in buckets.items()
        for item in bucket[used.get(class_i, 0) :]
    ]
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max_examples - len(selected)])
    rng.shuffle(selected)
    return selected


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


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def _error_prf(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    y_err = y != 0
    p_err = pred != 0
    tp = int((p_err & y_err).sum())
    fp = int((p_err & ~y_err).sum())
    fn = int((~p_err & y_err).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return f1, prec, rec


def _best_softmax_threshold(logits: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Threshold on P(error)=1-P(match). Prefer precision so heuristic spam drops."""
    if logits.ndim != 2 or logits.shape[0] == 0:
        return EDIT_SOFTMAX_FLOOR, 0.0, 0.0, 0.0
    probs = _softmax_rows(logits)
    pred = probs.argmax(axis=-1)
    error_p = 1.0 - probs[:, 0]
    candidates: list[tuple[float, float, float, float]] = []
    for t in np.concatenate(([0.0], np.linspace(0.20, 0.85, 27))):
        gated = np.where(error_p >= float(t), np.where(pred != 0, pred, 1), 0)
        f1, prec, rec = _error_prf(gated, y)
        candidates.append((f1, float(t), prec, rec))
    usable = [c for c in candidates if c[2] >= 0.55 and c[3] >= 0.25]
    if not usable:
        usable = [c for c in candidates if c[2] >= 0.45 and c[3] >= 0.20]
    pool = usable or candidates
    best = max(pool, key=lambda c: (c[0], c[2], -abs(c[1] - 0.45)))
    return max(best[1], EDIT_SOFTMAX_FLOOR), best[0], best[2], best[3]


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
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )
    best_f1 = -1.0
    best_val = float("inf")
    stale = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(_time_freq_mask(x)), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            n += 1
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        err_correct = 0
        n_err = 0
        all_logits = []
        all_y = []
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
                all_logits.append(logits.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
        val_loss /= max(len(val_loader), 1)
        acc = correct / max(total, 1)
        err_acc = err_correct / max(n_err, 1)
        logits_np = np.concatenate(all_logits) if all_logits else np.zeros((1, len(EDIT_CLASSES)))
        y_np = np.concatenate(all_y) if all_y else np.zeros(1, dtype=np.int64)
        thresh, f1, prec, rec = _best_softmax_threshold(logits_np, y_np)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_loss": val_loss,
            "acc": acc,
            "error_acc": err_acc,
            "error_f1": f1,
            "error_precision": prec,
            "error_recall": rec,
            "logit_threshold": thresh,
            "softmax_threshold": thresh,
        }
        history.append(row)
        print(
            f"{name} epoch {epoch} train={row['train_loss']:.4f} val={val_loss:.4f} "
            f"acc={acc:.3f} error_acc={err_acc:.3f} err_f1={f1:.3f} "
            f"p={prec:.3f} r={rec:.3f} thr={thresh:.2f} device={device_label(device)}"
        )
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": row,
            "logit_threshold": thresh,
            "softmax_threshold": thresh,
        }
        torch.save(ckpt, out_path.parent / f"{out_path.stem}_last.pt")
        improved = f1 > best_f1 + 1e-4 or (
            abs(f1 - best_f1) <= 1e-4 and val_loss <= best_val
        )
        if improved:
            best_f1 = max(best_f1, f1)
            best_val = min(best_val, val_loss)
            stale = 0
            torch.save(ckpt, out_path)
        else:
            stale += 1
            if stale >= 2:
                print(
                    f"{name} early stop at epoch {epoch} best_error_f1={best_f1:.3f}"
                )
                break
    (out_path.parent / f"{out_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} best_error_f1={best_f1:.3f}")
    return out_path


def _holdout_sample_dirs(root: Path, seed: int) -> set[Path]:
    dirs = [
        p
        for p in list_sample_dirs(root)
        if (p / "verified_score.musicxml").exists()
    ]
    dirs = sorted(dirs, key=lambda p: p.name)
    rng = random.Random(seed)
    shuffled = list(dirs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.1))
    if len(shuffled) > 1:
        n_val = min(n_val, len(shuffled) - 1)
    return set(shuffled[:n_val])


def _calibrate_edit_softmax_threshold(
    model: nn.Module,
    sample_dirs: list[Path],
    device: torch.device,
    stage1_dir: Path,
) -> tuple[float, dict]:
    """Sweep P(error) gate to maximize official hard type-aware melody F1."""
    from alignmodel.eval_melodies import eval_sample
    from alignmodel.pipeline import run_pipeline
    from alignmodel.types import pipeline_label_to_dict

    error_types = set(EDIT_CLASSES) - {"match"}
    model = model.to(device).eval()
    clip_rows: list[dict] = []
    for sample in sample_dirs:
        try:
            state = run_pipeline(sample, stages={1, 2}, device=str(device), weights_dir=stage1_dir)
        except Exception as exc:
            print(f"stage2 calib skip {sample.name}: {exc}")
            continue
        pending = [
            lab
            for lab in state.labels
            if lab.type in error_types
            and float(lab.end_time) - float(lab.start_time) >= MIN_EDIT_CROP_SEC
        ]
        kept = [lab for lab in state.labels if lab.type not in error_types]
        if not pending:
            clip_rows.append({"sample": sample, "kept": kept, "pending": [], "probs": None})
            continue
        mel = _load_mel(sample)
        crops = np.stack(
            [_mel_crop(mel, float(lab.start_time), float(lab.end_time)) for lab in pending]
        )
        probs_out = []
        with torch.no_grad():
            for i0 in range(0, len(crops), 64):
                batch = torch.from_numpy(crops[i0 : i0 + 64]).to(device)
                probs_out.append(torch.softmax(model(batch), dim=-1).detach().cpu().numpy())
        probs = np.concatenate(probs_out, axis=0)
        clip_rows.append(
            {"sample": sample, "kept": kept, "pending": pending, "probs": probs}
        )
    if not clip_rows:
        return EDIT_SOFTMAX_FLOOR, {"n_clips": 0, "reason": "no_clips"}

    candidates = []
    for t in np.concatenate(([0.0], np.linspace(0.20, 0.85, 27))):
        f1s: list[float] = []
        precs: list[float] = []
        recs: list[float] = []
        n_preds: list[int] = []
        n_golds: list[int] = []
        for row in clip_rows:
            kept = list(row["kept"])
            pending = row["pending"]
            probs = row["probs"]
            if pending and probs is not None:
                error_p = 1.0 - probs[:, 0]
                for lab, p_err in zip(pending, error_p):
                    if float(p_err) < float(t):
                        continue
                    kept.append(lab)
            labels = [pipeline_label_to_dict(lab) for lab in kept]
            scored = eval_sample(row["sample"], pred_labels=labels, soft=False)
            f1s.append(scored["melody_f1"])
            precs.append(scored["melody_precision"])
            recs.append(scored["melody_recall"])
            n_preds.append(scored["n_pred"])
            n_golds.append(scored["n_gold"])
        n = max(len(f1s), 1)
        mean_gold = sum(n_golds) / n
        candidates.append(
            {
                "t": float(t),
                "f1": sum(f1s) / n,
                "prec": sum(precs) / n,
                "rec": sum(recs) / n,
                "mean_n_pred": sum(n_preds) / n,
                "mean_n_gold": mean_gold,
            }
        )
    usable = [
        c
        for c in candidates
        if c["rec"] >= 0.15 and c["mean_n_pred"] <= max(10.0, 2.5 * max(c["mean_n_gold"], 1.0))
    ]
    pool = usable or candidates
    best = max(pool, key=lambda c: (c["f1"], c["prec"], c["t"]))
    thr = max(float(best["t"]), EDIT_SOFTMAX_FLOOR)
    print(
        f"stage2 calib (hard set-F1) thr={thr:.2f} "
        f"f1={best['f1']:.3f} p={best['prec']:.3f} r={best['rec']:.3f} "
        f"mean_n_pred={best['mean_n_pred']:.2f} gold={best['mean_n_gold']:.2f} "
        f"clips={len(clip_rows)}"
    )
    return thr, {"chosen": best, "n_clips": len(clip_rows), "sweep": candidates}


def _maybe_calibrate_stage2(
    ckpt_path: Path, dirs: list[Path], cfg: StageTrainConfig, device: torch.device
) -> None:
    calib_dirs = _calib_pool(dirs, cfg, n=20)
    if not calib_dirs:
        print("stage2 calib skipped (no non-holdout clips)")
        return
    stage1_dir = ckpt_path.parent / "_calib_stage1"
    s1 = ckpt_path.parent / "stage1.pt"
    if s1.exists():
        stage1_dir.mkdir(parents=True, exist_ok=True)
        dest = stage1_dir / "stage1.pt"
        if not dest.exists() or dest.stat().st_mtime < s1.stat().st_mtime:
            dest.write_bytes(s1.read_bytes())
    else:
        stage1_dir = ckpt_path.parent
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EditCropNet()
    model.load_state_dict(blob["model"])
    try:
        thr, info = _calibrate_edit_softmax_threshold(model, calib_dirs, device, stage1_dir)
    except Exception as exc:
        print(f"stage2 calib failed: {exc}")
        return
    blob["logit_threshold"] = thr
    blob["softmax_threshold"] = thr
    blob["calibration"] = {k: info[k] for k in info if k != "sweep"}
    torch.save(blob, ckpt_path)
    print(f"Updated {ckpt_path} softmax_threshold={thr:.2f}")


def _calib_pool(dirs: list[Path], cfg: StageTrainConfig, n: int = 20) -> list[Path]:
    holdout = _holdout_sample_dirs(cfg.data_root, cfg.seed)
    rng = random.Random(cfg.seed + 17)
    pool = [d for d in dirs if d not in holdout and (d / "verified_score.musicxml").exists()]
    rng.shuffle(pool)
    return pool[:n]


def _calibrate_rhythm_threshold(
    model: nn.Module,
    sample_dirs: list[Path],
    device: torch.device,
    weights_dir: Path,
) -> tuple[float, dict]:
    """Sweep RhythmNet logit threshold against official hard melody F1."""
    from alignmodel.eval_melodies import eval_sample
    from alignmodel.pipeline import run_pipeline
    from alignmodel.stages.dc_alignment import ensure_rhythm_pairs
    from alignmodel.stages.rhythm import flagged_rhythm_spans, merge_time_spans
    from alignmodel.types import PipelineConfig, pipeline_label_to_dict

    model = model.to(device).eval()
    cfg = PipelineConfig(weights_dir=str(weights_dir), rhythm_logit_override=None)
    clip_rows: list[dict] = []
    for sample in sample_dirs:
        try:
            state = run_pipeline(sample, stages={1, 2}, device=str(device), config=cfg)
        except Exception as exc:
            print(f"stage3 calib skip {sample.name}: {exc}")
            continue
        kept = list(state.labels)
        try:
            mel = _load_mel(sample)
            pairs = [
                pair
                for pair in ensure_rhythm_pairs(state, learned=None, mel=mel)
                if pair.kind in {"match", "substitute", "rest"}
            ]
            windows = merge_time_spans(
                flagged_rhythm_spans(pairs, state.config),
                gap=float(state.config.rhythm_merge_gap_sec),
                min_dur=float(state.config.min_candidate_sec),
            )
        except Exception as exc:
            print(f"stage3 calib windows skip {sample.name}: {exc}")
            clip_rows.append({"sample": sample, "kept": kept, "hits": []})
            continue
        hits = []
        for t0, t1 in windows:
            if t1 - t0 < 0.12:
                continue
            crop = torch.from_numpy(_mel_span_resize(mel, t0, t1)).unsqueeze(0).to(device)
            aux = torch.from_numpy(_rhythm_aux(mel, t0, t1)).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = float(model(crop, aux).squeeze().detach().cpu())
            hits.append((float(t0), float(t1), logit))
        clip_rows.append({"sample": sample, "kept": kept, "hits": hits})
    if not clip_rows:
        return 0.0, {"n_clips": 0, "reason": "no_clips"}

    logits = [hit[2] for row in clip_rows for hit in row["hits"]]
    grid = [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if logits:
        grid.extend(float(q) for q in np.quantile(np.asarray(logits), [0.2, 0.4, 0.6, 0.8]))
    candidates = []
    for t in sorted(set(round(float(x), 3) for x in grid)):
        f1s: list[float] = []
        precs: list[float] = []
        recs: list[float] = []
        n_preds: list[int] = []
        n_golds: list[int] = []
        for row in clip_rows:
            labels = [pipeline_label_to_dict(lab) for lab in row["kept"]]
            for t0, t1, logit in row["hits"]:
                if logit <= t:
                    continue
                labels.append(
                    {
                        "id": f"rcal-{len(labels)}",
                        "type": "rhythm_error",
                        "start_time": t0,
                        "end_time": t1,
                        "source": "pipeline",
                    }
                )
            scored = eval_sample(row["sample"], pred_labels=labels, soft=False)
            f1s.append(scored["melody_f1"])
            precs.append(scored["melody_precision"])
            recs.append(scored["melody_recall"])
            n_preds.append(scored["n_pred"])
            n_golds.append(scored["n_gold"])
        n = max(len(f1s), 1)
        mean_gold = sum(n_golds) / n
        candidates.append(
            {
                "t": float(t),
                "f1": sum(f1s) / n,
                "prec": sum(precs) / n,
                "rec": sum(recs) / n,
                "mean_n_pred": sum(n_preds) / n,
                "mean_n_gold": mean_gold,
            }
        )
    usable = [
        c
        for c in candidates
        if c["rec"] >= 0.10 and c["mean_n_pred"] <= max(12.0, 3.0 * max(c["mean_n_gold"], 1.0))
    ]
    pool = usable or candidates
    best = max(pool, key=lambda c: (c["f1"], c["prec"], c["t"]))
    print(
        f"stage3 calib (hard set-F1) thr={best['t']:.2f} "
        f"f1={best['f1']:.3f} p={best['prec']:.3f} r={best['rec']:.3f} "
        f"mean_n_pred={best['mean_n_pred']:.2f} gold={best['mean_n_gold']:.2f} "
        f"clips={len(clip_rows)}"
    )
    return float(best["t"]), {"chosen": best, "n_clips": len(clip_rows), "sweep": candidates}


def _maybe_calibrate_stage3(
    ckpt_path: Path, dirs: list[Path], cfg: StageTrainConfig, device: torch.device
) -> None:
    calib_dirs = _calib_pool(dirs, cfg, n=20)
    if not calib_dirs:
        print("stage3 calib skipped (no non-holdout clips)")
        return
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RhythmNet()
    model.load_state_dict(blob["model"])
    try:
        thr, info = _calibrate_rhythm_threshold(model, calib_dirs, device, ckpt_path.parent)
    except Exception as exc:
        print(f"stage3 calib failed: {exc}")
        return
    blob["logit_threshold"] = thr
    blob["calibration"] = {k: info[k] for k in info if k != "sweep"}
    torch.save(blob, ckpt_path)
    print(f"Updated {ckpt_path} logit_threshold={thr:.2f}")


def train_stages(cfg: StageTrainConfig) -> dict[str, Path]:
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    dirs = list_sample_dirs(cfg.data_root)
    if cfg.skip_holdout:
        holdout = _holdout_sample_dirs(cfg.data_root, cfg.seed)
        dirs = [sample for sample in dirs if sample not in holdout]
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
        if cfg.mine_heuristic_edits:
            mined = _heuristic_edit_items(
                dirs,
                device=cfg.device,
                cache_dir=cfg.output_dir / "heuristic_edit_cache",
                max_samples=cfg.heuristic_mine_max,
            )
            print(f"stage2 mined heuristic crops={len(mined)}")
            items.extend(mined)
        counts = [0] * len(EDIT_CLASSES)
        for it in items:
            counts[it["y"]] += 1
        train_items, val_items = _split_by_sample(items, cfg.val_fraction, cfg.seed)
        train_items = _limit_edit_training_items(
            train_items, cfg.stage2_max_train_examples, cfg.seed
        )
        train_counts = [0] * len(EDIT_CLASSES)
        for it in train_items:
            train_counts[it["y"]] += 1
        print(
            f"stage2 examples={len(items)} counts={dict(zip(EDIT_CLASSES, counts))} "
            f"train={len(train_items)} train_counts={dict(zip(EDIT_CLASSES, train_counts))} "
            f"val={len(val_items)}"
        )
        cache = _MelCache(maxsize=4096)
        weights = torch.ones(len(EDIT_CLASSES), dtype=torch.float32)
        weights[EDIT_CLASSES.index("match")] = MATCH_CLASS_WEIGHT
        train_ds = EditDataset(train_items, cache=cache, augment=True)
        val_ds = EditDataset(val_items, cache=cache, augment=False)
        batch = cfg.batch_size
        while True:
            try:
                if str(cfg.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                stage_cfg = StageTrainConfig(
                    data_root=cfg.data_root,
                    output_dir=cfg.output_dir,
                    epochs=cfg.epochs,
                    batch_size=batch,
                    lr=cfg.lr,
                    device=cfg.device,
                    seed=cfg.seed,
                    val_fraction=cfg.val_fraction,
                    stages=cfg.stages,
                    max_samples=cfg.max_samples,
                    skip_holdout=cfg.skip_holdout,
                    mine_heuristic_edits=cfg.mine_heuristic_edits,
                    heuristic_mine_max=cfg.heuristic_mine_max,
                    stage2_max_train_examples=cfg.stage2_max_train_examples,
                )
                written["stage2"] = _run_multiclass(
                    EditCropNet(),
                    train_ds,
                    val_ds,
                    stage_cfg,
                    cfg.output_dir / "stage2.pt",
                    weights,
                    name="stage2",
                )
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch <= 4:
                    raise
                print(f"stage2 OOM at batch {batch}; retrying {batch // 2}")
                torch.cuda.empty_cache()
                batch //= 2
        _maybe_calibrate_stage2(written["stage2"], dirs, cfg, resolve_device(cfg.device))

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
        _maybe_calibrate_stage3(written["stage3"], dirs, cfg, resolve_device(cfg.device))
    return written
