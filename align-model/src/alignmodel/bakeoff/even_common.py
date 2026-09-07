"""Helpers for even bakeoff versions only. Do not share with odd-version common.py."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from alignmodel.config import MELODY_ALIGN_TYPES, MELODY_NOTE_CLASSES, ModelConfig
from alignmodel.dataset import list_sample_dirs
from alignmodel.melody import gold_melodies_from_labels, load_bundle_notes
from alignmodel.melody_model import class_index
from alignmodel.model import AudioEncoder, HierarchicalFusion, ScoreEncoder
from alignmodel.types import schema12_document
from datacreate.melody import ScoreSoundingNote, WeakMelody, match_melodies_detail, padded_melody

DECODE_PAD_NOTES = 2
MATCH_I = class_index("match")


def encode_fused(
    audio: AudioEncoder,
    score: ScoreEncoder,
    fusion: HierarchicalFusion,
    cfg: ModelConfig,
    mel: Tensor,
    mel_mask: Tensor,
    pitch: Tensor,
    onset: Tensor,
    duration: Tensor,
    note_mask: Tensor,
    hop_sec: float,
) -> tuple[Tensor, Tensor]:
    audio_h = audio(mel, mel_mask, hop_sec)
    t_audio = audio_h.size(1)
    stride = cfg.audio_stride
    if mel_mask.size(-1) >= t_audio * stride:
        audio_mask = (
            mel_mask[:, : t_audio * stride]
            .view(mel_mask.size(0), t_audio, stride)
            .any(dim=-1)
        )
    else:
        audio_mask = torch.ones(audio_h.size(0), t_audio, dtype=torch.bool, device=mel.device)
    score_h = score(pitch, onset, duration, note_mask)
    fused_score, _ = fusion(score_h, audio_h, note_mask, audio_mask)
    return fused_score, audio_mask


def copies_and_coverage(
    type_logits: Tensor,
    copies_logits: Tensor,
    note_mask: Tensor,
    copies_y: Tensor,
    copies_loss_weight: float,
    coverage_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    copies_loss = nn.functional.cross_entropy(copies_logits, copies_y)
    probs = nn.functional.softmax(type_logits, dim=-1)
    mask_f = note_mask.float()
    denom = mask_f.sum().clamp_min(1)
    pred_rep = (probs[..., class_index("repetition")] * mask_f).sum() / denom
    coverage = nn.functional.relu(pred_rep - 0.30)
    extra = copies_loss_weight * copies_loss + coverage_loss_weight * coverage
    return extra, copies_loss, coverage


def decode_runs_no_tile(
    types: list[str],
    notes: list[ScoreSoundingNote],
    extra_copies: int,
    pad_notes: int = DECODE_PAD_NOTES,
    type_probs: Tensor | None = None,
    conf_min: float = 0.0,
) -> list[dict[str, Any]]:
    """Contiguous same-type non-match cores → schema 1.2. No 5-note tile split."""
    n = min(len(types), len(notes))
    labels: list[dict[str, Any]] = []
    i = 0
    while i < n:
        kind = types[i]
        if kind == "match" or kind not in MELODY_ALIGN_TYPES:
            i += 1
            continue
        j = i + 1
        while j < n and types[j] == kind:
            j += 1
        if type_probs is not None and conf_min > 0:
            cls = class_index(kind)
            mean_p = float(type_probs[i:j, cls].mean())
            if mean_p < conf_min:
                i = j
                continue
        lo, hi = i, j
        if hi <= lo:
            i = j
            continue
        span = padded_melody(notes, lo, hi, pad_notes)
        span_notes = notes[span.start_note_index : span.end_note_index + 1]
        item: dict[str, Any] = {
            "id": f"mel_{len(labels):03d}",
            "source": "melody",
            "type": MELODY_ALIGN_TYPES[kind],
            "start_time": round(span_notes[0].start, 4),
            "end_time": round(span_notes[-1].end, 4),
            "score_part": {
                "start_note_index": span.start_note_index,
                "end_note_index": span.end_note_index,
                "pad_notes": span.pad_notes,
                "start_measure": span.start_measure,
                "end_measure": span.end_measure,
            },
            "pitches": list(span.pitches),
            "note_ids": list(span.note_ids),
        }
        if kind == "repetition":
            item["extra_copies"] = 1 if extra_copies <= 0 else min(2, extra_copies)
            item["repeats_label_range"] = {
                "start_time": item["start_time"],
                "end_time": item["end_time"],
            }
        labels.append(item)
        i = j
    return labels


def decode_fault_runs(
    fault_prob: Tensor,
    type_logits: Tensor,
    notes: list[ScoreSoundingNote],
    extra_copies: int,
    threshold: float = 0.5,
    pad_notes: int = DECODE_PAD_NOTES,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Type only where fault-prob > threshold; one schema 1.2 label per fault run."""
    n = min(int(fault_prob.numel()), len(notes), int(type_logits.size(0)))
    types = ["match"] * n
    labels: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if float(fault_prob[i]) <= threshold:
            i += 1
            continue
        j = i + 1
        while j < n and float(fault_prob[j]) > threshold:
            j += 1
        logits = type_logits[i:j].mean(0).clone()
        logits[MATCH_I] = -1e9
        kind = MELODY_NOTE_CLASSES[int(logits.argmax())]
        if kind == "match" or kind not in MELODY_ALIGN_TYPES:
            i = j
            continue
        for k in range(i, j):
            types[k] = kind
        span = padded_melody(notes, i, j, pad_notes)
        span_notes = notes[span.start_note_index : span.end_note_index + 1]
        item: dict[str, Any] = {
            "id": f"mel_{len(labels):03d}",
            "source": "melody",
            "type": MELODY_ALIGN_TYPES[kind],
            "start_time": round(span_notes[0].start, 4),
            "end_time": round(span_notes[-1].end, 4),
            "score_part": {
                "start_note_index": span.start_note_index,
                "end_note_index": span.end_note_index,
                "pad_notes": span.pad_notes,
                "start_measure": span.start_measure,
                "end_measure": span.end_measure,
            },
            "pitches": list(span.pitches),
            "note_ids": list(span.note_ids),
        }
        if kind == "repetition":
            item["extra_copies"] = 1 if extra_copies <= 0 else min(2, extra_copies)
            item["repeats_label_range"] = {
                "start_time": item["start_time"],
                "end_time": item["end_time"],
            }
        labels.append(item)
        i = j
    return types, labels


def notes_from_tensors(pitch, onset, duration, n: int) -> list[ScoreSoundingNote]:
    notes: list[ScoreSoundingNote] = []
    for i in range(n):
        start = float(onset[i])
        dur = float(duration[i])
        notes.append(
            ScoreSoundingNote(
                index=i,
                pitch=int(pitch[i]),
                start=start,
                end=start + max(dur, 0.04),
                ql_start=0.0,
                ql_end=0.0,
                measure=None,
                note_id=f"note_{i:04d}",
            )
        )
    return notes


def write_pred_document(sample_id: str, labels: list[dict], extra_copies: int, path: Path) -> None:
    doc = schema12_document(
        sample_id=sample_id,
        labels=labels,
        annotator_id="align_melody",
        extra={"melody": {"extra_copies": extra_copies}},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def val_like_dirs(root: Path, n: int = 100, seed: int = 365) -> list[Path]:
    dirs = [p for p in list_sample_dirs(root) if (p / "verified_score.musicxml").exists()]
    dirs = sorted(dirs, key=lambda p: p.name)
    rng = random.Random(seed)
    shuffled = list(dirs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.1))
    if len(shuffled) > 1:
        n_val = min(n_val, len(shuffled) - 1)
    return shuffled[:n_val][:n]


def official_set_metrics(gold_labels: list[dict], pred_labels: list[dict]) -> dict[str, float]:
    gold = gold_melodies_from_labels(gold_labels)
    pred = [WeakMelody(pitches=[int(p) for p in (lab.get("pitches") or [])]) for lab in pred_labels]
    pred = [m for m in pred if m.pitches]
    detail = match_melodies_detail(gold, pred)
    return {
        "set_f1": float(detail["f1"]),
        "precision": float(detail["precision"]),
        "recall": float(detail["recall"]),
        "n_pred": float(len(pred)),
        "n_gold": float(len(gold)),
    }


def load_gold_labels(sample_dir: Path) -> list[dict]:
    gold_doc = json.loads((sample_dir / "labels.json").read_text(encoding="utf-8"))
    return gold_doc.get("labels") or []


@torch.no_grad()
def infer_and_eval_holdout(
    *,
    infer_fn: Callable,
    model: nn.Module,
    device: torch.device,
    data_root: Path,
    pred_dir: Path,
    n_eval: int = 100,
    seed: int = 365,
) -> dict:
    samples = val_like_dirs(data_root, n=n_eval, seed=seed)
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(samples, start=1):
        pred_path = pred_dir / f"{sample.name}.json"
        result = infer_fn(model, sample, device)
        write_pred_document(
            result["sample_id"],
            result["labels"],
            int(result.get("extra_copies") or 0),
            pred_path,
        )
        gold = load_gold_labels(sample)
        metrics = official_set_metrics(gold, result["labels"])
        rows.append({"sample": sample.name, **metrics})
        if i == 1 or i % 20 == 0:
            print(
                f"  holdout {i}/{len(samples)} f1={metrics['set_f1']:.3f} "
                f"n_pred={metrics['n_pred']:.0f} n_gold={metrics['n_gold']:.0f}",
                flush=True,
            )
    n = max(len(rows), 1)
    return {
        "n_samples": len(rows),
        "set_f1": round(sum(r["set_f1"] for r in rows) / n, 4),
        "precision": round(sum(r["precision"] for r in rows) / n, 4),
        "recall": round(sum(r["recall"] for r in rows) / n, 4),
        "mean_n_pred": round(sum(r["n_pred"] for r in rows) / n, 3),
        "mean_n_gold": round(sum(r["n_gold"] for r in rows) / n, 3),
        "samples": rows,
    }
