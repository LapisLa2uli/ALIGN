"""Training data and loop for transcribed-note repetition scoring."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alignmodel.stages.note_repetition_model import (
    FEATURE_NAMES,
    NoteRepetitionModelConfig,
    NoteRepetitionScorer,
    propose_note_repeat_candidates,
    save_note_repetition_model,
)
from alignmodel.types import TranscribedNote
from alignmodel.validated_targets import target_note_map


def _row_sequence(note_map_path: Path | Mapping[str, Any]):
    document = (
        target_note_map(note_map_path)
        if isinstance(note_map_path, Mapping)
        else json.loads(note_map_path.read_text(encoding="utf-8"))
    )
    clean = {
        int(row["clean_index"]): int(row["pitch_midi"])
        for row in document.get("clean_notes") or []
    }
    rendered = sorted(
        document.get("rendered_notes") or [],
        key=lambda row: int(row["rendered_index"]),
    )
    pitch_offsets = [
        int(row["pitch_midi_written"]) - clean[int(row["primary_clean_index"])]
        for row in rendered
        if row.get("primary_clean_index") is not None
        and int(row["primary_clean_index"]) in clean
        and str(row.get("relationship")) in {"match", "copy"}
    ]
    rendered_pitch_correction = (
        -int(round(float(np.median(pitch_offsets))))
        if pitch_offsets
        else 0
    )
    notes = []
    primary = []
    relationships = []
    for row in rendered:
        clean_index = row.get("primary_clean_index")
        relationship = str(row.get("relationship") or "extra")
        # Existing procedural maps can have a +2 error in rendered written
        # pitch. Lineage clean pitch is authoritative whenever available.
        pitch = (
            clean[int(clean_index)]
            if clean_index is not None and int(clean_index) in clean
            else int(row["pitch_midi_written"]) + rendered_pitch_correction
        )
        notes.append(
            TranscribedNote(
                int(pitch),
                float(row["start_sec"]),
                float(row["end_sec"]),
                1.0,
            )
        )
        primary.append(int(clean_index) if clean_index is not None else None)
        relationships.append(relationship)
    return notes, primary, relationships


def candidate_rows(
    note_map_path: Path | Mapping[str, Any], *, seed: int
) -> tuple[list[np.ndarray], list[float]]:
    notes, primary, relationships = _row_sequence(note_map_path)
    if len(notes) < 2:
        return [], []
    extra_mask = [
        relationship in {"copy", "extra"}
        for relationship in relationships
    ]
    candidates = propose_note_repeat_candidates(
        notes, extra_mask=extra_mask
    )
    positives = []
    negatives = []
    for candidate in candidates:
        length = candidate.repeat_i1 - candidate.repeat_i0
        target_is_copy = all(
            relationships[index] == "copy"
            for index in range(candidate.repeat_i0, candidate.repeat_i1)
        )
        lineage_equal = all(
            primary[candidate.source_i0 + offset] is not None
            and primary[candidate.source_i0 + offset]
            == primary[candidate.repeat_i0 + offset]
            for offset in range(length)
        )
        row = candidate.features
        if target_is_copy and lineage_equal:
            positives.append(row)
        else:
            negatives.append(row)
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[:24]
    continuation_fraction = FEATURE_NAMES.index(
        "continuation_pitch_fraction"
    )
    continuation_score = FEATURE_NAMES.index("continuation_alignment_score")
    divergent_hard_negatives = []
    for row in positives:
        if float(row[continuation_fraction]) <= 0.5:
            continue
        divergent = row.copy()
        # Preserve the source/replay prefix and restart evidence while making
        # only the expected post-replay continuation deliberately unrelated.
        divergent[continuation_fraction] = 0.0
        divergent[continuation_score] = -1.0
        divergent_hard_negatives.append(divergent)
    rng.shuffle(divergent_hard_negatives)
    natural_limit = max(24, 2 * len(positives))
    negatives = (
        negatives[:natural_limit]
        + divergent_hard_negatives[: len(positives)]
    )
    return (
        positives + negatives,
        [1.0] * len(positives) + [0.0] * len(negatives),
    )


def _manifest_maps(
    manifest: Path,
    split: str,
    maximum: int,
) -> list[Any]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    rows = list(document.get(split) or [])
    selected = []
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        if row.get("target_db") is not None and row.get("target_record") is not None:
            selected.append(row)
            if maximum and len(selected) >= maximum:
                break
            continue
        if str(row.get("corpus") or row.get("root")) != "procedural12k":
            raise ValueError("Repetition training rejected non-procedural row")
        path = Path(
            str(
                row.get("note_map")
                or Path(str(row["sample_dir"])) / "note_map.json"
            )
        )
        if path.is_file():
            selected.append(path)
        if maximum and len(selected) >= maximum:
            break
    return selected


def _copy_runs(relationships: list[str]) -> list[tuple[int, int]]:
    runs = []
    index = 0
    while index < len(relationships):
        if relationships[index] != "copy":
            index += 1
            continue
        end = index + 1
        while end < len(relationships) and relationships[end] == "copy":
            end += 1
        runs.append((index, end))
        index = end
    return runs


def _select_candidates(candidates, probabilities, threshold: float):
    ranked = sorted(
        zip(candidates, probabilities),
        key=lambda item: (
            -(
                float(item[1])
                + 0.15
                * np.log1p(item[0].repeat_i1 - item[0].repeat_i0)
            ),
            -(item[0].repeat_i1 - item[0].repeat_i0),
        ),
    )
    kept = []
    for candidate, probability in ranked:
        if float(probability) < threshold:
            continue
        length = candidate.repeat_i1 - candidate.repeat_i0
        if length == 1 and (
            float(probability) < max(0.90, threshold)
            or float(candidate.features[9]) < 0.5
        ):
            continue
        if any(
            candidate.repeat_i0 < other.repeat_i1
            and other.repeat_i0 < candidate.repeat_i1
            for other in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= 2:
            break
    return kept


@torch.no_grad()
def calibrate_span_threshold(
    model: NoteRepetitionScorer,
    note_maps: list[Any],
    *,
    device: str,
) -> tuple[float, dict]:
    clips = []
    for path in note_maps:
        notes, _primary, relationships = _row_sequence(path)
        extra_mask = [
            relationship in {"copy", "extra"}
            for relationship in relationships
        ]
        candidates = propose_note_repeat_candidates(
            notes, extra_mask=extra_mask
        )
        probabilities = (
            model(
                torch.from_numpy(
                    np.stack([candidate.features for candidate in candidates])
                ).to(device)
            )
            .sigmoid()
            .cpu()
            .numpy()
            if candidates
            else np.zeros(0, np.float32)
        )
        clips.append((candidates, probabilities, _copy_runs(relationships)))

    best_threshold = 0.9
    best_metrics = {"f1": -1.0}
    for threshold in np.linspace(0.50, 0.995, 100):
        tp = fp = fn = 0
        for candidates, probabilities, truth in clips:
            predicted = _select_candidates(
                candidates, probabilities, float(threshold)
            )
            used: set[int] = set()
            matched = 0
            for candidate in predicted:
                best = None
                best_iou = 0.0
                for index, (start, end) in enumerate(truth):
                    if index in used:
                        continue
                    overlap = max(
                        0,
                        min(candidate.repeat_i1, end)
                        - max(candidate.repeat_i0, start),
                    )
                    union = max(candidate.repeat_i1, end) - min(
                        candidate.repeat_i0, start
                    )
                    iou = overlap / max(union, 1)
                    if iou > best_iou:
                        best, best_iou = index, iou
                if best is not None and best_iou >= 0.5:
                    used.add(best)
                    matched += 1
            tp += matched
            fp += len(predicted) - matched
            fn += len(truth) - matched
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": 2
            * precision
            * recall
            / max(precision + recall, 1e-12),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if (metrics["f1"], metrics["precision"]) > (
            best_metrics["f1"],
            best_metrics.get("precision", 0.0),
        ):
            best_threshold, best_metrics = float(threshold), metrics
    return best_threshold, best_metrics


def build_repetition_tensors(
    manifest: Path,
    split: str,
    *,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for index, path in enumerate(
        _manifest_maps(manifest, split, max_samples)
    ):
        rows, targets = candidate_rows(path, seed=seed + index)
        features.extend(rows)
        labels.extend(targets)
    if not features:
        raise ValueError(f"No repetition candidates for {split}")
    return (
        torch.from_numpy(np.stack(features).astype(np.float32)),
        torch.tensor(labels, dtype=torch.float32),
    )


def _metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    predicted = logits.sigmoid() >= threshold
    truth = targets > 0.5
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def train_note_repetition_model(
    manifest: Path,
    output: Path,
    *,
    train_samples: int = 1000,
    val_samples: int = 200,
    epochs: int = 25,
    batch_size: int = 512,
    device: str = "cuda",
    seed: int = 365,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y = build_repetition_tensors(
        manifest, "train", max_samples=train_samples, seed=seed
    )
    val_x, val_y = build_repetition_tensors(
        manifest, "val", max_samples=val_samples, seed=seed + 10000
    )
    model = NoteRepetitionScorer().to(device)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negatives / max(positives, 1.0), device=device
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    history = []
    best_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        model.eval()
        with torch.no_grad():
            logits = model(val_x.to(device)).cpu()
        best_threshold = 0.5
        metrics = _metrics(logits, val_y, best_threshold)
        for threshold in np.linspace(0.35, 0.90, 23):
            candidate = _metrics(logits, val_y, float(threshold))
            if (candidate["f1"], candidate["precision"]) > (
                metrics["f1"],
                metrics["precision"],
            ):
                metrics = candidate
                best_threshold = float(threshold)
        row = {
            "epoch": epoch,
            "train_loss": total / max(steps, 1),
            "threshold": best_threshold,
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_f1={metrics['f1']:.4f} threshold={best_threshold:.3f}",
            flush=True,
        )
        # Component diagnostic only: candidate-row F1 cannot promote a model.
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            model.config.threshold = best_threshold
    assert best_state is not None
    model.load_state_dict(best_state)
    span_threshold, span_metrics = calibrate_span_threshold(
        model,
        _manifest_maps(manifest, "val", val_samples),
        device=device,
    )
    model.config.threshold = span_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    save_note_repetition_model(
        output,
        model,
        extra={
            "format_version": 2,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_rows": len(train_y),
            "val_rows": len(val_y),
            "best_f1": best_f1,
            "span_calibration": {
                "threshold": span_threshold,
                **span_metrics,
            },
            "history": history,
            "feature_names": FEATURE_NAMES,
            "hard_negatives": {
                "kind": "matching-prefix-divergent-continuation",
                "maximum_per_positive": 1,
                "natural_negative_ratio": 2,
            },
        },
    )
    output.with_suffix(".history.json").write_text(
        json.dumps(
            {
                "best_f1": best_f1,
                "threshold": model.config.threshold,
                "span_calibration": span_metrics,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
