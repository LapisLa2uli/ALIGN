"""Official note-wise comparison of Polytune, LadderSym, and AudioEval on 095-124.

Gold is the current primary ``labels.json``. Provenance is recorded per clip:
manual-source documents versus agent-copied documents. Timestamp scores are
diagnostics only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "align-model" / "src"), str(ROOT / "DataCreate" / "src")]

from datacreate.melody import (
    canonical_note_location,
    match_note_wise_labels_detail,
    parse_sounding_notes,
)

SCORED_TYPES = (
    "wrong_note",
    "missed_note",
    "extra_note",
    "rhythm_error",
    "repetition",
)
EVAL_IDS = [f"{index:03d}" for index in range(95, 125)]
MODELS = ("polytune", "laddersym", "audioeval")
REPORT_SCHEMA = "align-datacreate-095-124-three-system-note-wise-v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scored(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(label)
        for label in document.get("labels") or []
        if label.get("type") in SCORED_TYPES
    ]


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    credit = sum(float(row["credit"]) for row in rows)
    predicted = sum(int(row["predicted"]) for row in rows)
    gold = sum(int(row["gold"]) for row in rows)
    precision = credit / predicted if predicted else 0.0
    recall = credit / gold if gold else 0.0
    if not predicted and not gold:
        precision = recall = 1.0
    return {
        "credit": credit,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "pair_counts": {
            "full_credit": int(
                sum(int((row.get("pair_counts") or {}).get("full_credit") or 0) for row in rows)
            ),
            "half_credit": int(
                sum(int((row.get("pair_counts") or {}).get("half_credit") or 0) for row in rows)
            ),
            "zero_credit": int(
                sum(int((row.get("pair_counts") or {}).get("zero_credit") or 0) for row in rows)
            ),
        },
        "supports": {"predicted": predicted, "gold": gold},
        "clips": len(rows),
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    if not rows:
        return {
            "replicates": 0,
            "unit": "clip",
            "seed": seed,
            "lower_95": 0.0,
            "median": 0.0,
            "upper_95": 0.0,
        }
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(1000):
        selected = generator.integers(0, len(rows), size=len(rows))
        values.append(_aggregate([rows[int(index)] for index in selected])["f1"])
    return {
        "replicates": 1000,
        "unit": "clip",
        "seed": seed,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _audit_gold(sample_dir: Path) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    path = sample_dir / "labels.json"
    document = _read(path)
    labels = _scored(document)
    notes = parse_sounding_notes(sample_dir / "verified_score.musicxml")
    sources = Counter(str(label.get("source") or "") for label in labels)
    reasons: list[str] = []
    for label in labels:
        if canonical_note_location(label, score_event_count=len(notes)) is None:
            reasons.append("missing or invalid canonical score-event identity")
            break
        part = label.get("score_part")
        if isinstance(part, dict):
            try:
                first = int(part["start_note_index"])
                last = int(part["end_note_index"])
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid score_part")
                break
            expected = [int(note.pitch) for note in notes[first : last + 1]]
            supplied = label.get("pitches")
            if not isinstance(supplied, list) or [int(value) for value in supplied] != expected:
                reasons.append("pitches do not exactly validate the canonical range")
                break
    provenance = "agent" if sources.get("agent") else ("manual" if sources.get("manual") else "empty")
    audit = {
        "official_note_wise": "available" if not reasons else "unavailable",
        "reason": "validated" if not reasons else reasons[0],
        "gold_labels": len(labels),
        "schema_version": document.get("schema_version"),
        "annotator_id": document.get("annotator_id"),
        "label_sources": dict(sources),
        "gold_provenance": provenance,
        "labels_sha256": sha256(path),
        "score_event_count": len(notes),
        "type_counts": dict(Counter(str(label.get("type")) for label in labels)),
    }
    return (None, audit) if reasons else (labels, audit)


def _legacy_timestamp(gold: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        start = max(float(left["start_time"]), float(right["start_time"]))
        end = min(float(left["end_time"]), float(right["end_time"]))
        inter = max(0.0, end - start)
        union = max(float(left["end_time"]), float(right["end_time"])) - min(
            float(left["start_time"]), float(right["start_time"])
        )
        return inter / union if union > 0 else 0.0

    out = {}
    for name, (mode, threshold) in {
        "onset_50ms": ("onset", 0.05),
        "iou_0.3": ("iou", 0.3),
    }.items():
        used_g: set[int] = set()
        matches = 0
        for prediction in pred:
            best = None
            best_g = None
            for g_index, target in enumerate(gold):
                if g_index in used_g or prediction.get("type") != target.get("type"):
                    continue
                if mode == "onset":
                    score = abs(float(prediction["start_time"]) - float(target["start_time"]))
                    ok = score <= threshold
                    rank = -score
                else:
                    score = iou(prediction, target)
                    ok = score >= threshold
                    rank = score
                if ok and (best is None or rank > best):
                    best = rank
                    best_g = g_index
            if best_g is not None:
                used_g.add(best_g)
                matches += 1
        predicted = len(pred)
        gold_n = len(gold)
        precision = matches / predicted if predicted else 0.0
        recall = matches / gold_n if gold_n else 0.0
        out[name] = {
            "matches": matches,
            "predicted": predicted,
            "gold": gold_n,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
    return out


def _prediction_path(model: str, sample_id: str, samples_root: Path, freeze_root: Path) -> Path:
    if model == "audioeval":
        return freeze_root / "predictions" / f"{sample_id}.json"
    return samples_root / sample_id / f"labels_{model}.json"


def evaluate_model(
    model: str,
    samples_root: Path,
    freeze_root: Path,
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    per_type_rows: dict[str, list[dict[str, Any]]] = {kind: [] for kind in SCORED_TYPES}
    per_clip = []
    legacy_rows = []
    missing = []
    for sample_id, targets in gold.items():
        path = _prediction_path(model, sample_id, samples_root, freeze_root)
        if not path.is_file():
            missing.append(sample_id)
            continue
        predicted = _scored(_read(path))
        notes = parse_sounding_notes(samples_root / sample_id / "verified_score.musicxml")
        detail = match_note_wise_labels_detail(
            list(targets),
            predicted,
            score_event_count=len(notes),
        )
        if detail["status"] != "available":
            raise ValueError(f"{model}/{sample_id}: {detail}")
        rows.append(detail)
        legacy_rows.append(_legacy_timestamp(targets, predicted))
        per_clip.append(
            {
                "sample": sample_id,
                "credit": detail["credit"],
                "predicted": detail["predicted"],
                "gold": detail["gold"],
                "precision": detail["precision"],
                "recall": detail["recall"],
                "f1": detail["f1"],
                "pair_counts": detail["pair_counts"],
            }
        )
        for kind in SCORED_TYPES:
            per_type_rows[kind].append(
                match_note_wise_labels_detail(
                    [value for value in targets if value.get("type") == kind],
                    [value for value in predicted if value.get("type") == kind],
                    score_event_count=len(notes),
                )
            )
    if missing:
        return {
            "official_note_wise": {
                "status": "unavailable",
                "reason": f"missing predictions for {missing}",
                "missing_ids": missing,
            }
        }
    micro = _aggregate(rows)
    per_type = {kind: _aggregate(kind_rows) for kind, kind_rows in per_type_rows.items()}
    supported = [block for block in per_type.values() if block["gold"] > 0]
    legacy = {}
    for name in ("onset_50ms", "iou_0.3"):
        matches = sum(int(row[name]["matches"]) for row in legacy_rows)
        predicted = sum(int(row[name]["predicted"]) for row in legacy_rows)
        gold_n = sum(int(row[name]["gold"]) for row in legacy_rows)
        precision = matches / predicted if predicted else 0.0
        recall = matches / gold_n if gold_n else 0.0
        legacy[name] = {
            "matches": matches,
            "predicted": predicted,
            "gold": gold_n,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
    return {
        "official_note_wise": {
            **micro,
            "status": "available",
            "bootstrap_95_ci": _bootstrap(rows, 20260921),
            "per_type": per_type,
            "macro_f1": float(np.mean([block["f1"] for block in supported] or [0.0])),
            "matching_policy": (
                "exclusive one-to-one canonical score-event identity; "
                "1.0 same location and type; 0.5 same location different type; "
                "0 wrong location"
            ),
            "metric_schema": "align-note-wise-score-event-metric-v1",
        },
        "legacy_timestamp_diagnostics": legacy,
        "per_clip": per_clip,
    }


def subset_eval(
    model: str,
    samples_root: Path,
    freeze_root: Path,
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    ids: Sequence[str],
) -> dict[str, Any]:
    filtered = {sample_id: gold[sample_id] for sample_id in ids if sample_id in gold}
    return evaluate_model(model, samples_root, freeze_root, filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "DataCreate" / "samples")
    parser.add_argument(
        "--freeze-root",
        type=Path,
        default=ROOT / "align-model" / "runs" / "eval-datacreate-095-124-20260921",
    )
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    samples_root = args.samples.resolve()
    freeze_root = args.freeze_root.resolve()
    gold: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for sample_id in EVAL_IDS:
        labels, row = _audit_gold(samples_root / sample_id)
        audit[sample_id] = row
        if labels is not None:
            gold[sample_id] = labels
    manual_ids = [
        sample_id
        for sample_id, row in audit.items()
        if row["official_note_wise"] == "available" and row["gold_provenance"] == "manual"
    ]
    agent_ids = [
        sample_id
        for sample_id, row in audit.items()
        if row["official_note_wise"] == "available" and row["gold_provenance"] == "agent"
    ]
    models = {}
    for model in MODELS:
        models[model] = {
            "all_auditable": evaluate_model(model, samples_root, freeze_root, gold),
            "manual_source": subset_eval(model, samples_root, freeze_root, gold, manual_ids),
            "agent_source": subset_eval(model, samples_root, freeze_root, gold, agent_ids),
        }
    freeze_manifest = freeze_root / "freeze_manifest.json"
    report = {
        "schema_version": REPORT_SCHEMA,
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "created_utc": _utc(),
        "ids": EVAL_IDS,
        "subset": {
            "requested": EVAL_IDS,
            "official_note_wise_available": sorted(gold),
            "official_note_wise_unavailable": [
                sample_id for sample_id in EVAL_IDS if sample_id not in gold
            ],
            "manual_source_ids": manual_ids,
            "agent_source_ids": agent_ids,
        },
        "audit": audit,
        "scored_types": list(SCORED_TYPES),
        "models": models,
        "audioeval_freeze_manifest": str(freeze_manifest) if freeze_manifest.is_file() else None,
        "audioeval_freeze_sha256": sha256(freeze_manifest) if freeze_manifest.is_file() else None,
        "promotion": "experimental",
        "legacy_timestamp_used_for_selection": False,
        "gold_note": (
            "Primary labels.json is the working gold. Clips whose labels have "
            "source=agent were copied from AudioEval-family agent proposals and "
            "are not independent human annotations. Empty human files before that "
            "copy are not treated as clean negatives."
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "integrity.json").write_text(
        json.dumps(
            {
                "schema_version": "align-datacreate-095-124-integrity-v1",
                "report_sha256": sha256(report_path),
                "passed": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
