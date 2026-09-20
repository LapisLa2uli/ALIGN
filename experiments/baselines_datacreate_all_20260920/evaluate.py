"""Official note-wise evaluation of baseline GUI labels on DataCreate 001-030."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "align-model" / "src"), str(ROOT / "DataCreate" / "src")]

from datacreate.melody import (
    canonical_note_location,
    match_note_wise_labels_detail,
    parse_sounding_notes,
)

HERE = Path(__file__).resolve().parent
SCORED_TYPES = (
    "wrong_note",
    "missed_note",
    "extra_note",
    "rhythm_error",
    "repetition",
)
EVAL_IDS = [f"{index:03d}" for index in range(1, 31)]
MODELS = ("polytune", "laddersym")
LABEL_FILES = {
    "human": "labels.json",
    "polytune": "labels_polytune.json",
    "laddersym": "labels_laddersym.json",
}
REPORT_SCHEMA = "align-datacreate-baseline-note-wise-report-v1"


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


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    credit = sum(float(row["credit"]) for row in rows)
    predicted = sum(int(row["predicted"]) for row in rows)
    gold = sum(int(row["gold"]) for row in rows)
    precision = credit / predicted if predicted else 0.0
    recall = credit / gold if gold else 0.0
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
            "full_credit": int(sum(int((row.get("pair_counts") or {}).get("full_credit") or 0) for row in rows)),
            "half_credit": int(sum(int((row.get("pair_counts") or {}).get("half_credit") or 0) for row in rows)),
            "zero_credit": int(sum(int((row.get("pair_counts") or {}).get("zero_credit") or 0) for row in rows)),
        },
        "supports": {
            "predicted": predicted,
            "gold": gold,
        },
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        selected = generator.integers(0, len(rows), size=len(rows))
        values.append(_aggregate([rows[int(index)] for index in selected])["f1"])
    return {
        "replicates": 2000,
        "unit": "clip",
        "seed": seed,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _scored(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(label)
        for label in document.get("labels") or []
        if label.get("type") in SCORED_TYPES
    ]


def _audit_gold(sample_dir: Path) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    document = _read(sample_dir / LABEL_FILES["human"])
    labels = _scored(document)
    notes = parse_sounding_notes(sample_dir / "verified_score.musicxml")
    reasons = []
    for label in labels:
        if canonical_note_location(label, score_event_count=len(notes)) is None:
            reasons.append("missing or invalid canonical score-event identity")
            break
        part = label.get("score_part")
        if isinstance(part, dict):
            first = int(part["start_note_index"])
            last = int(part["end_note_index"])
            expected = [int(note.pitch) for note in notes[first : last + 1]]
            supplied = label.get("pitches")
            if not isinstance(supplied, list) or [int(value) for value in supplied] != expected:
                reasons.append("pitches do not exactly validate the canonical range")
                break
    audit = {
        "official_note_wise": "available" if not reasons else "unavailable",
        "reason": "validated" if not reasons else reasons[0],
        "gold_labels": len(labels),
        "schema_version": document.get("schema_version"),
        "score_event_count": len(notes),
    }
    return (None, audit) if reasons else (labels, audit)


def _iou(pred: Mapping[str, Any], gold: Mapping[str, Any]) -> float:
    start = max(float(pred["start_time"]), float(gold["start_time"]))
    end = min(float(pred["end_time"]), float(gold["end_time"]))
    inter = max(0.0, end - start)
    union = max(float(pred["end_time"]), float(gold["end_time"])) - min(
        float(pred["start_time"]), float(gold["start_time"])
    )
    return inter / union if union > 0 else 0.0


def _legacy_timestamp(gold: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    criteria = {
        "onset_50ms": ("onset", 0.05),
        "onset_100ms": ("onset", 0.10),
        "onset_200ms": ("onset", 0.20),
        "iou_0.3": ("iou", 0.3),
        "iou_0.5": ("iou", 0.5),
    }
    out = {}
    for name, (mode, threshold) in criteria.items():
        used_g: set[int] = set()
        used_p: set[int] = set()
        matches = 0
        for p_index, prediction in enumerate(pred):
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
                    score = _iou(prediction, target)
                    ok = score >= threshold
                    rank = score
                if ok and (best is None or rank > best):
                    best = rank
                    best_g = g_index
            if best_g is not None:
                used_g.add(best_g)
                used_p.add(p_index)
                matches += 1
        predicted = len(pred)
        target_n = len(gold)
        precision = matches / predicted if predicted else 0.0
        recall = matches / target_n if target_n else 0.0
        out[name] = {
            "matches": matches,
            "predicted": predicted,
            "gold": target_n,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
    return out


def evaluate_model(
    samples_root: Path,
    model: str,
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    per_type_rows: dict[str, list[dict[str, Any]]] = {kind: [] for kind in SCORED_TYPES}
    per_clip = []
    legacy_rows = []
    for sample_id, targets in gold.items():
        predicted = _scored(_read(samples_root / sample_id / LABEL_FILES[model]))
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
                "unevaluable_prediction_indices": detail["unevaluable_prediction_indices"],
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
    micro = _aggregate(rows)
    per_type = {kind: _aggregate(kind_rows) for kind, kind_rows in per_type_rows.items()}
    supported = [block for block in per_type.values() if block["gold"] > 0]
    legacy = {}
    for name in ("onset_50ms", "onset_100ms", "onset_200ms", "iou_0.3", "iou_0.5"):
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
            "bootstrap_95_ci": _bootstrap(rows, 20260920),
            "per_type": per_type,
            "macro_f1": float(np.mean([block["f1"] for block in supported] or [0.0])),
            "matching_policy": (
                "exclusive one-to-one canonical score-event identity; "
                "1.0 same location and type; 0.5 same location different type; "
                "0 wrong location"
            ),
            "metric_schema": "align-note-wise-score-event-metric-v1",
            "status": "available",
        },
        "legacy_timestamp_diagnostics": legacy,
        "per_clip": per_clip,
    }


def write_findings(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# DataCreate PolyTune / LadderSym findings",
        "",
        "The 2026-09-18 retrain scored 40 real clips with timestamp error-event F1 because many gold documents failed canonical location audit. This run writes score-linked schema 1.2 GUI label sets for every DataCreate clip and scores numeric IDs 001-030 with the official note-wise metric. The result stays experimental: official F1 is reported only on auditable gold clips, and timestamp numbers remain diagnostics.",
        "",
        f"Labeled clips: {report['subset']['labeled_clips']}. Evaluated IDs: 001-030 ({report['subset']['requested']}). Officially auditable: {report['subset']['official_note_wise_available']}. Unavailable: {report['subset']['official_note_wise_unavailable']}.",
        "",
        "| Model | Status | Precision | Recall | F1 | Credit | Pred / Gold | Full / half |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        official = report["models"][model]["official_note_wise"]
        pairs = official["pair_counts"]
        lines.append(
            "| {model} | {status} | {p:.6f} | {r:.6f} | {f:.6f} | {c:.1f} | {n}/{g} | {full}/{half} |".format(
                model=model,
                status=official["status"],
                p=official["precision"],
                r=official["recall"],
                f=official["f1"],
                c=official["credit"],
                n=official["predicted"],
                g=official["gold"],
                full=pairs["full_credit"],
                half=pairs["half_credit"],
            )
        )
    lines += [
        "",
        "Unavailable gold IDs: " + ", ".join(report["subset"]["unavailable_ids"] or ["none"]) + ".",
        "",
        "Per-type official F1 is in `report.json`. Legacy onset/IoU diagnostics are under `legacy_timestamp_diagnostics` and were not used for ranking.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "DataCreate" / "samples")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args(argv)
    samples_root = args.samples.resolve()
    gold = {}
    audit = {}
    for sample_id in EVAL_IDS:
        labels, row = _audit_gold(samples_root / sample_id)
        audit[sample_id] = row
        if labels is not None:
            gold[sample_id] = labels
    labeled = sorted(
        path.name
        for path in samples_root.iterdir()
        if path.is_dir()
        and (path / LABEL_FILES["polytune"]).is_file()
        and (path / LABEL_FILES["laddersym"]).is_file()
    )
    models = {model: evaluate_model(samples_root, model, gold) for model in MODELS}
    report = {
        "schema_version": REPORT_SCHEMA,
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "created_utc": _utc(),
        "subset": {
            "requested": EVAL_IDS,
            "labeled_clips": len(labeled),
            "labeled_ids": labeled,
            "official_note_wise_available": sorted(gold),
            "official_note_wise_unavailable": [
                sample_id for sample_id in EVAL_IDS if sample_id not in gold
            ],
            "unavailable_ids": [
                sample_id for sample_id in EVAL_IDS if sample_id not in gold
            ],
        },
        "audit": audit,
        "scored_types": list(SCORED_TYPES),
        "models": models,
        "promotion": "experimental",
        "legacy_timestamp_used_for_selection": False,
        "human_labels_json_touched": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_findings(report, output / "FINDINGS.md")
    (output / "integrity.json").write_text(
        json.dumps(
            {
                "schema_version": "align-datacreate-baseline-note-wise-integrity-v1",
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
