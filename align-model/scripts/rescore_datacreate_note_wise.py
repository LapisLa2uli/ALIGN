"""Gold-isolated canonical note-wise rescore of frozen DataCreate predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from alignmodel.melody import (
    canonical_note_location,
    match_note_wise_labels_detail,
    parse_sounding_notes,
)

import eval_datacreate_current as base
import eval_datacreate_error_heads_v3 as v3eval


SCHEMA = "align-datacreate-note-wise-rescore-v1"
SCORED_TYPES = {
    "wrong_note",
    "missed_note",
    "extra_note",
    "rhythm_error",
    "repetition",
}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    credit = sum(float(value["credit"]) for value in rows)
    predicted = sum(int(value["predicted"]) for value in rows)
    target = sum(int(value["gold"]) for value in rows)
    precision = credit / predicted if predicted else 0.0
    recall = credit / target if target else 0.0
    return {
        "credit": credit,
        "predicted": predicted,
        "gold": target,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def _bootstrap(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        selected = generator.integers(0, len(rows), size=len(rows))
        values.append(_aggregate([rows[index] for index in selected])["f1"])
    return {
        "replicates": 2000,
        "unit": "clip",
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _validated_gold(sample_dir: Path) -> tuple[list[dict[str, Any]] | None, str]:
    document = base._json(sample_dir / "labels.json")
    labels = [
        dict(value)
        for value in document.get("labels") or []
        if value.get("type") in SCORED_TYPES
    ]
    notes = parse_sounding_notes(sample_dir / "verified_score.musicxml")
    for label in labels:
        if canonical_note_location(label, score_event_count=len(notes)) is None:
            return None, "missing or invalid canonical score-event identity"
        part = label.get("score_part")
        if isinstance(part, dict):
            first = int(part["start_note_index"])
            last = int(part["end_note_index"])
            expected = [value.pitch for value in notes[first : last + 1]]
            if [int(value) for value in label.get("pitches") or []] != expected:
                return None, "pitch audit failed for canonical score range"
    return labels, "validated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, errors = v3eval._verify_freeze(args.freeze_manifest.resolve())
    if errors:
        raise ValueError(f"Frozen predictions failed integrity: {errors}")
    source_report = base._json(args.source_report.resolve())
    expected_ids = list(source_report["subset"]["sample_ids"])
    samples_root = Path(manifest["samples_root"])
    audit: dict[str, Any] = {}
    gold = {}
    for sample in expected_ids:
        labels, reason = _validated_gold(samples_root / sample)
        audit[sample] = {
            "official_note_wise": (
                "available" if labels is not None else "unavailable"
            ),
            "reason": reason,
        }
        if labels is not None:
            gold[sample] = labels
    documents = {
        name: v3eval._documents(manifest, name)
        for name in ("v3", "v3_balanced", "v2", "rules")
    }
    metrics = {}
    for name, by_sample in documents.items():
        rows = []
        per_type_rows = {kind: [] for kind in sorted(SCORED_TYPES)}
        for sample, targets in gold.items():
            predicted = [
                dict(value)
                for value in by_sample[sample].get("labels") or []
                if value.get("type") in SCORED_TYPES
            ]
            notes = parse_sounding_notes(
                samples_root / sample / "verified_score.musicxml"
            )
            detail = match_note_wise_labels_detail(
                targets,
                predicted,
                score_event_count=len(notes),
            )
            if detail["status"] != "available":
                raise ValueError(f"Prediction rescore unavailable: {name}/{sample}")
            rows.append(detail)
            for kind in per_type_rows:
                kind_detail = match_note_wise_labels_detail(
                    [value for value in targets if value.get("type") == kind],
                    [value for value in predicted if value.get("type") == kind],
                    score_event_count=len(notes),
                )
                per_type_rows[kind].append(kind_detail)
        micro = _aggregate(rows)
        metrics[name] = {
            **micro,
            "bootstrap_95_ci": _bootstrap(rows, 20260915),
            "per_type": {
                kind: _aggregate(kind_rows)
                for kind, kind_rows in per_type_rows.items()
            },
            "macro_f1": float(
                np.mean(
                    [
                        _aggregate(kind_rows)["f1"]
                        for kind, kind_rows in per_type_rows.items()
                        if _aggregate(kind_rows)["gold"] > 0
                    ]
                    or [0.0]
                )
            ),
        }
    report = {
        "schema_version": SCHEMA,
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "source_freeze_manifest": str(args.freeze_manifest.resolve()),
        "source_freeze_manifest_sha256": base._sha256(
            args.freeze_manifest.resolve()
        ),
        "source_legacy_report": str(args.source_report.resolve()),
        "source_legacy_report_sha256": base._sha256(args.source_report.resolve()),
        "subset": {
            "agent_labelled": len(expected_ids),
            "official_note_wise_available": len(gold),
            "official_note_wise_unavailable": len(expected_ids) - len(gold),
            "available_sample_ids": sorted(gold),
        },
        "audit": audit,
        "metrics": metrics,
        "legacy_timestamp_used_for_selection": False,
        "production_mutated": False,
        "lockbox_touched": False,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    base._atomic_json(output, report)
    base._atomic_json(
        output.with_name("integrity.json"),
        {
            "schema_version": "align-datacreate-note-wise-rescore-integrity-v1",
            "report_sha256": base._sha256(output),
            "freeze_manifest_sha256": base._sha256(
                args.freeze_manifest.resolve()
            ),
            "legacy_report_sha256": base._sha256(args.source_report.resolve()),
            "passed": True,
            "production_mutated": False,
            "lockbox_touched": False,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
