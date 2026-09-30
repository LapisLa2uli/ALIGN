"""Compare two DataCreate agent-label sets against human labels note-wise."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import eval_datacreate_current as base
from alignmodel.melody import (
    labels_with_canonical_locations,
    load_bundle_notes,
    micro_note_wise,
    official_label_metrics,
)


def _micro(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return micro_note_wise(
        [row["official_note_wise"] for row in rows]
    )


def _score_document(
    sample_dir: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    gold_doc = json.loads(
        (sample_dir / "labels.json").read_text(encoding="utf-8")
    )
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    notes = load_bundle_notes(sample_dir)
    gold_raw = [dict(value) for value in gold_doc.get("labels") or []]
    predicted_raw = [
        dict(value) for value in prediction.get("labels") or []
    ]
    gold = labels_with_canonical_locations(gold_raw, notes)
    predicted = labels_with_canonical_locations(predicted_raw, notes)
    official = official_label_metrics(
        gold, predicted, score_event_count=len(notes) or None
    )
    kinds = sorted(
        {
            str(value.get("type"))
            for value in (*gold, *predicted)
            if value.get("type")
        }
    )
    per_type = {
        kind: official_label_metrics(
            [
                value
                for value in gold
                if str(value.get("type")) == kind
            ],
            [
                value
                for value in predicted
                if str(value.get("type")) == kind
            ],
            score_event_count=len(notes) or None,
        )
        for kind in kinds
    }
    return {
        "sample": sample_dir.name,
        "gold_count": len(gold_raw),
        "prediction_count": len(predicted_raw),
        "annotator_id": prediction.get("annotator_id"),
        "method": (prediction.get("agent_labeling") or {}).get("method"),
        "gold_types": dict(
            Counter(str(value.get("type")) for value in gold_raw)
        ),
        "prediction_types": dict(
            Counter(str(value.get("type")) for value in predicted_raw)
        ),
        "official_note_wise": official,
        "per_type": per_type,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [
        row
        for row in rows
        if row["official_note_wise"].get("status") == "available"
    ]
    type_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in available:
        for kind, metric in row["per_type"].items():
            if metric.get("status") == "available":
                type_rows.setdefault(kind, []).append(
                    {"official_note_wise": metric}
                )
    return {
        "clips": len(rows),
        "available_clips": len(available),
        "unavailable": [
            row["sample"]
            for row in rows
            if row["official_note_wise"].get("status") != "available"
        ],
        "raw_predictions": sum(
            int(row["prediction_count"]) for row in rows
        ),
        "official_note_wise": {
            **_micro(available),
            "per_type": {
                kind: _micro(values)
                for kind, values in sorted(type_rows.items())
            },
        },
    }


def _paired_bootstrap(
    old_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if len(old_rows) != len(new_rows):
        raise ValueError("Paired bootstrap populations differ")
    generator = np.random.default_rng(seed)
    old_values = []
    new_values = []
    deltas = []
    for _ in range(replicates):
        selected = generator.integers(0, len(old_rows), len(old_rows))
        old = _micro([old_rows[int(index)] for index in selected])["f1"]
        new = _micro([new_rows[int(index)] for index in selected])["f1"]
        old_values.append(old)
        new_values.append(new)
        deltas.append(new - old)
    return {
        "replicates": replicates,
        "unit": "clip",
        "old_f1": {
            "lower_95": float(np.quantile(old_values, 0.025)),
            "median": float(np.quantile(old_values, 0.5)),
            "upper_95": float(np.quantile(old_values, 0.975)),
        },
        "new_f1": {
            "lower_95": float(np.quantile(new_values, 0.025)),
            "median": float(np.quantile(new_values, 0.5)),
            "upper_95": float(np.quantile(new_values, 0.975)),
        },
        "delta_new_minus_old": {
            "lower_95": float(np.quantile(deltas, 0.025)),
            "median": float(np.quantile(deltas, 0.5)),
            "upper_95": float(np.quantile(deltas, 0.975)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--old-labels", type=Path, required=True)
    parser.add_argument("--new-labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    samples = sorted(
        path
        for path in args.samples.iterdir()
        if path.is_dir()
        and (path / "labels.json").is_file()
        and (path / "verified_score.musicxml").is_file()
    )
    if len(samples) != 94:
        raise ValueError(f"Expected 94 samples, found {len(samples)}")
    old_rows = []
    new_rows = []
    for sample in samples:
        old_path = args.old_labels / f"{sample.name}.json"
        new_path = (
            args.new_labels / f"{sample.name}.json"
            if args.new_labels is not None
            else sample / "labels_agent.json"
        )
        old_rows.append(_score_document(sample, old_path))
        new_rows.append(_score_document(sample, new_path))
    nonempty_indices = [
        index
        for index, row in enumerate(old_rows)
        if int(row["gold_count"]) > 0
    ]
    old_nonempty = [old_rows[index] for index in nonempty_indices]
    new_nonempty = [new_rows[index] for index in nonempty_indices]
    old_summary = _summary(old_nonempty)
    new_summary = _summary(new_nonempty)
    old_f1 = float(old_summary["official_note_wise"]["f1"])
    new_f1 = float(new_summary["official_note_wise"]["f1"])
    report = {
        "schema_version": "align-datacreate-agent-label-comparison-v1",
        "metric": {
            "unit": "canonical score-event identity",
            "matching": "exclusive one-to-one",
            "exact_type_credit": 1.0,
            "wrong_type_exact_location_credit": 0.5,
            "wrong_location_credit": 0.0,
            "timestamp_metrics": "not used",
        },
        "population": {
            "all_samples": len(samples),
            "nonempty_human_gold_samples": len(nonempty_indices),
            "samples": [sample.name for sample in samples],
        },
        "old": old_summary,
        "new": new_summary,
        "delta_new_minus_old_f1": new_f1 - old_f1,
        "winner_on_nonempty_human_gold": (
            "new" if new_f1 > old_f1
            else "old" if old_f1 > new_f1
            else "tie"
        ),
        "paired_bootstrap": _paired_bootstrap(
            old_nonempty,
            new_nonempty,
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        ),
        "per_clip": [
            {
                "sample": old["sample"],
                "gold_count": old["gold_count"],
                "old_prediction_count": old["prediction_count"],
                "new_prediction_count": new["prediction_count"],
                "old_f1": old["official_note_wise"]["f1"],
                "new_f1": new["official_note_wise"]["f1"],
                "delta": (
                    float(new["official_note_wise"]["f1"])
                    - float(old["official_note_wise"]["f1"])
                ),
            }
            for old, new in zip(old_rows, new_rows)
        ],
    }
    base._atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "old_f1": old_f1,
                "new_f1": new_f1,
                "delta": new_f1 - old_f1,
                "winner": report["winner_on_nonempty_human_gold"],
                "nonempty_human_gold_samples": len(nonempty_indices),
                "paired_delta_ci": report["paired_bootstrap"][
                    "delta_new_minus_old"
                ],
            },
            indent=2,
        )
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
