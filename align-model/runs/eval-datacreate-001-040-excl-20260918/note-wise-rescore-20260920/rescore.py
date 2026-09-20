"""Rescore DataCreate 001-040 exclusion subset with the new note-wise F1.

Uses current local labelling outputs in labels_agent.json (error-heads-v3)
against labels.json gold. Locations are projected with
labels_with_canonical_locations before exclusive note-wise matching.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ALIGN = ROOT / "align-model"
sys.path[:0] = [
    str(ALIGN / "scripts"),
    str(ALIGN / "src"),
    str(ROOT / "DataCreate" / "src"),
]

import eval_datacreate_current as base
from alignmodel.melody import (
    labels_with_canonical_locations,
    load_bundle_notes,
    micro_note_wise,
    official_label_metrics,
)

SAMPLES = ROOT / "DataCreate" / "samples"
OUT = Path(__file__).resolve().parent
EXCLUDE = {"005", "007", "010", "012", "020", "026", "030", "034", "036"}
KEEP = [f"{index:03d}" for index in range(1, 41) if f"{index:03d}" not in EXCLUDE]
TYPE_ORDER = [
    "wrong_note",
    "extra_note",
    "missed_note",
    "rhythm_error",
    "repetition",
    "click",
    "bad_start",
    "bad_timbre",
    "squeak",
    "sliding",
]


def _types(labels: list[dict]) -> dict[str, int]:
    return dict(Counter(str(label.get("type")) for label in labels))


def _finish(rows: list[dict]) -> dict:
    return micro_note_wise(
        [
            row["official_note_wise"]
            if "official_note_wise" in row
            else row
            for row in rows
        ]
    )


def _bootstrap(details: list[dict], seed: int = 20260920) -> dict:
    if not details:
        return {
            "replicates": 2000,
            "unit": "clip",
            "lower_95": 0.0,
            "median": 0.0,
            "upper_95": 0.0,
        }
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        selected = generator.integers(0, len(details), size=len(details))
        values.append(
            _finish([details[int(index)] for index in selected])["f1"]
        )
    return {
        "replicates": 2000,
        "unit": "clip",
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def score_clip(sample: str) -> dict:
    sample_dir = SAMPLES / sample
    gold_doc = json.loads((sample_dir / "labels.json").read_text(encoding="utf-8"))
    pred_doc = json.loads(
        (sample_dir / "labels_agent.json").read_text(encoding="utf-8")
    )
    notes = load_bundle_notes(sample_dir)
    gold_raw = [dict(label) for label in gold_doc.get("labels") or []]
    pred_raw = [dict(label) for label in pred_doc.get("labels") or []]
    gold = labels_with_canonical_locations(gold_raw, notes)
    pred = labels_with_canonical_locations(pred_raw, notes)
    metrics = official_label_metrics(
        gold, pred, score_event_count=len(notes) or None
    )
    per_type = {}
    kinds = sorted(
        {str(label.get("type")) for label in gold + pred if label.get("type")}
    )
    for kind in kinds:
        per_type[kind] = official_label_metrics(
            [label for label in gold if str(label.get("type")) == kind],
            [label for label in pred if str(label.get("type")) == kind],
            score_event_count=len(notes) or None,
        )
    return {
        "sample": sample,
        "gold_annotator_id": gold_doc.get("annotator_id"),
        "pred_annotator_id": pred_doc.get("annotator_id"),
        "pred_method": (pred_doc.get("agent_labeling") or {}).get("method"),
        "n_gold_raw": len(gold_raw),
        "n_pred_raw": len(pred_raw),
        "gold_types": _types(gold_raw),
        "pred_types": _types(pred_raw),
        "score_event_count": len(notes),
        **metrics,
        "per_type": per_type,
    }


def summarize(rows: list[dict], name: str) -> dict:
    available = [row for row in rows if row.get("status") == "available"]
    unavailable = [row["sample"] for row in rows if row.get("status") != "available"]
    micro = _finish(available)
    type_rows: dict[str, list[dict]] = {}
    for row in available:
        for kind, metrics in (row.get("per_type") or {}).items():
            if metrics.get("status") == "available":
                type_rows.setdefault(kind, []).append(metrics)
    kinds = [kind for kind in TYPE_ORDER if kind in type_rows] + [
        kind for kind in sorted(type_rows) if kind not in TYPE_ORDER
    ]
    return {
        "subset": name,
        "clips": [row["sample"] for row in rows],
        "available_clips": [row["sample"] for row in available],
        "unavailable_clips": unavailable,
        "official_note_wise": {
            **micro,
            "bootstrap_95_ci": _bootstrap(
                [row["official_note_wise"] for row in available]
            ),
            "matching_policy": (
                "labels_with_canonical_locations then exclusive "
                "score-event identity; same type=1.0, different type=0.5, "
                "wrong location=0"
            ),
            "per_type": {kind: _finish(type_rows[kind]) for kind in kinds},
        },
        "legacy_pitch_similarity": {
            "f1": float(
                np.mean([row["legacy_pitch_similarity_f1"] for row in rows] or [0.0])
            ),
            "note": "clip-mean legacy pitch-list F1; not headline",
        },
        "per_clip": [
            {
                "sample": row["sample"],
                "status": row["status"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "credit": row["credit"],
                "gold": row["gold"],
                "predicted": row["predicted"],
                "n_gold_raw": row["n_gold_raw"],
                "n_pred_raw": row["n_pred_raw"],
                "gold_types": row["gold_types"],
                "pred_types": row["pred_types"],
                "legacy_pitch_similarity_f1": row["legacy_pitch_similarity_f1"],
            }
            for row in rows
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [score_clip(sample) for sample in KEEP]
    nonempty = [row for row in rows if row["n_gold_raw"] > 0]
    previous_official = [
        row
        for row in rows
        if row["sample"] in {"001", "006", "017", "019", "029", "033"}
    ]
    report = {
        "schema_version": "align-datacreate-001-040-excl-new-note-wise-v1",
        "model": {
            "local_labelling": "error-heads-v3 in labels_agent.json",
            "annotator_id": "cursor_agent_error_heads_v3",
            "inference_rerun": False,
            "reason": (
                "F1 formula changed; predictions already stored in "
                "labels_agent.json from the completed v3 freeze"
            ),
        },
        "subset": {
            "requested": "DataCreate samples 001-040",
            "excluded": sorted(EXCLUDE),
            "kept": KEEP,
        },
        "metric_change": {
            "earlier": (
                "Clip rejected if any gold pitch list failed to validate the "
                "claimed score_part; repetitions without extra_copies were "
                "unevaluable"
            ),
            "now": (
                "Project timed/schema-1.1 labels onto score notes with "
                "labels_with_canonical_locations (repair pitches from the "
                "score; default extra_copies=1 for repetitions), then exclusive "
                "note-wise identity matching"
            ),
        },
        "nonempty_gold": summarize(nonempty, "nonempty_gold"),
        "all_kept_clips": summarize(rows, "all_31_kept"),
        "previous_six_official_clips": summarize(
            previous_official, "previous_six_audited_clips"
        ),
    }
    base._atomic_json(OUT / "report.json", report)
    headline = report["nonempty_gold"]["official_note_wise"]
    print(
        json.dumps(
            {
                "report": str(OUT / "report.json"),
                "nonempty_gold_clips": report["nonempty_gold"]["available_clips"],
                "unavailable": report["nonempty_gold"]["unavailable_clips"],
                "f1": headline["f1"],
                "precision": headline["precision"],
                "recall": headline["recall"],
                "credit": headline["credit"],
                "predicted": headline["predicted"],
                "gold": headline["gold"],
                "ci": headline["bootstrap_95_ci"],
                "all_31_f1": report["all_kept_clips"]["official_note_wise"]["f1"],
                "previous_six_f1": report["previous_six_official_clips"][
                    "official_note_wise"
                ]["f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
