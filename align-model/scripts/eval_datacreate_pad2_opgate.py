"""Re-decode DataCreate 001-040 exclusions with pad=2 and operation gating.

Leaves the frozen production v3 decode_config.json unmodified. Overlays
pad_notes=2 (annotator identity) and require_path_operation_gate on the
in-memory SchemaDecodeConfig, then scores official note-wise F1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ALIGN = Path(__file__).resolve().parents[1]
ROOT = ALIGN.parent
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(ALIGN / "src"),
    str(ROOT / "DataCreate" / "src"),
]

import numpy as np
import torch

import eval_datacreate_current as base
import eval_datacreate_error_heads_v3 as frozen_eval
from alignmodel.melody import (
    labels_with_canonical_locations,
    load_bundle_notes,
    micro_note_wise,
    official_label_metrics,
)

SAMPLES = ROOT / "DataCreate" / "samples"
OUT_DEFAULT = ALIGN / "runs" / "eval-datacreate-pad2-opgate-20260920"
EXCLUDE = {"005", "007", "010", "012", "020", "026", "030", "034", "036"}
KEEP = [f"{index:03d}" for index in range(1, 41) if f"{index:03d}" not in EXCLUDE]
OVERLAY = {
    "pad_notes": 2,
    "identity_span": "padded",
    "require_path_operation_gate": True,
}
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
HEADLINE = "v3"


def _types(labels: list[dict]) -> dict[str, int]:
    return dict(Counter(str(label.get("type")) for label in labels))


def _finish(rows: list[dict]) -> dict:
    return micro_note_wise(
        [
            row["official_note_wise"] if "official_note_wise" in row else row
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
    ordered = sorted(values)
    return {
        "replicates": 2000,
        "unit": "clip",
        "lower_95": float(ordered[int(0.025 * (len(ordered) - 1))]),
        "median": float(ordered[len(ordered) // 2]),
        "upper_95": float(ordered[int(0.975 * (len(ordered) - 1))]),
    }


def _score_clip(sample: str, pred_path: Path) -> dict:
    gold_doc = base._json(SAMPLES / sample / "labels.json")
    pred_doc = base._json(pred_path)
    notes = load_bundle_notes(SAMPLES / sample)
    gold_raw = [dict(value) for value in gold_doc.get("labels") or []]
    pred_raw = [dict(value) for value in pred_doc.get("labels") or []]
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
        "pred_pipeline": pred_doc.get("pipeline") or {},
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
            }
            for row in rows
        ],
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    frozen_eval._defaults(parser)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    if args.output == parser.get_default("output"):
        args.output = OUT_DEFAULT
    return args


def decode(args: argparse.Namespace) -> list[dict]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    overlay_path = output / "decode_overlay.json"
    base._atomic_json(
        overlay_path,
        {
            "schema_version": "align-error-heads-v3-decode-overlay-v1",
            "frozen_decode_config": str(args.v3_decode_config.resolve()),
            "frozen_decode_config_sha256": base._sha256(
                args.v3_decode_config.resolve()
            ),
            "frozen_decode_config_unmodified": True,
            "overlay": OVERLAY,
            "rationale": {
                "pad_notes": (
                    "DataCreate gold writes pad_notes=2; official identity is "
                    "the padded start/end, not the core"
                ),
                "require_path_operation_gate": (
                    "Emit wrong only on SUBSTITUTE, extra only on EXTRA, "
                    "miss only on DELETE; skip MATCH rows"
                ),
            },
        },
    )
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    status = base._training_status(args.resource_status.resolve())
    stack = frozen_eval._load_v3_stack(args)
    stack["v3_config"] = replace(stack["v3_config"], **OVERLAY)
    stack["v3_balanced_config"] = replace(stack["v3_balanced_config"], **OVERLAY)
    rows = []
    for position, name in enumerate(KEEP, 1):
        sample = args.samples.resolve() / name
        print(f"pad2_opgate {position}/{len(KEEP)} {name}", flush=True)
        try:
            rows.append(frozen_eval._freeze_one(sample, output, stack))
        except BaseException as exc:
            rows.append(
                {
                    "sample": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    manifest = {
        "schema_version": "align-datacreate-pad2-opgate-freeze-v1",
        "created_utc": base._utc(),
        "kept": KEEP,
        "excluded": sorted(EXCLUDE),
        "overlay": OVERLAY,
        "training_resource_status": status,
        "rows": rows,
    }
    base._atomic_json(output / "freeze_manifest.json", manifest)
    return rows


def score(output: Path) -> dict:
    models = ("v3", "v3_balanced")
    by_model = {}
    for model in models:
        rows = [
            _score_clip(
                sample, output / "predictions" / model / f"{sample}.json"
            )
            for sample in KEEP
        ]
        nonempty = [row for row in rows if row["n_gold_raw"] > 0]
        by_model[model] = {
            "nonempty_gold": summarize(nonempty, "nonempty_gold"),
            "all_kept_clips": summarize(rows, "all_31_kept"),
        }
    headline = by_model[HEADLINE]["nonempty_gold"]["official_note_wise"]
    report = {
        "schema_version": "align-datacreate-pad2-opgate-report-v1",
        "model": {
            "local_labelling": "error-heads-v3 with pad=2 and operation gate",
            "frozen_production_decode_config_unmodified": True,
            "overlay": OVERLAY,
            "headline": HEADLINE,
        },
        "subset": {
            "requested": "DataCreate samples 001-040",
            "excluded": sorted(EXCLUDE),
            "kept": KEEP,
        },
        "metric": {
            "earlier_v3": (
                "pad_notes=1 padded identity and classifier-fired extra/miss/"
                "wrong, including MATCH-row wrongs"
            ),
            "limitation": (
                "Official note-wise F1 was 0 because gold uses pad_notes=2 "
                "and MATCH rows over-fired"
            ),
            "now": (
                "Padded identity with pad_notes=2; extra/miss/wrong gated on "
                "joint EXTRA/DELETE/SUBSTITUTE"
            ),
            "status": "experimental decode overlay; not promoted",
        },
        "by_model": by_model,
        "headline_nonempty_gold": headline,
    }
    base._atomic_json(output / "report.json", report)
    return report


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = _args()
    output = args.output.resolve()
    if not args.skip_decode:
        decode(args)
    report = score(output)
    headline = report["headline_nonempty_gold"]
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "overlay": OVERLAY,
                "nonempty_gold_clips": report["by_model"][HEADLINE][
                    "nonempty_gold"
                ]["available_clips"],
                "f1": headline["f1"],
                "precision": headline["precision"],
                "recall": headline["recall"],
                "credit": headline["credit"],
                "predicted": headline["predicted"],
                "gold": headline["gold"],
                "ci": headline["bootstrap_95_ci"],
                "all_31_f1": report["by_model"][HEADLINE]["all_kept_clips"][
                    "official_note_wise"
                ]["f1"],
                "v3_balanced_nonempty_f1": report["by_model"]["v3_balanced"][
                    "nonempty_gold"
                ]["official_note_wise"]["f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
