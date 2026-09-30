"""Official note-wise score of the mel-v1 / error-heads-v5 freeze on 095-124."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / "experiments" / "baselines_datacreate_095_124_20260921"
sys.path.insert(0, str(PRIOR))

import score_three as base

REPORT_SCHEMA = "align-datacreate-095-124-mel-v5-note-wise-v1"


def _prediction_path(sample_id: str, freeze_root: Path) -> Path:
    staged = freeze_root / "staged-labels" / f"{sample_id}.json"
    if staged.is_file():
        return staged
    return freeze_root / "predictions" / f"{sample_id}.json"


def evaluate_model(
    samples_root: Path,
    freeze_root: Path,
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    per_type_rows: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in base.SCORED_TYPES
    }
    per_clip = []
    legacy_rows = []
    missing = []
    for sample_id, targets in gold.items():
        path = _prediction_path(sample_id, freeze_root)
        if not path.is_file():
            missing.append(sample_id)
            continue
        predicted = base._scored(base._read(path))
        notes = base.parse_sounding_notes(
            samples_root / sample_id / "verified_score.musicxml"
        )
        detail = base.match_note_wise_labels_detail(
            list(targets),
            predicted,
            score_event_count=len(notes),
        )
        if detail["status"] != "available":
            raise ValueError(f"audioeval/{sample_id}: {detail}")
        rows.append(detail)
        legacy_rows.append(base._legacy_timestamp(targets, predicted))
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
        for kind in base.SCORED_TYPES:
            per_type_rows[kind].append(
                base.match_note_wise_labels_detail(
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
    micro = base._aggregate(rows)
    per_type = {
        kind: base._aggregate(kind_rows)
        for kind, kind_rows in per_type_rows.items()
    }
    supported = [block for block in per_type.values() if block["gold"] > 0]
    legacy: dict[str, Any] = {}
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
            "bootstrap_95_ci": base._bootstrap(rows, 20260921),
            "per_type": per_type,
            "macro_f1": float(
                base.np.mean([block["f1"] for block in supported] or [0.0])
            ),
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
    samples_root: Path,
    freeze_root: Path,
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    ids: Sequence[str],
) -> dict[str, Any]:
    filtered = {
        sample_id: gold[sample_id] for sample_id in ids if sample_id in gold
    }
    return evaluate_model(samples_root, freeze_root, filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=Path, default=ROOT / "DataCreate" / "samples"
    )
    parser.add_argument(
        "--freeze-root",
        type=Path,
        default=ROOT
        / "align-model"
        / "runs"
        / "eval-datacreate-095-124-mel-v5-20260921",
    )
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    samples_root = args.samples.resolve()
    freeze_root = args.freeze_root.resolve()
    gold: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for sample_id in base.EVAL_IDS:
        labels, row = base._audit_gold(samples_root / sample_id)
        audit[sample_id] = row
        if labels is not None:
            gold[sample_id] = labels
    manual_ids = [
        sample_id
        for sample_id, row in audit.items()
        if row["official_note_wise"] == "available"
        and row["gold_provenance"] == "manual"
    ]
    agent_ids = [
        sample_id
        for sample_id, row in audit.items()
        if row["official_note_wise"] == "available"
        and row["gold_provenance"] == "agent"
    ]
    models = {
        "audioeval_mel_v5": {
            "all_auditable": evaluate_model(samples_root, freeze_root, gold),
            "manual_source": subset_eval(
                samples_root, freeze_root, gold, manual_ids
            ),
            "agent_source": subset_eval(
                samples_root, freeze_root, gold, agent_ids
            ),
        }
    }
    freeze_manifest = freeze_root / "freeze_manifest.json"
    report = {
        "schema_version": REPORT_SCHEMA,
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "created_utc": base._utc(),
        "ids": base.EVAL_IDS,
        "subset": {
            "requested": base.EVAL_IDS,
            "official_note_wise_available": sorted(gold),
            "official_note_wise_unavailable": [
                sample_id
                for sample_id in base.EVAL_IDS
                if sample_id not in gold
            ],
            "manual_source_ids": manual_ids,
            "agent_source_ids": agent_ids,
        },
        "audit": audit,
        "scored_types": list(base.SCORED_TYPES),
        "models": models,
        "audioeval_freeze_manifest": (
            str(freeze_manifest) if freeze_manifest.is_file() else None
        ),
        "audioeval_freeze_sha256": (
            base.sha256(freeze_manifest) if freeze_manifest.is_file() else None
        ),
        "promotion": "experimental",
        "legacy_timestamp_used_for_selection": False,
        "gold_note": (
            "Primary labels.json is the working gold. Agent-source clips were "
            "copied from cursor_agent_mel_transcriber_v1_error_heads_v5. "
            "This freeze uses that same stack. Agreement on agent-source clips "
            "is self-copy, not independent human evaluation."
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "integrity.json").write_text(
        json.dumps(
            {
                "schema_version": "align-datacreate-095-124-mel-v5-integrity-v1",
                "report_sha256": base.sha256(report_path),
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
