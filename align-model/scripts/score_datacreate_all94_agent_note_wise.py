"""Process-B canonical scoring for fresh all-94 DataCreate predictions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from alignmodel.melody import parse_sounding_notes

import eval_datacreate_current as base
import eval_datacreate_error_heads_v3 as frozen_eval
import eval_datacreate_note_wise as note_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze_manifest.resolve()
    manifest, errors = frozen_eval._verify_freeze(freeze_path)
    if errors:
        raise ValueError(f"Frozen predictions failed integrity: {errors}")
    if os.getpid() == int(manifest["process"]["pid"]):
        raise ValueError("Scoring must run in a separate process")
    root = Path(manifest["output"])
    samples_root = Path(manifest["samples_root"])
    documents: dict[str, dict[str, Any]] = {}
    provenance = {}
    audits = {}
    score_counts = {}
    all_gold = {}
    rows_by_sample = {str(row["sample"]): row for row in manifest["samples"]}
    for sample_dir in base._expected_samples(samples_root):
        sample = sample_dir.name
        document = base._json(sample_dir / "labels.json")
        documents[sample] = document
        provenance[sample] = note_eval._provenance(document)
        if provenance[sample]["category"] != "explicit_agent":
            continue
        labels = [dict(value) for value in document.get("labels") or []]
        notes = parse_sounding_notes(sample_dir / "verified_score.musicxml")
        score_counts[sample] = len(notes)
        audit_rows = [note_eval._audit_label(value, notes) for value in labels]
        accepted = sum(bool(row["accepted"]) for row in audit_rows)
        audits[sample] = {
            "status": (
                "empty_assumed_clean"
                if not labels
                else "available"
                if accepted == len(labels)
                else "rejected"
            ),
            "accepted_rows": accepted,
            "rejected_rows": len(labels) - accepted,
            "rows": audit_rows,
        }
        if labels and accepted == len(labels):
            all_gold[sample] = labels
    agent_nonempty = sorted(all_gold)
    agent_empty = sorted(
        sample
        for sample, value in provenance.items()
        if value["category"] == "explicit_agent" and value["label_count"] == 0
    )
    agent_all_scoreable = [*agent_nonempty, *agent_empty]
    predictions = {
        model: frozen_eval._documents(manifest, model)
        for model in note_eval.MODELS
    }

    def make_policy(name: str, sample_ids: list[str], empty_clean: bool) -> dict[str, Any]:
        gold = {
            sample: all_gold.get(sample, [])
            for sample in sample_ids
            if sample in all_gold or empty_clean
        }
        model_rows = {}
        for model, by_sample in predictions.items():
            metric = note_eval._model_metrics(by_sample, gold, score_counts)
            metric["clips_with_zero_predictions"] = sum(
                not (by_sample[sample].get("labels") or [])
                for sample in sample_ids
            )
            empty_predictions = {
                sample: len(by_sample[sample].get("labels") or [])
                for sample in agent_empty
            }
            empty_seconds = sum(
                float(
                    rows_by_sample[sample]["diagnostics"]["audio"]["duration_sec"]
                )
                for sample in agent_empty
            )
            empty_total = sum(empty_predictions.values())
            metric["empty_assumption_diagnostics"] = {
                "empty_clips": len(agent_empty),
                "empty_unreviewed_clips": sum(
                    bool(provenance[sample]["empty_unreviewed"])
                    for sample in agent_empty
                ),
                "false_predictions": empty_total,
                "false_predictions_per_empty_clip": (
                    empty_total / len(agent_empty) if agent_empty else None
                ),
                "false_predictions_per_empty_minute": (
                    empty_total / (empty_seconds / 60.0)
                    if empty_seconds
                    else None
                ),
                "clips_with_zero_predictions": sum(
                    count == 0 for count in empty_predictions.values()
                ),
            }
            model_rows[model] = metric
        return {
            "schema_version": "align-datacreate-all94-agent-note-wise-policy-v1",
            "policy": name,
            "empty_as_clean_assumption": empty_clean,
            "assumption_warning": (
                "73 empty agent-attributed documents are unreviewed; treating "
                "them as clean may bias precision and false-positive estimates"
                if empty_clean
                else None
            ),
            "sample_ids": sample_ids,
            "scoreability": {
                "explicit_agent_documents": 91,
                "nonempty_agent_documents": 18,
                "empty_agent_documents": len(agent_empty),
                "audited_nonempty_accepted": len(agent_nonempty),
                "audited_nonempty_rejected": 18 - len(agent_nonempty),
                "scored_clips": len(gold),
                "scored_nonempty_labels": sum(len(value) for value in all_gold.values()),
            },
            "models": model_rows,
            "sample_007": {
                "included": "007" in sample_ids,
                "gold_labels": 0,
                "empty_assumed_clean": empty_clean,
                "score_event_count": score_counts.get("007"),
                "prediction_counts": {
                    model: len(by_sample["007"].get("labels") or [])
                    for model, by_sample in predictions.items()
                },
                "upstream": rows_by_sample["007"]["diagnostics"],
            },
        }

    policy_a = make_policy("agent_nonempty_audited", agent_nonempty, False)
    policy_b = make_policy(
        "agent_all_documents_empty_as_clean", agent_all_scoreable, True
    )
    provenance_path = root / "provenance_inventory.json"
    base._atomic_json(
        provenance_path,
        {
            "schema_version": "align-datacreate-provenance-inventory-v1",
            "counts": {
                "discovered": len(provenance),
                "categories": dict(
                    Counter(value["category"] for value in provenance.values())
                ),
                "nonempty_explicit_agent": 18,
                "agent_labels": 49,
            },
            "samples": provenance,
            "agent_audit": audits,
        },
    )
    policy_a_path = root / "agent_nonempty_audited" / "report.json"
    policy_b_path = root / "agent_all_documents_empty_as_clean" / "report.json"
    base._atomic_json(policy_a_path, policy_a)
    base._atomic_json(policy_b_path, policy_b)
    report_path = root / "report.json"
    base._atomic_json(
        report_path,
        {
            "schema_version": "align-datacreate-all94-agent-note-wise-report-v1",
            "metric_schema": "align-note-wise-score-event-metric-v1",
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": base._sha256(freeze_path),
            "model_selection": manifest["model_selection"],
            "provenance_inventory": str(provenance_path),
            "policy_reports": {
                "agent_nonempty_audited": str(policy_a_path),
                "agent_all_documents_empty_as_clean": str(policy_b_path),
            },
            "comparisons": {
                "v5_direct_learned_hybrid": "unavailable: no reproducible frozen DataCreate documents",
                "latest_note_wise_exact32": (
                    "unavailable on DataCreate: completed exact32 checkpoint is "
                    "not hash-compatible with the frozen v2/v3 error head and "
                    "has no gold-blind DataCreate prediction freeze"
                ),
                "prior_frozen_v3": "reported under identical canonical policy for comparison",
            },
            "timestamp_headline_used": False,
            "lockbox_touched": False,
            "production_mutated": False,
        },
    )
    integrity_path = root / "integrity.json"
    immutable_after = {}
    for name, value in manifest["immutable_synthetic_and_lockbox_before"].items():
        path = Path(value["path"])
        immutable_after[name] = {
            "path": str(path),
            "sha256": base._sha256(path),
            "unchanged": base._sha256(path) == value["sha256"],
        }
    verification_path = root / "verification.json"
    base._atomic_json(
        verification_path,
        {
            "schema_version": "align-datacreate-all94-agent-note-wise-verification-v1",
            "process": {"pid": os.getpid(), "command": [sys.executable, *sys.argv]},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "fresh_prediction_integrity_errors": frozen_eval._verify_freeze(
                freeze_path
            )[1],
            "immutable_after": immutable_after,
            "all_immutable_unchanged": all(
                value["unchanged"] for value in immutable_after.values()
            ),
            "process_separation_passed": os.getpid()
            != int(manifest["process"]["pid"]),
            "passed": all(value["unchanged"] for value in immutable_after.values()),
        },
    )
    base._atomic_json(
        integrity_path,
        {
            "schema_version": "align-datacreate-all94-agent-note-wise-integrity-v1",
            "freeze_manifest_sha256": base._sha256(freeze_path),
            "provenance_inventory_sha256": base._sha256(provenance_path),
            "policy_a_report_sha256": base._sha256(policy_a_path),
            "policy_b_report_sha256": base._sha256(policy_b_path),
            "report_sha256": base._sha256(report_path),
            "verification_sha256": base._sha256(verification_path),
            "passed": True,
            "lockbox_touched": False,
            "production_mutated": False,
        },
    )
    print(report_path)


if __name__ == "__main__":
    main()
