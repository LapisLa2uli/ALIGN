"""Canonical note-wise DataCreate evaluation from hash-verified frozen outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.melody import (
    canonical_note_location,
    match_note_wise_labels_detail,
    parse_sounding_notes,
)

import eval_datacreate_current as base
import eval_datacreate_error_heads_v3 as frozen_eval


REPORT_SCHEMA = "align-datacreate-canonical-note-wise-report-v1"
FREEZE_SCHEMA = "align-datacreate-canonical-note-wise-freeze-reference-v1"
INTEGRITY_SCHEMA = "align-datacreate-canonical-note-wise-integrity-v1"
MODELS = ("v3", "v3_balanced", "v2", "rules")
_EXPLICIT_USER_ID = re.compile(r"^annotator\d+$", re.IGNORECASE)
_EXPLICIT_HUMAN_TEXT = re.compile(r"\b(?:human|user|manual)\b", re.IGNORECASE)
_NOTE_ID = re.compile(r"^note_(\d+)$")


def _provenance(document: Mapping[str, Any]) -> dict[str, Any]:
    annotator_id = document.get("annotator_id")
    fields = {
        key: document.get(key)
        for key in (
            "annotator_id",
            "annotator",
            "annotator_type",
            "annotation_method",
            "created_by",
            "generated_by",
            "reviewed_by",
            "review_status",
            "annotation_complete",
        )
        if document.get(key) is not None
    }
    text = json.dumps(fields, sort_keys=True, default=str)
    if str(annotator_id) == "ai_f0_align":
        category = "explicit_agent"
        evidence = "document annotator_id == ai_f0_align"
    elif (
        isinstance(annotator_id, str)
        and _EXPLICIT_USER_ID.fullmatch(annotator_id)
    ) or _EXPLICIT_HUMAN_TEXT.search(text):
        category = "explicit_user"
        evidence = f"document-level provenance: {fields}"
    else:
        category = "ambiguous"
        evidence = "no explicit document-level agent/user/human provenance"
    labels = [value for value in document.get("labels") or [] if isinstance(value, Mapping)]
    return {
        "category": category,
        "evidence": evidence,
        "document_fields": fields,
        "schema_version": document.get("schema_version"),
        "label_count": len(labels),
        "label_types": dict(Counter(str(value.get("type")) for value in labels)),
        "label_sources_diagnostic_only": dict(
            Counter(str(value.get("source")) for value in labels)
        ),
        "empty_unreviewed": not labels and not base._explicit_complete(document),
    }


def _audit_label(
    label: Mapping[str, Any],
    notes: Sequence[Any],
) -> dict[str, Any]:
    value = dict(label)
    location = canonical_note_location(value, score_event_count=len(notes))
    reasons: list[str] = []
    if location is None:
        reasons.append("missing_or_invalid_canonical_identity")
    part = value.get("score_part")
    if isinstance(part, Mapping):
        try:
            first = int(part["start_note_index"])
            last = int(part["end_note_index"])
        except (KeyError, TypeError, ValueError):
            reasons.append("invalid_score_part")
        else:
            if first < 0 or last < first or last >= len(notes):
                reasons.append("score_part_out_of_bounds")
            else:
                expected_pitches = [int(note.pitch) for note in notes[first : last + 1]]
                supplied_pitches = value.get("pitches")
                if not isinstance(supplied_pitches, list) or [
                    int(item) for item in supplied_pitches
                ] != expected_pitches:
                    reasons.append("pitch_list_does_not_validate_score_part")
                supplied_ids = value.get("note_ids")
                if supplied_ids:
                    expected_ids = list(range(first, last + 1))
                    parsed_ids = []
                    for item in supplied_ids:
                        match = _NOTE_ID.fullmatch(str(item))
                        if match is None:
                            parsed_ids = []
                            break
                        parsed_ids.append(int(match.group(1)))
                    if parsed_ids != expected_ids:
                        reasons.append("note_ids_do_not_validate_score_part")
    return {
        "label_id": value.get("id"),
        "type": value.get("type"),
        "accepted": not reasons,
        "canonical_location": location,
        "reasons": reasons,
    }


def _finish(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    credit = sum(float(row["credit"]) for row in rows)
    predicted = sum(int(row["predicted"]) for row in rows)
    gold = sum(int(row["gold"]) for row in rows)
    precision = credit / predicted if predicted else 0.0
    recall = credit / gold if gold else 0.0
    if not predicted and not gold:
        precision = recall = 1.0
    full = sum(int(row["pair_counts"]["full_credit"]) for row in rows)
    half = sum(int(row["pair_counts"]["half_credit"]) for row in rows)
    zero = sum(int(row["pair_counts"]["zero_credit"]) for row in rows)
    return {
        "credit": credit,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "full_credit_matches": full,
        "half_credit_matches": half,
        "zero_credit_assignments": zero,
        "zero_credit_or_unmatched_predictions": predicted - full - half,
        "zero_credit_or_unmatched_gold": gold - full - half,
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        selected = generator.integers(0, len(rows), size=len(rows))
        values.append(_finish([rows[int(index)] for index in selected])["f1"])
    return {
        "replicates": 2000,
        "unit": "clip",
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _prediction_hashes(
    manifest: Mapping[str, Any],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    rows = {str(row["sample"]): row for row in manifest["samples"]}
    output = {}
    for model in MODELS:
        digest = hashlib.sha256()
        files = []
        for sample in sample_ids:
            path = Path(rows[sample]["paths"][model])
            sha = base._sha256(path)
            digest.update(f"{sample}:{sha}\n".encode("ascii"))
            files.append({"sample": sample, "path": str(path), "sha256": sha})
        output[model] = {
            "aggregate_sha256": digest.hexdigest(),
            "files": files,
        }
    return output


def _model_metrics(
    documents: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, list[dict[str, Any]]],
    score_counts: Mapping[str, int],
) -> dict[str, Any]:
    clip_rows = []
    per_type_rows: dict[str, list[dict[str, Any]]] = {}
    all_types = sorted(
        {
            str(label.get("type"))
            for labels in gold.values()
            for label in labels
        }
        | {
            str(label.get("type"))
            for sample in gold
            for label in documents[sample].get("labels") or []
        }
    )
    per_type_rows = {kind: [] for kind in all_types}
    for sample, targets in gold.items():
        predicted = [dict(value) for value in documents[sample].get("labels") or []]
        detail = match_note_wise_labels_detail(
            targets,
            predicted,
            score_event_count=score_counts[sample],
        )
        if detail["status"] != "available":
            raise ValueError(f"Prediction identity unavailable for {sample}")
        clip_rows.append(detail)
        for kind in all_types:
            per_type_rows[kind].append(
                match_note_wise_labels_detail(
                    [value for value in targets if str(value.get("type")) == kind],
                    [value for value in predicted if str(value.get("type")) == kind],
                    score_event_count=score_counts[sample],
                )
            )
    result = _finish(clip_rows)
    result["bootstrap_95_ci"] = _bootstrap(clip_rows, 20260915)
    result["matching_policy"] = (
        "exclusive maximum-weight canonical identity; same type=1, "
        "different type=0.5, different location=0"
    )
    result["per_type"] = {
        kind: _finish(rows) for kind, rows in per_type_rows.items()
    }
    result["macro_f1_supported_gold_types"] = float(
        np.mean(
            [
                value["f1"]
                for value in result["per_type"].values()
                if value["gold"] > 0
            ]
            or [0.0]
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--only",
        choices=("agent", "user", "manual-provenance-ambiguous"),
    )
    args = parser.parse_args()
    source_freeze = args.freeze_manifest.resolve()
    manifest, errors = frozen_eval._verify_freeze(source_freeze)
    if errors:
        raise ValueError(f"Frozen predictions failed integrity: {errors}")
    samples_root = Path(manifest["samples_root"])
    inventory = {}
    documents = {}
    for sample_dir in base._expected_samples(samples_root):
        document = base._json(sample_dir / "labels.json")
        documents[sample_dir.name] = document
        inventory[sample_dir.name] = _provenance(document)
    subsets = {
        "agent": sorted(
            sample
            for sample, row in inventory.items()
            if row["category"] == "explicit_agent" and row["label_count"] > 0
        ),
        "user": sorted(
            sample
            for sample, row in inventory.items()
            if row["category"] == "explicit_user"
        ),
        "manual-provenance-ambiguous": sorted(
            sample
            for sample, row in inventory.items()
            if row["category"] == "ambiguous"
            and row["label_count"] > 0
            and set(row["label_sources_diagnostic_only"]) == {"manual"}
        ),
    }
    if args.only:
        subsets = {args.only: subsets[args.only]}
    predictions = {
        model: frozen_eval._documents(manifest, model) for model in MODELS
    }
    provenance_summary = {
        "categories": dict(
            Counter(str(row["category"]) for row in inventory.values())
        ),
        "nonempty_by_category": dict(
            Counter(
                str(row["category"])
                for row in inventory.values()
                if int(row["label_count"]) > 0
            )
        ),
        "ambiguous_sample_ids": sorted(
            sample
            for sample, row in inventory.items()
            if row["category"] == "ambiguous"
        ),
    }
    output_root = args.output_root.resolve()
    for subset_name in subsets:
        if (output_root / subset_name / "report.json").exists():
            raise FileExistsError(output_root / subset_name / "report.json")
    for subset_name, sample_ids in subsets.items():
        subset_dir = output_root / subset_name
        audit = {}
        gold = {}
        score_counts = {}
        for sample in sample_ids:
            sample_dir = samples_root / sample
            labels = [
                dict(value) for value in documents[sample].get("labels") or []
            ]
            if not labels:
                audit[sample] = {
                    "official_note_wise": "unavailable",
                    "reason": (
                        "empty unreviewed document is not a verified clean negative"
                        if inventory[sample]["empty_unreviewed"]
                        else "document has no verifiable labels"
                    ),
                    "rows": [],
                }
                continue
            notes = parse_sounding_notes(sample_dir / "verified_score.musicxml")
            rows = [_audit_label(value, notes) for value in labels]
            accepted = sum(bool(row["accepted"]) for row in rows)
            audit[sample] = {
                "official_note_wise": (
                    "available" if accepted == len(rows) else "unavailable"
                ),
                "reason": (
                    "all label identities validated"
                    if accepted == len(rows)
                    else "clip rejected because at least one gold row is unevaluable"
                ),
                "accepted_rows": accepted,
                "rejected_rows": len(rows) - accepted,
                "rows": rows,
            }
            if subset_name == "manual-provenance-ambiguous" and accepted:
                gold[sample] = [
                    label
                    for label, row in zip(labels, rows)
                    if bool(row["accepted"])
                ]
                score_counts[sample] = len(notes)
                audit[sample]["official_note_wise"] = "available"
                audit[sample]["reason"] = (
                    "accepted rows scored; rejected rows excluded with reasons"
                )
            elif accepted == len(rows):
                gold[sample] = labels
                score_counts[sample] = len(notes)
        hashes = _prediction_hashes(manifest, sample_ids)
        freeze_reference = {
            "schema_version": FREEZE_SCHEMA,
            "source_freeze_manifest": str(source_freeze),
            "source_freeze_manifest_sha256": base._sha256(source_freeze),
            "source_freeze_integrity_verified": True,
            "prediction_models": list(MODELS),
            "prediction_hashes": hashes,
            "model_selection": manifest["model_selection"],
            "audio_inference_rerun": False,
            "gold_opened_only_after_source_freeze_verification": True,
            "sample_ids": sample_ids,
        }
        freeze_path = subset_dir / "freeze_manifest.json"
        base._atomic_json(freeze_path, freeze_reference)
        report = {
            "schema_version": REPORT_SCHEMA,
            "metric_schema": "align-note-wise-score-event-metric-v1",
            "subset": subset_name,
            "provenance_rule": (
                "agent: document annotator_id exactly ai_f0_align and nonempty; "
                "user: explicit document-level annotator/user/human evidence only; "
                "manual-provenance-ambiguous: no document provenance and every "
                "nonempty label has source manual, kept separate from both"
            ),
            "sample_ids": sample_ids,
            "documents": {sample: inventory[sample] for sample in sample_ids},
            "corpus_provenance_inventory": inventory,
            "corpus_provenance_summary": provenance_summary,
            "audit": audit,
            "counts": {
                "discovered": len(inventory),
                "subset_documents": len(sample_ids),
                "subset_labels": sum(
                    int(inventory[sample]["label_count"]) for sample in sample_ids
                ),
                "subset_label_types": dict(
                    Counter(
                        str(label.get("type"))
                        for sample in sample_ids
                        for label in documents[sample].get("labels") or []
                    )
                ),
                "subset_schema_versions": dict(
                    Counter(
                        str(inventory[sample]["schema_version"])
                        for sample in sample_ids
                    )
                ),
                "accepted_clips": len(gold),
                "rejected_clips": len(sample_ids) - len(gold),
                "identity_accepted_rows": sum(
                    int(value.get("accepted_rows", 0)) for value in audit.values()
                ),
                "identity_rejected_rows": sum(
                    int(value.get("rejected_rows", 0)) for value in audit.values()
                ),
                "official_scored_rows": sum(len(value) for value in gold.values()),
                "rejection_reasons": dict(
                    Counter(
                        reason
                        for value in audit.values()
                        for row in value.get("rows", [])
                        for reason in row.get("reasons", [])
                    )
                ),
            },
            "official_note_wise": (
                {
                    "status": "available",
                    "models": {
                        model: _model_metrics(
                            predictions[model], gold, score_counts
                        )
                        for model in MODELS
                    },
                }
                if gold
                else {
                    "status": "unavailable",
                    "reason": "no verifiable nonempty gold documents in subset",
                }
            ),
            "legacy_timestamp_diagnostics": {
                "included": False,
                "reason": "not used as headline, selection, or substitute",
            },
            "v4_v5": {
                "included": False,
                "reason": (
                    "source freeze contains only v3, balanced v3, v2, and rules "
                    "documents; v4/v5 cannot be reproduced from document-only "
                    "artifacts without reopening upstream inference state"
                ),
            },
            "agent_label_caveat": (
                "agreement with agent-generated labels is not independent accuracy"
                if subset_name == "agent"
                else None
            ),
            "lockbox_touched": False,
            "production_mutated": False,
        }
        report_path = subset_dir / "report.json"
        base._atomic_json(report_path, report)
        integrity_path = subset_dir / "integrity.json"
        base._atomic_json(
            integrity_path,
            {
                "schema_version": INTEGRITY_SCHEMA,
                "freeze_manifest_sha256": base._sha256(freeze_path),
                "report_sha256": base._sha256(report_path),
                "source_freeze_manifest_sha256": base._sha256(source_freeze),
                "passed": True,
                "lockbox_touched": False,
                "production_mutated": False,
            },
        )
    print(output_root)


if __name__ == "__main__":
    main()
