"""Gold-isolated DataCreate evaluation for completed error-heads-v3.

Run ``freeze`` and ``score`` as separate Python processes.  Freeze reads only
the permitted score/audio/cache/model inputs and emits v3, v3-balanced, v2,
and deterministic-rule documents.  Score verifies every frozen hash before it
opens labels and classifies their provenance from explicit metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import eval_datacreate_current as base
from alignmodel.joint.candidates import (
    add_score_repeat_hints,
    basic_pitch_candidate_union,
)
from alignmodel.joint.error_heads import (
    SchemaDecodeConfig,
    build_inference_rows,
    filter_prediction_for_schema,
    heuristic_prediction,
    infer_error_heads,
    schema12_document,
    schema12_document_v3,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import (
    TYPED_LOCATION_METRIC_SCHEMA,
    evaluate_typed_location_labels,
)
from alignmodel.transcription.basic_pitch import (
    FROZEN_DECODE_CONFIG,
    decode_frozen_basic_pitch,
    sanitize_basic_pitch_notes,
)


FREEZE_SCHEMA = "align-datacreate-error-heads-v3-freeze-v1"
REPORT_SCHEMA = "align-datacreate-error-heads-v3-agent-labels-report-v2"
HALF_CREDIT_REPORT_SCHEMA = (
    "legacy-align-datacreate-error-heads-v3-timestamp-half-credit-report-v1"
)
AGENT_PATTERN = re.compile(
    r"(?:\bagent\b|\bai(?:\b|[_-])|artificial intelligence|llm|gpt|claude|cursor|"
    r"copilot|gemini|codex|automated annotator)",
    re.IGNORECASE,
)
HUMAN_PATTERN = re.compile(
    r"(?:\bhuman\b|manually reviewed|human reviewed|human-review)",
    re.IGNORECASE,
)


def _decode_config(value: Mapping[str, Any]) -> SchemaDecodeConfig:
    if value.get("schema_version") != "align-error-heads-v3-decode-config-v1":
        raise ValueError("Unsupported error-heads-v3 decode config")
    return SchemaDecodeConfig.from_mapping(value)


def _immutable_assets(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "v3_report": args.v3_report.resolve(),
        "v3_experiment_config": args.v3_experiment_config.resolve(),
        "v3_decode_config": args.v3_decode_config.resolve(),
        "v3_synthetic_freeze_manifest": args.v3_synthetic_freeze.resolve(),
        "v3_synthetic_integrity": args.v3_synthetic_integrity.resolve(),
        "v3_synthetic_verification": args.v3_synthetic_verification.resolve(),
        "sealed_lockbox": args.sealed_lockbox.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        key: {"path": str(path), "sha256": base._sha256(path)}
        for key, path in paths.items()
    }


def _load_v3_stack(args: argparse.Namespace) -> dict[str, Any]:
    stack = base._load_completed_stack(args)
    report = base._json(args.v3_report.resolve())
    experiment = base._json(args.v3_experiment_config.resolve())
    config_path = args.v3_decode_config.resolve()
    config_value = base._json(config_path)
    if report.get("schema_version") != "align-error-heads-v3-report-v1":
        raise ValueError("error-heads-v3 report is not completed v3")
    if not (report.get("process_protocol") or {}).get(
        "freeze_verified_before_gold_open"
    ):
        raise ValueError("error-heads-v3 report lacks verified two-process freeze")
    if (report.get("training") or {}).get("performed") is not False:
        raise ValueError("v3 unexpectedly reports classifier training")
    if report.get("promotion_performed") or report.get("production_mutated"):
        raise ValueError("v3 report indicates production mutation")
    config_hash = base._sha256(config_path)
    if experiment.get("decode_config_sha256") != config_hash:
        raise ValueError("v3 decode config hash differs from experiment record")
    source_v2 = Path(str(experiment["v2_freeze_manifest"])).resolve()
    if base._sha256(source_v2) != experiment.get("v2_freeze_manifest_sha256"):
        raise ValueError("v3 source v2 frozen-upstream hash mismatch")
    config = _decode_config(config_value)
    balanced = _decode_config(config_value["balanced_policy"])
    stack["v3_config"] = config
    stack["v3_balanced_config"] = balanced
    stack["selection"]["error_heads_v3"] = {
        "report": str(args.v3_report.resolve()),
        "report_schema_version": report["schema_version"],
        "report_sha256": base._sha256(args.v3_report.resolve()),
        "experiment_config": str(args.v3_experiment_config.resolve()),
        "experiment_schema_version": experiment.get("schema_version"),
        "experiment_config_sha256": base._sha256(
            args.v3_experiment_config.resolve()
        ),
        "decode_config": str(config_path),
        "decode_config_schema_version": config_value.get("schema_version"),
        "decode_config_sha256": config_hash,
        "source_v2_freeze_manifest": str(source_v2),
        "source_v2_freeze_manifest_sha256": base._sha256(source_v2),
        "classifier_retrained": False,
        "classifier_checkpoint": stack["selection"]["error_heads_v2"]["path"],
        "classifier_checkpoint_sha256": stack["selection"]["error_heads_v2"][
            "checkpoint_sha256"
        ],
        "compatible": True,
        "policy": (
            "completed v3 maximum-F1 sequence-cluster decoder applied to the "
            "exact completed v2 head and hash-compatible frozen upstream"
        ),
        "balanced_policy_evaluated_as_alternate": True,
    }
    return stack


def _freeze_one(
    sample: Path, output: Path, stack: Mapping[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    wav = sample / "performance_audio.wav"
    score_path = sample / "verified_score.musicxml"
    if not wav.is_file() or not score_path.is_file():
        raise FileNotFoundError(
            f"required inputs: audio={wav.is_file()} score={score_path.is_file()}"
        )
    audio = base._audio_info(wav)
    phase = time.perf_counter()
    features, cache = base._cache_features(sample, output)
    feature_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    score = tuple(ScoreEventIndex.from_musicxml(score_path).events)
    if not score:
        raise ValueError("verified score has no sounding events")
    canonical = sanitize_basic_pitch_notes(
        decode_frozen_basic_pitch(features), features, FROZEN_DECODE_CONFIG
    )
    all_candidates = tuple(
        basic_pitch_candidate_union(
            features,
            configs=base._frontend_configs(stack["joint_payload"]),
            minimum_confidence=0.0,
        )
    )
    admitted = tuple(
        add_score_repeat_hints(
            [
                value
                for value in all_candidates
                if value.confidence >= stack["minimum_confidence"]
            ],
            score,
        )
    )
    if not admitted:
        raise ValueError("score/audio mismatch: no admitted candidates")
    candidate_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    path = stack["lattice"].decode(admitted, score)
    rows = build_inference_rows(stack["lattice"], admitted, score, path)
    raw_prediction = infer_error_heads(
        stack["heads_model"],
        rows,
        stack["heads_payload"]["thresholds"],
        device="cpu",
    )
    v2_prediction = filter_prediction_for_schema(
        raw_prediction, stack["heads_payload"]["schema_thresholds"]
    )
    documents = {
        "v3": schema12_document_v3(
            sample.name, rows, raw_prediction, score, stack["v3_config"]
        ),
        "v3_balanced": schema12_document_v3(
            sample.name,
            rows,
            raw_prediction,
            score,
            stack["v3_balanced_config"],
        ),
        "v2": schema12_document(
            sample.name, rows, v2_prediction, score, pad_notes=1
        ),
        "rules": schema12_document(
            sample.name, rows, heuristic_prediction(rows), score, pad_notes=1
        ),
    }
    diagnostics = base._path_diagnostics(admitted, score, path)
    diagnostics.update(
        {
            "canonical_transcription_count": len(canonical),
            "candidate_union_all_count": len(all_candidates),
            "retained_candidate_count": len(admitted),
            "candidate_to_score_count_ratio": len(admitted) / len(score),
            "canonical_to_score_count_ratio": len(canonical) / len(score),
            "audio": audio,
        }
    )
    hashes = stack["selection"]
    for name, document in documents.items():
        document["pipeline"].update(
            {
                "evaluation_model": name,
                "candidate_config_sha256": hashes["transcriber"][
                    "candidate_config_sha256"
                ],
                "joint_checkpoint_sha256": hashes["joint_aligner_decoder"][
                    "checkpoint_sha256"
                ],
                "error_heads_checkpoint_sha256": hashes["error_heads_v2"][
                    "checkpoint_sha256"
                ],
                "v3_decode_config_sha256": (
                    hashes["error_heads_v3"]["decode_config_sha256"]
                    if name.startswith("v3")
                    else None
                ),
                "diagnostics": diagnostics,
            }
        )
    paths = {
        name: output / "predictions" / name / f"{sample.name}.json"
        for name in documents
    }
    for name, document in documents.items():
        base._atomic_json(paths[name], document)
    finished = time.perf_counter()
    return {
        "sample": sample.name,
        "status": "succeeded",
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "hashes": {name: base._sha256(path) for name, path in paths.items()},
        "diagnostics": diagnostics,
        "feature_cache": cache,
        "runtime_seconds": {
            "features": feature_seconds,
            "candidate_and_score": candidate_seconds,
            "decode_and_heads": finished - phase,
            "total": finished - started,
        },
    }


def freeze(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    marker = output / "freeze_manifest.json"
    if marker.exists():
        raise FileExistsError(marker)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    status = base._training_status(args.resource_status.resolve())
    stack = _load_v3_stack(args)
    immutable_before = _immutable_assets(args)
    rows = []
    expected = base._expected_samples(args.samples.resolve())
    for position, sample in enumerate(expected, 1):
        print(f"freeze_v3 {position}/{len(expected)} {sample.name}", flush=True)
        if not sample.is_dir():
            rows.append(
                {"sample": sample.name, "status": "failed", "error": "missing"}
            )
            continue
        try:
            rows.append(_freeze_one(sample, output, stack))
        except BaseException as exc:
            rows.append(
                {
                    "sample": sample.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": base._utc(),
        "process": {"pid": os.getpid(), "command": [sys.executable, *sys.argv]},
        "gold_access": {
            "labels_opened": False,
            "forbidden_inputs_opened": [],
            "process_separation_required": True,
            "permitted_inputs": [
                "verified_score.musicxml",
                "performance_audio.wav or hash-validated Basic Pitch cache",
                "completed hash-compatible checkpoints/configs/reports",
                "training resource status",
            ],
        },
        "samples_root": str(args.samples.resolve()),
        "output": str(output),
        "expected": len(expected),
        "discovered": sum(path.is_dir() for path in expected),
        "inference_succeeded": sum(row["status"] == "succeeded" for row in rows),
        "inference_failed": sum(row["status"] == "failed" for row in rows),
        "samples": rows,
        "model_selection": stack["selection"],
        "immutable_synthetic_and_lockbox_before": immutable_before,
        "resource_coordination": {
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
            "decision": "CPU-only optimized cached inference; no GPU lease acquired",
            "status_snapshot": status,
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    base._atomic_json(marker, manifest)
    print(marker)


def _verify_freeze(path: Path) -> tuple[dict[str, Any], list[str]]:
    manifest = base._json(path)
    if manifest.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("Unsupported v3 DataCreate freeze manifest")
    errors = []
    for row in manifest["samples"]:
        for key, digest in (row.get("hashes") or {}).items():
            artifact = Path((row.get("paths") or {}).get(key, ""))
            if not artifact.is_file():
                errors.append(f"{row['sample']}:{key}:missing")
            elif base._sha256(artifact) != digest:
                errors.append(f"{row['sample']}:{key}:hash")
    return manifest, errors


def _metadata_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "source",
        "annotator",
        "annotator_id",
        "annotator_type",
        "annotation_method",
        "generated_by",
        "created_by",
        "comments",
        "comment",
        "notes",
        "schema_version",
        "created_at",
        "updated_at",
        "timestamp",
        "review_status",
        "reviewed",
        "annotation_complete",
    )
    return {key: document[key] for key in keys if key in document}


def _provenance(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata_projection(document)
    text = json.dumps(metadata, sort_keys=True, default=str)
    label_sources = Counter(
        str(label.get("source") or "<missing>")
        for label in document.get("labels") or []
        if isinstance(label, Mapping)
    )
    label_state = (
        "nonempty"
        if document.get("labels")
        else (
            "reviewed_empty"
            if base._explicit_complete(document)
            else "empty_unreviewed"
        )
    )
    explicit_agent = bool(AGENT_PATTERN.search(text))
    explicit_human = bool(HUMAN_PATTERN.search(text))
    if explicit_agent and not explicit_human:
        category = "explicit_agent"
    elif explicit_human or label_sources.get("manual", 0):
        category = "human_or_manual"
    elif not document.get("labels"):
        category = (
            "reviewed_empty"
            if base._explicit_complete(document)
            else "empty_unreviewed"
        )
    else:
        category = "ambiguous_nonempty"
    return {
        "category": category,
        "label_state": label_state,
        "metadata": metadata,
        "label_sources": dict(label_sources),
        "agent_pattern_match": explicit_agent,
        "human_pattern_match": explicit_human,
    }


def _documents(
    manifest: Mapping[str, Any], key: str
) -> dict[str, dict[str, Any]]:
    return {
        row["sample"]: base._json(Path(row["paths"][key]))
        for row in manifest["samples"]
        if row.get("status") == "succeeded" and key in (row.get("paths") or {})
    }


def _metrics(
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    documents: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, str | None],
    durations: Mapping[str, float],
) -> dict[str, Any]:
    labels = {
        sample: list(documents.get(sample, {}).get("labels") or [])
        for sample in gold
    }
    timestamp = base._event_report(gold, labels)
    four_types = {
        "wrong_note",
        "missed_note",
        "extra_note",
        "rhythm_error",
    }
    four_type_counts = [
        base._event_counts(
            gold[sample],
            labels.get(sample, ()),
            criterion="iou_0.3",
            types=four_types,
        )
        for sample in sorted(gold)
    ]
    totals = {
        key: sum(row[key] for row in four_type_counts)
        for key in ("correct", "predicted", "gold")
    }
    timestamp["requested_four_type"] = {
        **base._prf(totals["correct"], totals["predicted"], totals["gold"]),
        "bootstrap_f1": base._bootstrap_counts(
            four_type_counts, seed=20260915 + 403
        ),
        "types": sorted(four_types),
    }
    ranges = base._range_report(gold, labels, schemas)
    iou = timestamp["iou_0.3"]["micro"]
    false_count = float(iou["predicted"] - iou["matched"])
    minutes = sum(durations.get(sample, 0.0) for sample in gold) / 60.0
    timestamp["iou_0.3"].update(
        {
            "apparent_false_labels": false_count,
            "apparent_false_labels_per_clip": (
                false_count / len(gold) if gold else None
            ),
            "apparent_false_labels_per_minute": (
                false_count / minutes if minutes else None
            ),
            "evaluated_audio_minutes": minutes,
        }
    )
    return {
        "official_note_wise": {
            "status": "unavailable",
            "reason": (
                "schema 1.1 agent labels have not passed canonical score-event "
                "projection audit; timestamp and inferred ranges are diagnostics"
            ),
        },
        "legacy_timestamp_event": timestamp,
        "diagnostic_inferred_score_range": ranges,
        # Backward-compatible readers for the frozen legacy report.
        "timestamp_event": timestamp,
        "score_range": ranges,
    }


def _half_credit_bootstrap(
    clip_metrics: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict[str, Any] | None:
    if len(clip_metrics) < 5:
        return None
    generator = np.random.default_rng(seed)
    f1_values = []
    for _ in range(replicates):
        selected = generator.integers(
            0, len(clip_metrics), size=len(clip_metrics)
        )
        credit = sum(float(clip_metrics[index]["credit"]) for index in selected)
        predicted = sum(
            int(clip_metrics[index]["predicted"]) for index in selected
        )
        gold = sum(int(clip_metrics[index]["gold"]) for index in selected)
        precision = credit / predicted if predicted else 0.0
        recall = credit / gold if gold else 0.0
        f1_values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "unit": "clip",
        "replicates": replicates,
        "lower_95": float(np.quantile(f1_values, 0.025)),
        "median": float(np.quantile(f1_values, 0.5)),
        "upper_95": float(np.quantile(f1_values, 0.975)),
    }


def _half_credit_event_report(
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    type_mismatch_credit: float,
) -> dict[str, Any]:
    criteria = (
        "iou_0.3",
        "onset_50ms",
        "onset_100ms",
        "onset_250ms",
        "onset_500ms",
    )
    output: dict[str, Any] = {
        "schema_version": TYPED_LOCATION_METRIC_SCHEMA,
        "type_mismatch_credit": float(type_mismatch_credit),
        "assignment": "maximum-weight one-to-one Hungarian",
    }
    for criterion_index, criterion in enumerate(criteria):
        clips = [
            evaluate_typed_location_labels(
                list(documents.get(sample, {}).get("labels") or []),
                gold[sample],
                criterion=criterion,
                type_mismatch_credit=type_mismatch_credit,
            )
            for sample in sorted(gold)
        ]
        credit = sum(float(row["credit"]) for row in clips)
        predicted = sum(int(row["predicted"]) for row in clips)
        target = sum(int(row["gold"]) for row in clips)
        precision = credit / predicted if predicted else 0.0
        recall = credit / target if target else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output[criterion] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "credit": credit,
            "predicted": predicted,
            "gold": target,
            "pair_counts": {
                key: sum(int(row["pair_counts"][key]) for row in clips)
                for key in ("full_credit", "half_credit", "zero_credit")
            },
            "unmatched_predictions": sum(
                int(row["unmatched_predictions"]) for row in clips
            ),
            "unmatched_gold": sum(
                int(row["unmatched_gold"]) for row in clips
            ),
            "bootstrap_f1": _half_credit_bootstrap(
                clips, seed=20260915 + criterion_index
            ),
        }
    return output


def score(args: argparse.Namespace) -> None:
    freeze_path = args.freeze_manifest.resolve()
    manifest, errors = _verify_freeze(freeze_path)
    if errors:
        raise ValueError(f"Frozen predictions failed integrity: {errors}")
    if os.getpid() == int(manifest["process"]["pid"]):
        raise ValueError("Score must run in a separate process")
    output = Path(manifest["output"])
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    provenance_rows = []
    gold_by_sample: dict[str, list[dict[str, Any]]] = {}
    schemas: dict[str, str | None] = {}
    for sample_dir in base._expected_samples(Path(manifest["samples_root"])):
        label_path = sample_dir / "labels.json"
        if not label_path.is_file():
            provenance_rows.append(
                {"sample": sample_dir.name, "category": "missing_labels"}
            )
            continue
        document = base._json(label_path)
        provenance = _provenance(document)
        labels = [
            dict(label)
            for label in document.get("labels") or []
            if isinstance(label, Mapping)
        ]
        schemas[sample_dir.name] = document.get("schema_version")
        provenance_rows.append(
            {
                "sample": sample_dir.name,
                **provenance,
                "schema_version": document.get("schema_version"),
                "label_count": len(labels),
                "label_types": dict(
                    Counter(str(label.get("type")) for label in labels)
                ),
            }
        )
        gold_by_sample[sample_dir.name] = labels

    strict_ids = sorted(
        row["sample"]
        for row in provenance_rows
        if row.get("category") == "explicit_agent" and row.get("label_count", 0)
    )
    ambiguous_ids = sorted(
        row["sample"]
        for row in provenance_rows
        if row.get("category") == "ambiguous_nonempty"
    )
    scored_ids = strict_ids if strict_ids else ambiguous_ids
    selection_name = (
        "strict_explicit_agent_metadata"
        if strict_ids
        else "alternate_ambiguous_nonempty_no_explicit_human_marker"
    )
    gold = {sample: gold_by_sample[sample] for sample in scored_ids}
    durations = {
        row["sample"]: float(
            ((row.get("diagnostics") or {}).get("audio") or {}).get(
                "duration_sec", 0.0
            )
        )
        for row in manifest["samples"]
    }
    documents = {
        name: _documents(manifest, name)
        for name in ("v3", "v3_balanced", "v2", "rules")
    }
    metrics = {
        name: _metrics(gold, docs, schemas, durations)
        for name, docs in documents.items()
    }
    comparisons = {}
    for baseline in ("v2", "rules"):
        comparisons[baseline] = {
            "identical_subset_rerun": True,
            "timestamp_iou_0.3_f1": metrics[baseline]["timestamp_event"][
                "iou_0.3"
            ]["micro"]["f1"],
            "timestamp_iou_0.3_delta_v3_minus_baseline": (
                metrics["v3"]["timestamp_event"]["iou_0.3"]["micro"]["f1"]
                - metrics[baseline]["timestamp_event"]["iou_0.3"]["micro"]["f1"]
            ),
            "enhanced_range_f1": metrics[baseline]["score_range"][
                "enhanced_schema_1_1_diagnostic"
            ]["micro"]["f1"],
            "enhanced_range_delta_v3_minus_baseline": (
                metrics["v3"]["score_range"][
                    "enhanced_schema_1_1_diagnostic"
                ]["micro"]["f1"]
                - metrics[baseline]["score_range"][
                    "enhanced_schema_1_1_diagnostic"
                ]["micro"]["f1"]
            ),
        }
    selected_inference = [
        row for row in manifest["samples"] if row["sample"] in scored_ids
    ]
    upstream_totals = {
        key: int(
            sum(int((row.get("diagnostics") or {}).get(key, 0)) for row in selected_inference)
        )
        for key in (
            "score_event_count",
            "canonical_transcription_count",
            "candidate_union_all_count",
            "retained_candidate_count",
            "decoded_event_count",
            "mapped_event_count",
            "deleted_score_event_count",
        )
    }
    immutable_after = _immutable_assets(args)
    immutable_changes = [
        key
        for key, value in manifest[
            "immutable_synthetic_and_lockbox_before"
        ].items()
        if immutable_after.get(key, {}).get("sha256") != value.get("sha256")
    ]
    type_distribution = Counter(
        str(label.get("type"))
        for sample in scored_ids
        for label in gold[sample]
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": base._utc(),
        "counts": {
            "expected": manifest["expected"],
            "discovered": manifest["discovered"],
            "inference_succeeded": manifest["inference_succeeded"],
            "inference_failed": manifest["inference_failed"],
            "strict_agent_labelled_clips": len(strict_ids),
            "ambiguous_nonempty_clips": len(ambiguous_ids),
            "evaluated_clips": len(scored_ids),
            "evaluated_labels": sum(len(value) for value in gold.values()),
            "provenance_categories": dict(
                Counter(row["category"] for row in provenance_rows)
            ),
            "label_states": dict(
                Counter(row.get("label_state", "missing") for row in provenance_rows)
            ),
        },
        "subset": {
            "selection": selection_name,
            "sample_ids": scored_ids,
            "strict_explicit_agent_sample_ids": strict_ids,
            "alternate_ambiguous_sample_ids": ambiguous_ids,
            "label_type_distribution": dict(type_distribution),
            "schema_distribution": dict(
                Counter(str(schemas.get(sample)) for sample in scored_ids)
            ),
            "inclusion_rule": (
                "Strict: non-empty document-level metadata explicitly names "
                "an AI/LLM/agent and does not name human review. If strict is "
                "empty, score the separately named alternate containing "
                "non-empty labels with no explicit human/manual provenance. "
                "Label source alone and non-emptiness never prove agent origin."
            ),
        },
        "provenance_inventory": provenance_rows,
        "model_selection": manifest["model_selection"],
        "metrics": metrics,
        "comparisons": comparisons,
        "upstream_summary": {
            "totals": upstream_totals,
            "ratios_of_totals": {
                "canonical_to_score": (
                    upstream_totals["canonical_transcription_count"]
                    / upstream_totals["score_event_count"]
                ),
                "retained_candidates_to_score": (
                    upstream_totals["retained_candidate_count"]
                    / upstream_totals["score_event_count"]
                ),
                "mapped_events_to_score": (
                    upstream_totals["mapped_event_count"]
                    / upstream_totals["score_event_count"]
                ),
            },
            "mapped_score_coverage_mean": float(
                np.mean(
                    [
                        float(
                            (row.get("diagnostics") or {}).get(
                                "mapped_score_coverage", 0.0
                            )
                        )
                        for row in selected_inference
                    ]
                )
            ),
            "runtime_seconds_per_clip": {
                "mean": float(
                    np.mean(
                        [
                            float((row.get("runtime_seconds") or {}).get("total", 0.0))
                            for row in selected_inference
                        ]
                    )
                ),
                "median": float(
                    np.median(
                        [
                            float((row.get("runtime_seconds") or {}).get("total", 0.0))
                            for row in selected_inference
                        ]
                    )
                ),
            },
        },
        "sample_007": (
            {
                "included": True,
                "gold_labels": gold["007"],
                "predictions": {
                    name: documents[name]["007"] for name in documents
                },
                "diagnostics": next(
                    row["diagnostics"]
                    for row in manifest["samples"]
                    if row["sample"] == "007"
                ),
            }
            if "007" in scored_ids
            else {"included": False}
        ),
        "limitations": {
            "agent_labels_are_independent_human_ground_truth": False,
            "shared_model_bias_possible": True,
            "schema_1_2_claimed": False,
            "range_metric_status": (
                "enhanced schema-1.1 diagnostic only; score projection comes "
                "from optional pitches/score_part fields"
            ),
            "note_f1": (
                "not reported because there is no independent note "
                "transcription reference"
            ),
            "sparse_non_exhaustive": True,
            "false_label_rates_are_apparent": True,
        },
        "protocol": {
            "two_process_freeze_score": True,
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": base._sha256(freeze_path),
            "prediction_hashes_verified_before_labels_opened": True,
            "freeze_pid": manifest["process"]["pid"],
            "score_pid": os.getpid(),
            "freeze_command": manifest["process"]["command"],
            "score_command": [sys.executable, *sys.argv],
            "resource_coordination": manifest["resource_coordination"],
            "environment": manifest["environment"],
            "immutable_synthetic_and_lockbox_after": immutable_after,
            "immutable_changes": immutable_changes,
            "locked_synthetic_test_untouched": not immutable_changes,
        },
        "inference_samples": manifest["samples"],
    }
    base._atomic_json(output / "provenance_inventory.json", {
        "samples": provenance_rows
    })
    base._atomic_json(report_path, report)
    _, after_errors = _verify_freeze(freeze_path)
    integrity = {
        "schema_version": "align-datacreate-error-heads-v3-integrity-v1",
        "created_utc": base._utc(),
        "freeze_manifest_sha256": base._sha256(freeze_path),
        "report_sha256": base._sha256(report_path),
        "prediction_hash_errors": after_errors,
        "immutable_changes": immutable_changes,
        "passed": not after_errors and not immutable_changes,
    }
    base._atomic_json(output / "integrity.json", integrity)
    if not integrity["passed"]:
        raise RuntimeError("Post-score integrity failed")
    print(report_path)


def half_credit(args: argparse.Namespace) -> None:
    freeze_path = args.freeze_manifest.resolve()
    manifest, prediction_errors = _verify_freeze(freeze_path)
    if prediction_errors:
        raise ValueError(
            f"Frozen predictions failed integrity: {prediction_errors}"
        )
    source_report_path = args.source_report.resolve()
    source_integrity_path = source_report_path.with_name("integrity.json")
    source_report = base._json(source_report_path)
    source_integrity = base._json(source_integrity_path)
    if (
        source_integrity.get("report_sha256")
        != base._sha256(source_report_path)
        or not source_integrity.get("passed")
    ):
        raise ValueError("Source strict report integrity failed")

    output = args.half_credit_output.resolve()
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    immutable_before = _immutable_assets(args)

    # Gold is opened only after all frozen prediction and source-report hashes
    # have passed above. Rebuild provenance to ensure the exact subset remains.
    gold: dict[str, list[dict[str, Any]]] = {}
    schemas: dict[str, str | None] = {}
    strict_ids = []
    for sample_dir in base._expected_samples(Path(manifest["samples_root"])):
        label_path = sample_dir / "labels.json"
        if not label_path.is_file():
            continue
        document = base._json(label_path)
        provenance = _provenance(document)
        labels = [
            dict(label)
            for label in document.get("labels") or []
            if isinstance(label, Mapping)
        ]
        if provenance["category"] == "explicit_agent" and labels:
            strict_ids.append(sample_dir.name)
            gold[sample_dir.name] = labels
            schemas[sample_dir.name] = document.get("schema_version")
    strict_ids.sort()
    expected_ids = list(source_report["subset"]["sample_ids"])
    if strict_ids != expected_ids:
        raise ValueError(
            "Agent-labelled subset changed since strict report: "
            f"{strict_ids!r} != {expected_ids!r}"
        )

    documents = {
        name: _documents(manifest, name)
        for name in ("v3", "v3_balanced", "v2", "rules")
    }
    durations = {
        row["sample"]: float(
            ((row.get("diagnostics") or {}).get("audio") or {}).get(
                "duration_sec", 0.0
            )
        )
        for row in manifest["samples"]
    }
    strict = {
        name: _metrics(gold, docs, schemas, durations)["timestamp_event"]
        for name, docs in documents.items()
    }
    half = {
        name: _half_credit_event_report(
            gold,
            docs,
            type_mismatch_credit=args.type_mismatch_credit,
        )
        for name, docs in documents.items()
    }
    comparisons: dict[str, Any] = {}
    for name in documents:
        comparisons[name] = {}
        for criterion in (
            "iou_0.3",
            "onset_50ms",
            "onset_100ms",
            "onset_250ms",
            "onset_500ms",
        ):
            strict_metric = (
                strict[name][criterion]["micro"]
                if criterion.startswith("onset_")
                else strict[name][criterion]["micro"]
            )
            comparisons[name][criterion] = {
                "strict_f1": strict_metric["f1"],
                "half_credit_f1": half[name][criterion]["f1"],
                "absolute_delta": (
                    half[name][criterion]["f1"] - strict_metric["f1"]
                ),
            }

    immutable_after = _immutable_assets(args)
    immutable_changes = [
        key
        for key, value in immutable_before.items()
        if immutable_after[key]["sha256"] != value["sha256"]
    ]
    report = {
        "schema_version": HALF_CREDIT_REPORT_SCHEMA,
        "created_utc": base._utc(),
        "metric_config": {
            "schema_version": TYPED_LOCATION_METRIC_SCHEMA,
            "type_mismatch_credit": args.type_mismatch_credit,
            "location_criteria": [
                "iou_0.3",
                "onset_50ms",
                "onset_100ms",
                "onset_250ms",
                "onset_500ms",
            ],
            "assignment": "maximum-weight one-to-one Hungarian",
            "strict_metric_redefined": False,
            "range_type_mismatch_credit": {
                "already_implemented": True,
                "constant": "TYPE_MISMATCH_SCALE",
                "value": 0.5,
                "double_applied": False,
            },
        },
        "subset": source_report["subset"],
        "strict_type_aware_timestamp_unchanged": strict,
        "half_credit_timestamp": half,
        "strict_vs_half_credit": comparisons,
        "model_selection": source_report["model_selection"],
        "upstream_candidate_score_diagnostics": source_report[
            "upstream_summary"
        ],
        "note_evaluation": {
            "independent_performed_note_transcription_available": False,
            "note_f1_reported": False,
            "score_conditioned_proxy": {
                "reported": True,
                "values": source_report["upstream_summary"],
                "warning": (
                    "Candidate/decoded counts and score-conditioned alignment "
                    "coverage are diagnostics, not note F1. The verified score "
                    "is the intended music, not ground truth for performed "
                    "notes; performance errors make score agreement a proxy."
                ),
            },
        },
        "sample_007": {
            "included": False,
            "reason": "agent-attributed labels document is empty/unreviewed",
        },
        "protocol": {
            "existing_frozen_predictions_reused": True,
            "inference_rerun": False,
            "prediction_hashes_verified_before_gold_open": True,
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": base._sha256(freeze_path),
            "source_strict_report": str(source_report_path),
            "source_strict_report_sha256": base._sha256(source_report_path),
            "source_integrity": str(source_integrity_path),
            "source_integrity_sha256": base._sha256(source_integrity_path),
            "command": [sys.executable, *sys.argv],
            "process_pid": os.getpid(),
            "locked_synthetic_test_untouched": not immutable_changes,
            "immutable_changes": immutable_changes,
            "production_or_model_changes": False,
        },
        "limitations": source_report["limitations"],
    }
    base._atomic_json(report_path, report)
    _, after_errors = _verify_freeze(freeze_path)
    integrity = {
        "schema_version": (
            "align-datacreate-error-heads-v3-half-credit-integrity-v1"
        ),
        "created_utc": base._utc(),
        "report_sha256": base._sha256(report_path),
        "freeze_manifest_sha256": base._sha256(freeze_path),
        "prediction_hash_errors": after_errors,
        "immutable_changes": immutable_changes,
        "passed": not after_errors and not immutable_changes,
    }
    base._atomic_json(output / "integrity.json", integrity)
    if not integrity["passed"]:
        raise RuntimeError("Half-credit metric integrity failed")
    print(report_path)


def verify(args: argparse.Namespace) -> None:
    manifest, errors = _verify_freeze(args.freeze_manifest.resolve())
    output = Path(manifest["output"])
    report = output / "report.json"
    integrity = output / "integrity.json"
    result = {
        "prediction_errors": errors,
        "report_present": report.is_file(),
        "integrity_present": integrity.is_file(),
        "integrity": base._json(integrity) if integrity.is_file() else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors or not report.is_file() or not integrity.is_file():
        raise SystemExit(1)


def summarize(args: argparse.Namespace) -> None:
    report_path = args.report.resolve()
    report = base._json(report_path)
    target_types = (
        "wrong_note",
        "missed_note",
        "extra_note",
        "rhythm_error",
        "repetition",
    )

    def compact(name: str) -> dict[str, Any]:
        metrics = report["metrics"][name]
        timestamp = metrics["timestamp_event"]
        ranges = metrics["score_range"]
        return {
            "timestamp_iou_0.3": timestamp["iou_0.3"],
            "onset_tolerances": {
                key: timestamp[key]["micro"]
                for key in (
                    "onset_50ms",
                    "onset_100ms",
                    "onset_250ms",
                    "onset_500ms",
                )
            },
            "requested_four_type": timestamp["requested_four_type"],
            "layer2_three_type": timestamp["layer2"],
            "rhythm": timestamp["layer3_rhythm"],
            "repetition": timestamp["repetition"],
            "per_type_timestamp_iou_0.3": {
                key: timestamp["iou_0.3"]["per_type"].get(key)
                for key in target_types
            },
            "enhanced_schema_1_1_range": ranges[
                "enhanced_schema_1_1_diagnostic"
            ],
            "per_type_enhanced_range": {
                key: (ranges["per_type"].get(key) or {}).get("micro")
                for key in target_types
            },
            "formal_schema_1_2": ranges["formal_schema_1_2"],
        }

    summary = {
        "schema_version": "align-datacreate-error-heads-v3-headline-v1",
        "source_report": str(report_path),
        "source_report_sha256": base._sha256(report_path),
        "counts": report["counts"],
        "subset": report["subset"],
        "v3": compact("v3"),
        "v3_balanced": compact("v3_balanced"),
        "v2_identical_subset_rerun": compact("v2"),
        "rules_identical_subset_rerun": compact("rules"),
        "comparisons": report["comparisons"],
        "upstream_summary": report["upstream_summary"],
        "model_selection": report["model_selection"],
        "sample_007": report["sample_007"],
        "limitations": report["limitations"],
        "protocol": report["protocol"],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    base._atomic_json(output, summary)
    print(output)


def _defaults(parser: argparse.ArgumentParser) -> None:
    root = Path(__file__).resolve().parents[2]
    align = root / "align-model"
    v3 = align / "runs" / "joint-outputraw-full-v1" / "error-heads-v3"
    parser.add_argument("--samples", type=Path, default=root / "DataCreate" / "samples")
    parser.add_argument(
        "--output",
        type=Path,
        default=align
        / "runs"
        / "eval-datacreate-error-heads-v3-agent-labels-20260915",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-audit-v2"
        / "end-to-end-v2"
        / "weak-note-continuation-optimized"
        / "joint_decoder.pt",
    )
    parser.add_argument(
        "--error-heads-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "error-heads-v2"
        / "predicted"
        / "best.pt",
    )
    parser.add_argument(
        "--active-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "training-v1-optimized"
        / "last_checkpoint.pt",
    )
    parser.add_argument("--v3-report", type=Path, default=v3 / "report.json")
    parser.add_argument(
        "--v3-experiment-config", type=Path, default=v3 / "config.json"
    )
    parser.add_argument(
        "--v3-decode-config", type=Path, default=v3 / "decode_config.json"
    )
    parser.add_argument(
        "--v3-synthetic-freeze", type=Path, default=v3 / "freeze_manifest.json"
    )
    parser.add_argument(
        "--v3-synthetic-integrity", type=Path, default=v3 / "integrity.json"
    )
    parser.add_argument(
        "--v3-synthetic-verification", type=Path, default=v3 / "verification.json"
    )
    parser.add_argument(
        "--sealed-lockbox",
        type=Path,
        default=align
        / "data-audit"
        / "joint-outputraw-full-v1"
        / "LOCKBOX_SEALED.json",
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=align / "runs" / "TRAINING_RESOURCE_STATUS.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = phases.add_parser("freeze")
    _defaults(freeze_parser)
    freeze_parser.add_argument("--cpu-threads", type=int, default=2)
    freeze_parser.set_defaults(function=freeze)
    score_parser = phases.add_parser("score")
    _defaults(score_parser)
    score_parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / "eval-datacreate-error-heads-v3-agent-labels-20260915"
        / "freeze_manifest.json",
    )
    score_parser.set_defaults(function=score)
    half_credit_parser = phases.add_parser("half-credit")
    _defaults(half_credit_parser)
    half_credit_parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / "eval-datacreate-error-heads-v3-agent-labels-20260915"
        / "freeze_manifest.json",
    )
    half_credit_parser.add_argument(
        "--source-report",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / "eval-datacreate-error-heads-v3-agent-labels-20260915"
        / "report.json",
    )
    half_credit_parser.add_argument(
        "--half-credit-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / "eval-datacreate-error-heads-v3-agent-labels-20260915"
        / "half-credit-metric-v1",
    )
    half_credit_parser.add_argument(
        "--type-mismatch-credit", type=float, default=0.5
    )
    half_credit_parser.set_defaults(function=half_credit)
    verify_parser = phases.add_parser("verify")
    verify_parser.add_argument("--freeze-manifest", type=Path, required=True)
    verify_parser.set_defaults(function=verify)
    summary_parser = phases.add_parser("summarize")
    summary_parser.add_argument("--report", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
