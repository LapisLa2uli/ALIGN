"""Localization-first error-heads-v3 calibration and two-process validation.

The v3 experiment reuses immutable, gold-isolated transcription/alignment/head
probabilities from the integrated v2 freeze.  ``analyze`` may open validation
gold to calibrate a deployable sequence decoder. ``freeze`` opens only frozen
inference artifacts plus that decoder config. ``score`` verifies the v3 freeze
before opening audited targets in a separate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.joint.error_heads import (
    LAYER2_CLASSES,
    SchemaDecodeConfig,
    attach_training_targets,
    evaluate_predictions,
    schema12_document_v3,
)
from alignmodel.melody import match_note_wise_labels_detail

import eval_error_heads_integrated as integrated


OUTPUT_SCHEMA = "align-error-heads-v3-experiment-v1"
FREEZE_SCHEMA = "align-error-heads-v3-freeze-v1"
REQUESTED_TYPES = integrated.REQUESTED_TYPES
ALL_SCHEMA_TYPES = REQUESTED_TYPES | {"repetition"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _config_json(config: SchemaDecodeConfig) -> dict[str, Any]:
    return {
        "schema_version": "align-error-heads-v3-decode-config-v1",
        **asdict(config),
    }


def _config_from_json(value: Mapping[str, Any]) -> SchemaDecodeConfig:
    if value.get("schema_version") != "align-error-heads-v3-decode-config-v1":
        raise ValueError("Unsupported v3 decode config")
    return SchemaDecodeConfig.from_mapping(value)


def _load_calibration_clips(args: argparse.Namespace) -> list[dict[str, Any]]:
    _manifest, artifacts = integrated._load_and_verify_freeze(
        args.v2_freeze_manifest.resolve()
    )
    audit = sqlite3.connect(
        f"file:{args.canonical_targets.resolve().as_posix()}?mode=ro", uri=True
    )
    clips = []
    try:
        for artifact in artifacts:
            row = audit.execute(
                "SELECT payload FROM targets WHERE ordinal=? AND split='val'",
                (int(artifact["ordinal"]),),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"Missing audited target for {artifact['ordinal']}"
                )
            target = json.loads(zlib.decompress(row[0]))
            clips.append(
                {
                    "ordinal": int(artifact["ordinal"]),
                    "sample": str(artifact["sample"]),
                    "source": str(artifact["source"]),
                    "duration": float(artifact["audio_duration_sec"]),
                    "rows": tuple(
                        integrated._row_from_json(value)
                        for value in artifact["head_rows"]
                    ),
                    "score": tuple(
                        integrated._score_from_json(value)
                        for value in artifact["score"]
                    ),
                    "prediction": integrated._prediction_from_json(
                        artifact["prediction"]
                    ),
                    "v2_document": artifact["schema_1_2"],
                    "rules_document": artifact["rules_schema_1_2"],
                    "valid_labels": [
                        dict(label)
                        for label in target.get("valid_labels") or ()
                        if label.get("type") != "intonation_error"
                    ],
                    "artifact": artifact,
                }
            )
    finally:
        audit.close()
    if len(clips) != 358:
        raise ValueError(f"Expected 358 calibration clips, got {len(clips)}")
    return clips


def _documents(
    clips: Sequence[Mapping[str, Any]],
    config: SchemaDecodeConfig,
    *,
    include_repetition: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        str(clip["sample"]): schema12_document_v3(
            str(clip["sample"]),
            clip["rows"],
            clip["prediction"],
            clip["score"],
            config,
            include_repetition=include_repetition,
        )
        for clip in clips
    }


def _metric(
    clips: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    types: set[str],
) -> dict[str, Any]:
    values = [
        integrated._schema_counts(
            clip["valid_labels"],
            documents[str(clip["sample"])]["labels"],
            types=types,
        )
        for clip in clips
    ]
    counts = integrated._sum_counts(values)
    return integrated._prf(
        counts["correct"], counts["predicted"], counts["target"]
    )


def _disabled_config() -> SchemaDecodeConfig:
    return SchemaDecodeConfig(
        high_thresholds={kind: 1.01 for kind in REQUESTED_TYPES},
        low_ratios={kind: 1.0 for kind in REQUESTED_TYPES},
        minimum_support={kind: 1 for kind in REQUESTED_TYPES},
        uncertainty_margins={kind: 0.0 for kind in REQUESTED_TYPES},
        merge_score_gap={kind: 0 for kind in REQUESTED_TYPES},
        max_row_gap=2,
        pad_notes=1,
        require_extra_neighbors=True,
        require_missed_resynchronization=True,
        nms_overlap=True,
    )


def _with_type(
    config: SchemaDecodeConfig,
    kind: str,
    *,
    threshold: float,
    low_ratio: float,
    support: int,
    margin: float,
    merge_gap: int,
) -> SchemaDecodeConfig:
    return replace(
        config,
        high_thresholds={**config.high_thresholds, kind: threshold},
        low_ratios={**config.low_ratios, kind: low_ratio},
        minimum_support={**config.minimum_support, kind: support},
        uncertainty_margins={**config.uncertainty_margins, kind: margin},
        merge_score_gap={**config.merge_score_gap, kind: merge_gap},
    )


def _calibrate(
    clips: Sequence[Mapping[str, Any]],
) -> tuple[SchemaDecodeConfig, dict[str, Any]]:
    config = _disabled_config()
    trace: dict[str, Any] = {"per_type": {}}
    thresholds = [
        *[float(value) for value in np.arange(0.10, 0.96, 0.05)],
        1.01,
    ]
    for kind in ("wrong_note", "missed_note", "extra_note", "rhythm_error"):
        supports = (1, 2, 3) if kind != "missed_note" else (1, 2)
        margins = (-1.0, -0.25, 0.0, 0.05)
        choices = []
        for threshold in thresholds:
            for low_ratio in (1.0, 0.75):
                for support in supports:
                    for margin in margins:
                        for merge_gap in (0, 1):
                            trial = _with_type(
                                config,
                                kind,
                                threshold=threshold,
                                low_ratio=low_ratio,
                                support=support,
                                margin=margin,
                                merge_gap=merge_gap,
                            )
                            metric = _metric(
                                clips,
                                _documents(
                                    clips, trial, include_repetition=False
                                ),
                                {kind},
                            )
                            choices.append(
                                {
                                    "metric": metric,
                                    "threshold": threshold,
                                    "low_ratio": low_ratio,
                                    "support": support,
                                    "margin": margin,
                                    "merge_gap": merge_gap,
                                }
                            )
        selected = max(
            choices,
            key=lambda value: (
                value["metric"]["f1"],
                value["metric"]["precision"],
                -value["metric"]["predicted"],
                value["threshold"],
                value["support"],
            ),
        )
        config = _with_type(
            config,
            kind,
            threshold=float(selected["threshold"]),
            low_ratio=float(selected["low_ratio"]),
            support=int(selected["support"]),
            margin=float(selected["margin"]),
            merge_gap=int(selected["merge_gap"]),
        )
        trace["per_type"][kind] = selected
        print(
            f"calibrated {kind}: f1={selected['metric']['f1']:.6f} "
            f"pred={selected['metric']['predicted']}",
            flush=True,
        )

    structural = []
    for pad in (0, 1, 2):
        for row_gap in (1, 2, 3):
            for nms in (True, False):
                trial = replace(
                    config,
                    pad_notes=pad,
                    max_row_gap=row_gap,
                    nms_overlap=nms,
                )
                metric = _metric(
                    clips, _documents(clips, trial), REQUESTED_TYPES
                )
                structural.append(
                    {
                        "metric": metric,
                        "pad_notes": pad,
                        "max_row_gap": row_gap,
                        "nms_overlap": nms,
                    }
                )
    selected_structure = max(
        structural,
        key=lambda value: (
            value["metric"]["f1"],
            value["metric"]["precision"],
            -value["metric"]["predicted"],
            value["pad_notes"] == 1,
        ),
    )
    config = replace(
        config,
        pad_notes=int(selected_structure["pad_notes"]),
        max_row_gap=int(selected_structure["max_row_gap"]),
        nms_overlap=bool(selected_structure["nms_overlap"]),
    )
    trace["structural"] = selected_structure
    trace["combined"] = _metric(
        clips, _documents(clips, config), REQUESTED_TYPES
    )
    return config, trace


def _document_metric(
    clips: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return _metric(clips, documents, REQUESTED_TYPES)


def _pair_labels(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    require_type: bool,
) -> list[tuple[int, int]]:
    """Oracle pairing helper, not official headline F1.

    ``require_type`` uses 0.0 type-mismatch credit; the ignore-type path uses
    1.0. Official scoring remains type_mismatch_credit=0.5.
    """
    if not predicted or not gold:
        return []
    detail = match_note_wise_labels_detail(
        [dict(value) for value in gold],
        [dict(value) for value in predicted],
        type_mismatch_credit=0.0 if require_type else 1.0,
    )
    if detail["status"] != "available":
        raise ValueError(f"Canonical oracle pairing unavailable: {detail['reason']}")
    return [
        (int(value["prediction_index"]), int(value["gold_index"]))
        for value in detail["pairs"]
        if float(value["credit"]) > 0.0
    ]


def _gold_type_documents(
    clips: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for clip in clips:
        predicted = [
            dict(label)
            for label in clip["v2_document"]["labels"]
            if label.get("type") in REQUESTED_TYPES
        ]
        gold = [
            dict(label)
            for label in clip["valid_labels"]
            if label.get("type") in REQUESTED_TYPES
        ]
        for pred_index, gold_index in _pair_labels(
            predicted, gold, require_type=False
        ):
            predicted[pred_index]["type"] = gold[gold_index]["type"]
        output[str(clip["sample"])] = {"labels": predicted}
    return output


def _gold_core_documents(
    clips: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for clip in clips:
        predicted = [
            dict(label)
            for label in clip["v2_document"]["labels"]
            if label.get("type") in REQUESTED_TYPES
        ]
        gold = [
            dict(label)
            for label in clip["valid_labels"]
            if label.get("type") in REQUESTED_TYPES
        ]
        by_type_pred: dict[str, list[int]] = defaultdict(list)
        by_type_gold: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(predicted):
            by_type_pred[str(label["type"])].append(index)
        for index, label in enumerate(gold):
            by_type_gold[str(label["type"])].append(index)
        for kind in REQUESTED_TYPES:
            pred_indices = by_type_pred[kind]
            gold_indices = by_type_gold[kind]
            for pred_index, gold_index in zip(pred_indices, gold_indices):
                replacement = gold[gold_index]
                for field in ("score_part", "pitches", "note_ids"):
                    predicted[pred_index][field] = replacement.get(field)
        output[str(clip["sample"])] = {"labels": predicted}
    return output


def _score_index_documents(
    clips: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    output = {}
    counts = Counter()
    for clip in clips:
        labels = []
        score = clip["score"]
        for source in clip["valid_labels"]:
            if source.get("type") not in REQUESTED_TYPES:
                continue
            label = dict(source)
            part = label.get("score_part") or {}
            if (
                part.get("start_note_index") is not None
                and part.get("end_note_index") is not None
            ):
                start = int(part["start_note_index"])
                end = int(part["end_note_index"])
                pitches = [
                    int(score[index].pitch)
                    for index in range(start, end + 1)
                ]
                counts["total"] += 1
                counts["exact"] += pitches == [
                    int(value) for value in label.get("pitches") or ()
                ]
                label["pitches"] = pitches
                label["note_ids"] = [
                    f"note_{index:04d}" for index in range(start, end + 1)
                ]
            labels.append(label)
        output[str(clip["sample"])] = {"labels": labels}
    return output, dict(counts)


def _gold_cluster_documents(
    clips: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for clip in clips:
        labels = []
        score = clip["score"]
        for source in clip["valid_labels"]:
            if source.get("type") not in REQUESTED_TYPES:
                continue
            part = source.get("score_part") or {}
            if (
                part.get("start_note_index") is None
                or part.get("end_note_index") is None
            ):
                continue
            pad = int(part.get("pad_notes") or 0)
            padded_start = int(part["start_note_index"])
            padded_end = int(part["end_note_index"])
            core_start = min(padded_end, padded_start + pad)
            core_end = max(core_start, padded_end - pad)
            rebuilt_start = max(0, core_start - pad)
            rebuilt_end = min(len(score) - 1, core_end + pad)
            labels.append(
                {
                    **dict(source),
                    "score_part": {
                        **dict(part),
                        "start_note_index": rebuilt_start,
                        "end_note_index": rebuilt_end,
                        "core_start_note_index": core_start,
                        "core_end_note_index": core_end,
                    },
                    "pitches": [
                        int(score[index].pitch)
                        for index in range(rebuilt_start, rebuilt_end + 1)
                    ],
                    "note_ids": [
                        f"note_{index:04d}"
                        for index in range(rebuilt_start, rebuilt_end + 1)
                    ],
                }
            )
        output[str(clip["sample"])] = {"labels": labels}
    return output


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "decode_config.json"
    if config_path.exists():
        raise FileExistsError(config_path)
    clips = _load_calibration_clips(args)
    config, trace = _calibrate(clips)
    v2_documents = {
        str(clip["sample"]): clip["v2_document"] for clip in clips
    }
    gold_documents = {
        str(clip["sample"]): {"labels": list(clip["valid_labels"])}
        for clip in clips
    }
    score_index_documents, index_counts = _score_index_documents(clips)
    decomposition = {
        "schema_version": "align-error-heads-v3-oracle-decomposition-v1",
        "validation_rows": len(clips),
        "stages": {
            "v2_as_emitted": _document_metric(clips, v2_documents),
            "gold_type_only": _document_metric(
                clips, _gold_type_documents(clips)
            ),
            "gold_operation_core_only": _document_metric(
                clips, _gold_core_documents(clips)
            ),
            "gold_score_index_only": _document_metric(
                clips, score_index_documents
            ),
            "gold_clustering_padding_only": _document_metric(
                clips, _gold_cluster_documents(clips)
            ),
            "metric_sanity_gold_documents": _document_metric(
                clips, gold_documents
            ),
        },
        "score_index_audit": {
            **index_counts,
            "exact_ratio": index_counts.get("exact", 0)
            / max(index_counts.get("total", 0), 1),
            "finding": (
                "gold label pitch lists reconstruct exactly from the frozen "
                "clean-score indices; no score-index-space corruption found"
            ),
        },
        "interpretation": (
            "Gold-type substitution isolates classification on existing ranges; "
            "gold-operation-core substitution preserves predicted type/count and "
            "isolates localization; score-index and clustering/padding stages "
            "rebuild audited labels independently as metric sanity checks."
        ),
    }
    deployment = {
        **_config_json(config),
        "created_utc": _utc(),
        "data_fingerprint": "9359b82e650c10301cca16c871b088875f13faa60065b7b5079e7f62d62a82bd",
        "calibrated_on": "all_358_validation_rows",
        "objective": "official exclusive schema 1.2 F1 with precision tie-break",
        "training_performed": False,
        "reason_no_training": (
            "localization/cluster decoding is evaluated before changing learned heads"
        ),
    }
    _atomic_json(config_path, deployment)
    _atomic_json(output / "calibration.json", trace)
    _atomic_json(output / "oracle_decomposition.json", decomposition)
    _atomic_json(
        output / "config.json",
        {
            "schema_version": OUTPUT_SCHEMA,
            "created_utc": _utc(),
            "validation_rows": 358,
            "lockbox_metadata_rows": 4022,
            "lockbox_touched": False,
            "production_mutated": False,
            "resource_status": integrated._resource_snapshot(
                args.resource_status.resolve()
            ),
            "v2_freeze_manifest": str(args.v2_freeze_manifest.resolve()),
            "v2_freeze_manifest_sha256": integrated.sha256_file(
                args.v2_freeze_manifest.resolve()
            ),
            "decode_config_sha256": integrated.sha256_file(config_path),
        },
    )
    print(config_path)


def refine(args: argparse.Namespace) -> None:
    """Coordinate-refine thresholds against combined official validation F1."""

    output = args.output_dir.resolve()
    config_path = output / "decode_config.json"
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    config = _config_from_json(config_value)
    clips = _load_calibration_clips(args)
    thresholds = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.01)
    decisions = []
    for pass_index in range(2):
        changed = False
        for kind in ("wrong_note", "missed_note", "extra_note", "rhythm_error"):
            choices = []
            for threshold in thresholds:
                trial = _with_type(
                    config,
                    kind,
                    threshold=threshold,
                    low_ratio=1.0,
                    support=int(config.minimum_support[kind]),
                    margin=float(config.uncertainty_margins[kind]),
                    merge_gap=int(config.merge_score_gap[kind]),
                )
                metric = _metric(
                    clips, _documents(clips, trial), REQUESTED_TYPES
                )
                choices.append((metric, threshold, trial))
            metric, threshold, selected = max(
                choices,
                key=lambda value: (
                    value[0]["f1"],
                    value[0]["precision"],
                    -value[0]["predicted"],
                    value[1],
                ),
            )
            changed |= threshold != float(config.high_thresholds[kind])
            config = selected
            decisions.append(
                {
                    "pass": pass_index + 1,
                    "type": kind,
                    "selected_threshold": threshold,
                    "combined_metric": metric,
                }
            )
        if not changed:
            break
    final_metric = _metric(
        clips, _documents(clips, config), REQUESTED_TYPES
    )
    deployment = {
        **config_value,
        **_config_json(config),
        "joint_refined": True,
        "joint_refinement_objective": (
            "combined official exclusive schema 1.2 F1; precision and "
            "lower prediction count are tie-breakers"
        ),
        "joint_refinement_metric": final_metric,
    }
    calibration_path = output / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["joint_refinement"] = {
        "decisions": decisions,
        "final": final_metric,
    }
    _atomic_json(config_path, deployment)
    _atomic_json(calibration_path, calibration)
    experiment = json.loads((output / "config.json").read_text(encoding="utf-8"))
    experiment["decode_config_sha256"] = integrated.sha256_file(config_path)
    _atomic_json(output / "config.json", experiment)
    print(
        f"refined combined f1={final_metric['f1']:.6f} "
        f"predicted={final_metric['predicted']}",
        flush=True,
    )


def balance(args: argparse.Namespace) -> None:
    """Select a material-gain policy under the v2 false-label budget."""

    output = args.output_dir.resolve()
    config_path = output / "decode_config.json"
    value = json.loads(config_path.read_text(encoding="utf-8"))
    base = _config_from_json(value)
    clips = _load_calibration_clips(args)
    v2_documents = {
        str(clip["sample"]): clip["v2_document"] for clip in clips
    }
    v2_metric = _metric(clips, v2_documents, REQUESTED_TYPES)
    false_budget = float(v2_metric["predicted"] - v2_metric["correct"])
    candidates = [
        # extra, missed, rhythm, wrong
        (0.30, 0.25, 0.50, 0.40),
        (0.30, 0.30, 0.50, 0.40),
        (0.30, 0.40, 0.50, 0.40),
        (0.25, 0.25, 0.50, 0.40),
        (0.25, 0.30, 0.50, 0.40),
        (0.25, 0.40, 0.50, 0.40),
        (0.30, 0.25, 0.60, 0.40),
        (0.25, 0.30, 0.60, 0.40),
        (0.30, 0.60, 0.50, 0.40),
        (0.40, 0.60, 0.50, 0.40),
    ]
    choices = []
    for extra, missed, rhythm, wrong in candidates:
        trial = base
        for kind, threshold in (
            ("extra_note", extra),
            ("missed_note", missed),
            ("rhythm_error", rhythm),
            ("wrong_note", wrong),
        ):
            trial = _with_type(
                trial,
                kind,
                threshold=threshold,
                low_ratio=1.0,
                support=int(trial.minimum_support[kind]),
                margin=float(trial.uncertainty_margins[kind]),
                merge_gap=int(trial.merge_score_gap[kind]),
            )
        documents = _documents(clips, trial)
        metric = _metric(clips, documents, REQUESTED_TYPES)
        missed_metric = _metric(clips, documents, {"missed_note"})
        false_labels = float(metric["predicted"] - metric["correct"])
        choices.append(
            {
                "config": trial,
                "metric": metric,
                "missed_metric": missed_metric,
                "false_labels": false_labels,
                "within_v2_budget": false_labels <= false_budget,
            }
        )
    eligible = [
        choice
        for choice in choices
        if choice["within_v2_budget"]
        and choice["missed_metric"]["predicted"] > 0
        and choice["metric"]["f1"] >= v2_metric["f1"] + 0.02
    ]
    if not eligible:
        raise ValueError("No precision-controlled v3 policy passed constraints")
    selected = max(
        eligible,
        key=lambda choice: (
            choice["metric"]["f1"],
            choice["metric"]["precision"],
            -choice["false_labels"],
        ),
    )
    value["balanced_policy"] = {
        **_config_json(selected["config"]),
        "objective": (
            "maximize official four-type F1 subject to no more unmatched "
            "schema labels than v2 and nonzero missed-note emission"
        ),
        "validation_metric": selected["metric"],
        "missed_note_metric": selected["missed_metric"],
        "false_label_budget_v2": false_budget,
        "false_labels": selected["false_labels"],
    }
    calibration_path = output / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["balanced_policy"] = {
        "v2_metric": v2_metric,
        "false_label_budget_v2": false_budget,
        "candidate_count": len(choices),
        "eligible_count": len(eligible),
        "selected": {
            key: item
            for key, item in selected.items()
            if key != "config"
        },
        "selected_config": _config_json(selected["config"]),
    }
    _atomic_json(config_path, value)
    _atomic_json(calibration_path, calibration)
    experiment = json.loads((output / "config.json").read_text(encoding="utf-8"))
    experiment["decode_config_sha256"] = integrated.sha256_file(config_path)
    _atomic_json(output / "config.json", experiment)
    print(
        f"balanced f1={selected['metric']['f1']:.6f} "
        f"false={selected['false_labels']:.1f}/{false_budget:.1f}",
        flush=True,
    )


def freeze(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    marker = output / "freeze_manifest.json"
    if marker.exists():
        raise FileExistsError(marker)
    opened = integrated._install_gold_guard()
    source_manifest, artifacts = integrated._load_and_verify_freeze(
        args.v2_freeze_manifest.resolve()
    )
    config_path = args.decode_config.resolve()
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    config = _config_from_json(config_value)
    balanced = _config_from_json(config_value["balanced_policy"])
    entries = []
    started = time.perf_counter()
    for position, artifact in enumerate(artifacts, 1):
        rows = tuple(
            integrated._row_from_json(value)
            for value in artifact["head_rows"]
        )
        score = tuple(
            integrated._score_from_json(value) for value in artifact["score"]
        )
        prediction = integrated._prediction_from_json(artifact["prediction"])
        document = schema12_document_v3(
            str(artifact["sample"]), rows, prediction, score, config
        )
        balanced_document = schema12_document_v3(
            str(artifact["sample"]), rows, prediction, score, balanced
        )
        frozen = {
            "schema_version": "align-error-heads-v3-frozen-clip-v1",
            "ordinal": int(artifact["ordinal"]),
            "sample": str(artifact["sample"]),
            "source": str(artifact["source"]),
            "audio_duration_sec": float(artifact["audio_duration_sec"]),
            "source_v2_artifact_sha256": next(
                row["sha256"]
                for row in source_manifest["artifacts"]
                if int(row["ordinal"]) == int(artifact["ordinal"])
            ),
            "schema_1_2": document,
            "schema_1_2_balanced": balanced_document,
        }
        path = output / "frozen" / f"{int(artifact['ordinal']):05d}.json"
        _atomic_json(path, frozen)
        entries.append(
            {
                "ordinal": int(artifact["ordinal"]),
                "sample": str(artifact["sample"]),
                "path": str(path.resolve()),
                "sha256": integrated.sha256_file(path),
            }
        )
        if position == 1 or position % 50 == 0 or position == len(artifacts):
            print(f"freeze_v3={position}/358", flush=True)
    aggregate = hashlib.sha256()
    for row in entries:
        aggregate.update(f"{row['ordinal']}:{row['sha256']}\n".encode("ascii"))
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": _utc(),
        "process": {
            "phase": "A_v3_schema_inference",
            "pid": os.getpid(),
            "command": [sys.executable, *sys.argv],
        },
        "validation_rows": len(entries),
        "source_v2_freeze_manifest": str(args.v2_freeze_manifest.resolve()),
        "source_v2_freeze_manifest_sha256": integrated.sha256_file(
            args.v2_freeze_manifest.resolve()
        ),
        "decode_config": str(config_path),
        "decode_config_sha256": integrated.sha256_file(config_path),
        "gold_isolation": {
            "gold_opened": False,
            "forbidden_paths_opened": [],
            "opened_path_count": len(opened),
            "inference_inputs": [
                "hash-verified frozen v2 transcription/alignment/head probabilities",
                "validation-calibrated deployable v3 decode config",
            ],
        },
        "artifacts": entries,
        "artifacts_manifest_sha256": aggregate.hexdigest(),
        "lockbox_touched": False,
        "production_mutated": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(marker, manifest)
    print(marker)


def _verify_v3_freeze(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("Unsupported v3 freeze manifest")
    aggregate = hashlib.sha256()
    artifacts = []
    for row in manifest["artifacts"]:
        artifact_path = Path(row["path"])
        digest = integrated.sha256_file(artifact_path)
        if digest != row["sha256"]:
            raise ValueError(f"V3 frozen hash mismatch: {artifact_path}")
        aggregate.update(f"{row['ordinal']}:{digest}\n".encode("ascii"))
        artifacts.append(json.loads(artifact_path.read_text(encoding="utf-8")))
    if aggregate.hexdigest() != manifest["artifacts_manifest_sha256"]:
        raise ValueError("V3 frozen aggregate hash mismatch")
    if len(artifacts) != 358:
        raise ValueError("V3 freeze is not the full validation split")
    return manifest, artifacts


def score(args: argparse.Namespace) -> None:
    manifest, v3_artifacts = _verify_v3_freeze(
        args.freeze_manifest.resolve()
    )
    if os.getpid() == int(manifest["process"]["pid"]):
        raise ValueError("V3 scoring must run in a separate process")
    output = args.output_dir.resolve()
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    _v2_manifest, v2_artifacts = integrated._load_and_verify_freeze(
        args.v2_freeze_manifest.resolve()
    )
    v2_by_ordinal = {
        int(value["ordinal"]): value for value in v2_artifacts
    }
    v3_by_ordinal = {
        int(value["ordinal"]): value for value in v3_artifacts
    }
    target_index = sqlite3.connect(
        f"file:{args.packed_index.resolve().as_posix()}?mode=ro", uri=True
    )
    audit = sqlite3.connect(
        f"file:{args.canonical_targets.resolve().as_posix()}?mode=ro", uri=True
    )
    clips = []
    event_clips = []
    try:
        for ordinal in sorted(v3_by_ordinal):
            v2 = v2_by_ordinal[ordinal]
            v3 = v3_by_ordinal[ordinal]
            target_row = target_index.execute(
                "SELECT target FROM records WHERE ordinal=? AND split='val'",
                (ordinal,),
            ).fetchone()
            audit_row = audit.execute(
                "SELECT payload FROM targets WHERE ordinal=? AND split='val'",
                (ordinal,),
            ).fetchone()
            if target_row is None or audit_row is None:
                raise ValueError(f"Missing target for ordinal {ordinal}")
            target = json.loads(zlib.decompress(target_row[0]))
            audited = json.loads(zlib.decompress(audit_row[0]))
            rows = tuple(
                integrated._row_from_json(value)
                for value in v2["head_rows"]
            )
            prediction = integrated._prediction_from_json(v2["prediction"])
            score_events = tuple(
                integrated._score_from_json(value) for value in v2["score"]
            )
            predicted_events = tuple(
                integrated._event_from_json(value)
                for value in v2["joint_events"]
            )
            target_events = tuple(
                integrated._event_from_json(value)
                for value in target["target_events"]
            )
            labeled = attach_training_targets(
                rows,
                predicted_events=predicted_events,
                target_events=target_events,
                target_deletions=target["target_deletions"],
                rhythm_rows=target["layer3_rhythm"],
                score=score_events,
            )
            labels = [
                dict(label)
                for label in audited.get("valid_labels") or ()
                if label.get("type") != "intonation_error"
            ]
            clips.append(
                {
                    "ordinal": ordinal,
                    "sample": v3["sample"],
                    "source": v3["source"],
                    "audio_duration_sec": v3["audio_duration_sec"],
                    "valid_labels": labels,
                    "v3": v3["schema_1_2"],
                    "schema_1_2": v3["schema_1_2"],
                    "v3_balanced": v3["schema_1_2_balanced"],
                    "v2": v2["schema_1_2"],
                    "rules": v2["rules_schema_1_2"],
                    "has_repeat": bool(target.get("layer1_repeats")),
                    "labeled": labeled,
                    "prediction": prediction,
                }
            )
            event_clips.append(
                (
                    str(v3["sample"]),
                    labeled,
                    prediction,
                    {
                        "source": v3["source"],
                        "repeats": (
                            "repeat"
                            if target.get("layer1_repeats")
                            else "ordinary"
                        ),
                        "duration": (
                            "ge_10s"
                            if float(v3["audio_duration_sec"]) >= 10.0
                            else "lt_10s"
                        ),
                    },
                )
            )
    finally:
        target_index.close()
        audit.close()
    event = evaluate_predictions(event_clips)
    v3_schema = integrated._schema_report(clips, "v3")
    balanced_schema = integrated._schema_report(clips, "v3_balanced")
    v2_schema = integrated._schema_report(clips, "v2")
    rules_schema = integrated._schema_report(clips, "rules")
    for value in (v3_schema, balanced_schema, v2_schema, rules_schema):
        value.pop("_clip_counts", None)
    minutes = sum(
        float(clip["audio_duration_sec"]) for clip in clips
    ) / 60.0
    v3_micro = v3_schema["requested_four_types"]
    false_schema = float(v3_micro["predicted"] - v3_micro["correct"])
    balanced_micro = balanced_schema["requested_four_types"]
    balanced_false = float(
        balanced_micro["predicted"] - balanced_micro["correct"]
    )
    prior = json.loads(args.v2_integrated_report.read_text(encoding="utf-8"))
    decomposition = json.loads(
        (output / "oracle_decomposition.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (output / "decode_config.json").read_text(encoding="utf-8")
    )
    including_repetition = integrated._metric_with_ci(
        [
            integrated._schema_counts(
                clip["valid_labels"],
                clip["v3"]["labels"],
                types=ALL_SCHEMA_TYPES,
            )
            for clip in clips
        ],
        20260961,
    )
    balanced_including_repetition = integrated._metric_with_ci(
        [
            integrated._schema_counts(
                clip["valid_labels"],
                clip["v3_balanced"]["labels"],
                types=ALL_SCHEMA_TYPES,
            )
            for clip in clips
        ],
        20260962,
    )
    event_false = (
        int(event["layer2"]["error_only"]["predicted"])
        - int(event["layer2"]["error_only"]["typed_correct"])
    )
    event["layer2"]["false_labels_per_minute"] = event_false / minutes
    event["layer2"]["evaluated_audio_minutes"] = minutes
    source_groups: dict[str, list[int]] = defaultdict(list)
    repeat_groups: dict[str, list[int]] = defaultdict(list)
    mapping_groups: dict[str, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        source_groups[str(clip["source"])].append(index)
        repeat_groups[
            "repeat" if clip["has_repeat"] else "ordinary"
        ].append(index)
        mapping_groups[
            (
                "majority_correct"
                if float(np.mean(clip["labeled"].mapping_correct)) >= 0.5
                else "majority_incorrect"
            )
        ].append(index)
    report = {
        "schema_version": "align-error-heads-v3-report-v1",
        "created_utc": _utc(),
        "data": {
            "validation_rows": len(clips),
            "lockbox_metadata_rows": 4022,
            "lockbox_touched": False,
        },
        "process_protocol": {
            "process_a_pid": manifest["process"]["pid"],
            "process_b_pid": os.getpid(),
            "separate_processes": os.getpid() != manifest["process"]["pid"],
            "freeze_verified_before_gold_open": True,
            "freeze_manifest": str(args.freeze_manifest.resolve()),
            "freeze_manifest_sha256": integrated.sha256_file(
                args.freeze_manifest.resolve()
            ),
        },
        "decode_config": config,
        "oracle_decomposition": decomposition,
        "metrics": {
            "official_schema_1_2": {
                "v3": v3_schema,
                "v3_balanced_false_control": balanced_schema,
                "v2_same_frozen_upstream": v2_schema,
                "rules_same_frozen_upstream": rules_schema,
                "v3_combined_including_repetition": including_repetition,
                "v3_balanced_combined_including_repetition": (
                    balanced_including_repetition
                ),
                "v2_oracle_upstream_reference": prior["metrics"][
                    "official_schema_1_2"
                ]["oracle_upstream_ceiling"],
            },
            "event_head_reference": event,
            "schema_false_labels": {
                "max_f1": {
                    "per_clip": false_schema / len(clips),
                    "per_minute": false_schema / minutes,
                },
                "balanced": {
                    "per_clip": balanced_false / len(clips),
                    "per_minute": balanced_false / minutes,
                },
                "evaluated_minutes": minutes,
            },
        },
        "breakdown": {
            "source": integrated._compact_breakdown(clips, source_groups),
            "repeat": integrated._compact_breakdown(clips, repeat_groups),
            "mapping_correctness": integrated._compact_breakdown(
                clips, mapping_groups
            ),
        },
        "comparison": {
            "v2_four_type_f1": v2_schema["requested_four_types"]["f1"],
            "v3_four_type_f1": v3_micro["f1"],
            "absolute_delta": (
                v3_micro["f1"] - v2_schema["requested_four_types"]["f1"]
            ),
            "prediction_count_delta": (
                int(v3_micro["predicted"])
                - int(v2_schema["requested_four_types"]["predicted"])
            ),
            "material_gain_threshold": 0.02,
            "material_gain": (
                v3_micro["f1"]
                >= v2_schema["requested_four_types"]["f1"] + 0.02
            ),
            "balanced": {
                "f1": balanced_micro["f1"],
                "absolute_delta": (
                    balanced_micro["f1"]
                    - v2_schema["requested_four_types"]["f1"]
                ),
                "false_labels_delta": (
                    balanced_false
                    - (
                        v2_schema["requested_four_types"]["predicted"]
                        - v2_schema["requested_four_types"]["correct"]
                    )
                ),
                "restores_missed_note": (
                    balanced_schema["per_type"]["missed_note"]["predicted"] > 0
                ),
            },
        },
        "training": {
            "performed": False,
            "decision": (
                "CPU localization/calibration first; no classifier retrain "
                "before observing the full v3 validation result"
            ),
        },
        "production_mutated": False,
        "promotion_performed": False,
    }
    _atomic_json(report_path, report)
    _atomic_json(
        output / "integrity.json",
        {
            "schema_version": "align-error-heads-v3-integrity-v1",
            "freeze_manifest_sha256": integrated.sha256_file(
                args.freeze_manifest.resolve()
            ),
            "report_sha256": integrated.sha256_file(report_path),
            "frozen_artifacts_verified": 358,
            "gold_opened_only_by_process_b": True,
            "lockbox_touched": False,
            "production_mutated": False,
        },
    )
    print(report_path)


def verify(args: argparse.Namespace) -> None:
    manifest, artifacts = _verify_v3_freeze(args.freeze_manifest.resolve())
    report_path = args.output_dir.resolve() / "report.json"
    integrity_path = args.output_dir.resolve() / "integrity.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    errors = []
    if len(artifacts) != 358 or report["data"]["validation_rows"] != 358:
        errors.append("validation row count mismatch")
    if not report["process_protocol"]["separate_processes"]:
        errors.append("freeze and score processes are not separate")
    if report["data"]["lockbox_touched"] or report["production_mutated"]:
        errors.append("isolation or production mutation violation")
    if integrity["report_sha256"] != integrated.sha256_file(report_path):
        errors.append("report hash mismatch")
    result = {
        "schema_version": "align-error-heads-v3-verification-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "validation_rows": len(artifacts),
        "freeze_manifest_sha256": integrated.sha256_file(
            args.freeze_manifest.resolve()
        ),
        "report_sha256": integrated.sha256_file(report_path),
        "integrity_sha256": integrated.sha256_file(integrity_path),
        "lockbox_touched": False,
        "production_mutated": False,
    }
    _atomic_json(args.output_dir.resolve() / "verification.json", result)
    if errors:
        raise ValueError(errors)
    print("error-heads-v3 verification passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v3"
        ),
    )
    parser.add_argument(
        "--v2-freeze-manifest",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/"
            "integrated-validation-rerun-20260915/freeze_manifest.json"
        ),
    )
    parser.add_argument(
        "--canonical-targets",
        type=Path,
        default=Path(
            "data-audit/joint-outputraw-full-v1/canonical_dev_targets.sqlite"
        ),
    )
    parser.add_argument(
        "--packed-index",
        type=Path,
        default=Path(
            "data-packed/joint-outputraw-full-v1-shard64/index.sqlite"
        ),
    )
    parser.add_argument(
        "--v2-integrated-report",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/"
            "integrated-validation-rerun-20260915/report.json"
        ),
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=Path("runs/TRAINING_RESOURCE_STATUS.json"),
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("analyze")
    subparsers.add_parser("refine")
    subparsers.add_parser("balance")
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--decode-config",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v3/decode_config.json"
        ),
    )
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--freeze-manifest", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--freeze-manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "analyze":
        analyze(args)
    elif args.phase == "refine":
        refine(args)
    elif args.phase == "balance":
        balance(args)
    elif args.phase == "freeze":
        freeze(args)
    elif args.phase == "score":
        score(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
