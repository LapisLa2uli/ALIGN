"""Leakage-free train-only calibration and one-shot v5 validation."""

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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.joint.candidates import ScoreEvent
from alignmodel.joint.error_heads import (
    FEATURE_NAMES,
    LAYER2_CLASSES,
    HeadPrediction,
    HeadRow,
    SchemaDecodeConfig,
    blend_direct_learned_prediction,
    direct_operation_probabilities,
    direct_rhythm_probabilities,
    infer_error_heads,
    labeled_from_json,
    load_error_heads,
)

import eval_error_heads_integrated as integrated
import train_error_heads_v3 as v3
import train_error_heads_v4 as v4


PROTOCOL_SCHEMA = "align-error-heads-v5-protocol-v1"
FREEZE_SCHEMA = "align-error-heads-v5-freeze-v1"
REQUESTED_TYPES = integrated.REQUESTED_TYPES
ALL_TYPES = REQUESTED_TYPES | {"repetition"}
SELECTOR_KINDS = ("wrong_note", "extra_note", "missed_note", "rhythm_error")


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


def _id_hash(values: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in sorted(values)).encode("ascii")
    ).hexdigest()


def assert_protocol_disjoint(
    fit_ids: Sequence[int],
    calibration_ids: Sequence[int],
    validation_ids: Sequence[int],
    groups: Mapping[int, str],
) -> None:
    """Refuse row or lineage-group leakage across all protocol partitions."""

    fit = set(int(value) for value in fit_ids)
    calibration = set(int(value) for value in calibration_ids)
    validation = set(int(value) for value in validation_ids)
    if fit & calibration or fit & validation or calibration & validation:
        raise ValueError("V5 protocol row leakage detected")
    missing = (fit | calibration | validation) - set(groups)
    if missing:
        raise ValueError(f"V5 protocol lacks groups for {len(missing)} rows")
    partitions = (
        ("fit", fit),
        ("calibration", calibration),
        ("validation", validation),
    )
    group_sets = {
        name: {groups[value] for value in values}
        for name, values in partitions
    }
    for left_index, (left_name, _left) in enumerate(partitions):
        for right_name, _right in partitions[left_index + 1 :]:
            overlap = group_sets[left_name] & group_sets[right_name]
            if overlap:
                raise ValueError(
                    f"V5 protocol group leakage: {left_name}/{right_name}"
                )


def _membership(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    by_sample = {
        str(row["sample"]): row
        for split in ("train", "val")
        for row in document[split]
    }
    connection = sqlite3.connect(
        "data-packed/joint-outputraw-full-v1-shard64/index.sqlite"
    )
    try:
        rows = connection.execute(
            "SELECT ordinal,sample FROM records WHERE split IN ('train','val')"
        ).fetchall()
    finally:
        connection.close()
    by_ordinal = {}
    sample_to_ordinal = {}
    for ordinal, sample in rows:
        member = by_sample[str(sample)]
        group = (
            f"{member['source']}:{member.get('leakage_group') or member['clean_fingerprint']}"
        )
        by_ordinal[int(ordinal)] = {
            **member,
            "ordinal": int(ordinal),
            "group": group,
        }
        sample_to_ordinal[str(sample)] = int(ordinal)
    return by_ordinal, sample_to_ordinal


def _decode_blob(value: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(value))


def _load_training_clips(args: argparse.Namespace) -> list[dict[str, Any]]:
    membership, _sample_to_ordinal = _membership(args.split_manifest.resolve())
    model, payload = load_error_heads(
        args.error_heads_checkpoint.resolve(), device="cpu"
    )
    thresholds = payload["thresholds"]
    connection = sqlite3.connect(
        f"file:{args.examples_cache.resolve().as_posix()}?mode=ro", uri=True
    )
    clips = []
    try:
        labels_by_ordinal = {
            int(ordinal): json.loads(labels)
            for ordinal, labels in connection.execute(
                "SELECT ordinal,labels FROM audited_labels"
            )
        }
        rows = connection.execute(
            "SELECT ordinal,sample,source,predicted,score,metadata "
            "FROM clips WHERE split='train' ORDER BY ordinal"
        )
        for position, (
            ordinal,
            sample,
            source,
            predicted_blob,
            score_json,
            metadata_json,
        ) in enumerate(rows, 1):
            ordinal = int(ordinal)
            member = membership[ordinal]
            if member["split"] != "train":
                raise ValueError(f"Non-train row entered fitting: {ordinal}")
            labeled = labeled_from_json(_decode_blob(predicted_blob))
            prediction = infer_error_heads(
                model, labeled.rows, thresholds, device="cpu"
            )
            direct, diagnostics = direct_operation_probabilities(labeled.rows)
            clips.append(
                {
                    "ordinal": ordinal,
                    "sample": str(sample),
                    "source": str(source),
                    "group": str(member["group"]),
                    "rows": labeled.rows,
                    "labeled": labeled,
                    "score": tuple(
                        ScoreEvent(
                            index=int(value["index"]),
                            pitch=int(value["pitch"]),
                            ql_start=float(value["ql_start"]),
                            ql_end=float(value["ql_end"]),
                            source_indices=tuple(
                                int(item)
                                for item in value.get("source_indices")
                                or (value["index"],)
                            ),
                            measure=(
                                int(value["measure"])
                                if value.get("measure") is not None
                                else None
                            ),
                        )
                        for value in json.loads(score_json)
                    ),
                    "metadata": json.loads(metadata_json),
                    "prediction": prediction,
                    "learned_prediction": prediction,
                    "direct_probabilities": direct,
                    "direct_rhythm_probabilities": direct_rhythm_probabilities(
                        labeled.rows
                    ),
                    "direct_diagnostics": diagnostics,
                    "valid_labels": [
                        dict(label)
                        for label in labels_by_ordinal[ordinal]
                        if label.get("type") != "intonation_error"
                    ],
                }
            )
            if position == 1 or position % 500 == 0:
                print(f"train_cache={position}/4544", flush=True)
    finally:
        connection.close()
    if len(clips) != 4544:
        raise ValueError(f"Expected 4544 train rows, got {len(clips)}")
    return clips


_SELECTOR_FEATURES = (
    "upstream_path_margin",
    "upstream_extra_probability",
    "upstream_noise_probability",
    "upstream_delete_probability",
    "candidate_confidence",
    "previous_confidence",
    "next_confidence",
    "log_tempo_normalized_duration_ratio",
    "previous_ioi_ratio",
    "next_ioi_ratio",
    "local_global_tempo_ratio",
    "gap_before_sec",
    "gap_after_sec",
    "normalized_score_position",
    "is_replay_state",
    "deleted_count_normalized",
    "path_operation_wrong",
    "path_operation_extra",
    "path_operation_missed",
)


def _candidate_mask(rows: Sequence[HeadRow], kind: str) -> np.ndarray:
    output = np.zeros(len(rows), dtype=bool)
    operation_name = {
        "wrong_note": "path_operation_wrong",
        "extra_note": "path_operation_extra",
        "missed_note": "path_operation_missed",
    }.get(kind)
    for index, row in enumerate(rows):
        if row.is_copy:
            continue
        if kind == "rhythm_error":
            output[index] = bool(
                row.kind == "event"
                and row.score_span is not None
                and row.features[FEATURE_NAMES.index("is_replay_state")] <= 0.5
            )
        elif operation_name is not None:
            output[index] = bool(
                row.features[FEATURE_NAMES.index(operation_name)] > 0.5
                and (
                    kind != "missed_note"
                    or row.kind == "gap"
                )
            )
    return output


def _matrix(
    clip: Mapping[str, Any], kind: str
) -> tuple[np.ndarray, np.ndarray]:
    rows = clip["rows"]
    mask = _candidate_mask(rows, kind)
    indices = np.flatnonzero(mask)
    if kind == "rhythm_error":
        learned = np.asarray(
            clip["learned_prediction"].rhythm_probabilities, dtype=np.float32
        )
        direct = np.asarray(
            clip["direct_rhythm_probabilities"], dtype=np.float32
        )
    else:
        class_index = LAYER2_CLASSES.index(kind)
        learned = np.asarray(
            clip["learned_prediction"].layer2_probabilities, dtype=np.float32
        )[:, class_index]
        direct = np.asarray(
            clip["direct_probabilities"], dtype=np.float32
        )[:, class_index]
    feature_indices = [FEATURE_NAMES.index(name) for name in _SELECTOR_FEATURES]
    values = [
        [
            float(direct[index]),
            float(learned[index]),
            *[float(rows[index].features[item]) for item in feature_indices],
        ]
        for index in indices
    ]
    return np.asarray(values, dtype=np.float32), indices.astype(np.int64)


def _targets(
    clip: Mapping[str, Any], kind: str, indices: np.ndarray
) -> np.ndarray:
    labeled = clip["labeled"]
    if kind == "rhythm_error":
        target = labeled.rhythm.astype(np.int64)
        valid = labeled.rhythm_mask.astype(bool)
        return np.asarray(
            [target[index] if valid[index] else 0 for index in indices],
            dtype=np.int64,
        )
    class_index = LAYER2_CLASSES.index(kind)
    return (labeled.layer2[indices] == class_index).astype(np.int64)


def _fit_selector(clips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selector: dict[str, Any] = {
        "schema_version": "align-error-heads-v5-operation-selector-v1",
        "features": [
            "direct_reliability",
            "learned_probability",
            *_SELECTOR_FEATURES,
        ],
        "classes": {},
    }
    for kind in SELECTOR_KINDS:
        matrices = []
        targets = []
        for clip in clips:
            matrix, indices = _matrix(clip, kind)
            if len(matrix):
                matrices.append(matrix)
                targets.append(_targets(clip, kind, indices))
        x = np.concatenate(matrices)
        y = np.concatenate(targets)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-5] = 1.0
        normalized = (x - mean) / scale
        coefficient = np.zeros(normalized.shape[1], dtype=np.float64)
        intercept = 0.0
        positive_weight = len(y) / max(2.0 * float(y.sum()), 1.0)
        negative_weight = len(y) / max(2.0 * float((1 - y).sum()), 1.0)
        sample_weight = np.where(y > 0, positive_weight, negative_weight)
        for iteration in range(1000):
            logits = normalized @ coefficient + intercept
            predicted = 1.0 / (
                1.0 + np.exp(-np.clip(logits, -20.0, 20.0))
            )
            residual = sample_weight * (predicted - y)
            learning_rate = 0.07 / (1.0 + iteration / 300.0)
            coefficient -= learning_rate * (
                normalized.T @ residual / sample_weight.sum()
                + 0.08 * coefficient
            )
            intercept -= learning_rate * float(
                residual.sum() / sample_weight.sum()
            )
        selector["classes"][kind] = {
            "coefficient": coefficient.tolist(),
            "intercept": intercept,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "candidate_rows": int(len(y)),
            "positive_rows": int(y.sum()),
            "negative_rows": int(len(y) - y.sum()),
        }
    return selector


def _selector_outputs(
    clip: Mapping[str, Any], selector: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    layer2 = np.zeros(
        (len(clip["rows"]), len(LAYER2_CLASSES)), dtype=np.float32
    )
    rhythm = np.zeros(len(clip["rows"]), dtype=np.float32)
    for kind, state in selector["classes"].items():
        matrix, indices = _matrix(clip, kind)
        if not len(matrix):
            continue
        mean = np.asarray(state["mean"], dtype=np.float32)
        scale = np.asarray(state["scale"], dtype=np.float32)
        coefficient = np.asarray(state["coefficient"], dtype=np.float32)
        logits = (
            ((matrix - mean) / scale) @ coefficient
            + float(state["intercept"])
        )
        values = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        if kind == "rhythm_error":
            rhythm[indices] = values
        else:
            layer2[indices, LAYER2_CLASSES.index(kind)] = values
    return layer2, rhythm


def _blend(
    clip: Mapping[str, Any],
    *,
    weight: float,
    direct_source: str,
    selector: Mapping[str, Any] | None,
) -> HeadPrediction:
    learned = clip["learned_prediction"]
    if direct_source == "selector":
        if selector is None:
            raise ValueError("Selector policy lacks frozen selector")
        direct_layer2, direct_rhythm = _selector_outputs(clip, selector)
    else:
        direct_layer2 = np.asarray(
            clip["direct_probabilities"], dtype=np.float32
        )
        direct_rhythm = np.asarray(
            clip["direct_rhythm_probabilities"], dtype=np.float32
        )
    prediction = blend_direct_learned_prediction(
        learned, direct_layer2, direct_weight=weight
    )
    learned_rhythm = np.asarray(
        learned.rhythm_probabilities, dtype=np.float32
    )
    rhythm = weight * direct_rhythm + (1.0 - weight) * learned_rhythm
    return HeadPrediction(
        layer2=prediction.layer2,
        rhythm=tuple(bool(value >= 0.5) for value in rhythm),
        deviation_sec=prediction.deviation_sec,
        rhythm_subtype=prediction.rhythm_subtype,
        layer2_probabilities=prediction.layer2_probabilities,
        rhythm_probabilities=tuple(float(value) for value in rhythm),
    )


def _variant(
    clips: Sequence[Mapping[str, Any]],
    weight: float,
    source: str,
    selector: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    return [
        {
            **clip,
            "prediction": _blend(
                clip,
                weight=weight,
                direct_source=source,
                selector=selector,
            ),
        }
        for clip in clips
    ]


def _global_balanced(
    clips: Sequence[Mapping[str, Any]],
    config: SchemaDecodeConfig,
    false_budget_per_clip: float,
) -> tuple[SchemaDecodeConfig, dict[str, Any]]:
    choices = []
    for offset in (0.0, 0.10, 0.20, 0.30, 0.40):
        trial = replace(
            config,
            high_thresholds={
                kind: min(1.01, float(value) + offset)
                for kind, value in config.high_thresholds.items()
            },
        )
        metric = v4._metric(clips, trial)
        false_per_clip = (
            float(metric["predicted"] - metric["correct"]) / len(clips)
        )
        choices.append((metric, false_per_clip, offset, trial))
    eligible = [
        value for value in choices if value[1] <= false_budget_per_clip
    ]
    selected = max(
        eligible or choices,
        key=lambda value: (
            value[0]["f1"],
            value[0]["precision"],
            -value[1],
        ),
    )
    metric, false_per_clip, offset, trial = selected
    return trial, {
        "metric": metric,
        "false_labels_per_clip": false_per_clip,
        "global_threshold_offset": offset,
        "candidate_count": len(choices),
        "false_budget_per_clip": false_budget_per_clip,
    }


def fit(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    protocol_path = output / "protocol_manifest.json"
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    resource = integrated._resource_snapshot(args.resource_status.resolve())
    clips = _load_training_clips(args)
    membership, _sample_to_ordinal = _membership(
        args.split_manifest.resolve()
    )
    train_ids = [int(clip["ordinal"]) for clip in clips]
    validation_ids = sorted(
        ordinal
        for ordinal, row in membership.items()
        if row["split"] == "val"
    )
    calibration_groups = {
        row["group"]
        for row in membership.values()
        if row["split"] == "train"
        and int(hashlib.sha256(row["group"].encode()).hexdigest(), 16) % 5 == 0
    }
    calibration_ids = sorted(
        ordinal
        for ordinal in train_ids
        if membership[ordinal]["group"] in calibration_groups
    )
    fit_ids = sorted(set(train_ids) - set(calibration_ids))
    groups = {
        ordinal: str(row["group"]) for ordinal, row in membership.items()
    }
    assert_protocol_disjoint(
        fit_ids, calibration_ids, validation_ids, groups
    )
    by_id = {int(clip["ordinal"]): clip for clip in clips}
    fit_clips = [by_id[value] for value in fit_ids]
    calibration_clips = [by_id[value] for value in calibration_ids]
    selector = _fit_selector(fit_clips)
    candidates: dict[str, dict[str, Any]] = {}
    specifications = (
        ("learned_only", 0.0, "deterministic"),
        ("direct_only", 1.0, "deterministic"),
        ("hybrid_050", 0.50, "selector"),
        ("hybrid_075", 0.75, "selector"),
        ("hybrid_100", 1.00, "selector"),
    )
    for name, weight, source in specifications:
        selected_clips = _variant(
            calibration_clips,
            weight,
            source,
            selector if source == "selector" else None,
        )
        config, trace = v4._calibrate_variant(selected_clips)
        candidates[name] = {
            "weight": weight,
            "direct_source": source,
            "config": config,
            "metric": trace["final"],
            "trace": trace,
        }
        print(
            f"{name} train_cal_f1={trace['final']['f1']:.6f}",
            flush=True,
        )
    hybrid_name, hybrid = max(
        (
            (name, value)
            for name, value in candidates.items()
            if name.startswith("hybrid_")
        ),
        key=lambda value: (
            value[1]["metric"]["f1"],
            value[1]["metric"]["precision"],
        ),
    )
    hybrid_clips = _variant(
        calibration_clips,
        hybrid["weight"],
        hybrid["direct_source"],
        selector,
    )
    balanced_config, balanced_trace = _global_balanced(
        hybrid_clips,
        hybrid["config"],
        false_budget_per_clip=3.42,
    )
    selector_path = output / "selector.json"
    _atomic_json(selector_path, selector)
    decode = {
        "schema_version": "align-error-heads-v5-decode-config-v1",
        "created_utc": _utc(),
        "selection_data": "train_calibration_partition_only",
        "predeclared_candidates": [
            {
                "name": name,
                "direct_weight": weight,
                "direct_source": source,
            }
            for name, weight, source in specifications
        ],
        "multiplicity": {
            "candidate_count": len(specifications),
            "validation_selection_performed": False,
        },
        "learned_only": {
            "direct_weight": 0.0,
            "direct_source": "deterministic",
            "schema_config": v3._config_json(
                candidates["learned_only"]["config"]
            ),
        },
        "direct_only": {
            "direct_weight": 1.0,
            "direct_source": "deterministic",
            "schema_config": v3._config_json(
                candidates["direct_only"]["config"]
            ),
        },
        "hybrid_max_f1": {
            "name": hybrid_name,
            "direct_weight": hybrid["weight"],
            "direct_source": hybrid["direct_source"],
            "schema_config": v3._config_json(hybrid["config"]),
        },
        "hybrid_balanced": {
            "name": hybrid_name,
            "direct_weight": hybrid["weight"],
            "direct_source": hybrid["direct_source"],
            "schema_config": v3._config_json(balanced_config),
            "constraint": balanced_trace,
        },
    }
    decode_path = output / "decode_config.json"
    _atomic_json(decode_path, decode)
    calibration = {
        "schema_version": "align-error-heads-v5-train-calibration-v1",
        "fit_rows": len(fit_ids),
        "calibration_rows": len(calibration_ids),
        "validation_rows_opened": 0,
        "candidates": {
            name: {
                "direct_weight": value["weight"],
                "direct_source": value["direct_source"],
                "metric": value["metric"],
                "schema_config": v3._config_json(value["config"]),
            }
            for name, value in candidates.items()
        },
        "selected_hybrid": hybrid_name,
        "balanced": balanced_trace,
    }
    calibration_path = output / "calibration.json"
    _atomic_json(calibration_path, calibration)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "created_utc": _utc(),
        "split_method": (
            "deterministic group-disjoint 4/5 selector fit and 1/5 "
            "policy calibration; SHA-256(group) modulo 5"
        ),
        "fit_rows": len(fit_ids),
        "calibration_rows": len(calibration_ids),
        "validation_rows": len(validation_ids),
        "fit_ordinals": fit_ids,
        "calibration_ordinals": calibration_ids,
        "validation_ordinals": validation_ids,
        "fit_ordinals_sha256": _id_hash(fit_ids),
        "calibration_ordinals_sha256": _id_hash(calibration_ids),
        "validation_ordinals_sha256": _id_hash(validation_ids),
        "fit_groups": len({groups[value] for value in fit_ids}),
        "calibration_groups": len(
            {groups[value] for value in calibration_ids}
        ),
        "validation_groups": len(
            {groups[value] for value in validation_ids}
        ),
        "group_overlap": False,
        "validation_targets_opened": False,
        "selector_sha256": integrated.sha256_file(selector_path),
        "decode_config_sha256": integrated.sha256_file(decode_path),
        "calibration_sha256": integrated.sha256_file(calibration_path),
        "examples_cache_sha256": integrated.sha256_file(
            args.examples_cache.resolve()
        ),
        "split_manifest_sha256": integrated.sha256_file(
            args.split_manifest.resolve()
        ),
        "error_heads_checkpoint_sha256": integrated.sha256_file(
            args.error_heads_checkpoint.resolve()
        ),
        "resource_coordination": resource,
        "gpu_used": False,
        "lockbox_touched": False,
        "production_mutated": False,
    }
    _atomic_json(protocol_path, protocol)
    print(protocol_path)


def _verify_protocol(
    args: argparse.Namespace, *, inference_safe: bool = False
) -> dict[str, Any]:
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported v5 protocol")
    for path, key in (
        (args.selector.resolve(), "selector_sha256"),
        (args.decode_config.resolve(), "decode_config_sha256"),
        (args.calibration.resolve(), "calibration_sha256"),
    ):
        if integrated.sha256_file(path) != protocol[key]:
            raise ValueError(f"Frozen protocol hash mismatch: {path}")
    if inference_safe:
        fit = set(int(value) for value in protocol["fit_ordinals"])
        calibration = set(
            int(value) for value in protocol["calibration_ordinals"]
        )
        validation = set(
            int(value) for value in protocol["validation_ordinals"]
        )
        if (
            fit & calibration
            or fit & validation
            or calibration & validation
            or protocol.get("group_overlap") is not False
        ):
            raise ValueError("Frozen v5 protocol declares leakage")
    else:
        membership, _sample_to_ordinal = _membership(
            args.split_manifest.resolve()
        )
        groups = {
            ordinal: str(row["group"]) for ordinal, row in membership.items()
        }
        assert_protocol_disjoint(
            protocol["fit_ordinals"],
            protocol["calibration_ordinals"],
            protocol["validation_ordinals"],
            groups,
        )
    return protocol


def _validation_clips(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _verify_protocol(args, inference_safe=True)
    _feature_manifest, artifacts = v4._verify_freeze(
        args.v4_feature_manifest.resolve(), v4.FEATURE_FREEZE_SCHEMA
    )
    actual = sorted(int(value["ordinal"]) for value in artifacts)
    if actual != sorted(int(value) for value in protocol["validation_ordinals"]):
        raise ValueError("Validation feature IDs differ from frozen protocol")
    clips = []
    for artifact in artifacts:
        rows = tuple(v4._full_row(value) for value in artifact["rows"])
        learned = integrated._prediction_from_json(
            artifact["learned_prediction"]
        )
        clips.append(
            {
                **artifact,
                "rows": rows,
                "score": tuple(
                    integrated._score_from_json(value)
                    for value in artifact["score"]
                ),
                "prediction": learned,
                "learned_prediction": learned,
                "direct_probabilities": np.asarray(
                    artifact["direct_probabilities"], dtype=np.float32
                ),
                "direct_rhythm_probabilities": direct_rhythm_probabilities(
                    rows
                ),
            }
        )
    return protocol, clips


def freeze(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    marker = output / "freeze_manifest.json"
    if marker.exists():
        raise FileExistsError(marker)
    opened = integrated._install_gold_guard()
    protocol, clips = _validation_clips(args)
    selector = json.loads(args.selector.read_text(encoding="utf-8"))
    decode = json.loads(args.decode_config.read_text(encoding="utf-8"))
    entries = []
    for position, clip in enumerate(clips, 1):
        documents = {}
        for name in (
            "learned_only",
            "direct_only",
            "hybrid_max_f1",
            "hybrid_balanced",
        ):
            policy = decode[name]
            prediction = _blend(
                clip,
                weight=float(policy["direct_weight"]),
                direct_source=str(policy["direct_source"]),
                selector=(
                    selector
                    if policy["direct_source"] == "selector"
                    else None
                ),
            )
            documents[name] = v3.schema12_document_v3(
                clip["sample"],
                clip["rows"],
                prediction,
                clip["score"],
                v3._config_from_json(policy["schema_config"]),
            )
        value = {
            "schema_version": "align-error-heads-v5-clip-v1",
            "ordinal": int(clip["ordinal"]),
            "sample": clip["sample"],
            "source": clip["source"],
            "audio_duration_sec": clip["audio_duration_sec"],
            "documents": documents,
        }
        path = output / "frozen" / f"{int(clip['ordinal']):05d}.json"
        _atomic_json(path, value)
        entries.append(
            {
                "ordinal": int(clip["ordinal"]),
                "sample": clip["sample"],
                "path": str(path.resolve()),
                "sha256": integrated.sha256_file(path),
            }
        )
        if position == 1 or position % 50 == 0 or position == len(clips):
            print(f"v5_freeze={position}/358", flush=True)
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": _utc(),
        "process": {
            "phase": "A_validation_inference",
            "pid": os.getpid(),
            "command": [sys.executable, *sys.argv],
        },
        "protocol_manifest_sha256": integrated.sha256_file(
            args.protocol_manifest.resolve()
        ),
        "selector_sha256": integrated.sha256_file(args.selector.resolve()),
        "decode_config_sha256": integrated.sha256_file(
            args.decode_config.resolve()
        ),
        "validation_rows": len(entries),
        "validation_ordinals_sha256": _id_hash(
            [int(value["ordinal"]) for value in entries]
        ),
        "gold_isolation": {
            "gold_opened": False,
            "forbidden_paths_opened": [],
            "opened_path_count": len(opened),
        },
        "artifacts": entries,
        "artifacts_manifest_sha256": v4._artifact_manifest(entries),
        "lockbox_touched": False,
        "production_mutated": False,
    }
    if manifest["validation_ordinals_sha256"] != protocol[
        "validation_ordinals_sha256"
    ]:
        raise ValueError("Frozen validation IDs violate protocol")
    _atomic_json(marker, manifest)
    print(marker)


def _verify_freeze(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _verify_protocol(args)
    manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("Unsupported v5 freeze")
    if manifest["protocol_manifest_sha256"] != integrated.sha256_file(
        args.protocol_manifest.resolve()
    ):
        raise ValueError("V5 freeze uses another protocol")
    artifacts = []
    for row in manifest["artifacts"]:
        path = Path(row["path"])
        if integrated.sha256_file(path) != row["sha256"]:
            raise ValueError(f"V5 artifact hash mismatch: {path}")
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    if len(artifacts) != 358:
        raise ValueError("V5 freeze must contain 358 rows")
    if _id_hash([int(value["ordinal"]) for value in artifacts]) != protocol[
        "validation_ordinals_sha256"
    ]:
        raise ValueError("V5 scored IDs differ from protocol")
    return manifest, artifacts


def score(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    opened_marker = output / "VALIDATION_OPENED.json"
    if opened_marker.exists():
        raise RuntimeError(
            "V5 validation targets were already opened; one-shot scoring refused"
        )
    manifest, artifacts = _verify_freeze(args)
    if os.getpid() == int(manifest["process"]["pid"]):
        raise ValueError("V5 scoring must run in a separate process")
    _atomic_json(
        opened_marker,
        {
            "schema_version": "align-error-heads-v5-validation-open-v1",
            "opened_utc": _utc(),
            "process_b_pid": os.getpid(),
            "freeze_manifest_sha256": integrated.sha256_file(
                args.freeze_manifest.resolve()
            ),
            "purpose": "single predeclared independent validation score",
        },
    )
    audit = sqlite3.connect(
        f"file:{args.canonical_targets.resolve().as_posix()}?mode=ro", uri=True
    )
    clips = []
    try:
        for artifact in artifacts:
            row = audit.execute(
                "SELECT payload FROM targets WHERE ordinal=? AND split='val'",
                (artifact["ordinal"],),
            ).fetchone()
            if row is None:
                raise ValueError(f"Missing validation target {artifact['ordinal']}")
            target = json.loads(zlib.decompress(row[0]))
            clips.append(
                {
                    **artifact,
                    "valid_labels": [
                        dict(label)
                        for label in target.get("valid_labels") or ()
                        if label.get("type") != "intonation_error"
                    ],
                    **artifact["documents"],
                }
            )
    finally:
        audit.close()
    reports = {}
    minutes = sum(float(clip["audio_duration_sec"]) for clip in clips) / 60.0
    for index, name in enumerate(
        (
            "learned_only",
            "direct_only",
            "hybrid_max_f1",
            "hybrid_balanced",
        )
    ):
        value = integrated._schema_report(clips, name)
        value.pop("_clip_counts", None)
        value["combined_including_repetition"] = integrated._metric_with_ci(
            [
                integrated._schema_counts(
                    clip["valid_labels"],
                    clip[name]["labels"],
                    types=ALL_TYPES,
                )
                for clip in clips
            ],
            20260980 + index,
        )
        requested = value["requested_four_types"]
        false = float(requested["predicted"] - requested["correct"])
        value["false_labels"] = {
            "effective": false,
            "per_clip": false / len(clips),
            "per_minute": false / minutes,
        }
        reports[name] = value
    v4_report = json.loads(args.v4_report.read_text(encoding="utf-8"))
    learned = reports["learned_only"]["requested_four_types"]
    hybrid = reports["hybrid_max_f1"]["requested_four_types"]
    balanced = reports["hybrid_balanced"]["requested_four_types"]
    gate = {
        "material_delta_required_vs_frozen_learned": 0.02,
        "minimum_precision": 0.15,
        "maximum_false_labels_per_clip": 3.42,
        "delta_vs_frozen_learned": hybrid["f1"] - learned["f1"],
        "passed": bool(
            hybrid["f1"] >= learned["f1"] + 0.02
            and hybrid["precision"] >= 0.15
            and reports["hybrid_balanced"]["false_labels"]["per_clip"] <= 3.42
            and balanced["f1"] >= learned["f1"]
        ),
    }
    report = {
        "schema_version": "align-error-heads-v5-report-v1",
        "created_utc": _utc(),
        "data": {
            "fit_rows": json.loads(
                args.protocol_manifest.read_text(encoding="utf-8")
            )["fit_rows"],
            "calibration_rows": json.loads(
                args.protocol_manifest.read_text(encoding="utf-8")
            )["calibration_rows"],
            "validation_rows": 358,
            "lockbox_metadata_rows": 4022,
            "lockbox_touched": False,
        },
        "protocol": {
            "train_only_selection": True,
            "validation_opened_once": True,
            "process_a_pid": manifest["process"]["pid"],
            "process_b_pid": os.getpid(),
            "separate_processes": True,
            "freeze_verified_before_gold_open": True,
            "predeclared_candidates": json.loads(
                args.decode_config.read_text(encoding="utf-8")
            )["predeclared_candidates"],
        },
        "metrics": {"official_schema_1_2": reports},
        "comparison": {
            "v4_same_set_exploratory_not_independent": v4_report["metrics"][
                "official_schema_1_2"
            ]["hybrid_max_f1"]["requested_four_types"],
            "v5_minus_v4_exploratory_f1": (
                hybrid["f1"]
                - v4_report["metrics"]["official_schema_1_2"][
                    "hybrid_max_f1"
                ]["requested_four_types"]["f1"]
            ),
        },
        "promotion_gate": gate,
        "promotion_performed": False,
        "production_mutated": False,
    }
    report_path = output / "report.json"
    _atomic_json(report_path, report)
    _atomic_json(
        output / "integrity.json",
        {
            "schema_version": "align-error-heads-v5-integrity-v1",
            "protocol_manifest_sha256": integrated.sha256_file(
                args.protocol_manifest.resolve()
            ),
            "freeze_manifest_sha256": integrated.sha256_file(
                args.freeze_manifest.resolve()
            ),
            "validation_open_marker_sha256": integrated.sha256_file(
                opened_marker
            ),
            "report_sha256": integrated.sha256_file(report_path),
            "validation_open_count": 1,
            "lockbox_touched": False,
            "production_mutated": False,
        },
    )
    print(report_path)


def verify(args: argparse.Namespace) -> None:
    manifest, artifacts = _verify_freeze(args)
    report_path = args.output_dir.resolve() / "report.json"
    integrity_path = args.output_dir.resolve() / "integrity.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    errors = []
    if len(artifacts) != 358 or report["data"]["validation_rows"] != 358:
        errors.append("validation count mismatch")
    if integrity["validation_open_count"] != 1:
        errors.append("validation was not opened exactly once")
    if report["data"]["lockbox_touched"] or report["production_mutated"]:
        errors.append("isolation failed")
    if integrity["report_sha256"] != integrated.sha256_file(report_path):
        errors.append("report hash mismatch")
    result = {
        "schema_version": "align-error-heads-v5-verification-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "fit_rows": report["data"]["fit_rows"],
        "calibration_rows": report["data"]["calibration_rows"],
        "validation_rows": len(artifacts),
        "validation_open_count": integrity["validation_open_count"],
        "lockbox_touched": False,
        "production_mutated": False,
    }
    _atomic_json(args.output_dir.resolve() / "verification.json", result)
    if errors:
        raise ValueError(errors)
    print("error-heads-v5 verification passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/error-heads-v5"),
    )
    parser.add_argument(
        "--examples-cache",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/examples.sqlite"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data-audit/joint-outputraw-full-v1/split.json"),
    )
    parser.add_argument(
        "--error-heads-checkpoint",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/predicted/best.pt"
        ),
    )
    parser.add_argument(
        "--v4-feature-manifest",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v4/"
            "feature_freeze_manifest.json"
        ),
    )
    parser.add_argument(
        "--v4-report",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v4/report.json"
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
        "--resource-status",
        type=Path,
        default=Path("runs/TRAINING_RESOURCE_STATUS.json"),
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v5/"
            "protocol_manifest.json"
        ),
    )
    parser.add_argument(
        "--selector",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v5/selector.json"
        ),
    )
    parser.add_argument(
        "--decode-config",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v5/decode_config.json"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v5/calibration.json"
        ),
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v5/freeze_manifest.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("fit")
    subparsers.add_parser("freeze")
    subparsers.add_parser("score")
    subparsers.add_parser("verify")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "fit":
        fit(args)
    elif args.phase == "freeze":
        freeze(args)
    elif args.phase == "score":
        score(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
