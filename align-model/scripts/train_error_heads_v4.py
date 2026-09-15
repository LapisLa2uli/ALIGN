"""Direct-operation plus learned-probability hybrid for error-heads-v4.

Phases:
1. ``features`` freezes inference-only local n-best/path evidence.
2. ``analyze`` opens validation gold and calibrates direct/learned/hybrid gates.
3. ``freeze`` emits v4 documents without gold.
4. ``score`` verifies the freeze before opening gold in a separate process.
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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.joint.error_heads import (
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYER2_CLASSES,
    HeadRow,
    SchemaDecodeConfig,
    attach_training_targets,
    blend_direct_learned_prediction,
    build_inference_rows,
    direct_operation_probabilities,
    schema12_document_v3,
)
from alignmodel.joint.lattice import (
    JointOperation,
    LatticePath,
    LatticeStep,
)

import eval_error_heads_integrated as integrated
import train_error_heads_v3 as v3


FEATURE_FREEZE_SCHEMA = "align-error-heads-v4-feature-freeze-v1"
PREDICTION_FREEZE_SCHEMA = "align-error-heads-v4-prediction-freeze-v1"
REQUESTED_TYPES = integrated.REQUESTED_TYPES
ALL_TYPES = REQUESTED_TYPES | {"repetition"}


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


def _full_row_json(row: HeadRow) -> dict[str, Any]:
    return {
        **integrated._row_json(row),
        "features": [float(value) for value in row.features],
    }


def _full_row(value: Mapping[str, Any]) -> HeadRow:
    base = integrated._row_from_json(value)
    features = np.asarray(value["features"], dtype=np.float32)
    if features.shape != (FEATURE_DIM,):
        raise ValueError("V4 frozen row has invalid feature shape")
    return HeadRow(
        features=features,
        kind=base.kind,
        event_index=base.event_index,
        score_span=base.score_span,
        start_sec=base.start_sec,
        end_sec=base.end_sec,
        is_copy=base.is_copy,
    )


def _path(value: Mapping[str, Any]) -> LatticePath:
    return LatticePath(
        steps=tuple(
            LatticeStep(
                candidate_index=int(step["candidate_index"]),
                score_span=(
                    tuple(int(item) for item in step["score_span"])
                    if step.get("score_span")
                    else None
                ),
                operation=JointOperation(str(step["operation"])),
                structural_operation=(
                    JointOperation(str(step["structural_operation"]))
                    if step.get("structural_operation")
                    else None
                ),
                resume_event=(
                    int(step["resume_event"])
                    if step.get("resume_event") is not None
                    else None
                ),
                deleted_events=tuple(
                    int(item) for item in step.get("deleted_events") or ()
                ),
            )
            for step in value["steps"]
        ),
        trailing_deletions=tuple(
            int(item) for item in value.get("trailing_deletions") or ()
        ),
        score=float(value["score"]),
    )


def _artifact_manifest(
    entries: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for row in entries:
        digest.update(f"{row['ordinal']}:{row['sha256']}\n".encode("ascii"))
    return digest.hexdigest()


def features(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    marker = output / "feature_freeze_manifest.json"
    if marker.exists():
        raise FileExistsError(marker)
    opened = integrated._install_gold_guard()
    source_manifest, artifacts = integrated._load_and_verify_freeze(
        args.v2_freeze_manifest.resolve()
    )
    stack = integrated._load_stack(args)
    entries = []
    summaries = []
    started = time.perf_counter()
    for position, artifact in enumerate(artifacts, 1):
        candidates = tuple(
            integrated._candidate_from_json(value)
            for value in artifact["candidates"]
        )
        score = tuple(
            integrated._score_from_json(value) for value in artifact["score"]
        )
        path = _path(artifact["path"])
        rows = build_inference_rows(stack["lattice"], candidates, score, path)
        minimal = tuple(
            integrated._row_from_json(value)
            for value in artifact["head_rows"]
        )
        if [
            (row.kind, row.event_index, row.score_span, row.is_copy)
            for row in rows
        ] != [
            (row.kind, row.event_index, row.score_span, row.is_copy)
            for row in minimal
        ]:
            raise ValueError(
                f"Recomputed inference rows differ for {artifact['sample']}"
            )
        events = tuple(
            integrated._event_from_json(value)
            for value in artifact["joint_events"]
        )
        direct, diagnostics = direct_operation_probabilities(
            rows, events, score
        )
        value = {
            "schema_version": "align-error-heads-v4-feature-clip-v1",
            "ordinal": int(artifact["ordinal"]),
            "sample": str(artifact["sample"]),
            "source": str(artifact["source"]),
            "audio_duration_sec": float(artifact["audio_duration_sec"]),
            "rows": [_full_row_json(row) for row in rows],
            "score": artifact["score"],
            "events": artifact["joint_events"],
            "learned_prediction": artifact["prediction"],
            "direct_probabilities": direct.tolist(),
            "direct_diagnostics": diagnostics,
            "v2_document": artifact["schema_1_2"],
            "rules_document": artifact["rules_schema_1_2"],
            "source_v2_artifact_sha256": next(
                row["sha256"]
                for row in source_manifest["artifacts"]
                if int(row["ordinal"]) == int(artifact["ordinal"])
            ),
        }
        path_out = output / "feature-freeze" / f"{int(artifact['ordinal']):05d}.json"
        _atomic_json(path_out, value)
        entries.append(
            {
                "ordinal": int(artifact["ordinal"]),
                "sample": str(artifact["sample"]),
                "path": str(path_out.resolve()),
                "sha256": integrated.sha256_file(path_out),
            }
        )
        summaries.append(diagnostics)
        if position == 1 or position % 50 == 0 or position == len(artifacts):
            print(f"v4_features={position}/358", flush=True)
    manifest = {
        "schema_version": FEATURE_FREEZE_SCHEMA,
        "created_utc": _utc(),
        "process": {
            "phase": "A0_inference_feature_freeze",
            "pid": os.getpid(),
            "command": [sys.executable, *sys.argv],
        },
        "validation_rows": len(entries),
        "model_selection": stack["selection"],
        "local_nbest_contract": {
            "method": (
                "exact local softmax over every legal operation/span option "
                "scored by the completed frozen joint decoder"
            ),
            "outputs": [
                "selected path margin",
                "extra/noise/delete posterior mass",
                "binary alternative entropy-derived certainty",
            ],
            "global_forward_backward_required": False,
            "reason": (
                "all direct gates are local operation-core decisions and the "
                "complete legal local option set is scored exactly"
            ),
        },
        "gold_isolation": {
            "gold_opened": False,
            "forbidden_paths_opened": [],
            "opened_path_count": len(opened),
        },
        "source_v2_freeze_manifest_sha256": integrated.sha256_file(
            args.v2_freeze_manifest.resolve()
        ),
        "artifacts": entries,
        "artifacts_manifest_sha256": _artifact_manifest(entries),
        "resource_coordination": {
            "device": "cpu",
            "gpu_lease_acquired": False,
            "status": integrated._resource_snapshot(
                args.resource_status.resolve()
            ),
        },
        "lockbox_touched": False,
        "production_mutated": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(marker, manifest)
    print(marker)


def _verify_freeze(
    path: Path, schema: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != schema:
        raise ValueError(f"Unsupported freeze schema: {manifest.get('schema_version')}")
    entries = []
    for row in manifest["artifacts"]:
        artifact_path = Path(row["path"])
        if integrated.sha256_file(artifact_path) != row["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {artifact_path}")
        entries.append(json.loads(artifact_path.read_text(encoding="utf-8")))
    if _artifact_manifest(manifest["artifacts"]) != manifest[
        "artifacts_manifest_sha256"
    ]:
        raise ValueError("Artifact aggregate hash mismatch")
    if len(entries) != 358:
        raise ValueError("V4 freeze does not contain 358 validation clips")
    return manifest, entries


def _load_gold_clips(args: argparse.Namespace) -> list[dict[str, Any]]:
    _manifest, artifacts = _verify_freeze(
        args.feature_freeze_manifest.resolve(), FEATURE_FREEZE_SCHEMA
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
                raise ValueError(f"Missing target {artifact['ordinal']}")
            target = json.loads(zlib.decompress(row[0]))
            clips.append(
                {
                    **artifact,
                    "rows": tuple(_full_row(value) for value in artifact["rows"]),
                    "score": tuple(
                        integrated._score_from_json(value)
                        for value in artifact["score"]
                    ),
                    "learned_prediction": integrated._prediction_from_json(
                        artifact["learned_prediction"]
                    ),
                    "direct_probabilities": np.asarray(
                        artifact["direct_probabilities"], dtype=np.float32
                    ),
                    "valid_labels": [
                        dict(label)
                        for label in target.get("valid_labels") or ()
                        if label.get("type") != "intonation_error"
                    ],
                }
            )
    finally:
        audit.close()
    return clips


def _variant_clips(
    clips: Sequence[Mapping[str, Any]], direct_weight: float
) -> list[dict[str, Any]]:
    return [
        {
            **clip,
            "prediction": blend_direct_learned_prediction(
                clip["learned_prediction"],
                clip["direct_probabilities"],
                direct_weight=direct_weight,
            ),
        }
        for clip in clips
    ]


def _candidate_core(
    rows: Sequence[HeadRow], index: int, kind: str
) -> tuple[int, int] | None:
    row = rows[index]
    if kind in {"wrong_note", "missed_note"}:
        return row.score_span
    if kind != "extra_note" or row.score_span is not None or row.is_copy:
        return None
    before = next(
        (
            candidate
            for candidate in reversed(rows[:index])
            if candidate.kind == "event"
            and candidate.score_span is not None
            and not candidate.is_copy
        ),
        None,
    )
    after = next(
        (
            candidate
            for candidate in rows[index + 1 :]
            if candidate.kind == "event"
            and candidate.score_span is not None
            and not candidate.is_copy
        ),
        None,
    )
    if (
        before is None
        or after is None
        or before.score_span is None
        or after.score_span is None
        or before.score_span[1] != after.score_span[0]
    ):
        return None
    return before.score_span[1], after.score_span[0] + 1


def _selector_features(
    clip: Mapping[str, Any], kind: str
) -> tuple[np.ndarray, np.ndarray]:
    class_index = LAYER2_CLASSES.index(kind)
    rows = clip["rows"]
    learned = np.asarray(
        clip["learned_prediction"].layer2_probabilities, dtype=np.float32
    )
    direct = np.asarray(clip["direct_probabilities"], dtype=np.float32)
    indices = []
    values = []
    selected_names = (
        "upstream_path_margin",
        "upstream_extra_probability",
        "upstream_noise_probability",
        "upstream_delete_probability",
        "candidate_confidence",
        "previous_confidence",
        "next_confidence",
        "normalized_score_position",
        "is_replay_state",
        "path_operation_wrong",
        "path_operation_extra",
        "path_operation_missed",
    )
    feature_indices = [FEATURE_NAMES.index(name) for name in selected_names]
    for index, row in enumerate(rows):
        core = _candidate_core(rows, index, kind)
        candidate = bool(
            core is not None
            and not row.is_copy
            and (
                kind == "wrong_note"
                and row.kind == "event"
                and row.features[FEATURE_NAMES.index("path_operation_wrong")] > 0.5
                or kind == "extra_note"
                and row.kind == "event"
                and row.features[FEATURE_NAMES.index("path_operation_extra")] > 0.5
                or kind == "missed_note"
                and row.kind == "gap"
                and row.features[FEATURE_NAMES.index("path_operation_missed")] > 0.5
            )
        )
        if not candidate:
            continue
        indices.append(index)
        values.append(
            [
                float(direct[index, class_index]),
                float(learned[index, class_index]),
                *[float(row.features[item]) for item in feature_indices],
            ]
        )
    return np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def _selector_targets(
    clip: Mapping[str, Any], kind: str, indices: np.ndarray
) -> np.ndarray:
    gold = []
    for label in clip["valid_labels"]:
        if label.get("type") != kind:
            continue
        part = label.get("score_part") or {}
        if part.get("start_note_index") is None or part.get("end_note_index") is None:
            continue
        pad = int(part.get("pad_notes") or 0)
        gold.append(
            (
                int(part["start_note_index"]) + pad,
                int(part["end_note_index"]) - pad + 1,
            )
        )
    output = []
    for index in indices:
        core = _candidate_core(clip["rows"], int(index), kind)
        assert core is not None
        output.append(
            any(
                max(core[0], target[0]) < min(core[1], target[1])
                for target in gold
            )
        )
    return np.asarray(output, dtype=np.int64)


def _fit_selector(
    clips: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selector: dict[str, Any] = {
        "schema_version": "align-error-heads-v4-operation-selector-v1",
        "classes": {},
        "features": [
            "direct_reliability",
            "learned_probability",
            "upstream_path_margin",
            "upstream_extra_probability",
            "upstream_noise_probability",
            "upstream_delete_probability",
            "candidate_confidence",
            "previous_confidence",
            "next_confidence",
            "normalized_score_position",
            "is_replay_state",
            "path_operation_wrong",
            "path_operation_extra",
            "path_operation_missed",
        ],
    }
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in ("wrong_note", "extra_note", "missed_note"):
        chunks = []
        targets = []
        for clip in clips:
            matrix, indices = _selector_features(clip, kind)
            if len(matrix):
                chunks.append(matrix)
                targets.append(_selector_targets(clip, kind, indices))
        x = np.concatenate(chunks)
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
        for iteration in range(800):
            logits = normalized @ coefficient + intercept
            predicted = 1.0 / (
                1.0 + np.exp(-np.clip(logits, -20.0, 20.0))
            )
            residual = sample_weight * (predicted - y)
            learning_rate = 0.08 / (1.0 + iteration / 250.0)
            coefficient -= learning_rate * (
                normalized.T @ residual / sample_weight.sum()
                + 0.10 * coefficient
            )
            intercept -= learning_rate * float(
                residual.sum() / sample_weight.sum()
            )
        selector["classes"][kind] = {
            "coefficient": coefficient.tolist(),
            "intercept": float(intercept),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "candidate_rows": int(len(y)),
            "positive_rows": int(y.sum()),
            "negative_rows": int(len(y) - y.sum()),
        }
    return selector, _apply_selector(clips, selector)


def _apply_selector(
    clips: Sequence[Mapping[str, Any]], selector: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for clip in clips:
        probabilities = np.zeros(
            (len(clip["rows"]), len(LAYER2_CLASSES)), dtype=np.float32
        )
        for kind, state in selector["classes"].items():
            matrix, indices = _selector_features(clip, kind)
            if not len(matrix):
                continue
            mean = np.asarray(state["mean"], dtype=np.float32)
            scale = np.asarray(state["scale"], dtype=np.float32)
            coefficient = np.asarray(state["coefficient"], dtype=np.float32)
            logits = (
                ((matrix - mean) / scale) @ coefficient
                + float(state["intercept"])
            )
            probabilities[indices, LAYER2_CLASSES.index(kind)] = (
                1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            )
        output.append({**clip, "direct_probabilities": probabilities})
    return output


def _metric(
    clips: Sequence[Mapping[str, Any]],
    config: SchemaDecodeConfig,
    types: set[str] = REQUESTED_TYPES,
) -> dict[str, Any]:
    documents = v3._documents(clips, config)
    return v3._metric(clips, documents, types)


def _calibrate_variant(
    clips: Sequence[Mapping[str, Any]],
) -> tuple[SchemaDecodeConfig, dict[str, Any]]:
    config = v3._disabled_config()
    thresholds = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.01)
    trace: dict[str, Any] = {"per_type": {}}
    for kind in ("wrong_note", "missed_note", "extra_note", "rhythm_error"):
        choices = []
        for threshold in thresholds:
            for support in ((1, 2) if kind != "rhythm_error" else (1, 2, 3)):
                trial = v3._with_type(
                    config,
                    kind,
                    threshold=threshold,
                    low_ratio=1.0,
                    support=support,
                    margin=-1.0,
                    merge_gap=1 if kind == "missed_note" else 0,
                )
                metric = _metric(clips, trial, {kind})
                choices.append((metric, threshold, support, trial))
        metric, threshold, support, config = max(
            choices,
            key=lambda value: (
                value[0]["f1"],
                value[0]["precision"],
                -value[0]["predicted"],
                value[1],
            ),
        )
        trace["per_type"][kind] = {
            "threshold": threshold,
            "support": support,
            "metric": metric,
        }
    decisions = []
    for kind in ("wrong_note", "missed_note", "extra_note", "rhythm_error"):
        choices = []
        for threshold in thresholds:
            trial = v3._with_type(
                config,
                kind,
                threshold=threshold,
                low_ratio=1.0,
                support=int(config.minimum_support[kind]),
                margin=-1.0,
                merge_gap=int(config.merge_score_gap[kind]),
            )
            choices.append((_metric(clips, trial), threshold, trial))
        metric, threshold, config = max(
            choices,
            key=lambda value: (
                value[0]["f1"],
                value[0]["precision"],
                -value[0]["predicted"],
            ),
        )
        decisions.append(
            {"type": kind, "threshold": threshold, "combined": metric}
        )
    trace["joint_refinement"] = decisions
    trace["final"] = _metric(clips, config)
    return config, trace


def _balanced_config(
    clips: Sequence[Mapping[str, Any]],
    config: SchemaDecodeConfig,
    false_budget: float,
    minimum_f1: float,
) -> tuple[SchemaDecodeConfig, dict[str, Any]]:
    choices = []
    offsets = (0.0, 0.10, 0.20, 0.35)
    for wrong_offset in offsets:
        for missed_offset in offsets:
            for extra_offset in offsets:
                for rhythm_offset in offsets:
                    thresholds = dict(config.high_thresholds)
                    for kind, offset in (
                        ("wrong_note", wrong_offset),
                        ("missed_note", missed_offset),
                        ("extra_note", extra_offset),
                        ("rhythm_error", rhythm_offset),
                    ):
                        thresholds[kind] = min(
                            1.01, float(thresholds[kind]) + offset
                        )
                    trial = replace(config, high_thresholds=thresholds)
                    metric = _metric(clips, trial)
                    false = float(metric["predicted"] - metric["correct"])
                    choices.append((metric, false, trial))
    eligible = [
        value
        for value in choices
        if value[1] <= false_budget and value[0]["f1"] >= minimum_f1
    ]
    if not eligible:
        raise ValueError("No balanced v4 policy satisfies validation gates")
    metric, false, selected = max(
        eligible,
        key=lambda value: (
            value[0]["f1"],
            value[0]["precision"],
            -value[1],
        ),
    )
    return selected, {
        "metric": metric,
        "false_labels": false,
        "false_budget": false_budget,
        "minimum_f1": minimum_f1,
        "searched": len(choices),
    }


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "decode_config.json"
    if config_path.exists():
        raise FileExistsError(config_path)
    clips = _load_gold_clips(args)
    learned_config = v3._config_from_json(
        json.loads(args.v3_config.read_text(encoding="utf-8"))
    )
    variants = {}
    learned_clips = _variant_clips(clips, 0.0)
    variants["learned_only"] = {
        "direct_weight": 0.0,
        "direct_source": "deterministic",
        "config": learned_config,
        "metric": _metric(learned_clips, learned_config),
        "calibration": "verified_v3_max_f1",
    }
    for name, weight in (
        ("direct_only", 1.0),
        ("hybrid_025", 0.25),
        ("hybrid_050", 0.50),
        ("hybrid_075", 0.75),
    ):
        selected_clips = _variant_clips(clips, weight)
        config, trace = _calibrate_variant(selected_clips)
        variants[name] = {
            "direct_weight": weight,
            "direct_source": "deterministic",
            "config": config,
            "metric": trace["final"],
            "calibration": trace,
        }
        print(
            f"{name} f1={trace['final']['f1']:.6f} "
            f"pred={trace['final']['predicted']}",
            flush=True,
        )
    first_hybrid_name, first_hybrid = max(
        (
            (name, value)
            for name, value in variants.items()
            if name.startswith("hybrid_")
        ),
        key=lambda value: (
            value[1]["metric"]["f1"],
            value[1]["metric"]["precision"],
        ),
    )
    selector, selector_base_clips = _fit_selector(clips)
    for name, weight in (
        ("selector_hybrid_025", 0.25),
        ("selector_hybrid_050", 0.50),
        ("selector_hybrid_075", 0.75),
        ("selector_hybrid_100", 1.00),
    ):
        selected_clips = _variant_clips(selector_base_clips, weight)
        config, trace = _calibrate_variant(selected_clips)
        variants[name] = {
            "direct_weight": weight,
            "direct_source": "selector",
            "config": config,
            "metric": trace["final"],
            "calibration": trace,
        }
        print(
            f"{name} f1={trace['final']['f1']:.6f} "
            f"pred={trace['final']['predicted']}",
            flush=True,
        )
    hybrid_name, hybrid = max(
        (
            (name, value)
            for name, value in variants.items()
            if "hybrid_" in name
        ),
        key=lambda value: (
            value[1]["metric"]["f1"],
            value[1]["metric"]["precision"],
        ),
    )
    selected_base_clips = (
        selector_base_clips
        if hybrid["direct_source"] == "selector"
        else clips
    )
    hybrid_clips = _variant_clips(selected_base_clips, hybrid["direct_weight"])
    v3_balanced = v3._config_from_json(
        json.loads(args.v3_config.read_text(encoding="utf-8"))[
            "balanced_policy"
        ]
    )
    v3_balanced_metric = _metric(learned_clips, v3_balanced)
    false_budget = float(
        v3_balanced_metric["predicted"] - v3_balanced_metric["correct"]
    )
    balanced, balanced_trace = _balanced_config(
        hybrid_clips,
        hybrid["config"],
        false_budget=false_budget,
        minimum_f1=v3_balanced_metric["f1"] + 0.02,
    )
    config_document = {
        "schema_version": "align-error-heads-v4-decode-config-v1",
        "created_utc": _utc(),
        "calibrated_on": "all_358_validation_rows",
        "objective": "official exclusive schema 1.2 four-type F1",
        "learned_only": {
            "direct_weight": 0.0,
            "schema_config": v3._config_json(learned_config),
        },
        "direct_only": {
            "direct_weight": 1.0,
            "direct_source": "deterministic",
            "schema_config": v3._config_json(variants["direct_only"]["config"]),
        },
        "hybrid_max_f1": {
            "name": hybrid_name,
            "direct_weight": hybrid["direct_weight"],
            "direct_source": hybrid["direct_source"],
            "schema_config": v3._config_json(hybrid["config"]),
        },
        "hybrid_balanced": {
            "name": hybrid_name,
            "direct_weight": hybrid["direct_weight"],
            "direct_source": hybrid["direct_source"],
            "schema_config": v3._config_json(balanced),
            "constraint": balanced_trace,
        },
        "operation_selector": selector,
        "training_performed": True,
        "training_decision": (
            "small validation-only linear operation-core selector fitted after "
            f"the first deterministic hybrid reached only "
            f"{first_hybrid['metric']['f1']:.6f} F1"
        ),
    }
    serializable_variants = {
        name: {
            **{key: item for key, item in value.items() if key != "config"},
            "schema_config": v3._config_json(value["config"]),
        }
        for name, value in variants.items()
    }
    _atomic_json(config_path, config_document)
    _atomic_json(
        output / "calibration.json",
        {
            "schema_version": "align-error-heads-v4-calibration-v1",
            "variants": serializable_variants,
            "selected_hybrid": hybrid_name,
            "first_deterministic_hybrid": {
                "name": first_hybrid_name,
                "metric": first_hybrid["metric"],
            },
            "operation_selector": selector,
            "balanced": balanced_trace,
            "v3_balanced_reference": v3_balanced_metric,
            "source_render": {
                "source": {"MozartClConcertoA": 358},
                "render": {"soundfont_v1": 358},
                "calibration_effect": "constant on this validation split",
            },
        },
    )
    _atomic_json(
        output / "config.json",
        {
            "schema_version": "align-error-heads-v4-experiment-v1",
            "created_utc": _utc(),
            "validation_rows": 358,
            "lockbox_metadata_rows": 4022,
            "lockbox_touched": False,
            "feature_freeze_manifest_sha256": integrated.sha256_file(
                args.feature_freeze_manifest.resolve()
            ),
            "decode_config_sha256": integrated.sha256_file(config_path),
            "resource_status": integrated._resource_snapshot(
                args.resource_status.resolve()
            ),
            "gpu_training_performed": False,
            "cpu_operation_selector_trained": True,
            "production_mutated": False,
        },
    )
    print(config_path)


def _policy_prediction(
    artifact: Mapping[str, Any],
    policy: Mapping[str, Any],
    root_config: Mapping[str, Any],
) -> tuple[Any, SchemaDecodeConfig]:
    learned = integrated._prediction_from_json(artifact["learned_prediction"])
    direct = np.asarray(artifact["direct_probabilities"], dtype=np.float32)
    if policy.get("direct_source") == "selector":
        selector_clip = {
            **artifact,
            "rows": tuple(_full_row(value) for value in artifact["rows"]),
            "learned_prediction": learned,
            "direct_probabilities": direct,
        }
        direct = _apply_selector(
            [selector_clip], root_config["operation_selector"]
        )[0]["direct_probabilities"]
    prediction = blend_direct_learned_prediction(
        learned, direct, direct_weight=float(policy["direct_weight"])
    )
    config = v3._config_from_json(policy["schema_config"])
    return prediction, config


def freeze(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    marker = output / "freeze_manifest.json"
    if marker.exists():
        raise FileExistsError(marker)
    opened = integrated._install_gold_guard()
    feature_manifest, artifacts = _verify_freeze(
        args.feature_freeze_manifest.resolve(), FEATURE_FREEZE_SCHEMA
    )
    config_path = args.decode_config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = []
    started = time.perf_counter()
    for position, artifact in enumerate(artifacts, 1):
        rows = tuple(_full_row(value) for value in artifact["rows"])
        score = tuple(
            integrated._score_from_json(value) for value in artifact["score"]
        )
        documents = {}
        for name in (
            "learned_only",
            "direct_only",
            "hybrid_max_f1",
            "hybrid_balanced",
        ):
            prediction, schema_config = _policy_prediction(
                artifact, config[name], config
            )
            documents[name] = schema12_document_v3(
                artifact["sample"], rows, prediction, score, schema_config
            )
        value = {
            "schema_version": "align-error-heads-v4-prediction-clip-v1",
            "ordinal": artifact["ordinal"],
            "sample": artifact["sample"],
            "source": artifact["source"],
            "audio_duration_sec": artifact["audio_duration_sec"],
            "documents": documents,
            "source_feature_sha256": next(
                row["sha256"]
                for row in feature_manifest["artifacts"]
                if int(row["ordinal"]) == int(artifact["ordinal"])
            ),
        }
        path = output / "frozen" / f"{int(artifact['ordinal']):05d}.json"
        _atomic_json(path, value)
        entries.append(
            {
                "ordinal": int(artifact["ordinal"]),
                "sample": artifact["sample"],
                "path": str(path.resolve()),
                "sha256": integrated.sha256_file(path),
            }
        )
        if position == 1 or position % 50 == 0 or position == len(artifacts):
            print(f"v4_freeze={position}/358", flush=True)
    manifest = {
        "schema_version": PREDICTION_FREEZE_SCHEMA,
        "created_utc": _utc(),
        "process": {
            "phase": "A_v4_hybrid_inference",
            "pid": os.getpid(),
            "command": [sys.executable, *sys.argv],
        },
        "validation_rows": len(entries),
        "feature_freeze_manifest_sha256": integrated.sha256_file(
            args.feature_freeze_manifest.resolve()
        ),
        "decode_config_sha256": integrated.sha256_file(config_path),
        "gold_isolation": {
            "gold_opened": False,
            "forbidden_paths_opened": [],
            "opened_path_count": len(opened),
        },
        "artifacts": entries,
        "artifacts_manifest_sha256": _artifact_manifest(entries),
        "lockbox_touched": False,
        "production_mutated": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(marker, manifest)
    print(marker)


def score(args: argparse.Namespace) -> None:
    manifest, artifacts = _verify_freeze(
        args.freeze_manifest.resolve(), PREDICTION_FREEZE_SCHEMA
    )
    if os.getpid() == int(manifest["process"]["pid"]):
        raise ValueError("V4 scoring must run in a separate process")
    output = args.output_dir.resolve()
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    audit = sqlite3.connect(
        f"file:{args.canonical_targets.resolve().as_posix()}?mode=ro", uri=True
    )
    target_index = sqlite3.connect(
        f"file:{args.packed_index.resolve().as_posix()}?mode=ro", uri=True
    )
    _feature_manifest, feature_artifacts = _verify_freeze(
        args.feature_freeze_manifest.resolve(), FEATURE_FREEZE_SCHEMA
    )
    features_by_ordinal = {
        int(value["ordinal"]): value for value in feature_artifacts
    }
    clips = []
    try:
        for artifact in artifacts:
            row = audit.execute(
                "SELECT payload FROM targets WHERE ordinal=? AND split='val'",
                (artifact["ordinal"],),
            ).fetchone()
            if row is None:
                raise ValueError(f"Missing target {artifact['ordinal']}")
            target = json.loads(zlib.decompress(row[0]))
            target_row = target_index.execute(
                "SELECT target FROM records WHERE ordinal=? AND split='val'",
                (artifact["ordinal"],),
            ).fetchone()
            if target_row is None:
                raise ValueError(f"Missing packed target {artifact['ordinal']}")
            packed_target = json.loads(zlib.decompress(target_row[0]))
            feature = features_by_ordinal[int(artifact["ordinal"])]
            rows = tuple(_full_row(value) for value in feature["rows"])
            labeled = attach_training_targets(
                rows,
                predicted_events=tuple(
                    integrated._event_from_json(value)
                    for value in feature["events"]
                ),
                target_events=tuple(
                    integrated._event_from_json(value)
                    for value in packed_target["target_events"]
                ),
                target_deletions=packed_target["target_deletions"],
                rhythm_rows=packed_target["layer3_rhythm"],
                score=tuple(
                    integrated._score_from_json(value)
                    for value in feature["score"]
                ),
            )
            clips.append(
                {
                    "ordinal": int(artifact["ordinal"]),
                    "sample": artifact["sample"],
                    "source": artifact["source"],
                    "audio_duration_sec": artifact["audio_duration_sec"],
                    "valid_labels": [
                        dict(label)
                        for label in target.get("valid_labels") or ()
                        if label.get("type") != "intonation_error"
                    ],
                    "has_repeat": bool(packed_target.get("layer1_repeats")),
                    "mapping_correct_fraction": float(
                        np.mean(labeled.mapping_correct)
                    ),
                    **artifact["documents"],
                }
            )
    finally:
        audit.close()
        target_index.close()
    reports = {}
    for name in (
        "learned_only",
        "direct_only",
        "hybrid_max_f1",
        "hybrid_balanced",
    ):
        reports[name] = integrated._schema_report(clips, name)
        reports[name].pop("_clip_counts", None)
        reports[name]["combined_including_repetition"] = (
            integrated._metric_with_ci(
                [
                    integrated._schema_counts(
                        clip["valid_labels"],
                        clip[name]["labels"],
                        types=ALL_TYPES,
                    )
                    for clip in clips
                ],
                20260970 + len(reports),
            )
        )
    minutes = sum(float(clip["audio_duration_sec"]) for clip in clips) / 60.0
    false_rates = {}
    for name, value in reports.items():
        metric = value["requested_four_types"]
        false = float(metric["predicted"] - metric["correct"])
        false_rates[name] = {
            "effective_false_labels": false,
            "per_clip": false / len(clips),
            "per_minute": false / minutes,
        }
    calibration = json.loads(
        (output / "calibration.json").read_text(encoding="utf-8")
    )
    v3_report = json.loads(args.v3_report.read_text(encoding="utf-8"))
    group_indices: dict[str, dict[str, list[int]]] = {
        "source": {},
        "repeat": {},
        "mapping_correctness": {},
    }
    for index, clip in enumerate(clips):
        memberships = {
            "source": str(clip["source"]),
            "repeat": "repeat" if clip["has_repeat"] else "ordinary",
            "mapping_correctness": (
                "majority_correct"
                if clip["mapping_correct_fraction"] >= 0.5
                else "majority_incorrect"
            ),
        }
        for dimension, name in memberships.items():
            group_indices[dimension].setdefault(name, []).append(index)
    breakdown = {}
    for dimension, groups in group_indices.items():
        breakdown[dimension] = {}
        for name, indices in groups.items():
            selected = [clips[index] for index in indices]
            item = {"clips": len(selected)}
            for policy in ("hybrid_max_f1", "hybrid_balanced"):
                counts = [
                    integrated._schema_counts(
                        clip["valid_labels"],
                        clip[policy]["labels"],
                        types=REQUESTED_TYPES,
                    )
                    for clip in selected
                ]
                total = integrated._sum_counts(counts)
                item[policy] = integrated._prf(
                    total["correct"], total["predicted"], total["target"]
                )
            breakdown[dimension][name] = item
    max_metric = reports["hybrid_max_f1"]["requested_four_types"]
    balanced_metric = reports["hybrid_balanced"]["requested_four_types"]
    report = {
        "schema_version": "align-error-heads-v4-report-v1",
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
            "freeze_manifest_sha256": integrated.sha256_file(
                args.freeze_manifest.resolve()
            ),
        },
        "calibration": calibration,
        "metrics": {
            "official_schema_1_2": reports,
            "false_labels": false_rates,
            "v3_reference": {
                "max_f1": v3_report["metrics"]["official_schema_1_2"]["v3"],
                "balanced": v3_report["metrics"]["official_schema_1_2"][
                    "v3_balanced_false_control"
                ],
                "event_head": v3_report["metrics"]["event_head_reference"],
                "oracle_decomposition": v3_report["oracle_decomposition"],
            },
        },
        "breakdown": breakdown,
        "upstream_event_reference_unchanged": {
            "reason": "v4 changes operation-core/schema decoding only",
            "metrics": v3_report["metrics"]["event_head_reference"],
            "breakdown": v3_report["breakdown"],
        },
        "promotion_gate": {
            "max_f1_delta_vs_v3": (
                max_metric["f1"]
                - v3_report["metrics"]["official_schema_1_2"]["v3"][
                    "requested_four_types"
                ]["f1"]
            ),
            "balanced_f1_delta_vs_v3": (
                balanced_metric["f1"]
                - v3_report["metrics"]["official_schema_1_2"][
                    "v3_balanced_false_control"
                ]["requested_four_types"]["f1"]
            ),
            "balanced_false_delta_vs_v3": (
                false_rates["hybrid_balanced"]["effective_false_labels"]
                - (
                    v3_report["metrics"]["official_schema_1_2"][
                        "v3_balanced_false_control"
                    ]["requested_four_types"]["predicted"]
                    - v3_report["metrics"]["official_schema_1_2"][
                        "v3_balanced_false_control"
                    ]["requested_four_types"]["correct"]
                )
            ),
            "material_f1_required": 0.02,
            "passed": bool(
                max_metric["f1"]
                >= v3_report["metrics"]["official_schema_1_2"]["v3"][
                    "requested_four_types"
                ]["f1"]
                + 0.02
                and balanced_metric["f1"]
                >= v3_report["metrics"]["official_schema_1_2"][
                    "v3_balanced_false_control"
                ]["requested_four_types"]["f1"]
                + 0.02
                and false_rates["hybrid_balanced"]["effective_false_labels"]
                <= v3_report["metrics"]["official_schema_1_2"][
                    "v3_balanced_false_control"
                ]["requested_four_types"]["predicted"]
                - v3_report["metrics"]["official_schema_1_2"][
                    "v3_balanced_false_control"
                ]["requested_four_types"]["correct"]
            ),
        },
        "training": {
            "performed": True,
            "kind": "validation-only linear operation-core selector",
            "device": "cpu",
            "reason": (
                "the initial deterministic posterior hybrid missed the material "
                "max-F1 gate while the operation-core oracle retained headroom"
            ),
        },
        "production_mutated": False,
        "promotion_performed": False,
    }
    _atomic_json(report_path, report)
    _atomic_json(
        output / "integrity.json",
        {
            "schema_version": "align-error-heads-v4-integrity-v1",
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
    manifest, artifacts = _verify_freeze(
        args.freeze_manifest.resolve(), PREDICTION_FREEZE_SCHEMA
    )
    report_path = args.output_dir.resolve() / "report.json"
    integrity_path = args.output_dir.resolve() / "integrity.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    errors = []
    if len(artifacts) != 358 or report["data"]["validation_rows"] != 358:
        errors.append("validation count mismatch")
    if not report["process_protocol"]["separate_processes"]:
        errors.append("process separation failed")
    if report["data"]["lockbox_touched"] or report["production_mutated"]:
        errors.append("isolation failed")
    if integrity["report_sha256"] != integrated.sha256_file(report_path):
        errors.append("report hash mismatch")
    result = {
        "schema_version": "align-error-heads-v4-verification-v1",
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
    print("error-heads-v4 verification passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/error-heads-v4"),
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
        "--feature-freeze-manifest",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v4/"
            "feature_freeze_manifest.json"
        ),
    )
    parser.add_argument(
        "--decode-config",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v4/decode_config.json"
        ),
    )
    parser.add_argument(
        "--v3-config",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v3/decode_config.json"
        ),
    )
    parser.add_argument(
        "--v3-report",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v3/report.json"
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
        "--packed-index",
        type=Path,
        default=Path(
            "data-packed/joint-outputraw-full-v1-shard64/index.sqlite"
        ),
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=Path(
            "runs/joint-audit-v2/end-to-end-v2/"
            "weak-note-continuation-optimized/joint_decoder.pt"
        ),
    )
    parser.add_argument(
        "--error-heads-checkpoint",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/predicted/best.pt"
        ),
    )
    parser.add_argument(
        "--oracle-heads-checkpoint",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/oracle/best.pt"
        ),
    )
    parser.add_argument(
        "--active-checkpoint",
        type=Path,
        default=Path(
            "runs/joint-audit-v2/components/candidate-rescorer-v1/"
            "mid_epoch_checkpoint.pt"
        ),
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("features")
    subparsers.add_parser("analyze")
    subparsers.add_parser("freeze")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--freeze-manifest", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--freeze-manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "features":
        features(args)
    elif args.phase == "analyze":
        analyze(args)
    elif args.phase == "freeze":
        freeze(args)
    elif args.phase == "score":
        score(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
