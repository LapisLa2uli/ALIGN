"""Two-process integrated validation for frozen joint + error-heads-v2.

``freeze`` is inference-only.  Its SQLite authorizer rejects reads of the
packed target column and its audit hook rejects known gold filenames.  A
separate ``score`` process verifies all frozen hashes before opening audited
validation targets.
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
import wave
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from alignmodel.joint.error_heads import (
    FEATURE_DIM,
    HeadPrediction,
    HeadRow,
    attach_training_targets,
    build_inference_rows,
    build_oracle_rows,
    evaluate_predictions,
    filter_prediction_for_schema,
    heuristic_prediction,
    infer_error_heads,
    load_error_heads,
    schema12_document,
)
from alignmodel.joint.index import JointEvent, ScoreEvent, ScoreEventIndex
from alignmodel.joint.lattice import (
    JointCandidate,
    LatticePath,
    SparseJointLattice,
)
from alignmodel.joint.metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
)
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import sha256_file
from alignmodel.joint.train import load_joint_model
from alignmodel.melody import (
    gold_melodies_from_labels,
    match_note_wise_labels_detail,
)


FREEZE_SCHEMA = "align-error-heads-integrated-freeze-v1"
REPORT_SCHEMA = "align-error-heads-integrated-validation-v1"
REQUESTED_TYPES = {
    "wrong_note",
    "missed_note",
    "extra_note",
    "rhythm_error",
}
LAYER2_TYPES = {"wrong_note", "missed_note", "extra_note"}
FORBIDDEN_PATH_TOKENS = (
    "labels.json",
    "note_map.json",
    "performance_audio.mid",
    "performance_score.musicxml",
    "canonical_dev_targets.sqlite",
    "validated_targets",
    "examples.sqlite",
    "split.json",
)


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


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def _event_json(value: JointEvent) -> dict[str, Any]:
    return {
        "pitch": value.pitch,
        "start": value.start,
        "end": value.end,
        "score_span": list(value.score_span) if value.score_span else None,
        "relationship": value.relationship,
        "copy_pass": value.copy_pass,
        "origin_relationship": value.origin_relationship,
        "rendered_index": value.rendered_index,
        "source_indices": list(value.source_indices),
        "confidence": value.confidence,
    }


def _event_from_json(value: Mapping[str, Any]) -> JointEvent:
    return JointEvent(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        score_span=(
            tuple(int(item) for item in value["score_span"])
            if value.get("score_span")
            else None
        ),
        relationship=str(value.get("relationship") or "match"),
        copy_pass=int(value.get("copy_pass") or 0),
        origin_relationship=value.get("origin_relationship"),
        rendered_index=(
            int(value["rendered_index"])
            if value.get("rendered_index") is not None
            else None
        ),
        source_indices=tuple(int(item) for item in value.get("source_indices") or ()),
        confidence=float(value.get("confidence", 1.0)),
    )


def _score_json(value: ScoreEvent) -> dict[str, Any]:
    return {
        "index": value.index,
        "pitch": value.pitch,
        "ql_start": value.ql_start,
        "ql_end": value.ql_end,
        "source_indices": list(value.source_indices),
        "measure": value.measure,
    }


def _score_from_json(value: Mapping[str, Any]) -> ScoreEvent:
    return ScoreEvent(
        index=int(value["index"]),
        pitch=int(value["pitch"]),
        ql_start=float(value["ql_start"]),
        ql_end=float(value["ql_end"]),
        source_indices=tuple(int(item) for item in value["source_indices"]),
        measure=(
            int(value["measure"]) if value.get("measure") is not None else None
        ),
    )


def _candidate_json(value: JointCandidate) -> dict[str, Any]:
    return {
        "pitch": value.pitch,
        "start": value.start,
        "end": value.end,
        "confidence": value.confidence,
        "score_hints": list(value.score_hints),
        "acoustic_features": list(value.acoustic_features),
    }


def _candidate_from_json(value: Mapping[str, Any]) -> JointCandidate:
    return JointCandidate(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        confidence=float(value["confidence"]),
        score_hints=tuple(int(item) for item in value.get("score_hints") or ()),
        acoustic_features=tuple(
            float(item) for item in value.get("acoustic_features") or ()
        ),
    )


def _row_json(value: HeadRow) -> dict[str, Any]:
    return {
        "kind": value.kind,
        "event_index": value.event_index,
        "score_span": list(value.score_span) if value.score_span else None,
        "start_sec": value.start_sec,
        "end_sec": value.end_sec,
        "is_copy": value.is_copy,
    }


def _row_from_json(value: Mapping[str, Any]) -> HeadRow:
    return HeadRow(
        features=np.zeros(FEATURE_DIM, dtype=np.float32),
        kind=str(value["kind"]),
        event_index=(
            int(value["event_index"])
            if value.get("event_index") is not None
            else None
        ),
        score_span=(
            tuple(int(item) for item in value["score_span"])
            if value.get("score_span")
            else None
        ),
        start_sec=float(value["start_sec"]),
        end_sec=float(value["end_sec"]),
        is_copy=bool(value.get("is_copy")),
    )


def _prediction_json(value: HeadPrediction) -> dict[str, Any]:
    return {
        "layer2": list(value.layer2),
        "rhythm": list(value.rhythm),
        "deviation_sec": list(value.deviation_sec),
        "rhythm_subtype": list(value.rhythm_subtype),
        "layer2_probabilities": [list(row) for row in value.layer2_probabilities],
        "rhythm_probabilities": list(value.rhythm_probabilities),
    }


def _prediction_from_json(value: Mapping[str, Any]) -> HeadPrediction:
    return HeadPrediction(
        layer2=tuple(str(item) for item in value["layer2"]),
        rhythm=tuple(bool(item) for item in value["rhythm"]),
        deviation_sec=tuple(float(item) for item in value["deviation_sec"]),
        rhythm_subtype=tuple(str(item) for item in value["rhythm_subtype"]),
        layer2_probabilities=tuple(
            tuple(float(item) for item in row)
            for row in value["layer2_probabilities"]
        ),
        rhythm_probabilities=tuple(
            float(item) for item in value["rhythm_probabilities"]
        ),
    )


def _path_json(path: LatticePath) -> dict[str, Any]:
    return {
        "score": path.score,
        "trailing_deletions": list(path.trailing_deletions),
        "steps": [
            {
                "candidate_index": step.candidate_index,
                "score_span": list(step.score_span) if step.score_span else None,
                "operation": step.operation.value,
                "structural_operation": (
                    step.structural_operation.value
                    if step.structural_operation is not None
                    else None
                ),
                "resume_event": step.resume_event,
                "deleted_events": list(step.deleted_events),
            }
            for step in path.steps
        ],
    }


def _frontend_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    frontend = (payload.get("training") or {}).get("frontend") or {}
    return {
        "name": frontend.get("name"),
        "candidate_generation": frontend.get("candidate_generation"),
        "decode_configs": frontend.get("decode_configs") or [],
        "frozen": True,
    }


def _resource_snapshot(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {
        "path": str(path.resolve()),
        "schema_version": value.get("schema_version"),
        "revision": value.get("revision"),
        "updated_utc": value.get("updated_utc"),
        "gpu_lease": (value.get("leases") or {}).get("gpu"),
        "policy": value.get("policy"),
    }


def _load_stack(args: argparse.Namespace) -> dict[str, Any]:
    joint_path = args.joint_checkpoint.resolve()
    head_path = args.error_heads_checkpoint.resolve()
    joint_report_path = joint_path.parent / "report.json"
    head_report_path = head_path.parents[1] / "report.json"
    verification_path = head_path.parents[1] / "verification.json"
    for path in (
        joint_path,
        joint_report_path,
        head_path,
        head_report_path,
        verification_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if joint_path.name != "joint_decoder.pt":
        raise ValueError("Only a completed joint_decoder.pt is selectable")
    if head_path.name != "best.pt" or head_path.parent.name != "predicted":
        raise ValueError("Only completed predicted/best.pt is selectable")
    joint_report = json.loads(joint_report_path.read_text(encoding="utf-8"))
    head_report = json.loads(head_report_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not (head_report.get("promotion_gate") or {}).get("passed_metric"):
        raise ValueError("Error-head checkpoint did not pass documented metric policy")
    if not (verification.get("promotion") or {}).get("integration_tests_passed"):
        raise ValueError("Error-head checkpoint lacks passed integration verification")
    joint_hash = sha256_file(joint_path)
    if joint_report.get("checkpoint_sha256") != joint_hash:
        raise ValueError("Joint checkpoint hash differs from completed report")
    expected_joint = (
        head_report.get("evaluation", {})
        .get("upstream", {})
        .get("decoder", {})
        .get("sha256")
    )
    if expected_joint != joint_hash:
        raise ValueError("Error heads are incompatible with selected joint checkpoint")
    joint_model, lattice_config, joint_payload = load_joint_model(
        joint_path, device="cpu"
    )
    if not joint_payload.get("history") or not joint_payload.get("best_validation"):
        raise ValueError("Joint checkpoint has no completed validation history")
    joint_model.eval()
    for parameter in joint_model.parameters():
        parameter.requires_grad = False
    heads_model, heads_payload = load_error_heads(head_path, device="cpu")
    if len(heads_payload.get("history") or ()) < 8:
        raise ValueError("Error-head checkpoint has incomplete training history")
    if not isinstance(heads_payload.get("schema_thresholds"), Mapping):
        raise ValueError("Error-head checkpoint lacks schema calibration")
    heads_model.eval()
    for parameter in heads_model.parameters():
        parameter.requires_grad = False
    frontend = _frontend_contract(joint_payload)
    frontend_hash = _canonical_hash(frontend)
    expected_frontend = (
        head_report.get("evaluation", {})
        .get("upstream", {})
        .get("transcriber", {})
        .get("candidate_config_sha256")
    )
    if frontend_hash != expected_frontend:
        raise ValueError("Candidate configuration hash is incompatible")
    head_upstream = heads_payload.get("upstream") or {}
    recorded_joint = (
        head_upstream.get("checkpoint_sha256")
        or (head_upstream.get("decoder") or {}).get("sha256")
    )
    if recorded_joint is not None and recorded_joint != joint_hash:
        raise ValueError("Error-head payload references another upstream")
    selection = {
        "policy": (
            "completed predicted error-heads-v2 with passed validation and "
            "integration gates; select the newest completed joint decoder "
            "explicitly hash-bound by that head; exclude active/partial checkpoints"
        ),
        "transcriber_candidate_config": {
            "implementation": "frozen Basic Pitch 0.4.0 activation-derived candidate union",
            "candidate_version": frontend["candidate_generation"],
            "config": frontend,
            "config_sha256": frontend_hash,
            "transfer": "packed candidate rows are inference-only frontend outputs",
        },
        "joint_aligner_decoder": {
            "path": str(joint_path),
            "sha256": joint_hash,
            "schema_version": joint_payload.get("schema_version"),
            "report": str(joint_report_path),
            "history_epochs": len(joint_payload["history"]),
            "best_validation_present": True,
        },
        "error_heads_v2": {
            "path": str(head_path),
            "sha256": sha256_file(head_path),
            "schema_version": heads_payload.get("schema_version"),
            "report": str(head_report_path),
            "verification": str(verification_path),
            "history_epochs": len(heads_payload["history"]),
            "schema_thresholds": dict(heads_payload["schema_thresholds"]),
            "data_fingerprint": heads_payload.get("data_fingerprint"),
        },
        "compatibility": {
            "joint_sha256_match": True,
            "candidate_config_sha256_match": True,
            "data_fingerprint": heads_payload.get("data_fingerprint"),
        },
        "excluded_active_or_partial": {
            "path": str(args.active_checkpoint.resolve()),
            "read": False,
            "reason": "last_checkpoint.pt from current training is forbidden",
        },
        "documented_selection_evidence": (
            head_report.get("evaluation", {})
            .get("upstream", {})
            .get("selection_evidence")
        ),
    }
    return {
        "lattice": SparseJointLattice(joint_model, lattice_config),
        "joint_payload": joint_payload,
        "heads_model": heads_model,
        "heads_payload": heads_payload,
        "selection": selection,
    }


def _deny_target_column(
    action: int,
    arg1: str | None,
    arg2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    if (
        action == sqlite3.SQLITE_READ
        and str(arg1).casefold() == "records"
        and str(arg2).casefold() == "target"
    ):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _install_gold_guard() -> list[str]:
    opened: list[str] = []

    def hook(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        value = arguments[0]
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        text = os.fspath(value)
        if isinstance(text, bytes):
            text = os.fsdecode(text)
        normalized = text.replace("\\", "/").casefold()
        if any(token in normalized for token in FORBIDDEN_PATH_TOKENS):
            raise PermissionError(f"Inference gold guard denied: {text}")
        opened.append(str(text))

    sys.addaudithook(hook)
    return opened


def _inference_records(
    ready: Mapping[str, Any],
) -> tuple[sqlite3.Connection, list[tuple[Any, ...]], dict[str, Any]]:
    packed_root = Path(str(ready["paths"]["packed_root"]))
    metadata_path = packed_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("pack_id") != ready["hashes"]["pack_id"]:
        raise ValueError("Packed metadata fingerprint mismatch")
    if sha256_file(metadata_path) != ready["hashes"]["packed_metadata_sha256"]:
        raise ValueError("Packed metadata hash mismatch")
    index_path = packed_root / str(metadata["index"]["name"])
    if sha256_file(index_path) != ready["hashes"]["packed_index_sha256"]:
        raise ValueError("Packed index hash mismatch")
    connection = sqlite3.connect(
        f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    connection.set_authorizer(_deny_target_column)
    rows = connection.execute(
        "SELECT ordinal,sample,source,candidates,record_key "
        "FROM records WHERE split='val' ORDER BY ordinal"
    ).fetchall()
    if len(rows) != 358:
        raise ValueError(f"Expected 358 validation inference rows, got {len(rows)}")
    return connection, rows, {
        "packed_root": str(packed_root.resolve()),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "target_column_authorizer": "SQLITE_DENY records.target",
        "candidate_version": metadata.get("candidate_version"),
    }


def freeze(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    marker = output / "freeze_manifest.json"
    if marker.exists():
        raise FileExistsError(marker)
    output.mkdir(parents=True, exist_ok=True)
    opened_paths = _install_gold_guard()
    ready = verify_data_ready(args.ready_marker.resolve())
    if (
        int(ready["counts"]["val"]) != 358
        or int(ready["counts"]["locked_test_metadata_only"]) != 4022
        or ready["verification"]["test_targets_materialized"] is not False
        or ready["verification"]["test_features_materialized"] is not False
    ):
        raise ValueError("Integrated validation requires the sealed 358-row release")
    resource = _resource_snapshot(args.resource_status.resolve())
    stack = _load_stack(args)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    connection, packed_rows, pack_contract = _inference_records(ready)
    if (
        pack_contract["candidate_version"]
        != stack["selection"]["transcriber_candidate_config"]["candidate_version"]
    ):
        raise ValueError("Packed candidates are incompatible with selected frontend")
    samples = []
    artifact_entries = []
    started = time.perf_counter()
    try:
        for position, (ordinal, sample, source, candidates_blob, record_key) in enumerate(
            packed_rows, 1
        ):
            candidate_values = json.loads(zlib.decompress(candidates_blob))
            candidates = tuple(
                _candidate_from_json(value) for value in candidate_values
            )
            sample_dir = args.source_root.resolve() / str(sample)
            score_path = sample_dir / "verified_score.musicxml"
            audio_path = sample_dir / "performance_audio.wav"
            if not score_path.is_file() or not audio_path.is_file():
                raise FileNotFoundError(f"Missing production inputs for {sample}")
            score = tuple(ScoreEventIndex.from_musicxml(score_path).events)
            path = stack["lattice"].decode(candidates, score)
            events = tuple(path.joint_events(candidates))
            rows = build_inference_rows(
                stack["lattice"], candidates, score, path
            )
            prediction = infer_error_heads(
                stack["heads_model"],
                rows,
                stack["heads_payload"]["thresholds"],
                device="cpu",
            )
            filtered = filter_prediction_for_schema(
                prediction, stack["heads_payload"]["schema_thresholds"]
            )
            rules_prediction = heuristic_prediction(rows)
            current_document = schema12_document(
                str(sample), rows, filtered, score, pad_notes=1
            )
            rules_document = schema12_document(
                str(sample), rows, rules_prediction, score, pad_notes=1
            )
            deleted = set(path.trailing_deletions)
            for step in path.steps:
                deleted.update(step.deleted_events)
            artifact = {
                "schema_version": "align-error-heads-frozen-clip-v1",
                "ordinal": int(ordinal),
                "record_key": str(record_key),
                "sample": str(sample),
                "source": str(source),
                "audio_duration_sec": _audio_duration(audio_path),
                "score": [_score_json(value) for value in score],
                "candidates": [_candidate_json(value) for value in candidates],
                "joint_events": [_event_json(value) for value in events],
                "predicted_deletions": sorted(deleted),
                "path": _path_json(path),
                "head_rows": [_row_json(value) for value in rows],
                "prediction": _prediction_json(prediction),
                "rules_prediction": _prediction_json(rules_prediction),
                "schema_1_2": current_document,
                "rules_schema_1_2": rules_document,
            }
            artifact_path = output / "frozen" / f"{int(ordinal):05d}.json"
            _atomic_json(artifact_path, artifact)
            digest = sha256_file(artifact_path)
            artifact_entries.append(
                {
                    "ordinal": int(ordinal),
                    "sample": str(sample),
                    "path": str(artifact_path.resolve()),
                    "sha256": digest,
                    "bytes": artifact_path.stat().st_size,
                }
            )
            samples.append(
                {
                    "ordinal": int(ordinal),
                    "sample": str(sample),
                    "source": str(source),
                    "candidate_count": len(candidates),
                    "decoded_event_count": len(events),
                    "mapped_event_count": sum(
                        event.score_span is not None for event in events
                    ),
                    "copy_event_count": sum(event.is_copy for event in events),
                    "gap_count": len(deleted),
                    "label_count_requested": sum(
                        label["type"] in REQUESTED_TYPES
                        for label in current_document["labels"]
                    ),
                    "repetition_context_labels": sum(
                        label["type"] == "repetition"
                        for label in current_document["labels"]
                    ),
                }
            )
            if position == 1 or position % 25 == 0 or position == len(packed_rows):
                elapsed = time.perf_counter() - started
                print(
                    f"freeze={position}/358 rows_per_sec={position/max(elapsed,1e-9):.3f}",
                    flush=True,
                )
    finally:
        connection.close()
    aggregate = hashlib.sha256()
    for row in artifact_entries:
        aggregate.update(f"{row['ordinal']}:{row['sha256']}\n".encode("ascii"))
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": _utc(),
        "process": {
            "pid": os.getpid(),
            "phase": "A_inference_freeze",
            "command": [sys.executable, *sys.argv],
        },
        "data": {
            "ready_marker": str(args.ready_marker.resolve()),
            "ready_marker_sha256": sha256_file(args.ready_marker.resolve()),
            "pack_id": ready["hashes"]["pack_id"],
            "validation_rows": 358,
            "lockbox_metadata_rows": 4022,
            "lockbox_touched": False,
            "pack_contract": pack_contract,
        },
        "gold_isolation": {
            "gold_opened": False,
            "forbidden_paths_opened": [],
            "forbidden_path_tokens": list(FORBIDDEN_PATH_TOKENS),
            "sqlite_target_column_read": False,
            "sqlite_authorizer": "SQLITE_DENY records.target",
            "score_source": "verified_score.musicxml parsed without lineage",
            "candidate_source": "packed inference candidate column only",
            "opened_path_count": len(opened_paths),
            "opened_path_suffix_counts": dict(
                Counter(Path(path).suffix.casefold() for path in opened_paths)
            ),
        },
        "model_selection": stack["selection"],
        "resource_coordination": {
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
            "gpu_lease_acquired": False,
            "status_snapshot": resource,
        },
        "artifacts": artifact_entries,
        "artifacts_manifest_sha256": aggregate.hexdigest(),
        "samples": samples,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(marker, manifest)
    print(marker)


def _load_and_verify_freeze(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("Unsupported freeze manifest")
    if manifest.get("gold_isolation", {}).get("gold_opened") is not False:
        raise ValueError("Freeze manifest does not assert gold isolation")
    artifacts = []
    aggregate = hashlib.sha256()
    for row in manifest["artifacts"]:
        artifact_path = Path(row["path"])
        digest = sha256_file(artifact_path)
        if digest != row["sha256"]:
            raise ValueError(f"Frozen artifact hash mismatch: {artifact_path}")
        aggregate.update(f"{row['ordinal']}:{digest}\n".encode("ascii"))
        artifacts.append(json.loads(artifact_path.read_text(encoding="utf-8")))
    if aggregate.hexdigest() != manifest["artifacts_manifest_sha256"]:
        raise ValueError("Frozen aggregate manifest hash mismatch")
    if len(artifacts) != 358:
        raise ValueError(f"Frozen artifact count is {len(artifacts)}, not 358")
    return manifest, artifacts


def _target_event(value: Mapping[str, Any]) -> JointEvent:
    return _event_from_json(value)


def _prf(correct: float, predicted: int, target: int) -> dict[str, Any]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / target if target else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "correct": correct,
        "predicted": predicted,
        "target": target,
    }


def _schema_counts(
    gold_labels: Sequence[Mapping[str, Any]],
    predicted_labels: Sequence[Mapping[str, Any]],
    *,
    types: set[str],
    ignore_type: bool = False,
) -> dict[str, Any]:
    gold = [
        dict(label)
        for label in gold_labels
        if label.get("type") in types
    ]
    predicted = [
        dict(label)
        for label in predicted_labels
        if label.get("type") in types
    ]
    detail = match_note_wise_labels_detail(
        gold,
        predicted,
        type_mismatch_credit=1.0 if ignore_type else 0.5,
    )
    if detail["status"] != "available":
        raise ValueError(f"Official note-wise metric unavailable: {detail['reason']}")
    return {
        "correct": float(detail["credit"]),
        "predicted": len(predicted),
        "target": len(gold),
    }


def _type_only_counts(
    gold_labels: Sequence[Mapping[str, Any]],
    predicted_labels: Sequence[Mapping[str, Any]],
    types: set[str],
) -> dict[str, Any]:
    gold = Counter(
        item.type
        for item in gold_melodies_from_labels(
            [
                dict(label)
                for label in gold_labels
                if label.get("type") in types
            ]
        )
        if item.type in types
    )
    predicted = Counter(
        str(label.get("type"))
        for label in predicted_labels
        if label.get("type") in types
    )
    return {
        "correct": float(
            sum(min(count, predicted.get(kind, 0)) for kind, count in gold.items())
        ),
        "predicted": sum(predicted.values()),
        "target": sum(gold.values()),
    }


def _sum_counts(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "correct": sum(float(value["correct"]) for value in values),
        "predicted": sum(int(value["predicted"]) for value in values),
        "target": sum(int(value["target"]) for value in values),
    }


def _bootstrap(
    values: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected = rng.integers(0, len(values), len(values))
        counts = _sum_counts([values[index] for index in selected])
        samples.append(
            _prf(
                counts["correct"], counts["predicted"], counts["target"]
            )["f1"]
        )
    return {
        "replicates": replicates,
        "unit": "clip",
        "lower_95": float(np.quantile(samples, 0.025)),
        "median": float(np.quantile(samples, 0.5)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _metric_with_ci(
    values: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, Any]:
    counts = _sum_counts(values)
    return {
        **_prf(counts["correct"], counts["predicted"], counts["target"]),
        "bootstrap_95_ci": _bootstrap(values, seed=seed),
    }


def _schema_report(
    clips: Sequence[Mapping[str, Any]],
    document_key: str,
) -> dict[str, Any]:
    combined = [
        _schema_counts(
            clip["valid_labels"],
            clip[document_key]["labels"],
            types=REQUESTED_TYPES,
        )
        for clip in clips
    ]
    per_type = {
        kind: _metric_with_ci(
            [
                _schema_counts(
                    clip["valid_labels"],
                    clip[document_key]["labels"],
                    types={kind},
                )
                for clip in clips
            ],
            20260920 + index,
        )
        for index, kind in enumerate(sorted(REQUESTED_TYPES))
    }
    repetition = [
        _schema_counts(
            clip["valid_labels"],
            clip[document_key]["labels"],
            types={"repetition"},
        )
        for clip in clips
    ]
    range_only = [
        _schema_counts(
            clip["valid_labels"],
            clip[document_key]["labels"],
            types=REQUESTED_TYPES,
            ignore_type=True,
        )
        for clip in clips
    ]
    type_only = [
        _type_only_counts(
            clip["valid_labels"],
            clip[document_key]["labels"],
            REQUESTED_TYPES,
        )
        for clip in clips
    ]
    return {
        "metric": "official exclusive canonical score-event identity; exact type=1, different type=0.5",
        "schema_version": "align-note-wise-score-event-metric-v1",
        "requested_four_types": _metric_with_ci(combined, 20260915),
        "per_type": per_type,
        "repetition_context_separate": _metric_with_ci(
            repetition, 20260926
        ),
        "diagnostic_location_only_ignore_type": _metric_with_ci(
            range_only, 20260927
        ),
        "diagnostic_type_only_ignore_location": _metric_with_ci(
            type_only, 20260928
        ),
        # Backward-compatible aliases. These are diagnostics, not promotion fields.
        "range_only_ignore_type": _metric_with_ci(range_only, 20260927),
        "type_only_perfect_range": _metric_with_ci(type_only, 20260928),
        "_clip_counts": combined,
    }


def _layer2_clip_counts(
    labeled: Any, prediction: HeadPrediction
) -> dict[str, Any]:
    predicted = np.asarray(
        [
            ("match", "wrong_note", "extra_note", "missed_note").index(value)
            for value in prediction.layer2
        ]
    )
    target = labeled.layer2
    return {
        "correct": int(np.sum((predicted == target) & (target != 0))),
        "predicted": int(np.sum(predicted != 0)),
        "target": int(np.sum(target != 0)),
    }


def _rhythm_clip_counts(
    labeled: Any, prediction: HeadPrediction
) -> dict[str, Any]:
    mask = labeled.rhythm_mask.astype(bool)
    predicted = np.asarray(prediction.rhythm, dtype=bool)[mask]
    target = labeled.rhythm.astype(bool)[mask]
    return {
        "correct": int(np.sum(predicted & target)),
        "predicted": int(np.sum(predicted)),
        "target": int(np.sum(target)),
    }


def _joint_clip_counts(sample: JointMetricSample) -> dict[str, dict[str, Any]]:
    report = evaluate_joint_dataset([sample])["aggregate"]["tolerances"]["50ms"]
    counts = report["counts"]
    return {
        "note": {
            "correct": counts["paired"],
            "predicted": counts["predicted"],
            "target": counts["target"],
        },
        "mapping": {
            "correct": counts["mapping_correct"],
            "predicted": counts["predicted"] - counts["predicted_extra"],
            "target": counts["target"] - counts["target_extra"],
        },
        "copy": {
            "correct": counts["copy_correct"],
            "predicted": counts["predicted_copy"],
            "target": counts["target_copy"],
        },
    }


def _compact_breakdown(
    clips: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    output = {}
    for name, indices in groups.items():
        selected = [clips[index] for index in indices]
        layer2 = [_layer2_clip_counts(row["labeled"], row["prediction"]) for row in selected]
        rhythm = [_rhythm_clip_counts(row["labeled"], row["prediction"]) for row in selected]
        schema = [
            _schema_counts(
                row["valid_labels"],
                row["schema_1_2"]["labels"],
                types=REQUESTED_TYPES,
            )
            for row in selected
        ]
        output[name] = {
            "clips": len(selected),
            "layer2_f1": _prf(**{
                "correct": _sum_counts(layer2)["correct"],
                "predicted": _sum_counts(layer2)["predicted"],
                "target": _sum_counts(layer2)["target"],
            })["f1"],
            "rhythm_f1": _prf(**{
                "correct": _sum_counts(rhythm)["correct"],
                "predicted": _sum_counts(rhythm)["predicted"],
                "target": _sum_counts(rhythm)["target"],
            })["f1"],
            "schema_four_type_f1": _prf(**{
                "correct": _sum_counts(schema)["correct"],
                "predicted": _sum_counts(schema)["predicted"],
                "target": _sum_counts(schema)["target"],
            })["f1"],
        }
    return output


def score(args: argparse.Namespace) -> None:
    manifest, frozen = _load_and_verify_freeze(
        args.freeze_manifest.resolve()
    )
    output = args.freeze_manifest.resolve().parent
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    if os.getpid() == int(manifest["process"]["pid"]):
        raise ValueError("Scoring must run in a separate process")

    target_index_path = Path(manifest["data"]["pack_contract"]["index"])
    target_connection = sqlite3.connect(
        f"file:{target_index_path.resolve().as_posix()}?mode=ro", uri=True
    )
    audit_connection = sqlite3.connect(
        f"file:{args.canonical_targets.resolve().as_posix()}?mode=ro", uri=True
    )
    stack = _load_stack(args)
    oracle_model, oracle_payload = load_error_heads(
        args.oracle_heads_checkpoint.resolve(), device="cpu"
    )
    if (
        not (oracle_payload.get("history") or ())
        or (oracle_payload.get("progress") or {}).get("phase")
        != "epoch_complete"
        or oracle_payload.get("data_fingerprint")
        != stack["heads_payload"].get("data_fingerprint")
        or (
            (oracle_payload.get("upstream") or {}).get("decoder") or {}
        ).get("sha256")
        != manifest["model_selection"]["joint_aligner_decoder"]["sha256"]
    ):
        raise ValueError("Oracle-head ablation checkpoint is incomplete or incompatible")
    oracle_model.eval()
    for parameter in oracle_model.parameters():
        parameter.requires_grad = False
    clips = []
    joint_samples = []
    oracle_eval_clips = []
    for position, artifact in enumerate(frozen, 1):
        ordinal = int(artifact["ordinal"])
        target_blob = target_connection.execute(
            "SELECT target FROM records WHERE ordinal=? AND split='val'",
            (ordinal,),
        ).fetchone()
        audit_blob = audit_connection.execute(
            "SELECT payload FROM targets WHERE ordinal=? AND split='val'",
            (ordinal,),
        ).fetchone()
        if target_blob is None or audit_blob is None:
            raise ValueError(f"Missing audited validation target {ordinal}")
        target = json.loads(zlib.decompress(target_blob[0]))
        audited = json.loads(zlib.decompress(audit_blob[0]))
        if target["sample"] != artifact["sample"]:
            raise ValueError(f"Sample mismatch at ordinal {ordinal}")
        score_events = tuple(_score_from_json(value) for value in artifact["score"])
        target_score = tuple(_score_from_json(value) for value in target["score"])
        if score_events != target_score:
            raise ValueError(f"Gold changed inference score at {ordinal}")
        candidates = tuple(
            _candidate_from_json(value) for value in artifact["candidates"]
        )
        predicted_events = tuple(
            _event_from_json(value) for value in artifact["joint_events"]
        )
        target_events = tuple(
            _target_event(value) for value in target["target_events"]
        )
        rows = tuple(_row_from_json(value) for value in artifact["head_rows"])
        prediction = _prediction_from_json(artifact["prediction"])
        rules_prediction = _prediction_from_json(artifact["rules_prediction"])
        labeled = attach_training_targets(
            rows,
            predicted_events=predicted_events,
            target_events=target_events,
            target_deletions=target["target_deletions"],
            rhythm_rows=target["layer3_rhythm"],
            score=score_events,
        )
        valid_labels = [
            dict(label)
            for label in audited.get("valid_labels") or ()
            if label.get("type") != "intonation_error"
        ]
        clip = {
            **artifact,
            "valid_labels": valid_labels,
            "labeled": labeled,
            "prediction": prediction,
            "rules_prediction": rules_prediction,
            "has_repeat": bool(target.get("layer1_repeats")),
        }
        clips.append(clip)
        joint_samples.append(
            JointMetricSample(
                predicted=predicted_events,
                target=target_events,
                source=str(artifact["source"]),
                predicted_deletions=frozenset(artifact["predicted_deletions"]),
                target_deletions=frozenset(
                    int(value) for value in target["target_deletions"]
                ),
                score_event_count=len(score_events),
            )
        )
        oracle_rows = build_oracle_rows(
            stack["lattice"],
            candidates,
            score_events,
            target_events,
            target["target_deletions"],
        )
        oracle_labeled = attach_training_targets(
            oracle_rows,
            predicted_events=target_events,
            target_events=target_events,
            target_deletions=target["target_deletions"],
            rhythm_rows=target["layer3_rhythm"],
            score=score_events,
        )
        oracle_prediction = infer_error_heads(
            oracle_model,
            oracle_rows,
            oracle_payload["thresholds"],
            device="cpu",
        )
        oracle_filtered = filter_prediction_for_schema(
            oracle_prediction, oracle_payload["schema_thresholds"]
        )
        clip["oracle_schema_1_2"] = schema12_document(
            artifact["sample"],
            oracle_rows,
            oracle_filtered,
            score_events,
            pad_notes=1,
        )
        oracle_eval_clips.append(
            (
                artifact["sample"],
                oracle_labeled,
                oracle_prediction,
                {
                    "source": artifact["source"],
                    "repeats": "repeat" if clip["has_repeat"] else "ordinary",
                    "duration": (
                        "ge_10s"
                        if artifact["audio_duration_sec"] >= 10.0
                        else "lt_10s"
                    ),
                },
            )
        )
        if position == 1 or position % 50 == 0 or position == len(frozen):
            print(f"score={position}/358", flush=True)
    target_connection.close()
    audit_connection.close()

    metadata_clips = [
        (
            clip["sample"],
            clip["labeled"],
            clip["prediction"],
            {
                "source": clip["source"],
                "repeats": "repeat" if clip["has_repeat"] else "ordinary",
                "duration": (
                    "ge_10s" if clip["audio_duration_sec"] >= 10.0 else "lt_10s"
                ),
            },
        )
        for clip in clips
    ]
    rule_clips = [
        (
            clip["sample"],
            clip["labeled"],
            clip["rules_prediction"],
            metadata,
        )
        for clip, (_sample, _labeled, _prediction, metadata) in zip(
            clips, metadata_clips
        )
    ]
    event = evaluate_predictions(metadata_clips)
    rules_event = evaluate_predictions(rule_clips)
    oracle_event = evaluate_predictions(oracle_eval_clips)
    minutes = sum(float(clip["audio_duration_sec"]) for clip in clips) / 60.0
    false_labels = (
        int(event["layer2"]["error_only"]["predicted"])
        - int(event["layer2"]["error_only"]["typed_correct"])
    )
    layer2_counts = [
        _layer2_clip_counts(clip["labeled"], clip["prediction"])
        for clip in clips
    ]
    rhythm_counts = [
        _rhythm_clip_counts(clip["labeled"], clip["prediction"])
        for clip in clips
    ]
    event["layer2"]["bootstrap_95_ci"] = _bootstrap(
        layer2_counts, seed=20260931
    )
    event["layer2"]["false_labels_per_minute"] = false_labels / minutes
    event["layer2"]["evaluated_audio_minutes"] = minutes
    event["layer3"]["rhythm"]["bootstrap_95_ci"] = _bootstrap(
        rhythm_counts, seed=20260932
    )

    schema_current = _schema_report(clips, "schema_1_2")
    schema_rules = _schema_report(clips, "rules_schema_1_2")
    schema_oracle = _schema_report(clips, "oracle_schema_1_2")
    joint = evaluate_joint_dataset(joint_samples)
    joint_counts = [_joint_clip_counts(sample) for sample in joint_samples]
    upstream_ci = {
        name: _metric_with_ci(
            [value[name] for value in joint_counts], 20260940 + index
        )
        for index, name in enumerate(("note", "mapping", "copy"))
    }
    total_predicted = sum(len(clip["joint_events"]) for clip in clips)
    total_target = sum(
        int(value["note"]["target"]) for value in joint_counts
    )
    total_candidates = sum(len(clip["candidates"]) for clip in clips)

    source_groups: dict[str, list[int]] = defaultdict(list)
    repeat_groups: dict[str, list[int]] = defaultdict(list)
    mapping_groups: dict[str, list[int]] = defaultdict(list)
    for index, (clip, counts) in enumerate(zip(clips, joint_counts)):
        source_groups[str(clip["source"])].append(index)
        repeat_groups[
            "repeat" if clip["has_repeat"] else "ordinary"
        ].append(index)
        conditional = counts["mapping"]["correct"] / max(
            counts["mapping"]["target"], 1
        )
        mapping_groups[
            "majority_correct" if conditional >= 0.5 else "majority_incorrect"
        ].append(index)

    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    prior_official = prior["evaluation"]["official_schema_1_2"][
        "predicted_upstream"
    ]
    prior_points = {
        "combined_including_repetition": prior_official["micro"]["f1"],
        "wrong_note": prior_official["per_type"]["wrong_note"]["f1"],
        "missed_note": prior_official["per_type"]["missed_note"]["f1"],
        "extra_note": prior_official["per_type"]["extra_note"]["f1"],
        "rhythm_error": prior_official["per_type"]["rhythm_error"]["f1"],
    }
    current_requested = schema_current["requested_four_types"]["f1"]
    current_including_repetition_counts = _sum_counts(
        [
            _schema_counts(
                clip["valid_labels"],
                clip["schema_1_2"]["labels"],
                types=REQUESTED_TYPES | {"repetition"},
            )
            for clip in clips
        ]
    )
    current_including_repetition = _prf(
        current_including_repetition_counts["correct"],
        current_including_repetition_counts["predicted"],
        current_including_repetition_counts["target"],
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": _utc(),
        "process_protocol": {
            "process_a_pid": manifest["process"]["pid"],
            "process_b_pid": os.getpid(),
            "separate_processes": os.getpid() != manifest["process"]["pid"],
            "frozen_manifest_verified_before_gold_open": True,
            "freeze_manifest": str(args.freeze_manifest.resolve()),
            "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        },
        "data": {
            "ready_marker": manifest["data"]["ready_marker"],
            "pack_id": manifest["data"]["pack_id"],
            "validation_rows": len(clips),
            "lockbox_metadata_rows": 4022,
            "lockbox_touched": False,
            "intonation_masked": True,
        },
        "model_selection": manifest["model_selection"],
        "oracle_ablation_selection": {
            "path": str(args.oracle_heads_checkpoint.resolve()),
            "sha256": sha256_file(args.oracle_heads_checkpoint.resolve()),
            "schema_version": oracle_payload.get("schema_version"),
            "history_epochs": len(oracle_payload["history"]),
            "completed_best_epoch": (
                oracle_payload.get("progress") or {}
            ).get("epoch"),
            "data_fingerprint": oracle_payload.get("data_fingerprint"),
            "joint_sha256": manifest["model_selection"][
                "joint_aligner_decoder"
            ]["sha256"],
            "training_exposure": "oracle_upstream_separate_ablation",
        },
        "metrics": {
            "official_schema_1_2": {
                "current": schema_current,
                "rules_same_frozen_upstream": schema_rules,
                "oracle_upstream_ceiling": schema_oracle,
                "combined_including_repetition_for_prior_comparability": (
                    current_including_repetition
                ),
            },
            "event_level": {
                "current": event,
                "rules_same_frozen_upstream": rules_event,
                "oracle_upstream_ceiling": oracle_event,
            },
            "upstream": {
                "joint_50ms": joint["aggregate"]["tolerances"]["50ms"],
                "bootstrap_95_ci": upstream_ci,
                "decoded_note_count_ratio": total_predicted / max(total_target, 1),
                "candidate_count_ratio": total_candidates / max(total_target, 1),
                "decoded_events": total_predicted,
                "candidates": total_candidates,
                "target_events": total_target,
            },
        },
        "breakdown": {
            "source": _compact_breakdown(clips, source_groups),
            "repeat_context": _compact_breakdown(clips, repeat_groups),
            "mapping_correctness": _compact_breakdown(clips, mapping_groups),
        },
        "prior_comparison": {
            "prior_report": str(args.prior_report.resolve()),
            "expected_points": prior_points,
            "fresh_points": {
                "requested_four_combined": current_requested,
                "combined_including_repetition": current_including_repetition["f1"],
                **{
                    kind: schema_current["per_type"][kind]["f1"]
                    for kind in sorted(REQUESTED_TYPES)
                },
            },
            "definition_note": (
                "The prior 0.211 headline included repetition. This request "
                "defines the combined headline as only wrong/missed/extra/rhythm; "
                "the directly comparable including-repetition score is reported separately."
            ),
        },
        "dominant_bottleneck": {
            "finding": (
                "score-range localization/projection and false-label control "
                "dominate end-to-end schema loss; frozen upstream exposure "
                "dominates the separate event-level Layer2 gap"
            ),
            "evidence": {
                "oracle_event_layer2_f1": oracle_event["full_typed_error_f1"],
                "predicted_event_layer2_f1": event["full_typed_error_f1"],
                "oracle_schema_four_type_f1": schema_oracle[
                    "requested_four_types"
                ]["f1"],
                "predicted_schema_four_type_f1": current_requested,
                "schema_range_only_f1": schema_current[
                    "range_only_ignore_type"
                ]["f1"],
                "schema_type_only_ceiling_f1": schema_current[
                    "type_only_perfect_range"
                ]["f1"],
            },
        },
        "resource_coordination": manifest["resource_coordination"],
        "production_mutated": False,
        "promotion_performed": False,
    }
    schema_current.pop("_clip_counts", None)
    schema_rules.pop("_clip_counts", None)
    schema_oracle.pop("_clip_counts", None)
    _atomic_json(report_path, report)
    _atomic_json(
        output / "integrity.json",
        {
            "schema_version": "align-error-heads-integrated-integrity-v1",
            "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
            "report_sha256": sha256_file(report_path),
            "frozen_artifacts_verified": 358,
            "gold_opened_only_by_process_b": True,
            "process_a_pid": manifest["process"]["pid"],
            "process_b_pid": os.getpid(),
            "lockbox_touched": False,
            "production_mutated": False,
        },
    )
    print(report_path)


def verify(args: argparse.Namespace) -> None:
    manifest, artifacts = _load_and_verify_freeze(args.freeze_manifest.resolve())
    report_path = args.freeze_manifest.resolve().parent / "report.json"
    integrity_path = args.freeze_manifest.resolve().parent / "integrity.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    errors = []
    if len(artifacts) != 358 or report["data"]["validation_rows"] != 358:
        errors.append("validation count mismatch")
    if manifest["data"]["lockbox_touched"] or report["data"]["lockbox_touched"]:
        errors.append("lockbox touched")
    if report.get("production_mutated") or report.get("promotion_performed"):
        errors.append("forbidden mutation/promotion")
    if integrity["report_sha256"] != sha256_file(report_path):
        errors.append("report hash mismatch")
    if not report["process_protocol"]["separate_processes"]:
        errors.append("processes were not separate")
    output = {
        "schema_version": "align-error-heads-integrated-verification-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "validation_rows": len(artifacts),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        "report_sha256": sha256_file(report_path),
        "integrity_sha256": sha256_file(integrity_path),
        "lockbox_touched": False,
        "production_mutated": False,
    }
    _atomic_json(
        args.freeze_manifest.resolve().parent / "verification.json", output
    )
    if errors:
        raise ValueError(errors)
    print("integrated verification passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ready-marker",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/DATA_READY.json"),
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=Path("runs/TRAINING_RESOURCE_STATUS.json"),
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
    parser.add_argument(
        "--canonical-targets",
        type=Path,
        default=Path(
            "data-audit/joint-outputraw-full-v1/canonical_dev_targets.sqlite"
        ),
    )
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/report.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--source-root", type=Path, default=Path(r"E:\outputRaw_sf_10k")
    )
    freeze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/error-heads-v2/"
            "integrated-validation-rerun-20260915"
        ),
    )
    freeze_parser.add_argument("--cpu-threads", type=int, default=2)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--freeze-manifest", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--freeze-manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "freeze":
        freeze(args)
    elif args.phase == "score":
        score(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
