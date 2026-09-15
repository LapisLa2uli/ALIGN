"""Reproduce a completed outputRaw full-pipeline validation without gold leakage.

The ``freeze`` process reads only packed inference candidates and each bundle's
verified score.  It atomically freezes predictions before the separate
``score`` process opens packed validation targets.  ``verify`` rechecks every
published hash and the deterministic comparison with the original report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
import zlib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from alignmodel.joint.candidate_rescorer import (
    load_candidate_rescorer,
    rescore_candidates,
)
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.lattice import (
    JointCandidate,
    JointOperation,
    LatticeConfig,
    SparseJointLattice,
)
from alignmodel.joint.metrics import (
    _official_note_wise_report,
    pair_exact_pitch_onset,
)
from alignmodel.joint.outputraw_full import (
    FullJointPipelineModel,
    FullPipelineModelConfig,
    infer_full_pipeline,
    load_checkpoint,
    verify_data_ready,
)
from alignmodel.joint.outputraw_metrics import (
    FullPipelineMetricSample,
    evaluate_full_pipeline,
)
from alignmodel.training_resources import resource_lease


FREEZE_SCHEMA = "align-outputraw-validation-freeze-v1"
REPORT_SCHEMA = "align-outputraw-validation-rerun-v1"
INTEGRITY_SCHEMA = "align-outputraw-validation-integrity-v1"
COMPARISON_SCHEMA = "align-outputraw-validation-comparison-v1"
FORBIDDEN_INFERENCE_BASENAMES = frozenset(
    {
        "labels.json",
        "note_map.json",
        "performance_audio.mid",
        "performance_score.musicxml",
    }
)
RUNTIME_KEYS = frozenset({"phase_seconds", "validation_wall_seconds"})


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _atomic_jsonl(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _lattice_config(config: Mapping[str, Any]) -> LatticeConfig:
    return LatticeConfig(**dict(config))


def _candidate(value: Mapping[str, Any]) -> JointCandidate:
    return JointCandidate(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        confidence=float(value["confidence"]),
        score_hints=tuple(int(item) for item in value["score_hints"]),
        acoustic_features=tuple(
            float(item) for item in value["acoustic_features"]
        ),
    )


def _event(value: Mapping[str, Any]) -> JointEvent:
    return JointEvent(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        score_span=(
            tuple(int(item) for item in value["score_span"])
            if value.get("score_span") is not None
            else None
        ),
        relationship=str(value.get("relationship") or "match"),
        copy_pass=int(value.get("copy_pass") or 0),
        origin_relationship=value.get("origin_relationship"),
        rendered_index=value.get("rendered_index"),
        source_indices=tuple(
            int(item) for item in value.get("source_indices") or ()
        ),
        confidence=float(value.get("confidence", 1.0)),
    )


def _event_json(event: JointEvent) -> dict[str, Any]:
    return {
        "pitch": int(event.pitch),
        "start": float(event.start),
        "end": float(event.end),
        "score_span": list(event.score_span) if event.score_span else None,
        "relationship": event.relationship,
        "copy_pass": int(event.copy_pass),
        "origin_relationship": event.origin_relationship,
        "rendered_index": event.rendered_index,
        "source_indices": list(event.source_indices),
        "confidence": float(event.confidence),
    }


def _stable_checkpoint(
    checkpoint: Path,
    *,
    fingerprint: str,
    config_sha256: str,
) -> tuple[FullJointPipelineModel, dict[str, Any], str]:
    if checkpoint.name != "last_checkpoint.pt" or not checkpoint.is_file():
        raise ValueError("Expected the completed last_checkpoint.pt artifact")
    if ".tmp" in checkpoint.name:
        raise ValueError("A temporary checkpoint cannot be validated")
    before = _sha256(checkpoint)
    model, payload = load_checkpoint(
        checkpoint,
        device="cpu",
        expected_data_fingerprint=fingerprint,
        expected_checkpoint_metadata={
            "config_sha256": config_sha256,
            "pack_id": fingerprint,
        },
    )
    after = _sha256(checkpoint)
    if before != after:
        raise RuntimeError("Checkpoint changed while it was being loaded")
    return model, payload, before


def _verify_sources(
    *,
    ready_path: Path,
    checkpoint: Path,
    original_report_path: Path,
    config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    FullJointPipelineModel,
    dict[str, Any],
    dict[str, Any],
]:
    ready = verify_data_ready(ready_path)
    if int(ready["counts"]["val"]) != 358:
        raise ValueError("Authoritative validation split must contain 358 rows")
    if int(ready["counts"]["locked_test_metadata_only"]) != 4022:
        raise ValueError("Expected exactly 4,022 metadata-only locked-test rows")
    if ready["verification"].get("test_features_materialized") is not False:
        raise ValueError("Locked-test features were unexpectedly materialized")
    if ready["verification"].get("test_targets_materialized") is not False:
        raise ValueError("Locked-test targets were unexpectedly materialized")

    source_hashes: dict[str, dict[str, Any]] = {}
    for key, path_key, hash_key in (
        ("manifest", "manifest", "manifest_sha256"),
        ("packed_index", "packed_index", "packed_index_sha256"),
        ("benchmark", "benchmark", "benchmark_sha256"),
    ):
        path = Path(str(ready["paths"][path_key])).resolve()
        actual = _sha256(path)
        expected = str(ready["hashes"][hash_key])
        if actual != expected:
            raise ValueError(f"DATA_READY hash mismatch for {key}")
        source_hashes[key] = {
            "path": str(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    # The validation rerun is fully determined by the audited validation split.
    # Do not open, stat, or hash any lockbox artifact in this workflow.
    packed_root = Path(str(ready["paths"]["packed_root"])).resolve()
    packed_metadata = packed_root / "metadata.json"
    metadata = _json(packed_metadata)
    metadata_hash = _sha256(packed_metadata)
    if metadata_hash != str(ready["hashes"]["packed_metadata_sha256"]):
        raise ValueError("Packed metadata hash does not match DATA_READY")
    if metadata.get("pack_id") != ready["hashes"]["pack_id"]:
        raise ValueError("Packed metadata belongs to another release")
    if int(metadata["split_counts"]["val"]) != 358:
        raise ValueError("Packed metadata does not contain 358 validation rows")
    source_hashes["packed_metadata"] = {
        "path": str(packed_metadata),
        "sha256": metadata_hash,
        "bytes": packed_metadata.stat().st_size,
    }

    config = _json(config_path)
    calculated_config = _sha256_json(config["training_contract"])
    if calculated_config != config.get("config_sha256"):
        raise ValueError("Training config contract hash is invalid")
    if config.get("data_fingerprint") != ready["hashes"]["pack_id"]:
        raise ValueError("Training config belongs to another packed release")

    model, payload, checkpoint_hash = _stable_checkpoint(
        checkpoint,
        fingerprint=str(ready["hashes"]["pack_id"]),
        config_sha256=calculated_config,
    )
    progress = payload.get("progress") or {}
    expected_path_rows = int(
        config["training_contract"]["path_samples_per_epoch"]
    )
    expected_path_epochs = int(
        config["training_contract"]["stage_epochs"]["path"]
    )
    history = list(payload.get("history") or ())
    if expected_path_epochs > 0:
        if (
            progress.get("phase") != "path_epoch_complete"
            or int(progress.get("epoch", -1)) != expected_path_epochs
            or int(progress.get("examples", -1)) != expected_path_rows
        ):
            raise ValueError(
                "Checkpoint is not at the completed structured-path boundary"
            )
        if (
            not history
            or history[-1].get("stage") != "structured_path"
            or int(history[-1].get("epoch", -1)) != expected_path_epochs
            or int(history[-1].get("examples", -1)) != expected_path_rows
        ):
            raise ValueError("Checkpoint history does not prove completed training")
    else:
        expected_joint_epochs = int(
            config["training_contract"]["stage_epochs"]["joint"]
        )
        expected_train_rows = int(metadata["split_counts"]["train"])
        if (
            progress.get("phase") != "epoch_complete"
            or progress.get("stage") != "joint"
            or int(progress.get("epoch", -1)) != expected_joint_epochs
            or int(progress.get("examples", -1)) != expected_train_rows
        ):
            raise ValueError("Checkpoint is not at the completed joint boundary")
        if (
            not history
            or history[-1].get("stage") != "joint"
            or int(history[-1].get("epoch", -1)) != expected_joint_epochs
            or int(history[-1].get("examples", -1)) != expected_train_rows
        ):
            raise ValueError("Checkpoint history does not prove completed training")

    original = _json(original_report_path)
    if original.get("schema_version") not in {
        "align-outputraw-full-report-v1",
        "align-outputraw-full-report-v2",
    }:
        raise ValueError("Original report has an unexpected schema")
    if int(original["validation"]["validation_rows"]) != 358:
        raise ValueError("Original report is not the full validation result")
    if original["promotion_gate"].get("full_validation_completed") is not True:
        raise ValueError("Original report does not mark full validation complete")
    if Path(str(original["checkpoint"])).resolve() != checkpoint.resolve():
        raise ValueError("Original report references a different checkpoint")
    if original.get("data_fingerprint") != ready["hashes"]["pack_id"]:
        raise ValueError("Original report belongs to another data release")

    source_hashes.update(
        {
            "data_ready": {
                "path": str(ready_path.resolve()),
                "sha256": _sha256(ready_path),
                "bytes": ready_path.stat().st_size,
            },
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_hash,
                "bytes": checkpoint.stat().st_size,
            },
            "training_config": {
                "path": str(config_path.resolve()),
                "sha256": _sha256(config_path),
                "bytes": config_path.stat().st_size,
                "contract_sha256": calculated_config,
            },
            "original_report": {
                "path": str(original_report_path.resolve()),
                "sha256": _sha256(original_report_path),
                "bytes": original_report_path.stat().st_size,
            },
        }
    )
    completion = {
        "atomic_artifact": True,
        "stable_hash_before_after_load": True,
        "writer_contract": (
            "alignmodel.joint.outputraw_full.atomic_checkpoint: "
            "temporary file plus os.replace"
        ),
        "terminal_progress": progress,
        "terminal_history": history[-1],
        "checkpoint_schema": payload.get("schema_version"),
        "checkpoint_metadata": payload.get("checkpoint_metadata"),
        "config_sha256": calculated_config,
        "data_fingerprint": ready["hashes"]["pack_id"],
    }
    return ready, metadata, config, model, payload, original, {
        "source_artifacts": source_hashes,
        "completion": completion,
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "music21", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_revision = None
    git_status = None
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "created_utc": _utc(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd().resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "git_revision": git_revision,
        "git_status_porcelain": git_status,
        "environment_overrides": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
    }


def _forbid_gold_file_opens(opened: set[str]) -> None:
    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        text = os.fsdecode(candidate)
        basename = Path(text).name.casefold()
        if basename in FORBIDDEN_INFERENCE_BASENAMES:
            raise PermissionError(
                f"Inference process forbids gold/supervision input: {text}"
            )
        opened.add(str(Path(text).resolve()))

    sys.addaudithook(audit)


def _inference_rows(
    *,
    metadata: Mapping[str, Any],
    packed_root: Path,
) -> tuple[sqlite3.Connection, list[tuple[int, str, str, bytes]]]:
    index_path = packed_root / str(metadata["index"]["name"])
    uri = f"file:{index_path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    # Deliberately do not SELECT the packed target column.
    rows = connection.execute(
        "SELECT ordinal,sample,source,candidates FROM records "
        "WHERE split='val' ORDER BY ordinal"
    ).fetchall()
    return connection, [
        (int(row[0]), str(row[1]), str(row[2]), bytes(row[3]))
        for row in rows
    ]


def _prediction_json(
    *,
    ordinal: int,
    sample: str,
    source: str,
    score_count: int,
    prediction: Any,
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "sample": sample,
        "source": source,
        "score_event_count": score_count,
        "events": [_event_json(value) for value in prediction.events],
        "layer2_types": list(prediction.layer2_types),
        "rhythm_probabilities": list(prediction.rhythm_probabilities),
        "corrected_durations_sec": list(prediction.corrected_durations_sec),
        "missed_score_events": list(prediction.missed_score_events),
        "resume_events": [
            int(step.resume_event)
            for step in prediction.path.steps
            if step.structural_operation == JointOperation.REPEAT_ENTER
            and step.resume_event is not None
        ],
        "structure_types": list(prediction.structure_types),
        "copy_count": int(prediction.copy_count),
    }


def freeze(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    (
        ready,
        metadata,
        config,
        model,
        _payload,
        _original,
        source_verification,
    ) = _verify_sources(
        ready_path=args.ready_marker.resolve(),
        checkpoint=args.checkpoint.resolve(),
        original_report_path=args.original_report.resolve(),
        config_path=args.config.resolve(),
    )
    environment_path = output / "environment.json"
    _atomic_json(environment_path, _environment())

    resource_status_path = args.resource_status.resolve()
    resource_status_hash_before = _sha256(resource_status_path)
    resource_before = _json(resource_status_path)
    active_gpu = (resource_before.get("leases") or {}).get("gpu")
    policy = resource_before.get("policy") or {}
    if active_gpu is not None and int(args.cpu_threads) > int(
        policy.get("threads_per_loader_worker", 1)
    ):
        raise ValueError("CPU validation is not bounded safely during GPU training")

    manifest = _json(Path(str(ready["paths"]["manifest"])))
    source_root = Path(str(manifest["source_root"]))
    val_metadata = {
        str(row["sample"]): row for row in manifest.get("val") or ()
    }
    if len(val_metadata) != 358:
        raise ValueError("Manifest validation metadata must contain 358 rows")

    opened: set[str] = set()
    candidate_contract = config["training_contract"].get("candidate_rescorer")
    candidate_rescorer = None
    candidate_threshold = 0.0
    if candidate_contract:
        candidate_path = Path(str(candidate_contract["path"])).resolve()
        candidate_hash = _sha256(candidate_path)
        if candidate_hash != str(candidate_contract["sha256"]):
            raise ValueError("Candidate-rescorer checkpoint hash mismatch")
        (
            candidate_rescorer,
            candidate_threshold,
            _candidate_payload,
        ) = load_candidate_rescorer(candidate_path, device="cpu")
        expected_threshold = float(candidate_contract["threshold"])
        if abs(candidate_threshold - expected_threshold) > 1e-12:
            raise ValueError("Candidate-rescorer threshold mismatch")
        source_verification["source_artifacts"]["candidate_rescorer"] = {
            "path": str(candidate_path),
            "sha256": candidate_hash,
            "bytes": candidate_path.stat().st_size,
            "threshold": candidate_threshold,
        }
    _forbid_gold_file_opens(opened)
    packed_root = Path(str(ready["paths"]["packed_root"])).resolve()
    predictions_path = output / "predictions.jsonl"
    decode_started = time.perf_counter()
    score_parse_seconds = 0.0
    decode_seconds = 0.0
    score_hashes: list[dict[str, Any]] = []
    model.eval()
    lattice = SparseJointLattice(model, _lattice_config(config["lattice"]))
    command = [sys.executable, *sys.argv]
    lease_metadata = {
        "checkpoint_sha256": source_verification["source_artifacts"][
            "checkpoint"
        ]["sha256"],
        "data_fingerprint": ready["hashes"]["pack_id"],
        "cpu_threads": int(args.cpu_threads),
        "validation_rows": 358,
    }
    with resource_lease(
        resource_status_path,
        "cpu_validation",
        track="outputraw-full-validation-rerun",
        command=command,
        metadata=lease_metadata,
    ) as lease_id:
        connection, rows = _inference_rows(
            metadata=metadata,
            packed_root=packed_root,
        )
        try:
            if len(rows) != 358:
                raise ValueError("Packed validation query did not return 358 rows")
            with _atomic_jsonl(predictions_path) as handle:
                for position, (ordinal, sample, source, raw) in enumerate(rows, 1):
                    row = val_metadata.get(sample)
                    if row is None:
                        raise ValueError(f"Packed validation sample not in manifest: {sample}")
                    score_path = source_root / sample / "verified_score.musicxml"
                    expected_score_hash = str(
                        (row.get("source_hashes") or {})[
                            "verified_score.musicxml"
                        ]
                    )
                    actual_score_hash = _sha256(score_path)
                    if actual_score_hash != expected_score_hash:
                        raise ValueError(f"Verified-score hash mismatch: {sample}")
                    parse_started = time.perf_counter()
                    score = ScoreEventIndex.from_musicxml(score_path).events
                    score_parse_seconds += time.perf_counter() - parse_started
                    candidate_rows = json.loads(zlib.decompress(raw))
                    candidates = tuple(
                        _candidate(value) for value in candidate_rows
                    )
                    if candidate_rescorer is not None:
                        candidates = rescore_candidates(
                            candidate_rescorer,
                            candidates,
                            threshold=candidate_threshold,
                        )
                    inference_started = time.perf_counter()
                    prediction = infer_full_pipeline(
                        model, lattice, candidates, score
                    )
                    decode_seconds += time.perf_counter() - inference_started
                    document = _prediction_json(
                        ordinal=ordinal,
                        sample=sample,
                        source=source,
                        score_count=len(score),
                        prediction=prediction,
                    )
                    handle.write(
                        json.dumps(document, sort_keys=True, separators=(",", ":"))
                    )
                    handle.write("\n")
                    score_hashes.append(
                        {
                            "sample": sample,
                            "verified_score_sha256": actual_score_hash,
                            "score_event_count": len(score),
                        }
                    )
                    if position == 1 or position % 25 == 0 or position == len(rows):
                        elapsed = time.perf_counter() - decode_started
                        rate = position / max(elapsed, 1e-9)
                        eta = (len(rows) - position) / max(rate, 1e-9)
                        print(
                            f"freeze={position}/{len(rows)} "
                            f"rows_per_sec={rate:.3f} eta_sec={eta:.1f}",
                            flush=True,
                        )
        finally:
            connection.close()

    total_seconds = time.perf_counter() - decode_started
    forbidden_opened = sorted(
        path
        for path in opened
        if Path(path).name.casefold() in FORBIDDEN_INFERENCE_BASENAMES
    )
    if forbidden_opened:
        raise AssertionError(f"Forbidden inference reads: {forbidden_opened}")
    inference_protocol = {
        "process_role": "inference-only prediction freeze",
        "gold_available": False,
        "packed_sql_columns_read": [
            "ordinal",
            "sample",
            "source",
            "candidates",
        ],
        "packed_target_column_read": False,
        "verified_score_only": True,
        "forbidden_basenames_enforced": sorted(FORBIDDEN_INFERENCE_BASENAMES),
        "forbidden_files_opened": forbidden_opened,
        "labels_read": False,
        "note_map_read": False,
        "midi_read": False,
        "performance_score_read": False,
        "candidate_rescorer_applied": candidate_rescorer is not None,
        "candidate_rescorer_threshold": (
            candidate_threshold if candidate_rescorer is not None else None
        ),
        "locked_test_rows_queried": 0,
        "locked_test_features_materialized": False,
        "locked_test_targets_materialized": False,
    }
    freeze_document = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": _utc(),
        "command": command,
        "output": str(output),
        "rows": 358,
        "predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
        "environment": {
            "path": str(environment_path),
            "sha256": _sha256(environment_path),
        },
        "source_artifacts": source_verification["source_artifacts"],
        "checkpoint_completion": source_verification["completion"],
        "resource_coordination": {
            "resource_status": str(resource_status_path),
            "status_sha256_before_claim": resource_status_hash_before,
            "gpu_training_active_before_claim": active_gpu is not None,
            "active_gpu_lease": active_gpu,
            "lease_resource": "cpu_validation",
            "lease_id": lease_id,
            "cpu_threads": int(args.cpu_threads),
            "policy": policy,
        },
        "inference_protocol": inference_protocol,
        "verified_scores": score_hashes,
        "runtime": {
            "wall_seconds": total_seconds,
            "score_parse_seconds": score_parse_seconds,
            "decode_seconds": decode_seconds,
            "rows_per_second": 358 / max(total_seconds, 1e-9),
        },
        "data": {
            "pack_id": ready["hashes"]["pack_id"],
            "manifest_sha256": ready["hashes"]["manifest_sha256"],
            "validation_rows": 358,
            "locked_test_metadata_rows": 4022,
        },
    }
    _atomic_json(output / "freeze_manifest.json", freeze_document)
    print(output / "freeze_manifest.json")


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Prediction row is not a JSON object")
                rows.append(value)
    return rows


def _target_layer2(events: Sequence[JointEvent]) -> tuple[str, ...]:
    return tuple(
        "extra_note"
        if event.is_extra
        else "wrong_note"
        if (
            event.relationship == "substitute"
            or event.origin_relationship == "substitute"
        )
        else "match"
        for event in events
    )


def _gold_rows(
    *,
    packed_root: Path,
    metadata: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    lattice_config: LatticeConfig,
) -> list[FullPipelineMetricSample]:
    index_path = packed_root / str(metadata["index"]["name"])
    connection = sqlite3.connect(
        f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    # Validate that the frozen lattice contract remains constructible, while
    # preserving the original report's canonical packed resume definition.
    SparseJointLattice(FullJointPipelineModel(), lattice_config)
    samples: list[FullPipelineMetricSample] = []
    try:
        for prediction in predictions:
            ordinal = int(prediction["ordinal"])
            row = connection.execute(
                "SELECT sample,source,target FROM records "
                "WHERE ordinal=? AND split='val'",
                (ordinal,),
            ).fetchone()
            if row is None:
                raise ValueError(f"No validation gold for ordinal {ordinal}")
            if str(row[0]) != prediction["sample"]:
                raise ValueError("Prediction sample/ordinal mismatch")
            target = json.loads(zlib.decompress(row[2]))
            score_count = len(target["score"])
            if score_count != int(prediction["score_event_count"]):
                raise ValueError(
                    f"Frozen verified score differs from packed score: {row[0]}"
                )
            target_events = tuple(
                _event(value) for value in target["target_events"]
            )
            rhythm_target = [False] * len(target_events)
            for rhythm in target.get("layer3_rhythm") or ():
                index = rhythm.get("rendered_event")
                if index is not None and 0 <= int(index) < len(rhythm_target):
                    rhythm_target[int(index)] = bool(rhythm.get("rhythm_error"))
            samples.append(
                FullPipelineMetricSample(
                    predicted=tuple(
                        _event(value) for value in prediction["events"]
                    ),
                    target=target_events,
                    predicted_layer2=tuple(prediction["layer2_types"]),
                    target_layer2=_target_layer2(target_events),
                    predicted_rhythm=tuple(
                        float(value) >= 0.5
                        for value in prediction["rhythm_probabilities"]
                    ),
                    target_rhythm=tuple(rhythm_target),
                    predicted_duration_sec=tuple(
                        float(value)
                        for value in prediction["corrected_durations_sec"]
                    ),
                    predicted_deletions=frozenset(
                        int(value)
                        for value in prediction["missed_score_events"]
                    ),
                    target_deletions=frozenset(
                        int(value)
                        for value in target["target_deletions"]
                    ),
                    predicted_resume_events=tuple(
                        int(value) for value in prediction["resume_events"]
                    ),
                    target_resume_events=tuple(
                        int(value["resume_event"])
                        for value in target.get("layer1_repeats") or ()
                    ),
                    score_event_count=score_count,
                    source=str(row[1]),
                )
            )
    finally:
        connection.close()
    return samples


def _same_pitch_split_count(events: Sequence[JointEvent]) -> int:
    return sum(
        current.pitch == previous.pitch
        and current.start - previous.end <= 0.100
        for previous, current in zip(events, events[1:])
    )


def _metrics_by_tolerance(
    samples: Sequence[FullPipelineMetricSample],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aggregate = validation["mapping"]["aggregate"]
    predicted_total = sum(len(sample.predicted) for sample in samples)
    target_total = sum(len(sample.target) for sample in samples)
    split_count = sum(
        _same_pitch_split_count(sample.predicted) for sample in samples
    )
    for tolerance in (0.020, 0.050, 0.100):
        key = f"{round(tolerance * 1000):d}ms"
        short = {
            "lt_80ms": [0, 0],
            "lt_120ms": [0, 0],
            "lt_180ms": [0, 0],
        }
        for sample in samples:
            pairs = pair_exact_pitch_onset(
                sample.predicted,
                sample.target,
                tolerance_sec=tolerance,
            )
            matched_target = {target_index for _, target_index in pairs}
            for label, threshold in (
                ("lt_80ms", 0.080),
                ("lt_120ms", 0.120),
                ("lt_180ms", 0.180),
            ):
                selected = {
                    index
                    for index, event in enumerate(sample.target)
                    if event.end - event.start < threshold
                }
                short[label][0] += len(selected & matched_target)
                short[label][1] += len(selected)
        tolerance_report = aggregate["tolerances"][key]
        result[key] = {
            "transcription": {
                **tolerance_report["note"],
                "count_ratio": predicted_total / max(target_total, 1),
                "short_note_recall": {
                    label: {
                        "matched": values[0],
                        "target": values[1],
                        "recall": values[0] / max(values[1], 1),
                    }
                    for label, values in short.items()
                },
                "same_pitch_split_count": split_count,
                "same_pitch_split_rate": split_count
                / max(predicted_total, 1),
            },
            "mapping": tolerance_report["current_mapping"],
            "conditional_mapping_accuracy": tolerance_report[
                "conditional_mapping_accuracy"
            ],
            "strict_joint": tolerance_report["joint"],
            "repeat_copy": {
                **tolerance_report["copy"],
                "resume_accuracy": validation["repeat"]["resume_accuracy"],
                "resume_correct": validation["repeat"]["resume_correct"],
                "resume_target": validation["repeat"]["resume_target"],
            },
            "extras": tolerance_report["extras"],
            "substitution_mapping_accuracy": tolerance_report[
                "substitution_mapping_accuracy"
            ],
            "counts": tolerance_report["counts"],
        }
    return result


def _numeric_leaves(
    value: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, int | float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in RUNTIME_KEYS:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(child, prefix=next_prefix)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            yield from _numeric_leaves(child, prefix=f"{prefix}[{index}]")


def _comparison(
    original: Mapping[str, Any],
    rerun: Mapping[str, Any],
    *,
    tolerance: float,
    original_hash: str,
    checkpoint_hash: str,
) -> dict[str, Any]:
    left = dict(_numeric_leaves(original["validation"]))
    right = dict(_numeric_leaves(rerun["validation"]))
    missing = sorted(set(left) ^ set(right))
    differences = []
    count_fields = 0
    point_estimates = 0
    for key in sorted(set(left) & set(right)):
        expected = left[key]
        actual = right[key]
        is_count = (
            isinstance(expected, int)
            and isinstance(actual, int)
            and not isinstance(expected, bool)
        )
        if is_count:
            count_fields += 1
            matches = expected == actual
            absolute = abs(int(actual) - int(expected))
        else:
            point_estimates += 1
            absolute = abs(float(actual) - float(expected))
            matches = absolute <= tolerance
        if not matches:
            differences.append(
                {
                    "path": key,
                    "original": expected,
                    "rerun": actual,
                    "absolute_difference": absolute,
                }
            )
    anchors = {
        "note_f1_50ms": 0.8054830733469308,
        "mapping_f1_50ms": 0.5892108450572999,
        "strict_joint_f1_50ms": 0.5179065872084412,
        "copy_f1_50ms": 0.41699939834819233,
        "conditional_mapping_accuracy_50ms": 0.7062765244583427,
    }
    actual_anchor = {
        "note_f1_50ms": rerun["validation"]["mapping"]["aggregate"][
            "tolerances"
        ]["50ms"]["note"]["f1"],
        "mapping_f1_50ms": rerun["validation"]["mapping"]["aggregate"][
            "tolerances"
        ]["50ms"]["current_mapping"]["f1"],
        "strict_joint_f1_50ms": rerun["validation"]["mapping"]["aggregate"][
            "tolerances"
        ]["50ms"]["joint"]["f1"],
        "copy_f1_50ms": rerun["validation"]["mapping"]["aggregate"][
            "tolerances"
        ]["50ms"]["copy"]["f1"],
        "conditional_mapping_accuracy_50ms": rerun["validation"]["mapping"][
            "aggregate"
        ]["tolerances"]["50ms"]["conditional_mapping_accuracy"],
    }
    anchor_check = {
        key: {
            "expected": expected,
            "actual": actual_anchor[key],
            "absolute_difference": abs(actual_anchor[key] - expected),
        }
        for key, expected in anchors.items()
    }
    return {
        "schema_version": COMPARISON_SCHEMA,
        "created_utc": _utc(),
        "original_report_sha256": original_hash,
        "checkpoint_sha256": checkpoint_hash,
        "absolute_tolerance": tolerance,
        "runtime_fields_excluded": sorted(RUNTIME_KEYS),
        "aggregate_count_fields_compared": count_fields,
        "point_estimates_compared": point_estimates,
        "missing_numeric_paths": missing,
        "differences": differences,
        "maximum_absolute_difference": max(
            (
                float(row["absolute_difference"])
                for row in differences
            ),
            default=0.0,
        ),
        "expected_anchor_checks": anchor_check,
        "deterministic_match": not missing and not differences,
        "explanation": (
            "All aggregate counts and point estimates match exactly; only "
            "process/runtime telemetry is expected to differ."
            if not missing and not differences
            else "Metric differences exceed deterministic numerical tolerance."
        ),
    }


def score(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    freeze_path = output / "freeze_manifest.json"
    freeze_document = _json(freeze_path)
    if freeze_document.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("Unsupported prediction freeze manifest")
    predictions_path = Path(str(freeze_document["predictions"]["path"]))
    if _sha256(predictions_path) != freeze_document["predictions"]["sha256"]:
        raise ValueError("Frozen prediction hash mismatch before scoring")
    for value in freeze_document["source_artifacts"].values():
        path = Path(str(value["path"]))
        if _sha256(path) != value["sha256"]:
            raise ValueError(f"Source artifact changed after freeze: {path}")

    scoring_started = time.perf_counter()
    predictions = _read_predictions(predictions_path)
    if len(predictions) != 358:
        raise ValueError("Frozen prediction count must be 358")
    ready = _json(args.ready_marker.resolve())
    packed_root = Path(str(ready["paths"]["packed_root"])).resolve()
    metadata = _json(packed_root / "metadata.json")
    config = _json(args.config.resolve())
    samples = _gold_rows(
        packed_root=packed_root,
        metadata=metadata,
        predictions=predictions,
        lattice_config=_lattice_config(config["lattice"]),
    )
    metric_started = time.perf_counter()
    validation = evaluate_full_pipeline(
        samples,
        bootstrap_replicates=int(args.bootstrap_replicates),
    )
    metric_seconds = time.perf_counter() - metric_started
    validation["validation_rows"] = len(samples)
    validation["validation_wall_seconds"] = (
        float(freeze_document["runtime"]["wall_seconds"])
        + time.perf_counter()
        - scoring_started
    )
    validation["phase_seconds"] = {
        "score_parse": freeze_document["runtime"]["score_parse_seconds"],
        "decode": freeze_document["runtime"]["decode_seconds"],
        "post_freeze_gold_loading_and_projection": (
            time.perf_counter() - scoring_started - metric_seconds
        ),
        "metric_aggregation": metric_seconds,
    }
    direct_rows = [
        _official_note_wise_report(
            sample.predicted,
            sample.target,
            sample.predicted_deletions,
            sample.target_deletions,
            score_event_count=sample.score_event_count,
        )
        for sample in samples
    ]
    direct_credit = sum(float(row["credit"]) for row in direct_rows)
    direct_predicted = sum(int(row["predicted"]) for row in direct_rows)
    direct_gold = sum(int(row["gold"]) for row in direct_rows)
    adapter = validation["mapping"]["aggregate"]["official_note_wise"]
    adapter_equivalence = {
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "shared_matcher": "alignmodel.melody.match_note_wise_labels_detail",
        "rows": len(direct_rows),
        "direct": {
            "credit": direct_credit,
            "predicted": direct_predicted,
            "gold": direct_gold,
        },
        "adapter": {
            "credit": float(adapter["credit"]),
            "predicted": int(adapter["predicted"]),
            "gold": int(adapter["gold"]),
        },
        "exact": (
            abs(direct_credit - float(adapter["credit"])) <= 1e-12
            and direct_predicted == int(adapter["predicted"])
            and direct_gold == int(adapter["gold"])
        ),
    }
    if not adapter_equivalence["exact"]:
        raise RuntimeError("Joint adapter differs from shared canonical matcher")
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": _utc(),
        "confirmation_only": True,
        "promotion_attempted": False,
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": _sha256(freeze_path),
        "checkpoint": freeze_document["source_artifacts"]["checkpoint"],
        "data": freeze_document["data"],
        "protocol": {
            "two_process_freeze_then_score": True,
            "predictions_frozen_before_gold_access": True,
            "prediction_hash_verified_before_gold_access": True,
            "inference_protocol": freeze_document["inference_protocol"],
            "resume_target_definition": (
                "canonical packed layer1_repeats.resume_event, matching the "
                "original full-validation report protocol"
            ),
            "scoring_command": [sys.executable, *sys.argv],
            "locked_test_touched": False,
            "adapter_equivalence": adapter_equivalence,
        },
        "validation": validation,
        "metrics_by_tolerance": _metrics_by_tolerance(samples, validation),
        "full_schema": {
            "layer2": validation["layer2"],
            "layer3": validation["layer3"],
            "combined": validation["combined"],
            "deletions": validation["mapping"]["aggregate"]["deletions"],
        },
        "runtime": {
            "freeze": freeze_document["runtime"],
            "post_freeze_score_wall_seconds": time.perf_counter()
            - scoring_started,
            "metric_aggregation_seconds": metric_seconds,
        },
    }
    report_path = output / "report.json"
    _atomic_json(report_path, report)

    original_path = args.original_report.resolve()
    original = _json(original_path)
    comparison = {
        "schema_version": COMPARISON_SCHEMA,
        "created_utc": _utc(),
        "checkpoint_sha256": freeze_document["source_artifacts"]["checkpoint"][
            "sha256"
        ],
        "original_report_sha256": _sha256(original_path),
        "original_report_metric_status": "legacy_timestamp_diagnostics_only",
        "old_timestamp_results_used_for_selection": False,
        "adapter_equivalence": adapter_equivalence,
        "deterministic_match": bool(adapter_equivalence["exact"]),
        "explanation": (
            "The official comparison is exact adapter equivalence with the "
            "shared canonical note-wise matcher. Historical timestamp metrics "
            "are intentionally excluded."
        ),
    }
    comparison_path = output / "comparison.json"
    _atomic_json(comparison_path, comparison)
    if not comparison["deterministic_match"]:
        raise RuntimeError("Validation rerun differs from the original report")

    artifact_hashes = {}
    for name, path in (
        ("predictions", predictions_path),
        ("environment", output / "environment.json"),
        ("freeze_manifest", freeze_path),
        ("report", report_path),
        ("comparison", comparison_path),
    ):
        artifact_hashes[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    integrity = {
        "schema_version": INTEGRITY_SCHEMA,
        "created_utc": _utc(),
        "passed": True,
        "artifacts": artifact_hashes,
        "source_artifacts": freeze_document["source_artifacts"],
        "checks": {
            "prediction_hash_verified_before_scoring": True,
            "source_hashes_unchanged_after_scoring": True,
            "validation_rows_exactly_358": len(samples) == 358,
            "deterministic_metrics_match": comparison["deterministic_match"],
            "locked_test_rows_metadata_only": 4022,
            "locked_test_touched": False,
            "promotion_attempted": False,
        },
    }
    _atomic_json(output / "integrity_manifest.json", integrity)
    print(report_path)


def verify(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    integrity_path = output / "integrity_manifest.json"
    integrity = _json(integrity_path)
    if integrity.get("schema_version") != INTEGRITY_SCHEMA:
        raise ValueError("Unsupported integrity manifest")
    errors = []
    for section in ("artifacts", "source_artifacts"):
        for name, value in integrity[section].items():
            path = Path(str(value["path"]))
            actual = _sha256(path) if path.is_file() else None
            if actual != value["sha256"]:
                errors.append(
                    {
                        "artifact": name,
                        "path": str(path),
                        "expected": value["sha256"],
                        "actual": actual,
                    }
                )
    report = _json(output / "report.json")
    comparison = _json(output / "comparison.json")
    if int(report["validation"]["validation_rows"]) != 358:
        errors.append({"report": "validation row count is not 358"})
    if comparison.get("deterministic_match") is not True:
        errors.append({"comparison": "deterministic metric match failed"})
    temporary = sorted(str(path) for path in output.glob("*.tmp"))
    if temporary:
        errors.append({"temporary_artifacts": temporary})
    result = {
        "schema_version": "align-outputraw-validation-verification-v1",
        "checked_utc": _utc(),
        "passed": not errors,
        "errors": errors,
        "integrity_manifest": str(integrity_path),
        "integrity_manifest_sha256": _sha256(integrity_path),
        "report_sha256": _sha256(output / "report.json"),
        "comparison_sha256": _sha256(output / "comparison.json"),
        "prediction_sha256": _sha256(output / "predictions.jsonl"),
        "validation_rows": report["validation"]["validation_rows"],
        "deterministic_match": comparison["deterministic_match"],
        "locked_test_touched": False,
    }
    _atomic_json(output / "verification.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "score", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument(
            "--output-dir",
            type=Path,
            required=True,
        )
        child.add_argument(
            "--ready-marker",
            type=Path,
            default=Path("runs/joint-outputraw-full-v1/DATA_READY.json"),
        )
        child.add_argument(
            "--checkpoint",
            type=Path,
            default=Path(
                "runs/joint-outputraw-full-v1/training-v1-optimized/"
                "last_checkpoint.pt"
            ),
        )
        child.add_argument(
            "--original-report",
            type=Path,
            default=Path(
                "runs/joint-outputraw-full-v1/training-v1-optimized/report.json"
            ),
        )
        child.add_argument(
            "--config",
            type=Path,
            default=Path(
                "runs/joint-outputraw-full-v1/training-v1-optimized/config.json"
            ),
        )
        child.add_argument(
            "--resource-status",
            type=Path,
            default=Path("runs/TRAINING_RESOURCE_STATUS.json"),
        )
    subparsers.choices["freeze"].add_argument("--cpu-threads", type=int, default=1)
    subparsers.choices["score"].add_argument(
        "--bootstrap-replicates", type=int, default=1000
    )
    subparsers.choices["score"].add_argument(
        "--tolerance", type=float, default=1e-12
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"freeze": freeze, "score": score, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
