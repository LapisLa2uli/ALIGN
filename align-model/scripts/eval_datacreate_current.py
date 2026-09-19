"""Gold-isolated evaluation of the current completed stack on DataCreate.

Run ``freeze`` and ``evaluate`` as separate processes.  The freeze phase does
not open labels, candidate documents, note maps, MIDI, prior alignments, or
audit targets.  The evaluate phase first verifies every frozen prediction
hash, then opens labels and read-only diagnostic metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from alignmodel.joint.candidates import (
    CANDIDATE_GENERATION_VERSION,
    HIGH_RECALL_DECODE_CONFIGS,
    LEGACY_HIGH_RECALL_DECODE_CONFIGS,
    add_score_repeat_hints,
    basic_pitch_candidate_union,
)
from alignmodel.joint.error_heads import (
    HeadPrediction,
    build_inference_rows,
    filter_prediction_for_schema,
    heuristic_prediction,
    infer_error_heads,
    load_error_heads,
    schema12_document,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.lattice import JointOperation, SparseJointLattice
from alignmodel.joint.train import load_joint_model
from alignmodel.melody import (
    canonical_note_location,
    gold_melodies_from_labels,
    match_melodies_detail,
    match_note_wise_labels_detail,
    parse_sounding_notes,
    pred_melodies_from_labels,
)
from alignmodel.transcription.basic_pitch import (
    BASIC_PITCH_VERSION,
    FROZEN_DECODE_CONFIG,
    decode_frozen_basic_pitch,
    extract_sample_basic_pitch_features,
    load_audio_metadata,
    load_basic_pitch_cache,
    sanitize_basic_pitch_notes,
    save_basic_pitch_cache,
)
from alignmodel.training_resources import resource_lease


FREEZE_SCHEMA = "align-datacreate-real-freeze-v1"
REPORT_SCHEMA = "align-datacreate-real-evaluation-v2"
SCORED_MODEL_TYPES = {
    "wrong_note",
    "extra_note",
    "missed_note",
    "rhythm_error",
    "repetition",
}
LAYER2_TYPES = {"wrong_note", "extra_note", "missed_note"}
RHYTHM_TYPES = {"rhythm_error"}
REPETITION_TYPES = {"repetition"}
HUMAN_SOURCES = {"manual", "auto_confirmed", "auto_edited"}
AGENT_SOURCES = {"agent"}
_NOTE_ID = re.compile(r"^note_(\d+)$")
_FREEZE_SAMPLE_INPUTS = {
    "metadata.json",
    "performance_audio.wav",
    "verified_score.musicxml",
}
_EVALUATE_SAMPLE_INPUTS = {"labels.json", "verified_score.musicxml"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(
            value,
            stream,
            indent=2,
            sort_keys=True,
            default=lambda item: (
                item.item() if isinstance(item, np.generic) else str(item)
            ),
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _audio_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
    if frames <= 0 or sample_rate <= 0:
        raise ValueError("empty or invalid WAV")
    return {
        "duration_sec": frames / sample_rate,
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "sha256": _sha256(path),
    }


def _expected_samples(root: Path) -> list[Path]:
    return [root / f"{index:03d}" for index in range(1, 94)] + [
        root / "demo_001"
    ]


def _selected_samples(root: Path, sample_id: str | None = None) -> list[Path]:
    expected = _expected_samples(root)
    if sample_id is None:
        return expected
    by_name = {path.name: path for path in expected}
    if sample_id not in by_name:
        raise ValueError(
            f"Unknown DataCreate sample id {sample_id!r}; "
            "expected 001-093 or demo_001"
        )
    return [by_name[sample_id]]


class _SampleReadGuard:
    """Fail closed if a strict phase opens an undeclared sample file."""

    def __init__(self, samples: Sequence[Path], allowed: set[str], phase: str):
        self.roots = tuple(path.resolve() for path in samples)
        self.allowed = set(allowed)
        self.phase = phase
        self.opened: set[str] = set()
        self.violations: list[str] = []

    def _sample_relative(self, value: Any) -> str | None:
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        try:
            path = Path(os.fsdecode(value)).resolve()
        except (OSError, TypeError, ValueError):
            return None
        for root in self.roots:
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
        return None

    def __call__(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not arguments:
            return
        if event in {
            "os.remove",
            "os.rename",
            "os.rmdir",
            "os.mkdir",
            "os.chmod",
            "os.utime",
        }:
            affected = [
                name
                for name in (
                    self._sample_relative(arguments[0]),
                    self._sample_relative(arguments[1])
                    if event == "os.rename" and len(arguments) > 1
                    else None,
                )
                if name is not None
            ]
            if affected:
                self.violations.extend(affected)
                raise PermissionError(
                    f"{self.phase} may not mutate sample paths {affected!r}"
                )
            return
        if event != "open":
            return
        name = self._sample_relative(arguments[0])
        if name is None:
            return
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        writing = (
            isinstance(mode, str) and any(value in mode for value in "wax+")
        ) or (
            isinstance(flags, int)
            and bool(
                flags
                & (
                    os.O_WRONLY
                    | os.O_RDWR
                    | os.O_CREAT
                    | os.O_TRUNC
                    | os.O_APPEND
                )
            )
        )
        if writing or name not in self.allowed:
            self.violations.append(name)
            action = "write" if writing else "open"
            raise PermissionError(
                f"{self.phase} may not {action} sample input {name!r}"
            )
        self.opened.add(name)


def _install_sample_read_guard(
    samples: Sequence[Path], allowed: set[str], phase: str
) -> _SampleReadGuard:
    guard = _SampleReadGuard(samples, allowed, phase)
    sys.addaudithook(guard)
    return guard


def _protected_sample_stats(sample: Path) -> dict[str, Any]:
    output = {}
    for name in ("labels.json", "note_alignment_v2.json"):
        path = sample / name
        if not path.exists():
            output[name] = {"present": False}
            continue
        stat = path.stat()
        output[name] = {
            "present": True,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return output


def _training_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    value = _json(path)
    leases = value.get("leases") or {}
    lease = leases.get("gpu")
    ownership = value.get("ownership") or {}
    active_gpu_rows = [
        row
        for row in ownership.get("gpu_reserved") or []
        if (row or {}).get("phase") == "gpu_active"
    ]
    return {
        "path": str(path.resolve()),
        "schema_version": value.get("schema_version"),
        "updated_utc": value.get("updated_utc"),
        "gpu_lease_active": bool(lease),
        "gpu_lease": lease,
        "gpu_training_active": bool(lease) or bool(active_gpu_rows),
        "active_gpu_training": active_gpu_rows,
        "leases": dict(leases),
        "ownership": ownership,
        "policy": value.get("policy"),
        "recommended_concurrency": value.get("recommended_concurrency"),
    }


def _frontend_configs(payload: Mapping[str, Any]) -> Sequence[Any]:
    frontend = (payload.get("training") or {}).get("frontend") or {}
    version = frontend.get("candidate_generation") or (
        payload.get("training") or {}
    ).get("candidate_generation")
    return (
        HIGH_RECALL_DECODE_CONFIGS
        if version == CANDIDATE_GENERATION_VERSION
        else LEGACY_HIGH_RECALL_DECODE_CONFIGS
    )


def _frontend_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    training = payload.get("training") or {}
    frontend = training.get("frontend") or {}
    return {
        "name": frontend.get("name"),
        "candidate_generation": frontend.get("candidate_generation")
        or training.get("candidate_generation"),
        "decode_configs": frontend.get("decode_configs") or [],
        "frozen": True,
    }


def _checkpoint_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    history = payload.get("history") or []
    return {
        "schema_version": payload.get("schema_version"),
        "history_epochs": len(history),
        "best_validation_present": bool(payload.get("best_validation")),
        "progress": payload.get("progress"),
    }


def _load_completed_stack(args: argparse.Namespace) -> dict[str, Any]:
    joint = args.joint_checkpoint.resolve()
    joint_report_path = joint.parent / "report.json"
    heads = args.error_heads_checkpoint.resolve()
    heads_root = heads.parents[1]
    heads_report_path = heads_root / "report.json"
    for path in (joint, joint_report_path, heads, heads_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if joint.name != "joint_decoder.pt" or "last_checkpoint" in joint.name:
        raise ValueError("Joint checkpoint is not a completed joint_decoder.pt")
    if heads.name != "best.pt" or heads.parent.name != "predicted":
        raise ValueError("Error heads must use completed predicted/best.pt")

    joint_report = _json(joint_report_path)
    heads_report = _json(heads_report_path)
    if not (heads_report.get("promotion_gate") or {}).get(
        "full_validation_completed"
    ):
        raise ValueError("Error-head v2 report does not mark full validation complete")
    if heads_report.get("schema_version") != "align-frozen-error-heads-report-v2":
        raise ValueError("Error-head report is not v2")

    joint_hash = _sha256(joint)
    expected_joint_hash = (
        (heads_report.get("evaluation") or {})
        .get("upstream", {})
        .get("decoder", {})
        .get("sha256")
    )
    if expected_joint_hash != joint_hash:
        raise ValueError("Error-head report references a different joint checkpoint")
    if joint_report.get("checkpoint_sha256") != joint_hash:
        raise ValueError("Joint checkpoint hash differs from completed report")

    joint_model, lattice_config, joint_payload = load_joint_model(
        joint, device="cpu"
    )
    completion = _checkpoint_completion(joint_payload)
    if not completion["history_epochs"] or not completion["best_validation_present"]:
        raise ValueError("Joint checkpoint has no completed validation epoch")
    for parameter in joint_model.parameters():
        parameter.requires_grad = False
    lattice = SparseJointLattice(joint_model, lattice_config)

    heads_model, heads_payload = load_error_heads(heads, device="cpu")
    head_history = heads_payload.get("history") or []
    if len(head_history) < 8:
        raise ValueError("Error-head best checkpoint lacks eight completed epochs")
    schema_thresholds = heads_payload.get("schema_thresholds")
    if not isinstance(schema_thresholds, Mapping):
        raise ValueError("Error-head checkpoint lacks v2 schema calibration")
    for parameter in heads_model.parameters():
        parameter.requires_grad = False

    frontend = _frontend_contract(joint_payload)
    frontend_hash = _canonical_hash(frontend)
    expected_frontend_hash = (
        (heads_report.get("evaluation") or {})
        .get("upstream", {})
        .get("transcriber", {})
        .get("candidate_config_sha256")
    )
    if frontend_hash != expected_frontend_hash:
        raise ValueError("Candidate configuration differs from error-head contract")

    head_upstream = heads_payload.get("upstream") or {}
    recorded_head_hash = (
        head_upstream.get("checkpoint_sha256")
        or head_upstream.get("upstream_sha256")
        or (head_upstream.get("decoder") or {}).get("sha256")
    )
    if recorded_head_hash is not None and recorded_head_hash != joint_hash:
        raise ValueError("Error-head checkpoint itself references another upstream")

    minimum_confidence = float(
        (joint_payload.get("training") or {}).get(
            "minimum_candidate_confidence", 0.65
        )
    )
    return {
        "joint_model": joint_model,
        "lattice": lattice,
        "joint_payload": joint_payload,
        "heads_model": heads_model,
        "heads_payload": heads_payload,
        "minimum_confidence": minimum_confidence,
        "selection": {
            "selection_policy": (
                "newest completed non-diagnostic joint checkpoint explicitly "
                "hash-compatible with completed predicted error-heads-v2; "
                "exclude last/active checkpoints"
            ),
            "active_checkpoint_read": False,
            "transcriber": {
                "implementation": "Basic Pitch 0.4.0 frozen frontend",
                "checkpoint": (
                    "external pretrained Basic Pitch package model; no local "
                    "trainable transcriber checkpoint"
                ),
                "basic_pitch_version": BASIC_PITCH_VERSION,
                "candidate_version": frontend["candidate_generation"],
                "candidate_config_sha256": frontend_hash,
                "config": frontend,
            },
            "joint_aligner_decoder": {
                "path": str(joint),
                "report": str(joint_report_path.resolve()),
                "checkpoint_sha256": joint_hash,
                "checkpoint_schema_version": joint_payload.get("schema_version"),
                "report_schema_version": joint_report.get("schema_version"),
                "completion": completion,
            },
            "error_heads_v2": {
                "path": str(heads),
                "report": str(heads_report_path.resolve()),
                "checkpoint_sha256": _sha256(heads),
                "checkpoint_schema_version": heads_payload.get("schema_version"),
                "report_schema_version": heads_report.get("schema_version"),
                "history_epochs": len(head_history),
                "schema_thresholds": dict(schema_thresholds),
                "compatible": True,
                "compatibility_basis": {
                    "joint_sha256": joint_hash,
                    "candidate_config_sha256": frontend_hash,
                    "data_fingerprint": heads_payload.get("data_fingerprint"),
                },
            },
            "excluded_active_training": {
                "path": str(args.active_checkpoint.resolve()),
                "reason": "last_checkpoint.pt is active/half-written by policy",
                "read": False,
            },
            "report_selection_evidence": (
                (heads_report.get("evaluation") or {})
                .get("upstream", {})
                .get("selection_evidence")
            ),
        },
    }


def _load_production_checkpoint(
    path: Path,
) -> tuple[SparseJointLattice | None, Mapping[str, Any] | None, dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        return None, None, {
            "status": "not_runnable",
            "reason": "configured production checkpoint missing",
            "path": str(path),
        }
    report_path = path.parent / "report.json"
    if not report_path.is_file() or path.name != "joint_decoder.pt":
        return None, None, {
            "status": "not_runnable",
            "reason": "checkpoint lacks completed sibling report",
            "path": str(path),
        }
    model, config, payload = load_joint_model(path, device="cpu")
    completion = _checkpoint_completion(payload)
    if not completion["history_epochs"] or not completion["best_validation_present"]:
        return None, None, {
            "status": "not_runnable",
            "reason": "checkpoint lacks completed validation",
            "path": str(path),
        }
    for parameter in model.parameters():
        parameter.requires_grad = False
    return SparseJointLattice(model, config), payload, {
        "status": "alignment_evaluated",
        "path": str(path),
        "report": str(report_path.resolve()),
        "sha256": _sha256(path),
        "checkpoint_schema_version": payload.get("schema_version"),
        "completion": completion,
        "error_label_behavior": (
            "DataCreate production bridge emits no error labels; its real-test "
            "event-detection baseline is therefore the empty prediction set"
        ),
    }


def _cache_features(
    sample: Path,
    output: Path,
    *,
    fresh_features: bool = False,
) -> tuple[Any, dict[str, Any]]:
    wav = sample / "performance_audio.wav"
    metadata = load_audio_metadata(sample)
    destination = output / "feature-cache" / f"{sample.name}.npz"
    if fresh_features:
        features = extract_sample_basic_pitch_features(
            sample, cache_path=destination, force=True
        )
        source = None
        source_kind = "fresh_audio_inference"
    else:
        source = sample / "basic_pitch_cache.npz"
        features = load_basic_pitch_cache(source, wav, metadata)
        if features is not None:
            save_basic_pitch_cache(destination, features)
            source_kind = "sample_hash_validated_cache"
        else:
            features = extract_sample_basic_pitch_features(
                sample, cache_path=destination, force=False
            )
            source_kind = "fresh_audio_inference"
    return features, {
        "source_kind": source_kind,
        "source_path": (
            str(source) if source is not None and source.is_file() else None
        ),
        "sample_feature_cache_opened": (
            source is not None and source_kind == "sample_hash_validated_cache"
        ),
        "frozen_copy": str(destination.resolve()),
        "frozen_copy_sha256": _sha256(destination),
        "metadata": dict(features.metadata),
    }


def _path_diagnostics(
    candidates: Sequence[Any],
    score: Sequence[Any],
    path: Any,
) -> dict[str, Any]:
    events = tuple(path.joint_events(candidates))
    mapped = [event for event in events if event.score_span is not None]
    deleted = set(path.trailing_deletions)
    for step in path.steps:
        deleted.update(step.deleted_events)
    score_pitches = {int(event.pitch) for event in score}
    candidate_pitch_overlap = sum(
        int(candidate.pitch) in score_pitches for candidate in candidates
    )
    operations = Counter(
        str(step.operation.value if hasattr(step.operation, "value") else step.operation)
        for step in path.steps
    )
    return {
        "candidate_count": len(candidates),
        "decoded_event_count": len(events),
        "mapped_event_count": len(mapped),
        "unmapped_event_count": len(events) - len(mapped),
        "copy_event_count": sum(bool(event.is_copy) for event in events),
        "deleted_score_event_count": len(deleted),
        "score_event_count": len(score),
        "mapped_score_coverage": (
            len({index for event in mapped for index in range(*event.score_span)})
            / max(len(score), 1)
        ),
        "candidate_pitch_in_score_ratio": (
            candidate_pitch_overlap / max(len(candidates), 1)
        ),
        "path_score": float(path.score),
        "operations": dict(operations),
    }


def _freeze_one(
    sample: Path,
    output: Path,
    stack: Mapping[str, Any],
    production_lattice: SparseJointLattice | None,
    production_payload: Mapping[str, Any] | None,
    *,
    fresh_features: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    wav = sample / "performance_audio.wav"
    score_path = sample / "verified_score.musicxml"
    if not wav.is_file() or not score_path.is_file():
        raise FileNotFoundError(
            f"required inputs: audio={wav.is_file()} score={score_path.is_file()}"
        )
    audio = _audio_info(wav)
    phase = time.perf_counter()
    features, cache = _cache_features(
        sample, output, fresh_features=fresh_features
    )
    feature_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    index = ScoreEventIndex.from_musicxml(score_path)
    score = tuple(index.events)
    if not score:
        raise ValueError("verified score has no sounding events")
    canonical = sanitize_basic_pitch_notes(
        decode_frozen_basic_pitch(features), features, FROZEN_DECODE_CONFIG
    )
    configs = _frontend_configs(stack["joint_payload"])
    all_candidates = tuple(
        basic_pitch_candidate_union(
            features, configs=configs, minimum_confidence=0.0
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
        raise ValueError("score/audio mismatch: no admitted transcription candidates")
    candidate_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    path = stack["lattice"].decode(admitted, score)
    rows = build_inference_rows(stack["lattice"], admitted, score, path)
    prediction = infer_error_heads(
        stack["heads_model"],
        rows,
        stack["heads_payload"]["thresholds"],
        device="cpu",
    )
    filtered = filter_prediction_for_schema(
        prediction, stack["heads_payload"]["schema_thresholds"]
    )
    document = schema12_document(
        sample.name, rows, filtered, score, pad_notes=1
    )
    rules = schema12_document(
        sample.name, rows, heuristic_prediction(rows), score, pad_notes=1
    )
    current_seconds = time.perf_counter() - phase
    diagnostics = _path_diagnostics(admitted, score, path)
    diagnostics.update(
        {
            "canonical_transcription_count": len(canonical),
            "candidate_union_all_count": len(all_candidates),
            "retained_candidate_count": len(admitted),
            "candidate_to_score_count_ratio": len(admitted) / len(score),
            "canonical_to_score_count_ratio": len(canonical) / len(score),
            "audio": audio,
            "score_measure_min": min(
                (event.measure for event in score if event.measure is not None),
                default=None,
            ),
            "score_measure_max": max(
                (event.measure for event in score if event.measure is not None),
                default=None,
            ),
        }
    )
    document["pipeline"].update(
        {
            "candidate_config_sha256": stack["selection"]["transcriber"][
                "candidate_config_sha256"
            ],
            "joint_checkpoint_sha256": stack["selection"][
                "joint_aligner_decoder"
            ]["checkpoint_sha256"],
            "error_heads_checkpoint_sha256": stack["selection"][
                "error_heads_v2"
            ]["checkpoint_sha256"],
            "diagnostics": diagnostics,
        }
    )
    rules["pipeline"].update({"diagnostics": diagnostics})

    production = None
    production_seconds = 0.0
    if production_lattice is not None and production_payload is not None:
        phase = time.perf_counter()
        production_all = basic_pitch_candidate_union(
            features,
            configs=_frontend_configs(production_payload),
            minimum_confidence=0.0,
        )
        production_confidence = float(
            (production_payload.get("training") or {}).get(
                "minimum_candidate_confidence", 0.65
            )
        )
        production_candidates = tuple(
            add_score_repeat_hints(
                [
                    value
                    for value in production_all
                    if value.confidence >= production_confidence
                ],
                score,
            )
        )
        production_path = production_lattice.decode(
            production_candidates, score
        )
        production = {
            "schema_version": "align-datacreate-production-alignment-v1",
            "sample_id": sample.name,
            "labels": [],
            "diagnostics": {
                **_path_diagnostics(
                    production_candidates, score, production_path
                ),
                "candidate_union_all_count": len(production_all),
                "retained_candidate_count": len(production_candidates),
                "minimum_candidate_confidence": production_confidence,
            },
        }
        production_seconds = time.perf_counter() - phase

    paths = {
        "prediction": output / "predictions" / f"{sample.name}.json",
        "rules": output / "rules-baseline" / f"{sample.name}.json",
        "production": output / "production-baseline" / f"{sample.name}.json",
    }
    _atomic_json(paths["prediction"], document)
    _atomic_json(paths["rules"], rules)
    if production is not None:
        _atomic_json(paths["production"], production)
    finished = time.perf_counter()
    return {
        "sample": sample.name,
        "status": "succeeded",
        "paths": {
            key: str(path.resolve())
            for key, path in paths.items()
            if path.is_file()
        },
        "hashes": {
            key: _sha256(path) for key, path in paths.items() if path.is_file()
        },
        "diagnostics": diagnostics,
        "feature_cache": cache,
        "runtime_seconds": {
            "features": feature_seconds,
            "candidate_and_score": candidate_seconds,
            "current_stack": current_seconds,
            "production_alignment": production_seconds,
            "total": finished - started,
        },
    }


def freeze(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    marker = output / "freeze_manifest.json"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite frozen run: {marker}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.device != "cpu":
        raise ValueError("This evaluator permits CPU only while training is active")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(max(1, int(args.cpu_threads)))
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"

    expected = _selected_samples(args.samples.resolve(), args.sample_id)
    if any(
        output == sample.resolve() or output.is_relative_to(sample.resolve())
        for sample in expected
    ):
        raise ValueError("Output must be outside every sample directory")
    guard = (
        _install_sample_read_guard(expected, _FREEZE_SAMPLE_INPUTS, "freeze")
        if args.sample_id is not None
        else None
    )
    discovered = [path for path in expected if path.is_dir()]
    rows = []
    status_before = _training_status(args.resource_status.resolve())
    with resource_lease(
        args.resource_status.resolve(),
        "cpu",
        track=f"eval-datacreate-current:{output.name}",
        command=[sys.executable, *sys.argv],
        metadata={
            "device": "cpu",
            "cpu_threads": int(args.cpu_threads),
            "sample_ids": [path.name for path in expected],
        },
    ) as lease_id:
        status = _training_status(args.resource_status.resolve())
        if not status_before.get("gpu_training_active"):
            coordination = "CPU-only lease acquired; no active GPU lease observed"
        else:
            coordination = (
                "active GPU training observed; bounded CPU-only inference "
                "lease acquired"
            )
        stack = _load_completed_stack(args)
        if args.skip_production_baseline:
            production_lattice = production_payload = None
            production_selection = {
                "status": "skipped",
                "reason": "single-stack evaluation requested",
                "checkpoint_read": False,
            }
        else:
            production_lattice, production_payload, production_selection = (
                _load_production_checkpoint(args.production_checkpoint)
            )
        stack["selection"]["production_baseline"] = production_selection

        for position, sample in enumerate(expected, 1):
            print(f"freeze {position}/{len(expected)} {sample.name}", flush=True)
            if not sample.is_dir():
                rows.append(
                    {
                        "sample": sample.name,
                        "status": "failed",
                        "error": "sample directory missing",
                    }
                )
                continue
            try:
                rows.append(
                    _freeze_one(
                        sample,
                        output,
                        stack,
                        production_lattice,
                        production_payload,
                        fresh_features=bool(args.fresh_features),
                    )
                )
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
            "created_utc": _utc(),
            "process": {
                "pid": os.getpid(),
                "phase": "inference_freeze",
            },
            "gold_access": {
                "labels_opened": False,
                "forbidden_inputs_opened": list(
                    guard.violations if guard else ()
                ),
                "phase": "freeze",
                "process_separation_required": True,
                "read_guard_enforced": guard is not None,
                "opened_sample_inputs": sorted(guard.opened if guard else ()),
                "source_contract": (
                    "Only verified_score.musicxml, performance_audio.wav, "
                    "immutable metadata pitch convention, completed frozen "
                    "checkpoint/config/report files, and training resource "
                    "status were read."
                    if args.fresh_features
                    else (
                        "Only verified_score.musicxml, performance_audio.wav, "
                        "hash-validated Basic Pitch activations, metadata pitch "
                        "convention, completed checkpoint/config/report files, "
                        "and training resource status were read."
                    )
                ),
            },
            "samples_root": str(args.samples.resolve()),
            "output": str(output),
            "subset": {
                "requested": args.sample_id is not None,
                "sample_ids": [path.name for path in expected],
            },
            "discovered_expected": len(discovered),
            "expected": len(expected),
            "inference_succeeded": sum(
                row["status"] == "succeeded" for row in rows
            ),
            "inference_failed": sum(
                row["status"] == "failed" for row in rows
            ),
            "samples": rows,
            "model_selection": stack["selection"],
            "resource_coordination": {
                "device": "cpu",
                "cpu_threads": args.cpu_threads,
                "tensorflow_gpu_hidden": True,
                "lease": {
                    "resource": "cpu",
                    "lease_id": lease_id,
                    "track": f"eval-datacreate-current:{output.name}",
                },
                "decision": coordination,
                "status_before": status_before,
                "status_snapshot": status,
            },
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "basic_pitch": _package_version("basic-pitch"),
                "tensorflow": _package_version("tensorflow"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "basic_pitch_runtime": os.environ.get(
                    "ALIGN_BASIC_PITCH_RUNTIME", "tensorflow"
                ),
            },
            "command": [sys.executable, *sys.argv],
        }
        _atomic_json(marker, manifest)
    print(marker)


def _explicit_complete(document: Mapping[str, Any]) -> bool:
    for key in ("annotation_complete", "reviewed", "completed", "is_complete"):
        if document.get(key) is True:
            return True
    return str(document.get("review_status") or "").casefold() in {
        "complete",
        "completed",
        "reviewed",
    }


def _label_errors(
    document: Mapping[str, Any],
    *,
    duration: float | None,
    score_count: int | None,
) -> list[str]:
    errors = []
    if not isinstance(document.get("schema_version"), str):
        errors.append("missing schema_version")
    labels = document.get("labels")
    if not isinstance(labels, list):
        return [*errors, "labels is not an array"]
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            errors.append(f"label {index} is not an object")
            continue
        for key in ("id", "source", "start_time", "end_time", "type"):
            if label.get(key) is None:
                errors.append(f"label {index} missing {key}")
        try:
            start, end = float(label["start_time"]), float(label["end_time"])
            if end <= start:
                errors.append(f"label {index} has non-positive duration")
            if duration is not None and (start < -0.01 or end > duration + 0.25):
                errors.append(f"label {index} outside audio duration")
        except (KeyError, TypeError, ValueError):
            pass
        part = label.get("score_part")
        if isinstance(part, Mapping) and score_count is not None:
            try:
                first = int(part["start_note_index"])
                last = int(part["end_note_index"])
                if first < 0 or last < first or last >= score_count:
                    errors.append(f"label {index} score range outside score")
            except (KeyError, TypeError, ValueError):
                errors.append(f"label {index} malformed score range")
    return errors


def _inventory_gold(
    freeze_doc: Mapping[str, Any],
    *,
    gold_source: str = "human",
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    samples_root = Path(freeze_doc["samples_root"])
    frozen_by_id = {row["sample"]: row for row in freeze_doc["samples"]}
    subset_ids = (freeze_doc.get("subset") or {}).get("sample_ids")
    selected = (
        _selected_samples(samples_root)
        if not subset_ids
        else [samples_root / str(value) for value in subset_ids]
    )
    strict_subset = bool((freeze_doc.get("subset") or {}).get("requested"))
    inventory = []
    gold: dict[str, list[dict[str, Any]]] = {}
    for sample in selected:
        frozen = frozen_by_id.get(sample.name) or {}
        diagnostics = frozen.get("diagnostics") or {}
        duration = (diagnostics.get("audio") or {}).get("duration_sec")
        score_count = diagnostics.get("score_event_count")
        label_path = sample / "labels.json"
        metadata_path = sample / "metadata.json"
        alignment_path = sample / "note_alignment_v2.json"
        row: dict[str, Any] = {
            "sample": sample.name,
            "directory_present": sample.is_dir(),
            "required": {
                "verified_score.musicxml": (
                    sample / "verified_score.musicxml"
                ).is_file(),
                "performance_audio.wav": (
                    sample / "performance_audio.wav"
                ).is_file(),
                "metadata.json": metadata_path.is_file(),
                "labels.json": label_path.is_file(),
            },
            "inference_status": frozen.get("status"),
            "partial_take": None,
            "label_status": "missing",
            "note_alignment": {"present": alignment_path.is_file()},
        }
        if metadata_path.is_file() and not strict_subset:
            metadata = _json(metadata_path)
            row["partial_take"] = {
                "score_segment": metadata.get("score_segment"),
                "performance_trim": metadata.get("performance_trim"),
                "sample_id": metadata.get("sample_id"),
                "sample_id_matches_directory": str(
                    metadata.get("sample_id")
                )
                == sample.name,
            }
        if alignment_path.is_file() and not strict_subset:
            alignment = _json(alignment_path)
            row["note_alignment"].update(
                {
                    "format_version": alignment.get("format_version"),
                    "engine": alignment.get("engine"),
                    "checkpoint": (alignment.get("summary") or {}).get(
                        "checkpoint"
                    ),
                    "event_count": len(alignment.get("events") or []),
                    "labels_count": len(alignment.get("labels") or []),
                }
            )
        if label_path.is_file():
            document = _json(label_path)
            labels = document.get("labels") or []
            if gold_source == "agent":
                selected_labels = [
                    dict(label)
                    for label in labels
                    if str(label.get("source") or "") in AGENT_SOURCES
                ]
                if (
                    not selected_labels
                    and document.get("annotator_id") == "ai_f0_align"
                ):
                    selected_labels = [dict(label) for label in labels]
            else:
                selected_labels = [
                    dict(label)
                    for label in labels
                    if str(label.get("source") or "") in HUMAN_SOURCES
                ]
            explicit_complete = _explicit_complete(document)
            errors = _label_errors(
                document, duration=duration, score_count=score_count
            )
            range_count = sum(
                isinstance(label.get("pitches"), list)
                and bool(label.get("pitches"))
                and isinstance(label.get("score_part"), Mapping)
                for label in selected_labels
            )
            if selected_labels:
                status = (
                    "usable_agent_gold"
                    if gold_source == "agent"
                    else "usable_sparse_human_gold"
                )
                if frozen.get("status") == "succeeded":
                    gold[sample.name] = selected_labels
            elif explicit_complete:
                status = "reviewed_clean_empty"
            else:
                status = "empty_unreviewed"
            row.update(
                {
                    "label_status": status,
                    "label_schema_version": document.get("schema_version"),
                    "labels_total": len(labels),
                    "selected_labels": len(selected_labels),
                    "gold_source": gold_source,
                    "human_labels": len(selected_labels),
                    "human_range_labels": range_count,
                    "explicit_completeness_marker": explicit_complete,
                    "validation_errors": errors,
                    "human_types": dict(
                        Counter(
                            str(label.get("type"))
                            for label in selected_labels
                        )
                    ),
                }
            )
        inventory.append(row)
    return inventory, gold


def _note_id_indices(values: Any) -> tuple[int, ...] | None:
    if not isinstance(values, list) or not values:
        return None
    output = []
    for value in values:
        match = _NOTE_ID.fullmatch(str(value))
        if match is None:
            return None
        output.append(int(match.group(1)))
    return tuple(sorted(set(output)))


def _canonical_core_indices(
    label: Mapping[str, Any],
    *,
    score_event_count: int,
) -> tuple[tuple[int, ...] | None, str | None]:
    explicit = label.get("score_event_indices")
    if isinstance(explicit, list) and explicit:
        indices = tuple(sorted({int(value) for value in explicit}))
        basis = "explicit_score_event_indices"
    else:
        part = label.get("score_part")
        indices = None
        basis = None
        if isinstance(part, Mapping):
            core_start = part.get("core_start_note_index")
            core_end = part.get("core_end_note_index")
            if core_start is not None and core_end is not None:
                first, last = int(core_start), int(core_end)
                indices = tuple(range(first, last + 1))
                basis = "score_part_explicit_core"
        core_ids = _note_id_indices(label.get("core_note_ids"))
        if indices is None and core_ids is not None:
            indices = core_ids
            basis = "core_note_ids"
        if indices is None and isinstance(part, Mapping):
            try:
                first = int(part["start_note_index"])
                last = int(part["end_note_index"])
                pad = max(0, int(part.get("pad_notes") or 0))
            except (KeyError, TypeError, ValueError):
                return None, None
            core_first = min(last, first + pad)
            core_last = max(core_first, last - pad)
            indices = tuple(range(core_first, core_last + 1))
            basis = "score_part_range_minus_declared_padding"
    if (
        not indices
        or indices[0] < 0
        or indices[-1] >= int(score_event_count)
    ):
        return None, basis
    return indices, basis


def _canonical_projection_audit(
    label: Mapping[str, Any],
    notes: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = dict(label)
    indices, basis = _canonical_core_indices(
        value, score_event_count=len(notes)
    )
    reasons = []
    warnings = []
    if indices is None:
        reasons.append("missing_or_invalid_core_score_event_projection")
    part = value.get("score_part")
    if not isinstance(part, Mapping):
        reasons.append("missing_score_part")
    else:
        try:
            first = int(part["start_note_index"])
            last = int(part["end_note_index"])
        except (KeyError, TypeError, ValueError):
            reasons.append("invalid_score_part")
        else:
            if first < 0 or last < first or last >= len(notes):
                reasons.append("score_part_out_of_bounds")
            else:
                expected_pitches = [
                    int(note.pitch) for note in notes[first : last + 1]
                ]
                supplied_pitches = value.get("pitches")
                if supplied_pitches is not None and (
                    not isinstance(supplied_pitches, list)
                    or [int(item) for item in supplied_pitches]
                    != expected_pitches
                ):
                    reasons.append("pitch_list_does_not_validate_score_part")
                supplied_ids = _note_id_indices(value.get("note_ids"))
                if supplied_ids is not None and supplied_ids != tuple(
                    range(first, last + 1)
                ):
                    reasons.append("note_ids_do_not_validate_score_part")
    core_ids = _note_id_indices(value.get("core_note_ids"))
    if core_ids is not None and indices is not None and core_ids != indices:
        warning = "core_note_ids_disagree_with_core_score_part"
        if basis == "score_part_explicit_core":
            warnings.append(warning)
        else:
            reasons.append(warning)
    if indices is not None:
        value["score_event_indices"] = list(indices)
    location = canonical_note_location(
        value, score_event_count=len(notes)
    )
    if location is None:
        reasons.append("canonical_identity_unavailable")
    audit = {
        "label_id": value.get("id"),
        "type": value.get("type"),
        "accepted": not reasons,
        "projection_basis": basis,
        "canonical_location": location,
        "score_event_indices": list(indices) if indices is not None else None,
        "reasons": reasons,
        "warnings": warnings,
    }
    return value, audit


def _official_note_wise_report(
    samples_root: Path,
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    predicted: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if not gold:
        return {
            "status": "unavailable",
            "reason": "no selected nonempty gold labels",
        }
    sample_reports = {}
    all_details = []
    per_type_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample, target_labels in gold.items():
        notes = parse_sounding_notes(
            samples_root / sample / "verified_score.musicxml"
        )
        projected_gold = []
        gold_audit = []
        for label in target_labels:
            projected, audit = _canonical_projection_audit(label, notes)
            projected_gold.append(projected)
            gold_audit.append(audit)
        projected_predicted = []
        predicted_audit = []
        for label in predicted.get(sample, ()):
            projected, audit = _canonical_projection_audit(label, notes)
            projected_predicted.append(projected)
            predicted_audit.append(audit)
        rejected_gold = [
            row for row in gold_audit if not bool(row["accepted"])
        ]
        if rejected_gold:
            return {
                "status": "unavailable",
                "reason": (
                    f"{sample} gold failed canonical projection audit"
                ),
                "sample": sample,
                "gold_audit": gold_audit,
            }
        detail = match_note_wise_labels_detail(
            projected_gold,
            projected_predicted,
            score_event_count=len(notes),
        )
        if detail["status"] != "available":
            raise ValueError(
                f"Official note-wise metric unavailable for {sample}"
            )
        all_details.append(detail)
        types = sorted(
            {
                str(label.get("type"))
                for label in [*projected_gold, *projected_predicted]
            }
        )
        per_type = {}
        for kind in types:
            kind_detail = match_note_wise_labels_detail(
                [
                    label
                    for label in projected_gold
                    if str(label.get("type")) == kind
                ],
                [
                    label
                    for label in projected_predicted
                    if str(label.get("type")) == kind
                ],
                score_event_count=len(notes),
            )
            per_type[kind] = kind_detail
            per_type_details[kind].append(kind_detail)
        sample_reports[sample] = {
            "score_event_count": len(notes),
            "metrics": detail,
            "per_type": per_type,
            "gold": gold_audit,
            "predicted": predicted_audit,
        }

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        credit = sum(float(row["credit"]) for row in rows)
        predicted_count = sum(int(row["predicted"]) for row in rows)
        gold_count = sum(int(row["gold"]) for row in rows)
        return _prf(credit, predicted_count, gold_count)

    return {
        "status": "available",
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "matching_policy": (
            "exclusive maximum-weight one-to-one canonical score-event "
            "identity; exact location+type=1.0, exact location+different "
            "type=0.5, different location=0.0"
        ),
        "canonical_projection": (
            "score_event_indices derived from audited score_part core indices "
            "or core_note_ids; declared context padding is excluded"
        ),
        "micro": aggregate(all_details),
        "per_type": {
            kind: aggregate(rows)
            for kind, rows in sorted(per_type_details.items())
        },
        "samples": sample_reports,
    }


def _iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a0, a1 = float(left["start_time"]), float(left["end_time"])
    b0, b1 = float(right["start_time"]), float(right["end_time"])
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return intersection / union if union > 0.0 else 0.0


def _exclusive_hits(
    gold: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    score: Callable[[Mapping[str, Any], Mapping[str, Any]], float],
) -> int:
    if not gold or not predicted:
        return 0
    matrix = np.asarray(
        [
            [
                score(prediction, target)
                if str(prediction.get("type")) == str(target.get("type"))
                else 0.0
                for target in gold
            ]
            for prediction in predicted
        ],
        dtype=np.float64,
    )
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-matrix)
        return sum(matrix[row, column] > 0.0 for row, column in zip(rows, columns))
    except Exception:
        cells = sorted(
            (
                (float(matrix[row, column]), row, column)
                for row in range(len(predicted))
                for column in range(len(gold))
            ),
            reverse=True,
        )
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        hits = 0
        for value, row, column in cells:
            if value <= 0.0:
                break
            if row in used_rows or column in used_columns:
                continue
            used_rows.add(row)
            used_columns.add(column)
            hits += 1
        return hits


def _prf(correct: float, predicted: int, gold: int) -> dict[str, Any]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "matched": correct,
        "predicted": predicted,
        "gold": gold,
    }


def _event_counts(
    gold: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    *,
    criterion: str,
    types: set[str] | None = None,
) -> dict[str, int]:
    selected_gold = [
        label for label in gold if types is None or label.get("type") in types
    ]
    selected_pred = [
        label
        for label in predicted
        if types is None or label.get("type") in types
    ]
    if criterion.startswith("onset_"):
        tolerance = int(criterion.split("_")[1].removesuffix("ms")) / 1000.0

        def score(prediction: Mapping[str, Any], target: Mapping[str, Any]) -> float:
            delta = abs(
                float(prediction["start_time"]) - float(target["start_time"])
            )
            return 1.0 / (1.0 + delta) if delta <= tolerance else 0.0

    elif criterion == "iou_0.3":

        def score(prediction: Mapping[str, Any], target: Mapping[str, Any]) -> float:
            value = _iou(prediction, target)
            return value if value >= 0.3 else 0.0

    else:
        raise ValueError(criterion)
    return {
        "correct": _exclusive_hits(selected_gold, selected_pred, score),
        "predicted": len(selected_pred),
        "gold": len(selected_gold),
    }


def _bootstrap_counts(
    values: Sequence[Mapping[str, int]],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict[str, Any] | None:
    if len(values) < 5:
        return None
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(replicates):
        selected = rng.integers(0, len(values), size=len(values))
        correct = sum(values[index]["correct"] for index in selected)
        predicted = sum(values[index]["predicted"] for index in selected)
        gold = sum(values[index]["gold"] for index in selected)
        output.append(_prf(correct, predicted, gold)["f1"])
    return {
        "unit": "clip",
        "replicates": replicates,
        "lower_95": float(np.quantile(output, 0.025)),
        "median": float(np.quantile(output, 0.5)),
        "upper_95": float(np.quantile(output, 0.975)),
    }


def _event_report(
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    predicted: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    criteria = ("iou_0.3", "onset_50ms", "onset_100ms", "onset_250ms", "onset_500ms")
    report: dict[str, Any] = {}
    for criterion_index, criterion in enumerate(criteria):
        clip_counts = [
            _event_counts(
                gold[sample], predicted.get(sample, ()), criterion=criterion
            )
            for sample in sorted(gold)
        ]
        totals = {
            key: sum(row[key] for row in clip_counts)
            for key in ("correct", "predicted", "gold")
        }
        report[criterion] = {
            "micro": _prf(
                totals["correct"], totals["predicted"], totals["gold"]
            ),
            "bootstrap_f1": _bootstrap_counts(
                clip_counts, seed=20260915 + criterion_index
            ),
        }
    all_types = sorted(
        {
            str(label.get("type"))
            for labels in gold.values()
            for label in labels
        }
        | {
            str(label.get("type"))
            for labels in predicted.values()
            for label in labels
        }
    )
    per_type = {}
    for kind in all_types:
        counts = [
            _event_counts(
                gold[sample],
                predicted.get(sample, ()),
                criterion="iou_0.3",
                types={kind},
            )
            for sample in sorted(gold)
        ]
        totals = {
            key: sum(row[key] for row in counts)
            for key in ("correct", "predicted", "gold")
        }
        per_type[kind] = _prf(
            totals["correct"], totals["predicted"], totals["gold"]
        )
    report["iou_0.3"]["per_type"] = per_type
    report["iou_0.3"]["macro_f1"] = float(
        np.mean([value["f1"] for value in per_type.values()])
    )
    for name, types in (
        ("layer2", LAYER2_TYPES),
        ("layer3_rhythm", RHYTHM_TYPES),
        ("repetition", REPETITION_TYPES),
    ):
        counts = [
            _event_counts(
                gold[sample],
                predicted.get(sample, ()),
                criterion="iou_0.3",
                types=types,
            )
            for sample in sorted(gold)
        ]
        totals = {
            key: sum(row[key] for row in counts)
            for key in ("correct", "predicted", "gold")
        }
        report[name] = {
            **_prf(totals["correct"], totals["predicted"], totals["gold"]),
            "bootstrap_f1": _bootstrap_counts(
                counts, seed=20261000 + len(name)
            ),
            "types": sorted(types),
        }
    return report


def _range_counts(
    gold_labels: Sequence[Mapping[str, Any]],
    predicted_labels: Sequence[Mapping[str, Any]],
    types: set[str] | None = None,
) -> dict[str, Any]:
    selected_gold = [
        dict(label)
        for label in gold_labels
        if isinstance(label.get("pitches"), list)
        and bool(label.get("pitches"))
        and (types is None or label.get("type") in types)
    ]
    selected_pred = [
        dict(label)
        for label in predicted_labels
        if types is None or label.get("type") in types
    ]
    gold = gold_melodies_from_labels(selected_gold)
    predicted = pred_melodies_from_labels(selected_pred, [], pad_notes=1)
    detail = match_melodies_detail(gold, predicted)
    return {
        "correct": float(detail["n_matched"]),
        "predicted": len(predicted),
        "gold": len(gold),
        "f1": float(detail["f1"]),
        "precision": float(detail["precision"]),
        "recall": float(detail["recall"]),
    }


def _bootstrap_range(
    values: Sequence[Mapping[str, Any]], seed: int, replicates: int = 2000
) -> dict[str, Any] | None:
    integerized = [
        {
            "correct": float(row["correct"]),
            "predicted": int(row["predicted"]),
            "gold": int(row["gold"]),
        }
        for row in values
    ]
    if len(integerized) < 5:
        return None
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(replicates):
        selected = rng.integers(0, len(integerized), size=len(integerized))
        correct = sum(integerized[index]["correct"] for index in selected)
        predicted = sum(integerized[index]["predicted"] for index in selected)
        gold = sum(integerized[index]["gold"] for index in selected)
        output.append(_prf(correct, predicted, gold)["f1"])
    return {
        "unit": "clip",
        "replicates": replicates,
        "lower_95": float(np.quantile(output, 0.025)),
        "median": float(np.quantile(output, 0.5)),
        "upper_95": float(np.quantile(output, 0.975)),
    }


def _range_report(
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    predicted: Mapping[str, Sequence[Mapping[str, Any]]],
    schemas: Mapping[str, str | None],
) -> dict[str, Any]:
    def summarize(samples: Iterable[str], types: set[str] | None) -> dict[str, Any]:
        sample_ids = sorted(samples)
        rows = [
            _range_counts(
                gold[sample], predicted.get(sample, ()), types=types
            )
            for sample in sample_ids
        ]
        total_correct = sum(row["correct"] for row in rows)
        total_predicted = sum(row["predicted"] for row in rows)
        total_gold = sum(row["gold"] for row in rows)
        return {
            "clips": len(sample_ids),
            "micro": _prf(total_correct, total_predicted, total_gold),
            "macro_clip_f1": (
                float(np.mean([row["f1"] for row in rows])) if rows else None
            ),
            "bootstrap_f1": _bootstrap_range(rows, 20260915),
            "samples": {
                sample: row for sample, row in zip(sample_ids, rows)
            },
        }

    range_capable = [
        sample
        for sample, labels in gold.items()
        if any(
            isinstance(label.get("pitches"), list) and label.get("pitches")
            for label in labels
        )
    ]
    formal_schema12 = [
        sample for sample in range_capable if schemas.get(sample) == "1.2"
    ]
    per_type = {
        kind: summarize(range_capable, {kind})
        for kind in sorted(
            {
                str(label.get("type"))
                for sample in range_capable
                for label in gold[sample]
                if label.get("pitches")
            }
            | SCORED_MODEL_TYPES
        )
    }
    return {
        "metric": (
            "codebase official exclusive pitch-list assignment: hard LCS-Dice "
            "range hit, type mismatch receives half credit"
        ),
        "formal_schema_1_2": (
            summarize(formal_schema12, None)
            if formal_schema12
            else {
                "status": "unavailable",
                "reason": "no usable human gold document declares schema 1.2",
                "clips": 0,
            }
        ),
        "enhanced_schema_1_1_diagnostic": summarize(range_capable, None),
        "model_supported_types": summarize(
            range_capable, SCORED_MODEL_TYPES
        ),
        "per_type": per_type,
    }


def _prediction_documents(
    freeze_doc: Mapping[str, Any], key: str
) -> dict[str, dict[str, Any]]:
    output = {}
    for row in freeze_doc["samples"]:
        path_text = (row.get("paths") or {}).get(key)
        if row.get("status") == "succeeded" and path_text:
            output[row["sample"]] = _json(Path(path_text))
    return output


def _verify_frozen(freeze_path: Path) -> tuple[dict[str, Any], list[str]]:
    freeze_doc = _json(freeze_path)
    if freeze_doc.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("Unsupported freeze manifest")
    errors = []
    for row in freeze_doc["samples"]:
        for key, expected in (row.get("hashes") or {}).items():
            path_text = (row.get("paths") or {}).get(key)
            path = Path(path_text) if path_text else None
            if path is None or not path.is_file():
                errors.append(f"{row['sample']}:{key}:missing")
            elif _sha256(path) != expected:
                errors.append(f"{row['sample']}:{key}:hash mismatch")
    return freeze_doc, errors


def _historical_comparison(
    current_ranges: Mapping[str, Any],
    historical_path: Path,
) -> dict[str, Any]:
    if not historical_path.is_file():
        return {"status": "unavailable", "path": str(historical_path)}
    historical = _json(historical_path)
    current = current_ranges["enhanced_schema_1_1_diagnostic"]
    previous = (
        (historical.get("criteria") or {}).get("hard_type_sensitive") or {}
    )
    intersection = sorted(
        set(current.get("samples") or {})
        & {str(row.get("sample")) for row in historical.get("samples") or []}
    )
    current_intersection = [
        current["samples"][sample]["f1"] for sample in intersection
    ]
    return {
        "status": "comparable_under_same_codebase_pitch_list_definition",
        "path": str(historical_path.resolve()),
        "historical_checkpoint": historical.get("checkpoint"),
        "intersection_clips": len(intersection),
        "historical_macro_clip_f1": previous.get("mean_melody_f1"),
        "current_macro_clip_f1": (
            float(np.mean(current_intersection))
            if current_intersection
            else None
        ),
        "delta": (
            float(np.mean(current_intersection))
            - float(previous["mean_melody_f1"])
            if current_intersection and previous.get("mean_melody_f1") is not None
            else None
        ),
        "warning": (
            "Both are sparse enhanced-1.1 score-range diagnostics, not a "
            "formal schema-1.2 real-test score."
        ),
    }


def _sample_007(
    current: Mapping[str, Mapping[str, Any]],
    prior_path: Path,
) -> dict[str, Any]:
    document = current.get("007") or {}
    diagnostics = (document.get("pipeline") or {}).get("diagnostics") or {}
    labels = document.get("labels") or []
    result = {
        "tuning_use": False,
        "current": {
            "canonical_predicted_note_count": diagnostics.get(
                "canonical_transcription_count"
            ),
            "candidate_union_all_count": diagnostics.get(
                "candidate_union_all_count"
            ),
            "retained_candidate_count": diagnostics.get(
                "retained_candidate_count"
            ),
            "decoded_event_count": diagnostics.get("decoded_event_count"),
            "score_count": diagnostics.get("score_event_count"),
            "error_label_count": len(labels),
            "error_labels": [
                {
                    "type": label.get("type"),
                    "start_time": label.get("start_time"),
                    "end_time": label.get("end_time"),
                    "score_part": label.get("score_part"),
                }
                for label in labels
            ],
        },
        "prior_diagnostic_path": str(prior_path),
    }
    if prior_path.is_file():
        prior = _json(prior_path)
        streams = prior.get("streams") or {}
        result["prior"] = {
            "canonical_predicted_note_count": (
                streams.get("canonical_frozen") or {}
            ).get("count"),
            "candidate_union_all_count": (
                streams.get("joint_v2_all") or {}
            ).get("count"),
            "retained_candidate_count_0.65": (
                streams.get("joint_v2_confidence_065") or {}
            ).get("count"),
            "score_count": (prior.get("score") or {}).get("score_note_count"),
            "old_production_decoder_event_count": (
                prior.get("existing_artifacts") or {}
            ).get("joint_summary", {}).get("event_count"),
            "timing_free_coverage_upper_bound_0.65": (
                streams.get("joint_v2_confidence_065") or {}
            ).get("timing_free_sequence_coverage", {}).get("recall_upper_bound"),
            "independent_timing_reference_fitness": (
                prior.get("score") or {}
            ).get("timing_reference_quality", {}).get(
                "fitness_for_20_50_100ms_scoring"
            ),
        }
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    event = report["metrics"]["timestamp_event"]["iou_0.3"]["micro"]
    ranges = report["metrics"]["score_range"][
        "enhanced_schema_1_1_diagnostic"
    ]["micro"]
    official = report["metrics"]["official_note_wise"]
    if official.get("status") == "available":
        note = official["micro"]
        note_line = (
            "- Official canonical note-wise P/R/F1: "
            f"{note['precision']:.4f}/{note['recall']:.4f}/{note['f1']:.4f} "
            f"(support={note['gold']})"
        )
    else:
        note_line = (
            "- Official canonical note-wise score unavailable: "
            f"{official.get('reason')}"
        )
    sample = report["sample_007"]["current"]
    return "\n".join(
        [
            "# DataCreate current completed stack evaluation",
            "",
            f"- Discovered: {counts['discovered']} / {counts['expected']}",
            (
                f"- Inference: {counts['inference_succeeded']} succeeded, "
                f"{counts['inference_failed']} failed"
            ),
            (
                f"- Gold: {counts['gold_evaluable']} sparse non-empty human, "
                f"{counts['empty_unreviewed']} empty/unreviewed"
            ),
            note_line,
            (
                "- Legacy timestamp IoU>=0.3 diagnostic P/R/F1: "
                f"{event['precision']:.4f}/{event['recall']:.4f}/{event['f1']:.4f}"
            ),
            (
                "- Enhanced schema-1.1 pitch-range diagnostic P/R/F1: "
                f"{ranges['precision']:.4f}/{ranges['recall']:.4f}/{ranges['f1']:.4f}"
            ),
            (
                "- Sample 007 canonical/all/retained/decoded/score/labels: "
                f"{sample['canonical_predicted_note_count']}/"
                f"{sample['candidate_union_all_count']}/"
                f"{sample['retained_candidate_count']}/"
                f"{sample['decoded_event_count']}/"
                f"{sample['score_count']}/"
                f"{sample['error_label_count']}"
            ),
            "",
            "A one-sample result is a smoke test, not a validation estimate."
            if report["annotation_limitations"]["single_sample_smoke_test"]
            else (
                "Precision and false-label rates are apparent values against "
                "sparse, non-exhaustive annotations; empty files are not clean "
                "negatives."
            ),
            "",
        ]
    )


def _tree_sha256(path: Path) -> tuple[str | None, int]:
    if not path.exists():
        return None, 0
    if path.is_file():
        return _sha256(path), 1
    digest = hashlib.sha256()
    count = 0
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        count += 1
    return digest.hexdigest(), count


def _basic_pitch_asset() -> dict[str, Any]:
    package = importlib.import_module("basic_pitch")
    if hasattr(package, "build_icassp_2022_model_path"):
        path = Path(
            package.build_icassp_2022_model_path(package.FilenameSuffix.tf)
        )
    else:
        path = Path(package.ICASSP_2022_MODEL_PATH)
    digest, files = _tree_sha256(path)
    return {
        "path": str(path.resolve()),
        "tree_sha256": digest,
        "files": files,
        "hash_definition": (
            "SHA-256 over sorted relative path lengths, relative paths, and "
            "file bytes; plain file SHA-256 when the asset is one file"
        ),
    }


def summarize(args: argparse.Namespace) -> None:
    report_path = args.report.resolve()
    report = _json(report_path)
    rows = [
        row
        for row in report.get("inference_samples") or []
        if row.get("status") == "succeeded"
    ]

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [float((row.get("diagnostics") or {})[key]) for row in rows],
            dtype=np.float64,
        )

    def distribution(array: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }

    score = values("score_event_count")
    canonical = values("canonical_transcription_count")
    candidate_all = values("candidate_union_all_count")
    retained = values("retained_candidate_count")
    decoded = values("decoded_event_count")
    total_runtime = np.asarray(
        [
            float((row.get("runtime_seconds") or {}).get("total", 0.0))
            for row in rows
        ]
    )
    cache_sources = Counter(
        str((row.get("feature_cache") or {}).get("source_kind")) for row in rows
    )
    validation_failures = [
        row["sample"]
        for row in report.get("data_inventory") or []
        if row.get("validation_errors")
    ]
    metadata_mismatches = [
        row["sample"]
        for row in report.get("data_inventory") or []
        if (row.get("partial_take") or {}).get(
            "sample_id_matches_directory"
        )
        is False
    ]
    low_pitch_overlap = [
        row["sample"]
        for row in rows
        if float(
            (row.get("diagnostics") or {}).get(
                "candidate_pitch_in_score_ratio", 0.0
            )
        )
        < 0.5
    ]
    timestamp = report["metrics"]["timestamp_event"]
    ranges = report["metrics"]["score_range"]
    summary = {
        "schema_version": "align-datacreate-real-headline-v2",
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "counts": report["counts"],
        "headline_metrics": {
            "official_note_wise": {
                "status": "unavailable",
                "reason": (
                    "DataCreate schema 1.1 gold has not passed canonical "
                    "score-event projection audit"
                ),
            },
            "legacy_timestamp_iou_0.3": timestamp["iou_0.3"],
            "diagnostic_timestamp_onset_tolerances": {
                key: timestamp[key]
                for key in (
                    "onset_50ms",
                    "onset_100ms",
                    "onset_250ms",
                    "onset_500ms",
                )
            },
            "legacy_timestamp_layer2": timestamp["layer2"],
            "legacy_timestamp_layer3_rhythm": timestamp["layer3_rhythm"],
            "legacy_timestamp_repetition": timestamp["repetition"],
            "diagnostic_enhanced_schema_1_1_score_range": ranges[
                "enhanced_schema_1_1_diagnostic"
            ],
            "formal_schema_1_2": ranges["formal_schema_1_2"],
            "model_supported_score_range": ranges["model_supported_types"],
        },
        "upstream_counts": {
            "totals": {
                "score": int(np.sum(score)),
                "canonical_transcription": int(np.sum(canonical)),
                "candidate_union_all": int(np.sum(candidate_all)),
                "retained_candidates": int(np.sum(retained)),
                "decoded_events": int(np.sum(decoded)),
            },
            "ratios_of_totals": {
                "canonical_to_score": float(np.sum(canonical) / np.sum(score)),
                "candidate_union_all_to_score": float(
                    np.sum(candidate_all) / np.sum(score)
                ),
                "retained_candidates_to_score": float(
                    np.sum(retained) / np.sum(score)
                ),
                "decoded_events_to_score": float(
                    np.sum(decoded) / np.sum(score)
                ),
            },
            "per_clip_canonical_to_score": distribution(canonical / score),
            "per_clip_retained_to_score": distribution(retained / score),
            "mapped_score_coverage": distribution(
                values("mapped_score_coverage")
            ),
        },
        "runtime_seconds_per_sample": distribution(total_runtime),
        "runtime_seconds_sum": float(np.sum(total_runtime)),
        "feature_cache_sources": dict(cache_sources),
        "data_integrity": {
            "label_validation_failure_samples": validation_failures,
            "metadata_sample_id_mismatches": metadata_mismatches,
            "low_candidate_pitch_overlap_samples": low_pitch_overlap,
            "score_audio_structural_failures": report["counts"][
                "inference_failed"
            ],
        },
        "model_assets": {
            "basic_pitch": _basic_pitch_asset(),
            "selection": report["model_selection"],
        },
        "historical_comparison": report["historical_comparison"],
        "sample_007": report["sample_007"],
        "limitations": report["annotation_limitations"],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output}")
    _atomic_json(output, summary)
    _atomic_json(
        output.with_name("headline_integrity.json"),
        {
            "summary_sha256": _sha256(output),
            "source_report_sha256": _sha256(report_path),
            "created_utc": _utc(),
        },
    )
    print(output)


def evaluate(args: argparse.Namespace) -> None:
    evaluation_started = _utc()
    freeze_path = args.freeze_manifest.resolve()
    freeze_doc, integrity_errors = _verify_frozen(freeze_path)
    if integrity_errors:
        raise ValueError(f"Frozen prediction integrity failed: {integrity_errors}")
    freeze_pid = int((freeze_doc.get("process") or {}).get("pid", -1))
    if freeze_pid == os.getpid():
        raise ValueError("Scoring must run in a process separate from freeze")
    output = Path(freeze_doc["output"])
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {report_path}")
    samples_root = Path(freeze_doc["samples_root"])
    subset_ids = list((freeze_doc.get("subset") or {}).get("sample_ids") or [])
    strict_subset = bool((freeze_doc.get("subset") or {}).get("requested"))
    selected_samples = [samples_root / str(value) for value in subset_ids]
    protected_before = {
        sample.name: _protected_sample_stats(sample)
        for sample in selected_samples
    }
    evaluate_guard = (
        _install_sample_read_guard(
            selected_samples, _EVALUATE_SAMPLE_INPUTS, "evaluate"
        )
        if strict_subset
        else None
    )

    # Gold is opened only after the separate-process freeze hash check above.
    inventory, gold = _inventory_gold(
        freeze_doc, gold_source=args.gold_source
    )
    current_docs = _prediction_documents(freeze_doc, "prediction")
    rule_docs = _prediction_documents(freeze_doc, "rules")
    production_docs = _prediction_documents(freeze_doc, "production")
    current_labels = {
        sample: list(document.get("labels") or [])
        for sample, document in current_docs.items()
    }
    rule_labels = {
        sample: list(document.get("labels") or [])
        for sample, document in rule_docs.items()
    }
    production_labels = {
        sample: list(document.get("labels") or [])
        for sample, document in production_docs.items()
    }
    official_note_wise = _official_note_wise_report(
        samples_root, gold, current_labels
    )
    schemas = {
        row["sample"]: row.get("label_schema_version") for row in inventory
    }
    timestamp = _event_report(gold, current_labels)
    ranges = _range_report(gold, current_labels, schemas)
    rules_timestamp = _event_report(gold, rule_labels)
    rules_ranges = _range_report(gold, rule_labels, schemas)
    production_timestamp = _event_report(gold, production_labels)

    total_minutes = sum(
        float(
            (
                (
                    next(
                        row
                        for row in freeze_doc["samples"]
                        if row["sample"] == sample
                    ).get("diagnostics")
                    or {}
                ).get("audio")
                or {}
            ).get("duration_sec", 0.0)
        )
        for sample in gold
    ) / 60.0
    iou_micro = timestamp["iou_0.3"]["micro"]
    false_labels = int(iou_micro["predicted"] - iou_micro["matched"])
    timestamp["iou_0.3"]["apparent_false_labels_per_minute"] = (
        false_labels / total_minutes if total_minutes else None
    )
    timestamp["iou_0.3"]["evaluated_audio_minutes"] = total_minutes

    label_status = Counter(row["label_status"] for row in inventory)
    freeze_rows = freeze_doc["samples"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": _utc(),
        "evaluation_started_utc": evaluation_started,
        "freeze_created_utc": freeze_doc["created_utc"],
        "counts": {
            "expected": freeze_doc["expected"],
            "discovered": freeze_doc["discovered_expected"],
            "inference_succeeded": freeze_doc["inference_succeeded"],
            "inference_failed": freeze_doc["inference_failed"],
            "gold_evaluable": len(gold),
            "empty_unreviewed": label_status["empty_unreviewed"],
            "reviewed_clean_empty": label_status["reviewed_clean_empty"],
            "missing_labels": label_status["missing"],
            "usable_human_labels": sum(len(labels) for labels in gold.values()),
            "formal_schema_1_2_gold_clips": sum(
                schemas.get(sample) == "1.2" for sample in gold
            ),
            "note_alignment_artifacts": sum(
                bool(row["note_alignment"]["present"]) for row in inventory
            ),
            "partial_take_metadata": sum(
                bool((row.get("partial_take") or {}).get("score_segment"))
                for row in inventory
            ),
        },
        "model_selection": freeze_doc["model_selection"],
        "metrics": {
            "official_note_wise": official_note_wise,
            "legacy_timestamp_event": timestamp,
            "diagnostic_inferred_score_range": ranges,
            # Backward-compatible legacy aliases.
            "timestamp_event": timestamp,
            "score_range": ranges,
            "current_rules_same_upstream": {
                "timestamp_event": rules_timestamp,
                "score_range": rules_ranges,
            },
            "configured_production_datacreate_bridge": {
                "timestamp_event": production_timestamp,
                "note": (
                    "The configured production DataCreate bridge performs "
                    "alignment but emits no error labels."
                ),
            },
        },
        "historical_comparison": (
            {
                "status": "not_opened",
                "reason": "strict subset evaluation forbids prior predictions",
            }
            if strict_subset
            else _historical_comparison(
                ranges, args.historical_summary.resolve()
            )
        ),
        "sample_007": (
            {
                "status": "not_opened",
                "reason": "strict subset evaluation forbids prior diagnostics",
                "current": {
                    "canonical_predicted_note_count": None,
                    "candidate_union_all_count": None,
                    "retained_candidate_count": None,
                    "decoded_event_count": None,
                    "score_count": None,
                    "error_label_count": 0,
                },
            }
            if strict_subset
            else _sample_007(
                current_docs, args.sample_007_diagnostic.resolve()
            )
        ),
        "data_inventory": inventory,
        "inference_samples": freeze_rows,
        "annotation_limitations": {
            "sparse_non_exhaustive": True,
            "empty_unreviewed_are_clean_negatives": False,
            "precision_warning": (
                "Agreement against an agent annotation is not independent "
                "accuracy; unmatched predictions may be valid unannotated "
                "errors and timestamp false-label rates are apparent values."
                if args.gold_source == "agent"
                else (
                    "Human files have no corpus-level exhaustive review marker. "
                    "Unmatched predictions and false-labels/minute may be valid "
                    "unannotated errors and are reported as apparent values."
                )
            ),
            "unsupported_human_types": sorted(
                {
                    str(label.get("type"))
                    for labels in gold.values()
                    for label in labels
                }
                - SCORED_MODEL_TYPES
            ),
            "note_f1": (
                "not reported: DataCreate has no independent human note "
                "transcription ground truth"
            ),
            "formal_schema_1_2": (
                "unavailable unless a usable human labels document declares "
                "schema 1.2"
            ),
            "single_sample_smoke_test": strict_subset,
            "validation_estimate": not strict_subset,
            "threshold_tuning_performed": False,
        },
        "protocol": {
            "two_process_freeze_evaluate": True,
            "freeze_process_pid": freeze_pid,
            "evaluate_process_pid": os.getpid(),
            "process_separation_passed": freeze_pid != os.getpid(),
            "freeze_manifest": str(freeze_path),
            "freeze_prediction_hashes_verified_before_gold": True,
            "prediction_files_mutated_during_evaluation": False,
            "gold_source": args.gold_source,
            "schema_1_1_scoreability": (
                "official only after score_part/core note projection audit"
            ),
            "sample_read_guard_enforced": evaluate_guard is not None,
            "opened_sample_inputs": sorted(
                evaluate_guard.opened if evaluate_guard else ()
            ),
            "forbidden_sample_inputs_opened": list(
                evaluate_guard.violations if evaluate_guard else ()
            ),
            "old_predictions_opened": False,
            "prior_alignments_opened": False,
            "note_maps_opened": False,
            "midi_opened": False,
            "audit_targets_opened": False,
            "lockbox_accessed": False,
            "sample_files_mutated": False,
            "tuning_performed": False,
            "freeze_command": freeze_doc["command"],
            "evaluate_command": [sys.executable, *sys.argv],
            "resource_coordination": freeze_doc["resource_coordination"],
            "environment": freeze_doc["environment"],
        },
    }
    _atomic_json(output / "data_inventory.json", {"samples": inventory})
    _atomic_json(report_path, report)
    (output / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    _freeze_after, after_errors = _verify_frozen(freeze_path)
    protected_after = {
        sample.name: _protected_sample_stats(sample)
        for sample in selected_samples
    }
    protected_unchanged = protected_before == protected_after
    label_hashes = {
        sample.name: _sha256(sample / "labels.json")
        for sample in selected_samples
        if (sample / "labels.json").is_file()
    }
    integrity = {
        "schema_version": "align-datacreate-real-integrity-v1",
        "checked_utc": _utc(),
        "freeze_manifest_sha256": _sha256(freeze_path),
        "report_sha256": _sha256(report_path),
        "prediction_hash_errors": after_errors,
        "process_separation_passed": freeze_pid != os.getpid(),
        "protected_sample_stats_before": protected_before,
        "protected_sample_stats_after": protected_after,
        "protected_sample_stats_unchanged": protected_unchanged,
        "labels_sha256": label_hashes,
        "note_alignment_content_opened": False,
        "lockbox_accessed": False,
        "passed": not after_errors and protected_unchanged,
    }
    _atomic_json(output / "integrity.json", integrity)
    if after_errors or not protected_unchanged:
        raise RuntimeError("Frozen predictions or protected sample files changed")
    print(report_path)


def verify(args: argparse.Namespace) -> None:
    freeze_doc, errors = _verify_frozen(args.freeze_manifest.resolve())
    output = Path(freeze_doc["output"])
    report_path = output / "report.json"
    result = {
        "freeze_predictions_ok": not errors,
        "errors": errors,
        "report_present": report_path.is_file(),
        "report_sha256": _sha256(report_path) if report_path.is_file() else None,
        "freeze_manifest_sha256": _sha256(args.freeze_manifest.resolve()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors or not report_path.is_file():
        raise SystemExit(1)


def _default_paths(parser: argparse.ArgumentParser) -> None:
    root = Path(__file__).resolve().parents[2]
    align = root / "align-model"
    parser.add_argument(
        "--samples", type=Path, default=root / "DataCreate" / "samples"
    )
    parser.add_argument(
        "--sample-id",
        help="Evaluate one expected sample id (001-093 or demo_001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=align
        / "runs"
        / "eval-datacreate-joint-current-20260915-v1",
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
        "--production-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-audit-v2"
        / "path-1000-v4-interval"
        / "joint_decoder.pt",
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
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=align / "runs" / "TRAINING_RESOURCE_STATUS.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    _default_paths(freeze_parser)
    freeze_parser.add_argument("--device", default="cpu", choices=("cpu",))
    freeze_parser.add_argument("--cpu-threads", type=int, default=2)
    freeze_parser.add_argument(
        "--fresh-features",
        action="store_true",
        help="Ignore sample activation caches and infer directly from audio",
    )
    freeze_parser.add_argument(
        "--skip-production-baseline",
        action="store_true",
        help="Run only the selected current completed model stack",
    )
    freeze_parser.set_defaults(function=freeze)

    evaluate_parser = subparsers.add_parser("evaluate")
    _default_paths(evaluate_parser)
    evaluate_parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / "eval-datacreate-joint-current-20260915-v1"
        / "freeze_manifest.json",
    )
    evaluate_parser.add_argument(
        "--historical-summary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / "eval-datacreate-typed"
        / "summary.json",
    )
    evaluate_parser.add_argument(
        "--sample-007-diagnostic",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / "real-audio-diagnostics"
        / "007"
        / "diagnostic-v1-20260915"
        / "report.json",
    )
    evaluate_parser.add_argument(
        "--gold-source",
        choices=("human", "agent"),
        default="human",
    )
    evaluate_parser.set_defaults(function=evaluate)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--freeze-manifest", type=Path, required=True)
    verify_parser.set_defaults(function=verify)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--report", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
