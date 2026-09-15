from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import signal
import sqlite3
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    import psutil
except ImportError:  # Optional telemetry; never required for model correctness.
    psutil = None

from alignmodel.joint.error_heads import (
    FEATURE_DIM,
    LAYER2_CLASSES,
    ErrorHeadsConfig,
    FrozenUpstreamErrorHeads,
    attach_training_targets,
    build_inference_rows,
    build_oracle_rows,
    calibrate_thresholds,
    decode_probabilities,
    error_head_loss,
    evaluate_predictions,
    heuristic_prediction,
    labeled_from_json,
    labeled_to_json,
    load_error_heads,
    measured_class_weights,
    save_checkpoint_atomic,
    schema12_document,
    sha256_file,
    stack_labeled_rows,
)
from alignmodel.joint.lattice import SparseJointLattice
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.joint.train import load_joint_model
from alignmodel.joint.outputraw_full import verify_data_ready


OUTPUT_SCHEMA = "align-frozen-error-heads-run-v1"
DEFAULT_STATUS_PATH = Path("runs/TRAINING_RESOURCE_STATUS.json")
_WORKER_DATASET: PackedJointDataset | None = None
_WORKER_LATTICE: SparseJointLattice | None = None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _utilization(process: Any | None) -> dict[str, float | None]:
    if psutil is None or process is None:
        return {
            "process_cpu_percent": None,
            "machine_cpu_percent": None,
            "rss_mb": None,
        }
    return {
        "process_cpu_percent": process.cpu_percent(None),
        "machine_cpu_percent": psutil.cpu_percent(None),
        "rss_mb": process.memory_info().rss / 1024**2,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )


def _validate_completed_upstream(
    checkpoint: Path,
    packed_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    checkpoint = checkpoint.resolve()
    if checkpoint.name != "joint_decoder.pt" or ".tmp" in checkpoint.name:
        raise ValueError(
            "Frozen upstream must be a completed joint_decoder.pt, never a "
            "last/active checkpoint"
        )
    report_path = checkpoint.parent / "report.json"
    if not report_path.is_file():
        raise ValueError("Completed upstream checkpoint has no sibling report.json")
    model, lattice_config, payload = load_joint_model(checkpoint)
    if not payload.get("best_validation") or not payload.get("history"):
        raise ValueError("Upstream checkpoint has no completed validation epoch")
    frontend = (payload.get("training") or {}).get("frontend") or {}
    candidate_version = packed_metadata.get("candidate_version")
    if frontend.get("candidate_generation") != candidate_version:
        raise ValueError(
            "Packed candidate configuration differs from upstream checkpoint"
        )
    decode_configs = frontend.get("decode_configs") or []
    transcriber_config = {
        "name": frontend.get("name"),
        "candidate_generation": candidate_version,
        "decode_configs": decode_configs,
        "frozen": True,
    }
    checkpoint_hash = sha256_file(checkpoint)
    runs_root = next(
        (parent for parent in checkpoint.parents if parent.name == "runs"),
        None,
    )
    compatible = []
    if runs_root is not None:
        for candidate in runs_root.rglob("joint_decoder.pt"):
            lowered_parts = {part.casefold() for part in candidate.parts}
            if any(
                token in part
                for part in lowered_parts
                for token in ("smoke", "bench", "dev")
            ):
                continue
            if not (candidate.parent / "report.json").is_file():
                continue
            try:
                candidate_payload = torch.load(
                    candidate, map_location="cpu", weights_only=False
                )
            except (OSError, RuntimeError, ValueError):
                continue
            candidate_frontend = (
                (candidate_payload.get("training") or {}).get("frontend") or {}
            )
            if (
                candidate_frontend.get("candidate_generation")
                != candidate_version
                or not candidate_payload.get("best_validation")
                or not candidate_payload.get("history")
            ):
                continue
            compatible.append(
                {
                    "path": str(candidate.resolve()),
                    "modified_ns": candidate.stat().st_mtime_ns,
                }
            )
    compatible.sort(key=lambda value: int(value["modified_ns"]), reverse=True)
    if compatible and Path(compatible[0]["path"]) != checkpoint:
        raise ValueError(
            "Configured upstream is not the newest valid completed compatible "
            "joint decoder"
        )
    contract = {
        "selection_policy": "newest valid completed checkpoint compatible with pack",
        "selection_evidence": {
            "eligible_non_diagnostic_checkpoints": compatible,
            "selected_rank": 1,
        },
        "transcriber": {
            "implementation": "Basic Pitch 0.4.0 frozen activation frontend",
            "checkpoint": "external pretrained frontend embedded in feature release",
            "candidate_config_sha256": _canonical_hash(transcriber_config),
            "candidate_version": candidate_version,
            "config": transcriber_config,
        },
        "aligner": {
            "implementation": "SparseJointLattice path aligner",
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "decoder": {
            "implementation": "JointEdgeScorer replay-aware decoder",
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_hash,
            "schema_version": payload.get("schema_version"),
        },
        "shared_aligner_decoder_checkpoint": True,
        "report": str(report_path.resolve()),
        "frozen": True,
        "active_checkpoint_read": False,
    }
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return contract, model, lattice_config


def _worker_initialize(
    packed_root: str,
    checkpoint: str,
    threads: int,
) -> None:
    global _WORKER_DATASET, _WORKER_LATTICE
    torch.set_num_threads(max(1, int(threads)))
    _WORKER_DATASET = PackedJointDataset(
        Path(packed_root),
        verify_records=False,
        load_feature_arrays=False,
    )
    model, lattice_config, _payload = load_joint_model(Path(checkpoint))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    _WORKER_LATTICE = SparseJointLattice(model, lattice_config)


def _encode(value: Mapping[str, Any]) -> bytes:
    return zlib.compress(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        level=1,
    )


def _decode(value: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(value))


def _extract_ordinal(ordinal: int) -> tuple[Any, ...]:
    if _WORKER_DATASET is None or _WORKER_LATTICE is None:
        raise RuntimeError("Extraction worker is not initialized")
    packed = _WORKER_DATASET[int(ordinal)]
    example = packed.training_example()
    path = _WORKER_LATTICE.decode(example.candidates, example.score)
    predicted_events = tuple(path.joint_events(example.candidates))
    predicted_rows = build_inference_rows(
        _WORKER_LATTICE, example.candidates, example.score, path
    )
    predicted = attach_training_targets(
        predicted_rows,
        predicted_events=predicted_events,
        target_events=example.target_events,
        target_deletions=example.target_deletions,
        rhythm_rows=packed.target.get("layer3_rhythm") or (),
        score=example.score,
    )
    oracle_rows = build_oracle_rows(
        _WORKER_LATTICE,
        example.candidates,
        example.score,
        example.target_events,
        example.target_deletions,
    )
    oracle = attach_training_targets(
        oracle_rows,
        predicted_events=example.target_events,
        target_events=example.target_events,
        target_deletions=example.target_deletions,
        rhythm_rows=packed.target.get("layer3_rhythm") or (),
        score=example.score,
    )
    duration = (
        float(example.candidates[-1].end)
        if example.candidates
        else (
            float(example.target_events[-1].end)
            if example.target_events
            else 0.0
        )
    )
    metadata = {
        "source": packed.source,
        "repeats": "repeat" if packed.target.get("layer1_repeats") else "ordinary",
        "duration": (
            "lt_5s" if duration < 5.0 else "5_to_10s" if duration < 10.0 else "ge_10s"
        ),
        "duration_sec": duration,
        "candidate_count": len(example.candidates),
        "predicted_event_count": len(predicted_events),
        "target_event_count": len(example.target_events),
        "score_event_count": len(example.score),
    }
    score_json = [
        {
            "index": int(value.index),
            "pitch": int(value.pitch),
            "ql_start": float(value.ql_start),
            "ql_end": float(value.ql_end),
            "measure": value.measure,
        }
        for value in example.score
    ]
    return (
        int(ordinal),
        packed.split,
        packed.sample,
        packed.source,
        _encode(labeled_to_json(predicted)),
        _encode(labeled_to_json(oracle)),
        json.dumps(score_json, sort_keys=True),
        json.dumps(metadata, sort_keys=True),
    )


def _open_cache(
    path: Path,
    *,
    pack_id: str,
    upstream_sha256: str,
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS clips("
        "ordinal INTEGER PRIMARY KEY,split TEXT NOT NULL CHECK(split IN ('train','val')),"
        "sample TEXT UNIQUE NOT NULL,source TEXT NOT NULL,predicted BLOB NOT NULL,"
        "oracle BLOB NOT NULL,score TEXT NOT NULL,metadata TEXT NOT NULL)"
    )
    expected = {
        "schema_version": "align-frozen-error-example-sqlite-v1",
        "pack_id": pack_id,
        "upstream_sha256": upstream_sha256,
        "feature_dim": str(FEATURE_DIM),
        "locked_test_touched": "false",
    }
    saved = dict(connection.execute("SELECT key,value FROM metadata"))
    if saved and any(saved.get(key) != str(value) for key, value in expected.items()):
        raise ValueError("Existing error-head cache belongs to another contract")
    connection.executemany(
        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
        [(key, str(value)) for key, value in expected.items()],
    )
    connection.commit()
    return connection


def extract_examples(
    *,
    ready: Mapping[str, Any],
    checkpoint: Path,
    output_dir: Path,
    workers: int,
    max_train: int | None,
    max_val: int | None,
) -> Path:
    cache_path = output_dir / "examples.sqlite"
    connection = _open_cache(
        cache_path,
        pack_id=str(ready["hashes"]["pack_id"]),
        upstream_sha256=sha256_file(checkpoint),
    )
    packed = PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    )
    try:
        ordinals: list[int] = []
        for split, limit in (("train", max_train), ("val", max_val)):
            selected = packed.ordinals(split)
            if limit is not None:
                selected = selected[: max(0, int(limit))]
            ordinals.extend(selected)
        existing = {
            int(row[0]) for row in connection.execute("SELECT ordinal FROM clips")
        }
        pending = [value for value in ordinals if value not in existing]
        started = time.perf_counter()
        completed = len(ordinals) - len(pending)
        process = psutil.Process() if psutil is not None else None
        if process is not None:
            process.cpu_percent(None)
        if pending:
            with ProcessPoolExecutor(
                max_workers=max(1, workers),
                initializer=_worker_initialize,
                initargs=(
                    str(ready["paths"]["packed_root"]),
                    str(checkpoint.resolve()),
                    1,
                ),
            ) as pool:
                for result in pool.map(_extract_ordinal, pending, chunksize=1):
                    connection.execute(
                        "INSERT OR REPLACE INTO clips VALUES(?,?,?,?,?,?,?,?)",
                        result,
                    )
                    completed += 1
                    if completed % 8 == 0:
                        connection.commit()
                    if (
                        completed == 1
                        or completed % 10 == 0
                        or completed == len(ordinals)
                    ):
                        elapsed = time.perf_counter() - started
                        built = max(completed - (len(ordinals) - len(pending)), 1)
                        rate = built / max(elapsed, 1e-9)
                        eta = max(len(ordinals) - completed, 0) / max(rate, 1e-9)
                        progress = {
                            "phase": "extract",
                            "rows": completed,
                            "total_rows": len(ordinals),
                            "rows_per_sec": rate,
                            "eta_seconds": eta,
                            "workers": workers,
                            "utilization": _utilization(process),
                            "locked_test_touched": False,
                        }
                        _atomic_json(output_dir / "progress.json", progress)
                        print(
                            f"extract={completed}/{len(ordinals)} "
                            f"rows_per_sec={rate:.3f} eta_sec={eta:.1f}",
                            flush=True,
                        )
        connection.commit()
        counts = dict(
            connection.execute(
                "SELECT split,COUNT(*) FROM clips GROUP BY split"
            )
        )
        expected_train = (
            min(int(ready["counts"]["train"]), max_train)
            if max_train is not None
            else int(ready["counts"]["train"])
        )
        expected_val = (
            min(int(ready["counts"]["val"]), max_val)
            if max_val is not None
            else int(ready["counts"]["val"])
        )
        if counts != {"train": expected_train, "val": expected_val}:
            raise ValueError(f"Extracted split counts mismatch: {counts}")
        _atomic_json(
            output_dir / "extraction_report.json",
            {
                "schema_version": OUTPUT_SCHEMA,
                "cache": str(cache_path.resolve()),
                "counts": counts,
                "workers": workers,
                "pack_id": ready["hashes"]["pack_id"],
                "upstream_sha256": sha256_file(checkpoint),
                "actual_predicted_paths": True,
                "oracle_upstream_ablation": True,
                "sequence_consistent_target_attachment": True,
                "locked_test_touched": False,
            },
        )
        return cache_path
    finally:
        packed.close()
        connection.close()


def _load_clips(
    cache_path: Path,
    *,
    split: str,
    variant: str,
) -> list[tuple[str, Any, list[dict[str, Any]], dict[str, Any]]]:
    if variant not in {"predicted", "oracle"}:
        raise ValueError(variant)
    connection = sqlite3.connect(f"file:{cache_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f"SELECT sample,{variant},score,metadata FROM clips "
            "WHERE split=? ORDER BY ordinal",
            (split,),
        )
        return [
            (
                str(sample),
                labeled_from_json(_decode(blob)),
                json.loads(score),
                json.loads(metadata),
            )
            for sample, blob, score, metadata in rows
        ]
    finally:
        connection.close()


def _tensor_targets(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.asarray(value[selected])).to(device)
        for name, value in arrays.items()
        if name != "features"
    }


def _sanity_checks(
    arrays: Mapping[str, np.ndarray],
    *,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    count = min(128, len(arrays["features"]))
    if count < 8:
        raise ValueError("Sanity checks require at least eight training rows")
    device = torch.device("cpu")
    features = torch.from_numpy(arrays["features"][:count]).to(device)
    targets = {
        name: torch.from_numpy(value[:count]).to(device)
        for name, value in arrays.items()
        if name != "features"
    }
    weights = measured_class_weights(
        arrays["layer2"], len(LAYER2_CLASSES)
    )
    torch.manual_seed(seed)
    model = FrozenUpstreamErrorHeads(
        ErrorHeadsConfig(hidden_dim=48, dropout=0.0)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _step in range(40):
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss, _detail = error_head_loss(
            output, targets, layer2_weights=weights
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite overfit sanity loss")
        loss.backward()
        if not any(
            parameter.grad is not None and torch.any(parameter.grad != 0)
            for parameter in model.parameters()
        ):
            raise RuntimeError("No gradient reached downstream heads")
        optimizer.step()
        losses.append(float(loss.detach()))
    report = {
        "smoke_forward_backward": "passed",
        "gradient_flow": "passed",
        "overfit_rows": count,
        "overfit_initial_loss": losses[0],
        "overfit_final_loss": losses[-1],
        "overfit_loss_reduced": losses[-1] < losses[0],
    }
    if not report["overfit_loss_reduced"]:
        raise RuntimeError("Tiny-subset overfit sanity did not reduce loss")
    _atomic_json(output_dir / "sanity.json", report)
    return report


def _benchmark(
    arrays: Mapping[str, np.ndarray],
    *,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    device = torch.device("cpu")
    weights = measured_class_weights(
        arrays["layer2"], len(LAYER2_CLASSES)
    )
    results = []
    for batch_size in (2048, 8192, 32768):
        selected = np.arange(min(batch_size, len(arrays["features"])))
        for amp in ("float32", "bfloat16"):
            torch.manual_seed(seed)
            model = FrozenUpstreamErrorHeads()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            features = torch.from_numpy(arrays["features"][selected]).to(device)
            targets = _tensor_targets(arrays, selected, device)
            started = time.perf_counter()
            ok = True
            error = None
            iterations = 3
            try:
                for _ in range(iterations):
                    optimizer.zero_grad(set_to_none=True)
                    context = (
                        torch.autocast("cpu", dtype=torch.bfloat16)
                        if amp == "bfloat16"
                        else torch.autocast("cpu", enabled=False)
                    )
                    with context:
                        output = model(features)
                        loss, _detail = error_head_loss(
                            output, targets, layer2_weights=weights
                        )
                    loss.backward()
                    optimizer.step()
            except (RuntimeError, TypeError) as exc:
                ok = False
                error = str(exc)
            seconds = time.perf_counter() - started
            results.append(
                {
                    "batch_size": len(selected),
                    "amp_dtype": amp,
                    "torch_compile": False,
                    "workers": 0,
                    "iterations": iterations,
                    "seconds": seconds,
                    "rows_per_sec": (
                        len(selected) * iterations / max(seconds, 1e-9)
                        if ok
                        else 0.0
                    ),
                    "ok": ok,
                    "error": error,
                }
            )
    selected = max(
        (row for row in results if row["ok"]),
        key=lambda row: row["rows_per_sec"],
    )
    profile = {
        "schema_version": "align-error-head-hardware-profile-v1",
        "device": "cpu",
        "results": results,
        "selected": selected,
        "compile": {
            "selected": False,
            "reason": "Small tabular heads; eager avoids graph startup and Windows compiler dependency.",
        },
        "loader": {
            "source": "resumable SQLite extraction cache",
            "workers": 0,
            "prefetch": 0,
            "packed_extraction_workers": None,
        },
    }
    _atomic_json(output_dir / "benchmark.json", profile)
    return profile


def _probabilities(
    model: FrozenUpstreamErrorHeads,
    arrays: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layer2 = []
    rhythm = []
    deviation = []
    subtype = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays["features"]), batch_size):
            features = torch.from_numpy(
                arrays["features"][start : start + batch_size]
            ).to(device)
            output = model(features)
            layer2.append(torch.softmax(output.layer2_logits, -1).cpu().numpy())
            rhythm.append(torch.sigmoid(output.rhythm_logit).cpu().numpy())
            deviation.append(output.deviation_sec.cpu().numpy())
            subtype.append(output.rhythm_subtype_logits.cpu().numpy())
    return (
        np.concatenate(layer2),
        np.concatenate(rhythm),
        np.concatenate(deviation),
        np.concatenate(subtype),
    )


def _clip_predictions(
    clips: Sequence[tuple[str, Any, Any, dict[str, Any]]],
    probabilities: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    thresholds: Mapping[str, Any],
) -> list[tuple[str, Any, Any, Mapping[str, Any]]]:
    layer2, rhythm, deviation, subtype = probabilities
    output = []
    cursor = 0
    for sample, rows, _score, metadata in clips:
        end = cursor + len(rows.rows)
        prediction = decode_probabilities(
            rows.rows,
            layer2[cursor:end],
            rhythm[cursor:end],
            deviation[cursor:end],
            subtype[cursor:end],
            thresholds,
        )
        output.append((sample, rows, prediction, metadata))
        cursor = end
    if cursor != len(layer2):
        raise RuntimeError("Clip prediction cursor diverged")
    return output


def _train_variant(
    *,
    variant: str,
    cache_path: Path,
    output_dir: Path,
    data_fingerprint: str,
    upstream: Mapping[str, Any],
    epochs: int,
    seed: int,
    benchmark: Mapping[str, Any],
    resume: bool,
) -> Path:
    stop_signal: dict[str, int | None] = {"number": None}

    def request_stop(number: int, _frame: Any) -> None:
        # Finish the current optimizer step, then atomically persist the exact
        # sampler cursor/RNG state before honoring termination.
        stop_signal["number"] = int(number)

    for signal_name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, signal_name, None)
        if number is not None:
            signal.signal(number, request_stop)
    train_clips = _load_clips(cache_path, split="train", variant=variant)
    val_clips = _load_clips(cache_path, split="val", variant=variant)
    train = stack_labeled_rows([value[1] for value in train_clips])
    val = stack_labeled_rows([value[1] for value in val_clips])
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    if variant == "predicted":
        _sanity_checks(train, output_dir=output_dir, seed=seed)
    selected_profile = benchmark["selected"]
    batch_size = min(int(selected_profile["batch_size"]), len(train["features"]))
    amp_name = str(selected_profile["amp_dtype"])
    device = torch.device("cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    last_path = variant_dir / "last_checkpoint.pt"
    resume_payload = None
    if resume and last_path.is_file():
        model, resume_payload = load_error_heads(
            last_path,
            device=device,
            expected_data_fingerprint=data_fingerprint,
            expected_upstream={"decoder": upstream["decoder"]},
        )
    else:
        model = FrozenUpstreamErrorHeads().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    scaler = None
    class_weights = measured_class_weights(
        train["layer2"], len(LAYER2_CLASSES)
    )
    history = list(resume_payload.get("history") or []) if resume_payload else []
    best_score = max(
        (float(row.get("selection_score", -math.inf)) for row in history),
        default=-math.inf,
    )
    best_path = variant_dir / "best.pt"
    process = psutil.Process() if psutil is not None else None
    first_epoch = 1
    resume_position = 0
    resume_loss = 0.0
    resume_components: dict[str, float] = {}
    resume_batches = 0
    elapsed_before = 0.0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if resume_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        _restore_rng_state(resume_payload["rng_state"])
        progress = resume_payload.get("progress") or {}
        if progress.get("phase") == "training":
            first_epoch = int(progress["epoch"])
            sampler = progress.get("sampler") or {}
            if int(sampler.get("seed", -1)) != seed + first_epoch:
                raise ValueError("Resume sampler seed mismatch")
            resume_position = int(sampler.get("position", 0))
            resume_loss = float(progress.get("total_loss", 0.0))
            resume_components = {
                str(name): float(value)
                for name, value in (progress.get("components") or {}).items()
            }
            resume_batches = int(progress.get("optimizer_steps", 0))
            elapsed_before = float(progress.get("elapsed_seconds", 0.0))
        else:
            first_epoch = int(progress.get("epoch", 0)) + 1
    if first_epoch > epochs:
        if not best_path.is_file():
            raise RuntimeError("Resume is complete but has no best checkpoint")
        return best_path
    for epoch in range(first_epoch, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(
            len(train["features"])
        )
        started = time.perf_counter()
        continuing = epoch == first_epoch and resume_position > 0
        total_loss = resume_loss if continuing else 0.0
        completed = resume_position if continuing else 0
        components = dict(resume_components) if continuing else {}
        batch_index = resume_batches if continuing else 0
        for batch_index, start in enumerate(
            range(completed, len(order), batch_size), batch_index + 1
        ):
            selected = order[start : start + batch_size]
            features = torch.from_numpy(train["features"][selected]).to(device)
            targets = _tensor_targets(train, selected, device)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cpu", dtype=torch.bfloat16)
                if amp_name == "bfloat16"
                else torch.autocast("cpu", enabled=False)
            )
            with context:
                output = model(features)
                loss, detail = error_head_loss(
                    output, targets, layer2_weights=class_weights
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
            completed += len(selected)
            for name, value in detail.items():
                components[name] = components.get(name, 0.0) + float(value.detach())
            elapsed = time.perf_counter() - started
            total_elapsed = (elapsed_before if continuing else 0.0) + elapsed
            rate = completed / max(total_elapsed, 1e-9)
            eta = (len(order) - completed) / max(rate, 1e-9)
            progress = {
                "phase": "training",
                "variant": variant,
                "epoch": epoch,
                "sampler": {
                    "seed": seed + epoch,
                    "position": completed,
                    "rows": len(order),
                },
                "optimizer_steps": batch_index,
                "total_loss": total_loss,
                "components": components,
                "elapsed_seconds": total_elapsed,
                "rows_per_sec": rate,
                "eta_seconds": eta,
                "utilization": _utilization(process),
                "termination_signal": stop_signal["number"],
            }
            if (
                batch_index % 5 == 0
                or completed == len(order)
                or stop_signal["number"] is not None
            ):
                save_checkpoint_atomic(
                    variant_dir / "last_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=progress,
                    history=history,
                    data_fingerprint=data_fingerprint,
                    upstream=upstream,
                )
                _atomic_json(output_dir / "progress.json", progress)
            if stop_signal["number"] is not None:
                raise SystemExit(128 + int(stop_signal["number"]))
        scheduler.step()
        val_probabilities = _probabilities(
            model, val, batch_size=batch_size, device=device
        )
        thresholds = calibrate_thresholds(
            val_probabilities[0],
            val_probabilities[1],
            val,
            deviation_predictions=val_probabilities[2],
        )
        evaluated = evaluate_predictions(
            _clip_predictions(val_clips, val_probabilities, thresholds)
        )
        score = (
            float(evaluated["full_typed_error_f1"])
            + 0.25 * float(evaluated["layer3"]["rhythm"]["f1"])
        )
        row = {
            "epoch": epoch,
            "mean_loss": total_loss / max(completed, 1),
            "rows": completed,
            "rows_per_sec": completed / max(time.perf_counter() - started, 1e-9),
            "component_batch_means": {
                name: value / max(batch_index, 1)
                for name, value in components.items()
            },
            "validation": evaluated,
            "thresholds": thresholds,
            "selection_score": score,
        }
        history.append(row)
        if score > best_score:
            best_score = score
            save_checkpoint_atomic(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                progress={"phase": "epoch_complete", "variant": variant, "epoch": epoch},
                history=history,
                data_fingerprint=data_fingerprint,
                upstream=upstream,
                thresholds=thresholds,
            )
        save_checkpoint_atomic(
            variant_dir / "last_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            progress={"phase": "epoch_complete", "variant": variant, "epoch": epoch},
            history=history,
            data_fingerprint=data_fingerprint,
            upstream=upstream,
            thresholds=thresholds,
        )
        _atomic_json(variant_dir / "history.json", {"history": history})
        print(
            f"variant={variant} epoch={epoch} "
            f"typed_f1={evaluated['full_typed_error_f1']:.6f} "
            f"rhythm_f1={evaluated['layer3']['rhythm']['f1']:.6f}",
            flush=True,
        )
        resume_position = 0
        resume_loss = 0.0
        resume_components = {}
        resume_batches = 0
        elapsed_before = 0.0
    return best_path


def _score_events(score_rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    from alignmodel.joint.index import ScoreEvent

    return [
        ScoreEvent(
            index=int(row["index"]),
            pitch=int(row["pitch"]),
            ql_start=float(row["ql_start"]),
            ql_end=float(row["ql_end"]),
            source_indices=(int(row["index"]),),
            measure=row.get("measure"),
        )
        for row in score_rows
    ]


def evaluate(
    *,
    cache_path: Path,
    output_dir: Path,
    data_fingerprint: str,
    upstream: Mapping[str, Any],
    expected_val_rows: int,
) -> dict[str, Any]:
    reports = {}
    predictions_by_variant = {}
    for variant in ("predicted", "oracle"):
        clips = _load_clips(cache_path, split="val", variant=variant)
        if len(clips) != expected_val_rows:
            raise ValueError(
                f"{variant} validation has {len(clips)}, expected {expected_val_rows}"
            )
        arrays = stack_labeled_rows([value[1] for value in clips])
        model, payload = load_error_heads(
            output_dir / variant / "best.pt",
            expected_data_fingerprint=data_fingerprint,
            expected_upstream={"decoder": upstream["decoder"]},
        )
        probabilities = _probabilities(
            model, arrays, batch_size=32768, device=torch.device("cpu")
        )
        predictions = _clip_predictions(
            clips, probabilities, payload["thresholds"]
        )
        reports[variant] = evaluate_predictions(predictions)
        predictions_by_variant[variant] = (clips, predictions)

    predicted_clips = predictions_by_variant["predicted"][0]
    heuristic_clips = [
        (
            sample,
            rows,
            heuristic_prediction(rows.rows),
            metadata,
        )
        for sample, rows, _score, metadata in predicted_clips
    ]
    heuristic = evaluate_predictions(heuristic_clips)
    schema_path = output_dir / "validation_predictions_schema_1_2.jsonl"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        prediction_lookup = {
            sample: prediction
            for sample, _rows, prediction, _metadata in predictions_by_variant[
                "predicted"
            ][1]
        }
        for sample, rows, score_rows, _metadata in predicted_clips:
            document = schema12_document(
                sample,
                rows.rows,
                prediction_lookup[sample],
                _score_events(score_rows),
            )
            stream.write(json.dumps(document, sort_keys=True))
            stream.write("\n")
    os.replace(temporary, schema_path)
    report = {
        "schema_version": "align-frozen-error-heads-evaluation-v1",
        "validation_rows": expected_val_rows,
        "predicted_upstream": reports["predicted"],
        "oracle_upstream": reports["oracle"],
        "oracle_ceiling_delta": {
            "typed_error_f1": (
                reports["oracle"]["full_typed_error_f1"]
                - reports["predicted"]["full_typed_error_f1"]
            ),
            "rhythm_f1": (
                reports["oracle"]["layer3"]["rhythm"]["f1"]
                - reports["predicted"]["layer3"]["rhythm"]["f1"]
            ),
        },
        "current_rules_heuristics": heuristic,
        "historical_baselines": {
            "EditCropNet": {
                "status": "not_directly_runnable_on_verified_pack",
                "reason": "requires performance_mel crop inputs absent from this packed release",
                "registry_result": "historical 20-clip calibration F1 0.129; not same validation",
            },
            "RhythmNet": {
                "status": "not_directly_runnable_on_verified_pack",
                "reason": "requires mel crops absent from this packed release",
                "registry_result": "historical model suppressed at logit threshold 20; not same validation",
            },
        },
        "schema_1_2_predictions": str(schema_path.resolve()),
        "schema_1_2_rows": expected_val_rows,
        "upstream": upstream,
        "data_fingerprint": data_fingerprint,
        "intonation_masked": True,
        "locked_test_touched": False,
    }
    _atomic_json(output_dir / "evaluation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ready-marker",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/DATA_READY.json"),
    )
    parser.add_argument(
        "--upstream-checkpoint",
        type=Path,
        default=Path(
            "runs/joint-audit-v2/end-to-end-v2/"
            "weak-note-continuation-optimized/joint_decoder.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/error-heads-v1"),
    )
    parser.add_argument(
        "--resource-status", type=Path, default=DEFAULT_STATUS_PATH
    )
    parser.add_argument(
        "--phase", choices=("all", "extract", "train", "evaluate"), default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume atomic mid-epoch checkpoints when present.",
    )
    args = parser.parse_args()

    ready = verify_data_ready(args.ready_marker)
    if int(ready["counts"]["train"]) != 4544 or int(ready["counts"]["val"]) != 358:
        raise ValueError("Error heads require the exact 4,544/358 verified release")
    if (
        int(ready["counts"].get("locked_test_metadata_only", -1)) != 4022
        or ready["verification"].get("test_features_materialized") is not False
        or ready["verification"].get("test_targets_materialized") is not False
    ):
        raise ValueError("Lockbox contract is not sealed metadata-only")
    packed_metadata = json.loads(
        (
            Path(str(ready["paths"]["packed_root"])) / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    upstream, _model, _lattice_config = _validate_completed_upstream(
        args.upstream_checkpoint, packed_metadata
    )
    del _model, _lattice_config
    data_fingerprint = str(ready["hashes"]["pack_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resource_snapshot = (
        json.loads(args.resource_status.read_text(encoding="utf-8"))
        if args.resource_status.is_file()
        else {}
    )
    lease_id = None
    if args.device == "cuda":
        from alignmodel.training_resources import (
            ResourceBusyError,
            claim_resource,
            release_resource,
        )

        try:
            lease_id = claim_resource(
                args.resource_status,
                "gpu",
                track="error-heads-v1",
                command=sys.argv,
                metadata={"data_fingerprint": data_fingerprint},
            )
        except ResourceBusyError:
            raise RuntimeError(
                "GPU lease is active; rerun on CPU or wait for release"
            ) from None
        atexit.register(
            release_resource, args.resource_status, "gpu", lease_id
        )
    config = {
        "schema_version": OUTPUT_SCHEMA,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data_ready": str(args.ready_marker.resolve()),
        "data_fingerprint": data_fingerprint,
        "split_counts": {"train": 4544, "val": 358},
        "locked_test_metadata_rows": 4022,
        "upstream": upstream,
        "upstream_frozen": True,
        "target_policy": {
            "exact_audited_lineage_only": True,
            "targets_available_at_inference": False,
            "intonation_masked": True,
            "replay_rhythm_excluded": True,
        },
        "resource_coordination": {
            "status_path": str(args.resource_status.resolve()),
            "observed_gpu_lease": (resource_snapshot.get("leases") or {}).get("gpu"),
            "requested_device": args.device,
            "gpu_lease_id": lease_id,
            "bounded_extraction_workers": args.workers,
        },
        "production_weights_modified": False,
        "locked_test_touched": False,
    }
    _atomic_json(args.output_dir / "config.json", config)

    cache_path = args.output_dir / "examples.sqlite"
    if args.phase in {"all", "extract"}:
        cache_path = extract_examples(
            ready=ready,
            checkpoint=args.upstream_checkpoint,
            output_dir=args.output_dir,
            workers=max(1, args.workers),
            max_train=args.max_train,
            max_val=args.max_val,
        )
    if args.phase == "extract":
        return
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    if args.phase in {"all", "train"}:
        predicted_train = _load_clips(
            cache_path, split="train", variant="predicted"
        )
        train_arrays = stack_labeled_rows(
            [value[1] for value in predicted_train]
        )
        benchmark = _benchmark(
            train_arrays, output_dir=args.output_dir, seed=args.seed
        )
        benchmark["loader"]["packed_extraction_workers"] = args.workers
        _atomic_json(args.output_dir / "benchmark.json", benchmark)
        for variant in ("predicted", "oracle"):
            _train_variant(
                variant=variant,
                cache_path=cache_path,
                output_dir=args.output_dir,
                data_fingerprint=data_fingerprint,
                upstream=upstream,
                epochs=max(1, args.epochs),
                seed=args.seed,
                benchmark=benchmark,
                resume=args.resume,
            )
    if args.phase == "train":
        return

    expected_val = (
        min(358, args.max_val) if args.max_val is not None else 358
    )
    evaluation = evaluate(
        cache_path=cache_path,
        output_dir=args.output_dir,
        data_fingerprint=data_fingerprint,
        upstream=upstream,
        expected_val_rows=expected_val,
    )
    material_delta = (
        evaluation["predicted_upstream"]["full_typed_error_f1"]
        - evaluation["current_rules_heuristics"]["full_typed_error_f1"]
    )
    rhythm_delta = (
        evaluation["predicted_upstream"]["layer3"]["rhythm"]["f1"]
        - evaluation["current_rules_heuristics"]["layer3"]["rhythm"]["f1"]
    )
    report = {
        "schema_version": "align-frozen-error-heads-report-v1",
        "data_fingerprint": data_fingerprint,
        "train_rows": (
            min(4544, args.max_train)
            if args.max_train is not None
            else 4544
        ),
        "validation_rows": expected_val,
        "evaluation": evaluation,
        "promotion_gate": {
            "full_validation_completed": expected_val == 358,
            "integration_tests_required": True,
            "delta_typed_f1_vs_rules": material_delta,
            "delta_rhythm_f1_vs_rules": rhythm_delta,
            "material_improvement_threshold": 0.02,
            "passed_metric": (
                expected_val == 358
                and material_delta >= 0.02
                and rhythm_delta >= 0.0
            ),
            "failure_reason": (
                "Layer 3 rhythm did not improve over current rules"
                if rhythm_delta < 0.0
                else None
            ),
            "production_promotion_performed": False,
        },
        "staged_unfreezing": {
            "frozen_baseline_measured": True,
            "evaluated": False,
            "reason": (
                "Deferred because the main outputRaw full-pipeline track still "
                "holds the repository GPU lease; no resource contention or "
                "upstream weight mutation was permitted."
            ),
            "active_gpu_track": (
                ((resource_snapshot.get("leases") or {}).get("gpu") or {}).get(
                    "track"
                )
            ),
        },
        "locked_test_touched": False,
    }
    _atomic_json(args.output_dir / "report.json", report)
    print(
        f"predicted_typed_f1="
        f"{evaluation['predicted_upstream']['full_typed_error_f1']:.6f} "
        f"oracle_typed_f1="
        f"{evaluation['oracle_upstream']['full_typed_error_f1']:.6f} "
        f"delta_vs_rules={material_delta:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
