from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import random
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import psutil
import torch

from alignmodel.joint.candidate_rescorer import (
    CandidateRescorer,
    load_candidate_rescorer,
    rescore_candidates_with_indices,
)
from alignmodel.joint.lattice import LatticeConfig, SparseJointLattice
from alignmodel.joint.outputraw_full import (
    FullJointPipelineModel,
    FullPipelineAugmentConfig,
    FullPipelineLossConfig,
    FullPipelineModelConfig,
    atomic_checkpoint,
    load_checkpoint,
    restore_rng_state,
    verify_data_ready,
)
from alignmodel.joint.outputraw_train import (
    atomic_json,
    evaluate_packed_validation,
    iter_local_batches,
    iter_prepared_local_batches,
    load_stage_profile,
    require_exclusive_cuda,
    train_local_batch,
)
from alignmodel.joint.packed_data import PackedCursor, PackedJointDataset
from alignmodel.joint.prepared_local import PreparedLocalDataset
from alignmodel.training_resources import (
    DEFAULT_STATUS_PATH,
    claim_resource,
    release_resource,
)


STAGES = ("acoustic", "structure", "errors", "joint")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer(
    model: FullJointPipelineModel,
    *,
    learning_rate: float,
    weight_decay: float,
    fused: bool,
    legacy_path_lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    regular_parameters = []
    legacy_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            legacy_parameters
            if name.startswith("legacy_path.")
            else regular_parameters
        )
        target.append(parameter)
    parameters: list[Any] = []
    if regular_parameters:
        parameters.append({"params": regular_parameters})
    if legacy_parameters:
        parameters.append(
            {
                "params": legacy_parameters,
                "lr": learning_rate * legacy_path_lr_scale,
            }
        )
    kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(parameters, **kwargs)


def _amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if name == "float32":
        return None
    if device.type != "cuda":
        return torch.bfloat16 if name == "bfloat16" else None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported profiled AMP dtype {name!r}")


def _checkpoint_progress(
    *,
    stage_index: int,
    stage: str,
    epoch: int,
    cursor: PackedCursor,
    optimizer_steps: int,
    examples: int,
    edges: int,
    total_loss: float,
    batches: int,
    elapsed_seconds: float,
    phase_seconds: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "phase": "local",
        "stage_index": stage_index,
        "stage": stage,
        "epoch": epoch,
        "cursor": cursor.to_dict(),
        "optimizer_steps": optimizer_steps,
        "examples": examples,
        "edges": edges,
        "total_loss": total_loss,
        "batches": batches,
        "elapsed_seconds": elapsed_seconds,
        "phase_seconds": dict(phase_seconds),
    }


def _train_local_epoch(
    *,
    model: FullJointPipelineModel,
    dataset: PackedJointDataset,
    prepared_cache: PreparedLocalDataset | None,
    lattice_config: LatticeConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    loss_config: FullPipelineLossConfig,
    stage_index: int,
    stage: str,
    epoch: int,
    seed: int,
    batch_edges: int,
    accumulation: int,
    workers: int,
    prefetch: int,
    checkpoint_every: int,
    output_dir: Path,
    data_fingerprint: str,
    checkpoint_metadata: Mapping[str, Any],
    history: list[dict[str, Any]],
    resume_progress: Mapping[str, Any] | None,
    max_examples: int | None,
    fixed_order: bool,
) -> dict[str, Any]:
    if resume_progress is not None:
        cursor = PackedCursor.from_dict(resume_progress["cursor"])
        total_loss = float(resume_progress.get("total_loss", 0.0))
        examples = int(resume_progress.get("examples", 0))
        edges = int(resume_progress.get("edges", 0))
        optimizer_steps = int(resume_progress.get("optimizer_steps", 0))
        completed_batches = int(resume_progress.get("batches", 0))
        elapsed_before_resume = float(
            resume_progress.get("elapsed_seconds", 0.0)
        )
        phase_seconds = {
            str(name): float(value)
            for name, value in (
                resume_progress.get("phase_seconds") or {}
            ).items()
        }
    else:
        cursor = dataset.cursor(
            "train",
            epoch=0 if fixed_order else stage_index * 10_000 + epoch,
            seed=seed,
        )
        total_loss = 0.0
        examples = edges = optimizer_steps = 0
        completed_batches = 0
        elapsed_before_resume = 0.0
        phase_seconds = {}
    initial_examples = examples
    initial_edges = edges

    def add_phase(name: str, seconds: float) -> None:
        phase_seconds[name] = phase_seconds.get(name, 0.0) + seconds

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    lattice = SparseJointLattice(model, lattice_config)
    optimizer.zero_grad(set_to_none=True)
    epoch_started = time.perf_counter()
    runtime_process = psutil.Process()
    runtime_process.cpu_percent(None)
    psutil.cpu_percent(None)
    initial_io = runtime_process.io_counters()
    last_checkpoint = examples
    batch_position = 0
    batches = (
        iter_prepared_local_batches(
            dataset,
            prepared_cache,
            cursor,
            max_edges=batch_edges,
            max_samples=(
                max(max_examples - examples, 0)
                if max_examples is not None
                else None
            ),
        )
        if prepared_cache is not None
        else iter_local_batches(
            dataset,
            cursor,
            lattice=lattice,
            max_edges=batch_edges,
            workers=workers,
            prefetch=prefetch,
            max_samples=(
                max(max_examples - examples, 0)
                if max_examples is not None
                else None
            ),
        )
    )
    batch_iterator = iter(batches)
    while True:
        data_started = time.perf_counter()
        try:
            next_cursor, batch = next(batch_iterator)
        except StopIteration:
            break
        add_phase("data_and_collation", time.perf_counter() - data_started)
        if max_examples is not None and examples >= max_examples:
            break
        batch_position += 1
        loss, detail, batch_timing = train_local_batch(
            model,
            batch,
            device=device,
            loss_config=loss_config,
            amp_dtype=amp_dtype,
        )
        for name, seconds in batch_timing.items():
            add_phase(name, seconds)
        scaled_loss = loss / max(1, accumulation)
        backward_started = time.perf_counter()
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        synchronize()
        add_phase("backward", time.perf_counter() - backward_started)
        total_loss += float(loss.detach().float().cpu())
        examples += len(batch.samples)
        edges += batch.edge_count
        completed_batches += 1
        should_step = batch_position % max(1, accumulation) == 0
        if should_step:
            optimizer_started = time.perf_counter()
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                2.0,
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            synchronize()
            add_phase("optimizer", time.perf_counter() - optimizer_started)
        cursor = next_cursor
        elapsed = time.perf_counter() - epoch_started
        if (
            batch_position == 1
            or batch_position % 10 == 0
            or cursor.position == len(dataset.ordinals("train"))
        ):
            completed_now = max(examples - initial_examples, 1)
            rate = completed_now / max(elapsed, 1e-9)
            total = (
                min(len(dataset.ordinals("train")), max_examples)
                if max_examples is not None
                else len(dataset.ordinals("train"))
            )
            eta = max(total - examples, 0) / max(rate, 1e-9)
            memory = runtime_process.memory_info()
            current_io = runtime_process.io_counters()
            utilization = {
                "process_cpu_percent": runtime_process.cpu_percent(None),
                "machine_cpu_percent": psutil.cpu_percent(None),
                "rss_mb": memory.rss / 1024**2,
                "read_mib": (
                    current_io.read_bytes - initial_io.read_bytes
                )
                / 1024**2,
                "write_mib": (
                    current_io.write_bytes - initial_io.write_bytes
                )
                / 1024**2,
                "vram_allocated_mb": (
                    torch.cuda.memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
                "vram_reserved_mb": (
                    torch.cuda.memory_reserved(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            print(
                f"stage={stage} epoch={epoch} examples={examples}/{total} "
                f"edges={edges} loss={total_loss / completed_batches:.5f} "
                f"rows_per_sec={rate:.3f} eta_sec={eta:.1f} "
                f"phase_seconds={json.dumps(phase_seconds, sort_keys=True)} "
                f"utilization={json.dumps(utilization, sort_keys=True)} "
                f"components={json.dumps(detail, sort_keys=True)}",
                flush=True,
            )
        if (
            should_step
            and examples - last_checkpoint >= checkpoint_every
        ):
            progress = _checkpoint_progress(
                stage_index=stage_index,
                stage=stage,
                epoch=epoch,
                cursor=cursor,
                optimizer_steps=optimizer_steps,
                examples=examples,
                edges=edges,
                total_loss=total_loss,
                batches=completed_batches,
                elapsed_seconds=elapsed_before_resume + elapsed,
                phase_seconds=phase_seconds,
            )
            checkpoint_started = time.perf_counter()
            atomic_checkpoint(
                output_dir / "last_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                progress=progress,
                data_fingerprint=data_fingerprint,
                history=history,
                checkpoint_metadata=checkpoint_metadata,
            )
            add_phase("checkpoint_io", time.perf_counter() - checkpoint_started)
            atomic_json(
                output_dir / "progress.json",
                {
                    **progress,
                    "updated_unix": time.time(),
                    "elapsed_seconds": elapsed_before_resume + elapsed,
                    "phase_seconds": phase_seconds,
                    "locked_test_touched": False,
                },
            )
            last_checkpoint = examples
    if batch_position % max(1, accumulation):
        optimizer_started = time.perf_counter()
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            2.0,
        )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        synchronize()
        add_phase("optimizer", time.perf_counter() - optimizer_started)
    scheduler_started = time.perf_counter()
    scheduler.step()
    add_phase("scheduler", time.perf_counter() - scheduler_started)
    elapsed = time.perf_counter() - epoch_started
    elapsed_total = elapsed_before_resume + elapsed
    return {
        "stage": stage,
        "epoch": epoch,
        "examples": examples,
        "edges": edges,
        "optimizer_steps": optimizer_steps,
        "batches": completed_batches,
        "mean_batch_loss": total_loss / max(completed_batches, 1),
        "train_wall_seconds": elapsed_total,
        "examples_per_second": examples / max(elapsed_total, 1e-9),
        "edges_per_second": edges / max(elapsed_total, 1e-9),
        "segment_examples_per_second": (
            (examples - initial_examples) / max(elapsed, 1e-9)
        ),
        "segment_edges_per_second": (
            (edges - initial_edges) / max(elapsed, 1e-9)
        ),
        "phase_seconds": phase_seconds,
        "cursor": cursor.to_dict(),
    }


def _train_structured_path(
    *,
    model: FullJointPipelineModel,
    dataset: PackedJointDataset,
    lattice_config: LatticeConfig,
    epochs: int,
    samples_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    accumulation: int,
    device: torch.device,
    seed: int,
    output_dir: Path,
    data_fingerprint: str,
    checkpoint_metadata: Mapping[str, Any],
    history: list[dict[str, Any]],
    resume_payload: Mapping[str, Any] | None,
    resume_progress: Mapping[str, Any] | None,
    checkpoint_every: int,
    candidate_rescorer: CandidateRescorer | None,
    candidate_threshold: float,
) -> None:
    if epochs <= 0:
        return
    model.freeze_for_stage("joint")
    model.to(device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    if resume_payload is not None and resume_progress is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if resume_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        first_epoch = int(resume_progress["epoch"]) + int(
            resume_progress.get("phase") == "path_epoch_complete"
        )
    else:
        first_epoch = 1
    for epoch in range(first_epoch, epochs + 1):
        lattice = SparseJointLattice(model, lattice_config)
        if (
            resume_progress is not None
            and resume_progress.get("phase") == "structured_path"
            and int(resume_progress["epoch"]) == epoch
        ):
            cursor = PackedCursor.from_dict(resume_progress["cursor"])
            total_loss = float(resume_progress.get("total_loss", 0.0))
            trained = int(resume_progress.get("examples", 0))
            elapsed_before_resume = float(
                resume_progress.get("elapsed_seconds", 0.0)
            )
            phase_seconds = {
                str(name): float(value)
                for name, value in (
                    resume_progress.get("phase_seconds") or {}
                ).items()
            }
        else:
            cursor = dataset.cursor(
                "train", epoch=90_000 + epoch, seed=seed
            )
            total_loss = 0.0
            trained = 0
            elapsed_before_resume = 0.0
            phase_seconds = {}
        initial_trained = trained

        def add_phase(name: str, seconds: float) -> None:
            phase_seconds[name] = phase_seconds.get(name, 0.0) + seconds

        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        iterator = iter(
            dataset.iter_from_cursor(cursor, workers=0, prefetch=0)
        )
        while True:
            data_started = time.perf_counter()
            try:
                next_cursor, packed = next(iterator)
            except StopIteration:
                break
            add_phase("data", time.perf_counter() - data_started)
            if trained >= samples_per_epoch:
                break
            reconstruction_started = time.perf_counter()
            example = packed.training_example()
            candidates = example.candidates
            gold_spans = example.gold_spans
            gold_keep_unlinked = example.gold_keep_unlinked
            if candidate_rescorer is not None:
                candidates, kept_indices = rescore_candidates_with_indices(
                    candidate_rescorer,
                    candidates,
                    threshold=candidate_threshold,
                    score=example.score,
                )
                gold_spans = tuple(gold_spans[index] for index in kept_indices)
                gold_keep_unlinked = tuple(
                    gold_keep_unlinked[index] for index in kept_indices
                )
            add_phase(
                "target_reconstruction",
                time.perf_counter() - reconstruction_started,
            )
            forward_started = time.perf_counter()
            loss = lattice.nll(
                candidates,
                example.score,
                gold_spans,
                gold_keep_unlinked,
            ).float() / max(len(candidates), 1)
            add_phase("structured_forward", time.perf_counter() - forward_started)
            if not torch.isfinite(loss) or float(loss.detach()) < -1e-4:
                raise FloatingPointError(
                    f"Invalid path NLL for {packed.sample}: {loss}"
                )
            backward_started = time.perf_counter()
            (loss / max(1, accumulation)).backward()
            add_phase("structured_backward", time.perf_counter() - backward_started)
            total_loss += float(loss.detach())
            trained += 1
            if trained % max(1, accumulation) == 0:
                optimizer_started = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                add_phase("optimizer", time.perf_counter() - optimizer_started)
            cursor = next_cursor
            if trained == 1 or trained % 25 == 0:
                elapsed = time.perf_counter() - started
                rate = (trained - initial_trained) / max(elapsed, 1e-9)
                eta = (samples_per_epoch - trained) / max(rate, 1e-9)
                print(
                    f"stage=structured_path epoch={epoch} "
                    f"device={device.type} "
                    f"examples={trained}/{samples_per_epoch} "
                    f"nll={total_loss / trained:.6f} "
                    f"rows_per_sec={rate:.3f} eta_sec={eta:.1f} "
                    f"phase_seconds={json.dumps(phase_seconds, sort_keys=True)}",
                    flush=True,
                )
            if (
                trained % max(1, checkpoint_every) == 0
                and trained % max(1, accumulation) == 0
            ):
                progress = {
                    "phase": "structured_path",
                    "epoch": epoch,
                    "cursor": cursor.to_dict(),
                    "examples": trained,
                    "total_loss": total_loss,
                    "elapsed_seconds": (
                        elapsed_before_resume
                        + time.perf_counter()
                        - started
                    ),
                    "phase_seconds": phase_seconds,
                }
                checkpoint_started = time.perf_counter()
                atomic_checkpoint(
                    output_dir / "last_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    progress=progress,
                    data_fingerprint=data_fingerprint,
                    history=history,
                    checkpoint_metadata=checkpoint_metadata,
                )
                add_phase(
                    "checkpoint_io",
                    time.perf_counter() - checkpoint_started,
                )
        if trained % max(1, accumulation):
            optimizer_started = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            add_phase("optimizer", time.perf_counter() - optimizer_started)
        scheduler_started = time.perf_counter()
        scheduler.step()
        add_phase("scheduler", time.perf_counter() - scheduler_started)
        elapsed = elapsed_before_resume + time.perf_counter() - started
        history.append(
            {
                "stage": "structured_path",
                "epoch": epoch,
                "examples": trained,
                "mean_path_nll": total_loss / max(trained, 1),
                "train_wall_seconds": elapsed,
                "examples_per_second": trained / max(elapsed, 1e-9),
                "phase_seconds": phase_seconds,
            }
        )
        progress = {
            "phase": "path_epoch_complete",
            "epoch": epoch,
            "cursor": cursor.to_dict(),
            "examples": trained,
            "total_loss": total_loss,
            "elapsed_seconds": elapsed,
            "phase_seconds": phase_seconds,
        }
        atomic_checkpoint(
            output_dir / "last_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            progress=progress,
            data_fingerprint=data_fingerprint,
            history=history,
            checkpoint_metadata=checkpoint_metadata,
        )
        atomic_json(
            output_dir / "progress.json",
            {
                **progress,
                "updated_unix": time.time(),
                "locked_test_touched": False,
            },
        )
        resume_progress = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ready-marker",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/DATA_READY.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/training-v1"),
    )
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--actual-training-profile", type=Path)
    parser.add_argument("--initialize-checkpoint", type=Path, required=True)
    parser.add_argument("--initialize-full-checkpoint", type=Path)
    parser.add_argument("--candidate-rescorer-checkpoint", type=Path)
    parser.add_argument("--minimum-candidate-f1", type=float, default=0.85)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--prepared-local-cache", type=Path)
    parser.add_argument("--prepared-max-open-shards", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--path-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=DEFAULT_STATUS_PATH,
    )
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--acoustic-epochs", type=int, default=1)
    parser.add_argument("--structure-epochs", type=int, default=1)
    parser.add_argument("--errors-epochs", type=int, default=1)
    parser.add_argument("--joint-epochs", type=int, default=3)
    parser.add_argument("--path-epochs", type=int, default=1)
    parser.add_argument("--path-samples-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--path-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--local-checkpoint-every", type=int)
    parser.add_argument("--path-checkpoint-every", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument(
        "--overfit-mode",
        action="store_true",
        help="Reuse one deterministic tiny subset across stages/epochs.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--path-component-dim", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--structure-path-weight", type=float, default=0.0)
    parser.add_argument(
        "--path-transfer-mode",
        choices=("exact", "inflate"),
        default="exact",
    )
    parser.add_argument("--minimum-transfer-coverage", type=float, default=1.0)
    parser.add_argument(
        "--freeze-pretrained-path",
        action="store_true",
        help="Preserve the validated legacy path while training new full heads.",
    )
    parser.add_argument("--legacy-path-lr-scale", type=float, default=0.05)
    parser.add_argument("--max-options", type=int, default=12)
    parser.add_argument("--max-states", type=int, default=48)
    parser.add_argument("--max-delete-events", type=int, default=24)
    args = parser.parse_args()
    pause_request = args.output_dir / "PAUSE_REQUESTED.json"
    if pause_request.is_file():
        raise RuntimeError(
            f"Run is paused; remove {pause_request} only after reconfiguration"
        )
    if args.overfit_mode and args.max_train_examples is None:
        raise ValueError("--overfit-mode requires --max-train-examples")

    ready = verify_data_ready(args.ready_marker)
    data_fingerprint = str(ready["hashes"]["pack_id"])
    candidate_rescorer = None
    candidate_threshold = 0.0
    candidate_rescorer_metadata = None
    if args.candidate_rescorer_checkpoint is not None:
        before_hash = _sha256_file(args.candidate_rescorer_checkpoint)
        candidate_rescorer, candidate_threshold, candidate_payload = (
            load_candidate_rescorer(
                args.candidate_rescorer_checkpoint,
                device="cpu",
            )
        )
        after_hash = _sha256_file(args.candidate_rescorer_checkpoint)
        if before_hash != after_hash:
            raise RuntimeError("Candidate rescorer changed while being read")
        candidate_validation = candidate_payload.get("validation") or {}
        if float(candidate_validation.get("f1", -1.0)) < args.minimum_candidate_f1:
            raise ValueError("Candidate rescorer did not pass the validation gate")
        candidate_rescorer_metadata = {
            "path": str(args.candidate_rescorer_checkpoint.resolve()),
            "sha256": before_hash,
            "threshold": candidate_threshold,
            "validation": {
                key: value
                for key, value in candidate_validation.items()
                if key != "sample_counts"
            },
        }
    profile_document = json.loads(
        args.hardware_profile.read_text(encoding="utf-8")
    )
    if profile_document.get("device_profiled") != args.device:
        raise ValueError("Hardware profile device does not match training device")
    profile = load_stage_profile(args.hardware_profile, ready)
    actual_profile_document = None
    if args.actual_training_profile is not None:
        actual_profile_document = json.loads(
            args.actual_training_profile.read_text(encoding="utf-8")
        )
        if actual_profile_document.get("pack_id") != data_fingerprint:
            raise ValueError("Actual training profile belongs to another pack")
        selected_actual = actual_profile_document.get("selected") or {}
        if not selected_actual.get("ok"):
            raise ValueError("Actual training profile has no valid selection")
        profile = replace(
            profile,
            batch_edges=int(selected_actual["batch_edge_limit"]),
            amp_dtype=str(selected_actual["amp_dtype"]),
            fused_optimizer=bool(selected_actual["fused_optimizer"]),
            workers=0,
            prefetch=0,
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        lease_id = claim_resource(
            args.resource_status,
            "gpu",
            track="outputraw-full-training",
            command=sys.argv,
            metadata={"data_fingerprint": data_fingerprint},
        )
        atexit.register(
            release_resource,
            args.resource_status,
            "gpu",
            lease_id,
        )
        require_exclusive_cuda()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but CUDA is unavailable")
    verification = ready.get("verification") or {}
    if ready.get("schema_version") == "align-joint-data-ready-v1" and (
        verification.get("deep") is not True
        or int(verification.get("record_checks", -1))
        != int(ready["counts"]["packed"])
    ):
        raise ValueError(
            "Metadata-only training requires a complete deep-verified packed release"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lattice_config = LatticeConfig(
        max_options_per_candidate=args.max_options,
        max_states=args.max_states,
        max_delete_events=args.max_delete_events,
        noise_inference_bias=-6.0,
        continuation_feature_enabled=True,
        continuation_score_weight=0.35,
        continuation_hard_negative_copies=1,
        repeat_fragment_penalty=0.25,
    )
    model_config = FullPipelineModelConfig(
        path_component_dim=args.path_component_dim,
        residual_scale=args.residual_scale,
        structure_path_weight=args.structure_path_weight,
    )
    training_contract = {
        "data_fingerprint": data_fingerprint,
        "model": asdict(model_config),
        "path_transfer": {
            "mode": args.path_transfer_mode,
            "minimum_coverage": args.minimum_transfer_coverage,
            "frozen_during_local_training": args.freeze_pretrained_path,
            "legacy_path_lr_scale": args.legacy_path_lr_scale,
        },
        "candidate_rescorer": candidate_rescorer_metadata,
        "loss": asdict(FullPipelineLossConfig()),
        "augmentation": asdict(FullPipelineAugmentConfig()),
        "lattice": asdict(lattice_config),
        "stage_epochs": {
            "acoustic": args.acoustic_epochs,
            "structure": args.structure_epochs,
            "errors": args.errors_epochs,
            "joint": args.joint_epochs,
            "path": args.path_epochs,
        },
        "path_samples_per_epoch": args.path_samples_per_epoch,
        "learning_rate": args.learning_rate,
        "path_learning_rate": args.path_learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation": args.gradient_accumulation,
        "seed": args.seed,
        "overfit_mode": args.overfit_mode,
        "max_train_examples": args.max_train_examples,
        "profile": asdict(profile),
    }
    config_sha256 = _sha256_json(training_contract)
    base_checkpoint_metadata = {
        "config_sha256": config_sha256,
        "pack_id": data_fingerprint,
    }
    initialization: Mapping[str, Any]
    resume_payload: Mapping[str, Any] | None = None
    if args.resume_checkpoint is not None:
        model, resume_payload = load_checkpoint(
            args.resume_checkpoint,
            device=device,
            expected_data_fingerprint=data_fingerprint,
            expected_checkpoint_metadata={
                "config_sha256": config_sha256,
                "pack_id": data_fingerprint,
            },
        )
        existing_config = args.output_dir / "config.json"
        initialization = (
            json.loads(existing_config.read_text(encoding="utf-8")).get(
                "initialization"
            )
            if existing_config.is_file()
            else {}
        ) or {}
        history = list(resume_payload.get("history") or [])
        restore_rng_state(resume_payload["rng_state"])
    elif args.initialize_full_checkpoint is not None:
        model, warm_payload = load_checkpoint(
            args.initialize_full_checkpoint,
            device=device,
            expected_data_fingerprint=data_fingerprint,
        )
        if model.config.path_component_dim != 32:
            raise ValueError("Full warm start is not exact-32d")
        if args.minimum_transfer_coverage != 1.0:
            raise ValueError("Full warm start requires 100% transfer coverage")
        model.config = model_config
        initialization = {
            "mode": "exact_full_checkpoint",
            "path": str(args.initialize_full_checkpoint),
            "sha256": _sha256_file(args.initialize_full_checkpoint),
            "transfer_coverage": 1.0,
            "loaded_parameter_tensors": len(warm_payload["state_dict"]),
        }
        history = []
    else:
        model = FullJointPipelineModel(model_config).to(device)
        initialization = model.initialize_path(
            args.initialize_checkpoint,
            transfer_mode=args.path_transfer_mode,
            minimum_coverage=args.minimum_transfer_coverage,
        )
        history = []
    if profile.torch_compile:
        model.compile(mode="reduce-overhead")
    atomic_json(
        args.output_dir / "config.json",
        {
            "schema_version": "align-outputraw-full-train-config-v1",
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "model": asdict(model.config),
            "loss": asdict(FullPipelineLossConfig()),
            "augmentation": asdict(FullPipelineAugmentConfig()),
            "lattice": asdict(lattice_config),
            "profile": asdict(profile),
            "actual_training_profile": actual_profile_document,
            "training_contract": training_contract,
            "config_sha256": config_sha256,
            "initialization": initialization,
            "candidate_rescorer": candidate_rescorer_metadata,
            "data_fingerprint": data_fingerprint,
            "metadata_only_training_reads": True,
            "prepared_local_cache": (
                str(args.prepared_local_cache.resolve())
                if args.prepared_local_cache is not None
                else None
            ),
            "intonation_masked": True,
            "locked_test_touched": False,
        },
    )

    epoch_counts = (
        args.acoustic_epochs,
        args.structure_epochs,
        args.errors_epochs,
        args.joint_epochs,
    )
    with ExitStack() as stack:
        dataset = stack.enter_context(
            PackedJointDataset(
                Path(str(ready["paths"]["packed_root"])),
                manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
                verify_records=False,
                load_feature_arrays=False,
            )
        )
        prepared_cache = (
            stack.enter_context(
                PreparedLocalDataset(
                    args.prepared_local_cache,
                    pack_id=data_fingerprint,
                    lattice_config=lattice_config,
                    max_open_shards=max(
                        1, args.prepared_max_open_shards
                    ),
                )
            )
            if args.prepared_local_cache is not None
            else None
        )
        checkpoint_metadata = {
            **base_checkpoint_metadata,
            "packed_cache_version": str(dataset.metadata["schema_version"]),
            "prepared_local_cache_version": (
                str(prepared_cache.metadata["schema_version"])
                if prepared_cache is not None
                else None
            ),
            "prepared_local_fingerprint": (
                str(prepared_cache.metadata["cache_fingerprint"])
                if prepared_cache is not None
                else None
            ),
        }
        amp_dtype = _amp_dtype(profile.amp_dtype, device)
        resume_progress = (
            dict(resume_payload.get("progress") or {})
            if resume_payload is not None
            else None
        )
        path_resume = (
            resume_progress
            if resume_progress is not None
            and resume_progress.get("phase")
            in {"structured_path", "path_epoch_complete"}
            else None
        )
        local_stages = (
            ()
            if path_resume is not None
            else tuple(enumerate(zip(STAGES, epoch_counts)))
        )
        for stage_index, (stage, epochs) in local_stages:
            if epochs <= 0:
                continue
            if (
                resume_progress is not None
                and int(resume_progress.get("stage_index", -1)) > stage_index
            ):
                continue
            model.freeze_for_stage(stage)
            if args.freeze_pretrained_path:
                for parameter in model.legacy_path.parameters():
                    parameter.requires_grad = False
            optimizer = _optimizer(
                model,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                fused=profile.fused_optimizer and device.type == "cuda",
                legacy_path_lr_scale=args.legacy_path_lr_scale,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _epoch: 1.0
            )
            scaler = (
                torch.amp.GradScaler("cuda")
                if device.type == "cuda" and amp_dtype == torch.float16
                else None
            )
            if (
                resume_payload is not None
                and resume_progress is not None
                and resume_progress.get("phase") in {"local", "epoch_complete"}
                and int(resume_progress.get("stage_index", -1)) == stage_index
            ):
                optimizer.load_state_dict(
                    resume_payload["optimizer_state_dict"]
                )
                if resume_payload.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(
                        resume_payload["scheduler_state_dict"]
                    )
                if scaler is not None and resume_payload.get(
                    "scaler_state_dict"
                ) is not None:
                    scaler.load_state_dict(resume_payload["scaler_state_dict"])
                first_epoch = int(resume_progress["epoch"]) + int(
                    resume_progress.get("phase") == "epoch_complete"
                )
            else:
                first_epoch = 1
                resume_progress = None
            for epoch in range(first_epoch, epochs + 1):
                epoch_resume = (
                    resume_progress
                    if resume_progress is not None
                    and int(resume_progress.get("epoch", -1)) == epoch
                    else None
                )
                row = _train_local_epoch(
                    model=model,
                    dataset=dataset,
                    prepared_cache=prepared_cache,
                    lattice_config=lattice_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=device,
                    amp_dtype=amp_dtype,
                    loss_config=FullPipelineLossConfig(),
                    stage_index=stage_index,
                    stage=stage,
                    epoch=epoch,
                    seed=args.seed,
                    batch_edges=profile.batch_edges,
                    accumulation=max(1, args.gradient_accumulation),
                    workers=profile.workers,
                    prefetch=profile.prefetch,
                    checkpoint_every=max(
                        1,
                        args.local_checkpoint_every
                        if args.local_checkpoint_every is not None
                        else args.checkpoint_every,
                    ),
                    output_dir=args.output_dir,
                    data_fingerprint=data_fingerprint,
                    checkpoint_metadata=checkpoint_metadata,
                    history=history,
                    resume_progress=epoch_resume,
                    max_examples=args.max_train_examples,
                    fixed_order=args.overfit_mode,
                )
                history.append(row)
                atomic_json(
                    args.output_dir / "history.json", {"history": history}
                )
                epoch_progress = {
                    "phase": "epoch_complete",
                    "stage_index": stage_index,
                    "stage": stage,
                    "epoch": epoch,
                    "cursor": row["cursor"],
                    "examples": row["examples"],
                    "edges": row["edges"],
                    "optimizer_steps": row["optimizer_steps"],
                }
                atomic_checkpoint(
                    args.output_dir / "last_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=epoch_progress,
                    data_fingerprint=data_fingerprint,
                    history=history,
                    checkpoint_metadata=checkpoint_metadata,
                )
                atomic_checkpoint(
                    args.output_dir
                    / "checkpoints"
                    / f"{stage}-epoch-{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=epoch_progress,
                    data_fingerprint=data_fingerprint,
                    history=history,
                    checkpoint_metadata=checkpoint_metadata,
                )
                atomic_json(
                    args.output_dir / "progress.json",
                    {
                        **epoch_progress,
                        "updated_unix": time.time(),
                        "locked_test_touched": False,
                    },
                )
                resume_progress = None

        _train_structured_path(
            model=model,
            dataset=dataset,
            lattice_config=lattice_config,
            epochs=args.path_epochs,
            samples_per_epoch=min(
                args.path_samples_per_epoch, len(dataset.ordinals("train"))
            ),
            learning_rate=args.path_learning_rate,
            weight_decay=args.weight_decay,
            accumulation=max(1, args.gradient_accumulation),
            device=torch.device(args.path_device),
            seed=args.seed,
            output_dir=args.output_dir,
            data_fingerprint=data_fingerprint,
            checkpoint_metadata=checkpoint_metadata,
            history=history,
            resume_payload=resume_payload if path_resume is not None else None,
            resume_progress=path_resume,
            checkpoint_every=max(
                1,
                args.path_checkpoint_every
                if args.path_checkpoint_every is not None
                else args.checkpoint_every,
            ),
            candidate_rescorer=candidate_rescorer,
            candidate_threshold=candidate_threshold,
        )
        model.to("cpu")
        validation = evaluate_packed_validation(
            model,
            dataset,
            lattice_config=lattice_config,
            limit=args.max_val_examples,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            candidate_rescorer=candidate_rescorer,
            candidate_threshold=candidate_threshold,
        )
        combined = float(validation["combined"]["f1"])
        lower = float(validation["combined"]["bootstrap"]["lower_95"])
        full_validation = (
            args.max_val_examples is None
            and int(validation["validation_rows"])
            == int(ready["counts"]["val"])
        )
        report = {
            "schema_version": "align-outputraw-full-report-v2",
            "created_unix": time.time(),
            "data_ready": str(args.ready_marker.resolve()),
            "data_fingerprint": data_fingerprint,
            "checkpoint": str(
                (args.output_dir / "last_checkpoint.pt").resolve()
            ),
            "initialization": initialization,
            "candidate_rescorer": candidate_rescorer_metadata,
            "profile": profile_document,
            "history": history,
            "validation": validation,
            "validation_metric_status": "official_note_wise_available",
            "promotion_gate": {
                "full_validation_completed": full_validation,
                "official_metric": "align-note-wise-score-event-metric-v1",
                "official_metric_available": True,
                "combined_note_wise_f1_threshold": 0.83,
                "bootstrap_note_wise_lower_threshold": 0.80,
                "combined_note_wise_f1": combined,
                "bootstrap_note_wise_lower_95": lower,
                "passed": (
                    full_validation and combined >= 0.83 and lower >= 0.80
                ),
            },
            "locked_test_touched": False,
        }
        atomic_json(args.output_dir / "report.json", report)
        print(
            f"combined_note_wise_f1={combined:.6f} "
            f"bootstrap_note_wise_lower_95={lower:.6f} "
            f"promotion_passed={report['promotion_gate']['passed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
