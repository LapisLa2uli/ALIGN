"""Packed-data training utilities for the outputRaw full joint model."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch

try:
    import psutil
except ImportError:  # Optional unless exclusive-GPU process checks are requested.
    psutil = None

from .index import JointEvent
from .lattice import (
    JointOperation,
    LatticeConfig,
    SparseJointLattice,
    StructuralState,
)
from .metrics import pair_exact_pitch_onset
from .outputraw_full import (
    FullJointPipelineModel,
    FullPipelineAugmentConfig,
    FullPipelineLossConfig,
    FullPipelineTargets,
    augment_difficult_timbre,
    duration_supervision_weight,
    full_pipeline_loss,
    infer_full_pipeline,
)
from .outputraw_metrics import (
    FullPipelineMetricSample,
    evaluate_full_pipeline,
)
from .packed_data import PackedCursor, PackedJointDataset, PackedSample
from .prepared_local import PreparedLocalDataset

if TYPE_CHECKING:
    from .candidate_rescorer import CandidateRescorer


@dataclass(frozen=True)
class PreparedLocalSample:
    edge_features: torch.Tensor
    groups: tuple[tuple[int, int, int], ...]
    keep: torch.Tensor
    boundary: torch.Tensor
    split: torch.Tensor
    emission: torch.Tensor
    structure: torch.Tensor
    layer2: torch.Tensor
    rhythm: torch.Tensor
    duration_target: torch.Tensor
    duration_weight: torch.Tensor
    rearticulation_weight: torch.Tensor
    copy_count_target: int
    sample: str

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])


@dataclass(frozen=True)
class PreparedLocalBatch:
    edge_features: torch.Tensor
    targets: FullPipelineTargets
    samples: tuple[str, ...]

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])


@dataclass(frozen=True)
class StageProfile:
    batch_edges: int
    amp_dtype: str
    fused_optimizer: bool
    torch_compile: bool
    workers: int
    prefetch: int


def active_cuda_python_jobs() -> list[dict[str, Any]]:
    if psutil is None:
        raise RuntimeError(
            "psutil is required to verify exclusive CUDA process ownership"
        )
    jobs = []
    ancestors = {
        process.pid
        for process in psutil.Process(os.getpid()).parents()
    }
    for process in psutil.process_iter(("pid", "name", "cmdline", "status")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        lowered = command.lower()
        if (
            process.pid != os.getpid()
            and process.pid not in ancestors
            and "python" in str(process.info.get("name") or "").lower()
            and (
                "--device cuda" in lowered
                or "--local-device cuda" in lowered
                or "--path-device cuda" in lowered
                or "cuda:" in lowered
            )
        ):
            jobs.append(
                {
                    "pid": process.pid,
                    "ppid": process.ppid(),
                    "command": command,
                    "status": process.info.get("status"),
                }
            )
    parent_pids = {int(row["ppid"]) for row in jobs}
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"ppid", "status"}
        }
        for row in jobs
        if int(row["pid"]) not in parent_pids
        and row.get("status") != psutil.STATUS_STOPPED
    ]


def require_exclusive_cuda() -> None:
    jobs = active_cuda_python_jobs()
    if jobs:
        raise RuntimeError(
            "Refusing GPU use while another CUDA Python job is active: "
            f"{[row['pid'] for row in jobs]}"
        )


def load_stage_profile(
    hardware_profile: Path,
    ready: Mapping[str, Any],
) -> StageProfile:
    document = json.loads(hardware_profile.read_text(encoding="utf-8"))
    selected = document.get("selected") or {}
    loader = ready.get("loader") or {}
    if not selected.get("ok"):
        raise ValueError("Hardware profile has no successful selected setting")
    return StageProfile(
        batch_edges=int(selected["edges"]),
        amp_dtype=str(selected["amp_dtype"]),
        fused_optimizer=bool(selected.get("fused_optimizer")),
        torch_compile=selected.get("torch_compile") == "enabled",
        workers=int(loader.get("workers", 0)),
        prefetch=int(loader.get("prefetch", 0)),
    )


def _candidate_event(candidate: Any) -> JointEvent:
    return JointEvent(
        pitch=int(candidate.pitch),
        start=float(candidate.start),
        end=float(candidate.end),
        score_span=None,
        relationship="extra",
        confidence=float(candidate.confidence),
    )


def _operation_target(row: torch.Tensor) -> int:
    return int(torch.argmax(row[: len(tuple(JointOperation))]).item())


def _structure_target(operation: JointOperation) -> int:
    if operation == JointOperation.REPEAT_ENTER:
        return 1
    if operation == JointOperation.REPLAY:
        return 2
    if operation == JointOperation.CONTINUE:
        return 3
    return 0


def _layer2_target(
    operation: JointOperation,
    target: JointEvent | None,
) -> int:
    if operation == JointOperation.DELETE:
        return 3
    if target is None:
        return 2
    if (
        target.relationship == "substitute"
        or target.origin_relationship == "substitute"
    ):
        return 1
    if target.is_extra:
        return 2
    return 0


def _same_pitch_rearticulation(
    target_events: Sequence[JointEvent],
    index: int | None,
) -> bool:
    if index is None:
        return False
    event = target_events[index]
    neighbors = []
    if index:
        neighbors.append(target_events[index - 1])
    if index + 1 < len(target_events):
        neighbors.append(target_events[index + 1])
    return any(
        neighbor.pitch == event.pitch
        and (
            abs(neighbor.start - event.end) <= 0.100
            or abs(event.start - neighbor.end) <= 0.100
        )
        for neighbor in neighbors
    )


def prepare_local_sample(
    packed: PackedSample,
    lattice: SparseJointLattice,
) -> PreparedLocalSample:
    """Convert one exact packed sample into vectorized edge supervision."""

    example = packed.training_example()
    rows, groups = lattice.local_warmup_edges(
        example.candidates,
        example.score,
        example.gold_spans,
        example.gold_keep_unlinked,
    )
    edge_features = torch.tensor(rows, dtype=torch.float32)
    gold_indices = [start + offset for start, _end, offset in groups]
    operations = [
        tuple(JointOperation)[_operation_target(edge_features[index])]
        for index in gold_indices
    ]
    pairs = pair_exact_pitch_onset(
        tuple(_candidate_event(value) for value in example.candidates),
        example.target_events,
        tolerance_sec=0.050,
    )
    candidate_to_target = {left: right for left, right in pairs}
    rhythm_by_event = {
        int(row["rendered_event"]): bool(row["rhythm_error"])
        for row in packed.target.get("layer3_rhythm") or []
        if row.get("rendered_event") is not None
        and bool(row.get("supervision_mask", True))
    }

    keep = []
    boundary = []
    split = []
    layer2 = []
    rhythm = []
    duration_target = []
    duration_weight = []
    rearticulation_weight = []
    for index, (candidate, operation) in enumerate(
        zip(example.candidates, operations)
    ):
        target_index = candidate_to_target.get(index)
        target = (
            example.target_events[target_index]
            if target_index is not None
            else None
        )
        is_keep = bool(
            example.gold_spans[index] is not None
            or example.gold_keep_unlinked[index]
        )
        rearticulation = _same_pitch_rearticulation(
            example.target_events, target_index
        )
        strong_onset = bool(
            candidate.acoustic_features
            and float(candidate.acoustic_features[0]) >= 0.60
        )
        candidate_duration = max(float(candidate.end - candidate.start), 1e-3)
        target_duration = (
            max(float(target.end - target.start), 1e-3)
            if target is not None
            else candidate_duration
        )
        keep.append(int(is_keep))
        boundary.append(int(is_keep))
        split.append(int(rearticulation))
        layer2.append(_layer2_target(operation, target))
        rhythm.append(
            int(
                target_index is not None
                and rhythm_by_event.get(target_index, False)
            )
        )
        duration_target.append(
            math.log(target_duration / candidate_duration)
        )
        duration_weight.append(duration_supervision_weight(target_duration))
        rearticulation_weight.append(
            2.5 if rearticulation and strong_onset else 1.0
        )

    copy_count = max(
        (
            int(row.get("copy_count") or 0)
            for row in packed.target.get("layer1_repeats") or []
        ),
        default=0,
    )
    return PreparedLocalSample(
        edge_features=edge_features,
        groups=tuple(groups),
        keep=torch.tensor(keep, dtype=torch.long),
        boundary=torch.tensor(boundary, dtype=torch.long),
        split=torch.tensor(split, dtype=torch.long),
        emission=torch.tensor(
            [_operation_target(edge_features[index]) for index in gold_indices],
            dtype=torch.long,
        ),
        structure=torch.tensor(
            [_structure_target(value) for value in operations],
            dtype=torch.long,
        ),
        layer2=torch.tensor(layer2, dtype=torch.long),
        rhythm=torch.tensor(rhythm, dtype=torch.long),
        duration_target=torch.tensor(duration_target, dtype=torch.float32),
        duration_weight=torch.tensor(duration_weight, dtype=torch.float32),
        rearticulation_weight=torch.tensor(
            rearticulation_weight, dtype=torch.float32
        ),
        copy_count_target=min(max(copy_count, 0), 2),
        sample=packed.sample,
    )


def collate_local_samples(
    samples: Sequence[PreparedLocalSample],
) -> PreparedLocalBatch:
    if not samples:
        raise ValueError("Cannot collate an empty local batch")
    groups: list[tuple[int, int, int]] = []
    edge_offset = 0
    clip_index = []
    for clip, sample in enumerate(samples):
        groups.extend(
            (start + edge_offset, end + edge_offset, gold)
            for start, end, gold in sample.groups
        )
        clip_index.extend([clip] * len(sample.groups))
        edge_offset += sample.edge_count

    def joined(name: str) -> torch.Tensor:
        return torch.cat(
            [getattr(sample, name) for sample in samples], dim=0
        )

    return PreparedLocalBatch(
        edge_features=torch.cat(
            [sample.edge_features for sample in samples], dim=0
        ),
        targets=FullPipelineTargets(
            groups=groups,
            keep=joined("keep"),
            boundary=joined("boundary"),
            split=joined("split"),
            emission=joined("emission"),
            structure=joined("structure"),
            layer2=joined("layer2"),
            rhythm=joined("rhythm"),
            duration_target=joined("duration_target"),
            duration_weight=joined("duration_weight"),
            rearticulation_weight=joined("rearticulation_weight"),
            clip_index=torch.tensor(clip_index, dtype=torch.long),
            copy_count_target=torch.tensor(
                [sample.copy_count_target for sample in samples],
                dtype=torch.long,
            ),
        ),
        samples=tuple(sample.sample for sample in samples),
    )


def iter_local_batches(
    dataset: PackedJointDataset,
    cursor: PackedCursor,
    *,
    lattice: SparseJointLattice,
    max_edges: int,
    workers: int,
    prefetch: int,
    max_samples: int | None = None,
) -> Iterable[tuple[PackedCursor, PreparedLocalBatch]]:
    pending: list[PreparedLocalSample] = []
    pending_edges = 0
    last_cursor = cursor
    consumed = 0
    for next_cursor, packed in dataset.iter_from_cursor(
        cursor, workers=workers, prefetch=prefetch
    ):
        if max_samples is not None and consumed >= max_samples:
            break
        prepared = prepare_local_sample(packed, lattice)
        if pending and pending_edges + prepared.edge_count > max_edges:
            yield last_cursor, collate_local_samples(pending)
            pending = []
            pending_edges = 0
        pending.append(prepared)
        pending_edges += prepared.edge_count
        last_cursor = next_cursor
        consumed += 1
    if pending:
        yield last_cursor, collate_local_samples(pending)


def iter_prepared_local_batches(
    dataset: PackedJointDataset,
    prepared: PreparedLocalDataset,
    cursor: PackedCursor,
    *,
    max_edges: int,
    max_samples: int | None = None,
) -> Iterable[tuple[PackedCursor, PreparedLocalBatch]]:
    """Batch exact precomputed rows while preserving packed cursor semantics."""

    if cursor.pack_id != dataset.metadata["pack_id"]:
        raise ValueError("Prepared local cursor belongs to another packed release")
    order = dataset.deterministic_order(
        cursor.split,
        epoch=cursor.epoch,
        seed=cursor.seed,
    )
    if not 0 <= cursor.position <= len(order):
        raise ValueError("Prepared local cursor position is out of range")
    pending = []
    pending_edges = 0
    last_cursor = cursor
    consumed = 0
    for offset, source_ordinal in enumerate(
        order[cursor.position :],
        cursor.position,
    ):
        if max_samples is not None and consumed >= max_samples:
            break
        sample = prepared[source_ordinal]
        next_cursor = PackedCursor(
            cursor.pack_id,
            cursor.split,
            cursor.epoch,
            cursor.seed,
            offset + 1,
        )
        if pending and pending_edges + sample.edge_count > max_edges:
            yield last_cursor, collate_local_samples(pending)
            pending = []
            pending_edges = 0
        pending.append(sample)
        pending_edges += sample.edge_count
        last_cursor = next_cursor
        consumed += 1
    if pending:
        yield last_cursor, collate_local_samples(pending)


def move_targets(
    targets: FullPipelineTargets,
    device: torch.device,
) -> FullPipelineTargets:
    return FullPipelineTargets(
        groups=targets.groups,
        keep=targets.keep.to(device, non_blocking=True),
        boundary=targets.boundary.to(device, non_blocking=True),
        split=targets.split.to(device, non_blocking=True),
        emission=targets.emission.to(device, non_blocking=True),
        structure=targets.structure.to(device, non_blocking=True),
        layer2=targets.layer2.to(device, non_blocking=True),
        rhythm=targets.rhythm.to(device, non_blocking=True),
        duration_target=targets.duration_target.to(device, non_blocking=True),
        duration_weight=targets.duration_weight.to(device, non_blocking=True),
        rearticulation_weight=targets.rearticulation_weight.to(
            device, non_blocking=True
        ),
        clip_index=targets.clip_index.to(device, non_blocking=True),
        copy_count_target=targets.copy_count_target.to(
            device, non_blocking=True
        ),
    )


def train_local_batch(
    model: FullJointPipelineModel,
    batch: PreparedLocalBatch,
    *,
    device: torch.device,
    loss_config: FullPipelineLossConfig,
    amp_dtype: torch.dtype | None,
    augment_config: FullPipelineAugmentConfig = FullPipelineAugmentConfig(),
) -> tuple[torch.Tensor, Mapping[str, float], Mapping[str, float]]:
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    transfer_started = time.perf_counter()
    features = batch.edge_features
    if device.type == "cuda":
        features = features.pin_memory()
    features = features.to(device, non_blocking=True)
    targets = move_targets(batch.targets, device)
    synchronize()
    transfer_seconds = time.perf_counter() - transfer_started
    forward_started = time.perf_counter()
    features = augment_difficult_timbre(
        features, targets.groups, augment_config
    )
    context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with context:
        result = full_pipeline_loss(model, features, targets, loss_config)
    synchronize()
    forward_seconds = time.perf_counter() - forward_started
    detail = {
        name: float(value.detach().float().cpu())
        for name, value in result.components.items()
    }
    return (
        result.total,
        detail,
        {
            "host_to_device": transfer_seconds,
            "forward": forward_seconds,
        },
    )


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


def _target_resume_events(
    events: Sequence[JointEvent],
    lattice: SparseJointLattice,
) -> tuple[int, ...]:
    """Reconstruct resume anchors using the decoder's structural semantics."""

    state = StructuralState()
    resume_events = []
    for event in events:
        if event.score_span is None:
            continue
        transition = lattice._transition(
            state,
            event.score_span,
            allow_long_delete=True,
        )
        if transition is None:
            raise ValueError("Gold event sequence has an invalid lattice transition")
        destination, structural, _deleted = transition
        if structural == JointOperation.REPEAT_ENTER:
            resume_events.append(int(destination.resume_event))
        state = destination
    return tuple(resume_events)


@torch.inference_mode()
def evaluate_packed_validation(
    model: FullJointPipelineModel,
    dataset: PackedJointDataset,
    *,
    lattice_config: LatticeConfig,
    limit: int | None = None,
    rhythm_threshold: float = 0.5,
    bootstrap_replicates: int = 1000,
    progress_every: int = 25,
    seed: int = 20260915,
    candidate_rescorer: CandidateRescorer | None = None,
    candidate_threshold: float = 0.0,
) -> dict[str, object]:
    """Decode full validation and report every promotion-gate metric."""

    ordinals = dataset.ordinals("val")
    if limit is not None:
        random.Random(
            f"{dataset.metadata['pack_id']}:{seed}:validation-diagnostic"
        ).shuffle(ordinals)
        ordinals = ordinals[:limit]
    lattice = SparseJointLattice(model, lattice_config)
    samples = []
    started = time.perf_counter()
    phase_seconds = {
        "data": 0.0,
        "target_reconstruction": 0.0,
        "decode": 0.0,
        "metric_projection": 0.0,
    }
    for position, ordinal in enumerate(ordinals, 1):
        phase_started = time.perf_counter()
        packed = dataset[ordinal]
        phase_seconds["data"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        example = packed.training_example()
        if candidate_rescorer is not None:
            from .candidate_rescorer import rescore_candidates

            example = replace(
                example,
                candidates=rescore_candidates(
                    candidate_rescorer,
                    example.candidates,
                    threshold=candidate_threshold,
                    score=example.score,
                ),
            )
        phase_seconds["target_reconstruction"] += (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        prediction = infer_full_pipeline(
            model, lattice, example.candidates, example.score
        )
        phase_seconds["decode"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        rhythm_target = [False] * len(example.target_events)
        for row in packed.target.get("layer3_rhythm") or []:
            index = row.get("rendered_event")
            if index is not None and 0 <= int(index) < len(rhythm_target):
                rhythm_target[int(index)] = bool(row.get("rhythm_error"))
        predicted_resume = tuple(
            int(step.resume_event)
            for step in prediction.path.steps
            if step.structural_operation == JointOperation.REPEAT_ENTER
            and step.resume_event is not None
        )
        target_resume = _target_resume_events(example.target_events, lattice)
        samples.append(
            FullPipelineMetricSample(
                predicted=prediction.events,
                target=example.target_events,
                predicted_layer2=prediction.layer2_types,
                target_layer2=_target_layer2(example.target_events),
                predicted_rhythm=tuple(
                    value >= rhythm_threshold
                    for value in prediction.rhythm_probabilities
                ),
                target_rhythm=tuple(rhythm_target),
                predicted_duration_sec=prediction.corrected_durations_sec,
                predicted_deletions=frozenset(
                    prediction.missed_score_events
                ),
                target_deletions=example.target_deletions,
                predicted_resume_events=predicted_resume,
                target_resume_events=target_resume,
                score_event_count=len(example.score),
                source=packed.source,
            )
        )
        phase_seconds["metric_projection"] += (
            time.perf_counter() - phase_started
        )
        if (
            position == 1
            or position % progress_every == 0
            or position == len(ordinals)
        ):
            elapsed = time.perf_counter() - started
            rate = position / max(elapsed, 1e-9)
            eta = (len(ordinals) - position) / max(rate, 1e-9)
            print(
                f"validation={position}/{len(ordinals)} "
                f"rows_per_sec={rate:.3f} eta_sec={eta:.1f}",
                f"phase_seconds={json.dumps(phase_seconds, sort_keys=True)}",
                flush=True,
            )
    metric_started = time.perf_counter()
    report = evaluate_full_pipeline(
        samples,
        bootstrap_replicates=bootstrap_replicates,
    )
    phase_seconds["metric_aggregation"] = (
        time.perf_counter() - metric_started
    )
    report["validation_rows"] = len(ordinals)
    report["validation_wall_seconds"] = time.perf_counter() - started
    report["phase_seconds"] = phase_seconds
    return report


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
