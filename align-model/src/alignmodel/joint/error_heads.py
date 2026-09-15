"""Frozen-upstream Layer 2/3 error heads for the verified outputRaw release.

The inference half of this module accepts only candidates, a clean score, and
an already-decoded joint path.  Audited lineage is attached in a separate
function used by training/evaluation, which makes accidental gold access in
deployment straightforward to test.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .index import JointEvent, ScoreEvent
from .lattice import (
    CONTINUATION_FEATURE_INDEX,
    FEATURE_DIM as EDGE_FEATURE_DIM,
    JointCandidate,
    JointOperation,
    LatticePath,
    LatticeStep,
    SparseJointLattice,
    StructuralState,
)


SCHEMA_VERSION = "align-frozen-error-heads-v1"
CACHE_SCHEMA_VERSION = "align-frozen-error-examples-v1"
LAYER2_CLASSES = ("match", "wrong_note", "extra_note", "missed_note")
RHYTHM_SUBTYPES = ("none", "short", "long")
_OPERATIONS = tuple(JointOperation)

FEATURE_NAMES = (
    *(f"edge_{index}" for index in range(EDGE_FEATURE_DIM)),
    "upstream_edge_score",
    "upstream_path_margin",
    "upstream_extra_probability",
    "upstream_noise_probability",
    "upstream_delete_probability",
    "global_tempo_sec_per_ql",
    "local_tempo_sec_per_ql",
    "local_global_tempo_ratio",
    "observed_duration_sec",
    "score_duration_ql",
    "log_tempo_normalized_duration_ratio",
    "previous_performance_interval",
    "next_performance_interval",
    "previous_score_interval",
    "next_score_interval",
    "previous_interval_error",
    "next_interval_error",
    "previous_ioi_ratio",
    "next_ioi_ratio",
    "score_span_width",
    "is_mapped",
    "is_unlinked",
    "is_score_gap",
    "is_copy",
    "is_replay_state",
    "candidate_confidence",
    "previous_confidence",
    "next_confidence",
    "gap_before_sec",
    "gap_after_sec",
    "normalized_score_position",
    "normalized_candidate_position",
    "deleted_count_normalized",
    "path_operation_match",
    "path_operation_wrong",
    "path_operation_extra",
    "path_operation_missed",
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ErrorHeadsConfig:
    feature_dim: int = FEATURE_DIM
    hidden_dim: int = 128
    dropout: float = 0.10
    layer2_classes: int = len(LAYER2_CLASSES)
    rhythm_subtypes: int = len(RHYTHM_SUBTYPES)


@dataclass(frozen=True)
class HeadRow:
    """One inference-safe event or score-gap row."""

    features: np.ndarray
    kind: str
    event_index: int | None
    score_span: tuple[int, int] | None
    start_sec: float
    end_sec: float
    is_copy: bool


@dataclass(frozen=True)
class LabeledHeadRows:
    """Rows plus exact audited targets; never accepted by inference."""

    rows: tuple[HeadRow, ...]
    layer2: np.ndarray
    rhythm: np.ndarray
    rhythm_mask: np.ndarray
    deviation_sec: np.ndarray
    deviation_mask: np.ndarray
    rhythm_subtype: np.ndarray
    mapping_correct: np.ndarray
    target_event_index: np.ndarray

    def validate(self) -> None:
        count = len(self.rows)
        for name in (
            "layer2",
            "rhythm",
            "rhythm_mask",
            "deviation_sec",
            "deviation_mask",
            "rhythm_subtype",
            "mapping_correct",
            "target_event_index",
        ):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} does not align with rows")


@dataclass(frozen=True)
class ErrorHeadOutput:
    layer2_logits: torch.Tensor
    rhythm_logit: torch.Tensor
    deviation_sec: torch.Tensor
    rhythm_subtype_logits: torch.Tensor


@dataclass(frozen=True)
class HeadPrediction:
    layer2: tuple[str, ...]
    rhythm: tuple[bool, ...]
    deviation_sec: tuple[float, ...]
    rhythm_subtype: tuple[str, ...]
    layer2_probabilities: tuple[tuple[float, ...], ...]
    rhythm_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class SchemaDecodeConfig:
    """Inference-only controls for sequence-consistent schema 1.2 decoding."""

    high_thresholds: Mapping[str, float]
    low_ratios: Mapping[str, float]
    minimum_support: Mapping[str, int]
    uncertainty_margins: Mapping[str, float]
    merge_score_gap: Mapping[str, int]
    max_row_gap: int = 2
    pad_notes: int = 1
    require_extra_neighbors: bool = True
    require_missed_resynchronization: bool = True
    nms_overlap: bool = True


@dataclass(frozen=True)
class _PathEvidence:
    edge: np.ndarray
    edge_score: float
    margin: float
    extra_probability: float
    noise_probability: float
    delete_probability: float


class FrozenUpstreamErrorHeads(nn.Module):
    """Compact shared trunk with discrete edit/rhythm and deviation heads."""

    def __init__(self, config: ErrorHeadsConfig = ErrorHeadsConfig()) -> None:
        super().__init__()
        if config.feature_dim != FEATURE_DIM:
            raise ValueError(f"Expected feature dimension {FEATURE_DIM}")
        self.config = config
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.layer2_head = nn.Linear(config.hidden_dim, config.layer2_classes)
        self.rhythm_head = nn.Linear(config.hidden_dim, 1)
        self.deviation_head = nn.Linear(config.hidden_dim, 1)
        self.rhythm_subtype_head = nn.Linear(
            config.hidden_dim, config.rhythm_subtypes
        )

    def forward(self, features: torch.Tensor) -> ErrorHeadOutput:
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"Expected [N, {FEATURE_DIM}] features")
        hidden = self.trunk(self.input_norm(features))
        return ErrorHeadOutput(
            layer2_logits=self.layer2_head(hidden),
            rhythm_logit=self.rhythm_head(hidden).squeeze(-1),
            deviation_sec=self.deviation_head(hidden).squeeze(-1),
            rhythm_subtype_logits=self.rhythm_subtype_head(hidden),
        )


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _edge_bias(
    lattice: SparseJointLattice,
    source: StructuralState,
    step: LatticeStep,
    edge: Sequence[float],
) -> float:
    value = (
        lattice.config.noise_inference_bias
        if step.operation == JointOperation.NOISE
        else 0.0
    )
    if (
        lattice.config.continuation_feature_enabled
        and lattice.config.continuation_score_weight
        and step.structural_operation
        in {JointOperation.REPEAT_ENTER, JointOperation.REPLAY}
    ):
        value += (
            lattice.config.continuation_score_weight
            * edge[CONTINUATION_FEATURE_INDEX]
        )
    if (
        lattice.config.repeat_fragment_penalty
        and source.mode == "replay"
        and step.structural_operation == JointOperation.REPEAT_ENTER
    ):
        value -= lattice.config.repeat_fragment_penalty
    return float(value)


def _path_evidence(
    lattice: SparseJointLattice,
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    path: LatticePath,
) -> tuple[_PathEvidence, ...]:
    """Score local alternatives from each state on the actual decoded path."""

    if len(path.steps) != len(candidates):
        raise ValueError("Joint path must contain one step per candidate")
    tempo = lattice._tempo_scale(candidates, score)
    state = StructuralState()
    all_edges: list[list[float]] = []
    all_biases: list[float] = []
    groups: list[tuple[int, int, int]] = []
    operation_ids: list[list[int]] = []
    for candidate_index, (candidate, selected) in enumerate(
        zip(candidates, path.steps)
    ):
        previous = candidates[candidate_index - 1] if candidate_index else None
        start = len(all_edges)
        selected_offset: int | None = None
        group_operations: list[int] = []
        for operation in (JointOperation.EXTRA, JointOperation.NOISE):
            step = LatticeStep(
                candidate_index,
                None,
                operation,
                None,
                state.resume_event if state.mode == "replay" else None,
            )
            edge = lattice._edge_features(
                operation,
                candidate,
                None,
                state,
                state,
                0,
                score,
                tempo,
                previous,
                candidates,
                candidate_index,
            )
            if selected.score_span is None and selected.operation == operation:
                selected_offset = len(all_edges) - start
            all_edges.append(edge)
            all_biases.append(_edge_bias(lattice, state, step, edge))
            group_operations.append(_OPERATIONS.index(operation))
        for span in lattice._span_options(
            candidate,
            candidates,
            score,
            forced=selected.score_span,
        ):
            transition = lattice._transition(
                state,
                span,
                allow_long_delete=span == selected.score_span,
            )
            if transition is None:
                continue
            destination, structural, deleted = transition
            base = (
                JointOperation.MATCH
                if candidate.pitch == score[span[0]].pitch
                else JointOperation.SUBSTITUTE
            )
            scored = structural or (JointOperation.DELETE if deleted else base)
            step = LatticeStep(
                candidate_index,
                span,
                base,
                structural,
                (
                    destination.resume_event
                    if destination.mode == "replay"
                    else (
                        state.resume_event
                        if structural == JointOperation.CONTINUE
                        else None
                    )
                ),
                deleted,
            )
            edge = lattice._edge_features(
                scored,
                candidate,
                span,
                state,
                destination,
                len(deleted),
                score,
                tempo,
                previous,
                candidates,
                candidate_index,
            )
            if selected.score_span == span:
                selected_offset = len(all_edges) - start
            all_edges.append(edge)
            all_biases.append(_edge_bias(lattice, state, step, edge))
            group_operations.append(_OPERATIONS.index(scored))
        if selected_offset is None:
            raise RuntimeError(
                f"Decoded edge missing from local options at candidate {candidate_index}"
            )
        groups.append((start, len(all_edges), selected_offset))
        operation_ids.append(group_operations)
        if selected.score_span is not None:
            transition = lattice._transition(
                state, selected.score_span, allow_long_delete=True
            )
            if transition is None:
                raise RuntimeError("Decoded path contains an invalid transition")
            state = transition[0]

    device = next(lattice.scorer.parameters()).device
    with torch.inference_mode():
        values = (
            lattice.scorer(
                torch.tensor(all_edges, dtype=torch.float32, device=device)
            )
            + torch.tensor(all_biases, dtype=torch.float32, device=device)
        ).float().cpu()
    output = []
    for (start, end, selected_offset), operations in zip(groups, operation_ids):
        scores = values[start:end]
        probabilities = torch.softmax(scores, dim=0)
        selected_index = selected_offset
        alternatives = torch.cat(
            (scores[:selected_index], scores[selected_index + 1 :])
        )
        margin = (
            float(scores[selected_index] - torch.max(alternatives))
            if alternatives.numel()
            else 0.0
        )
        operation_array = np.asarray(operations)
        probability_array = probabilities.numpy()
        output.append(
            _PathEvidence(
                edge=np.asarray(
                    all_edges[start + selected_index], dtype=np.float32
                ),
                edge_score=float(scores[selected_index]),
                margin=margin,
                extra_probability=float(
                    probability_array[
                        operation_array == _OPERATIONS.index(JointOperation.EXTRA)
                    ].sum()
                ),
                noise_probability=float(
                    probability_array[
                        operation_array == _OPERATIONS.index(JointOperation.NOISE)
                    ].sum()
                ),
                delete_probability=float(
                    probability_array[
                        operation_array == _OPERATIONS.index(JointOperation.DELETE)
                    ].sum()
                ),
            )
        )
    return tuple(output)


def _tempo_context(
    events: Sequence[JointEvent],
    score: Sequence[ScoreEvent],
) -> tuple[float, tuple[float, ...]]:
    if not events or not score:
        return 0.5, (0.5,) * len(events)
    global_tempo = max(
        0.08,
        min(
            2.5,
            (events[-1].end - events[0].start)
            / max(score[-1].ql_end - score[0].ql_start, 0.25),
        ),
    )
    anchors: list[tuple[int, float]] = []
    previous_index: int | None = None
    previous_event: JointEvent | None = None
    for event_index, event in enumerate(events):
        if event.score_span is None or event.is_copy:
            continue
        current_index = event.score_span[0]
        if previous_index is not None and previous_event is not None:
            score_delta = (
                score[current_index].ql_start - score[previous_index].ql_start
            )
            time_delta = event.start - previous_event.start
            if 0.0 < score_delta <= 8.0 and 0.0 < time_delta <= 8.0:
                anchors.append((event_index, time_delta / score_delta))
        previous_index = current_index
        previous_event = event
    local = []
    for index in range(len(events)):
        nearby = [
            value
            for position, value in anchors
            if abs(position - index) <= 4 and 0.08 <= value <= 2.5
        ]
        local.append(
            float(np.median(nearby[-5:])) if nearby else float(global_tempo)
        )
    return float(global_tempo), tuple(local)


def _semantic_operation(event: JointEvent | None, score: Sequence[ScoreEvent]) -> str:
    if event is None:
        return "missed_note"
    if event.score_span is None:
        return "extra_note"
    return (
        "match"
        if event.pitch == score[event.score_span[0]].pitch
        else "wrong_note"
    )


def _event_feature(
    *,
    index: int,
    events: Sequence[JointEvent],
    score: Sequence[ScoreEvent],
    evidence: _PathEvidence,
    global_tempo: float,
    local_tempos: Sequence[float],
) -> np.ndarray:
    event = events[index]
    previous = events[index - 1] if index else None
    following = events[index + 1] if index + 1 < len(events) else None
    span = event.score_span
    score_duration = (
        score[span[1] - 1].ql_end - score[span[0]].ql_start
        if span is not None
        else 0.0
    )
    observed_duration = max(event.end - event.start, 1e-3)
    local_tempo = local_tempos[index]
    duration_ratio = (
        math.log(
            observed_duration / max(score_duration * local_tempo, 1e-3)
        )
        if score_duration > 0.0
        else 0.0
    )
    previous_perf_interval = (
        event.pitch - previous.pitch if previous is not None else 0
    )
    next_perf_interval = (
        following.pitch - event.pitch if following is not None else 0
    )
    previous_score_interval = 0
    next_score_interval = 0
    previous_ioi_ratio = next_ioi_ratio = 0.0
    if (
        previous is not None
        and previous.score_span is not None
        and span is not None
    ):
        previous_score_interval = (
            score[span[0]].pitch - score[previous.score_span[0]].pitch
        )
        score_ioi = (
            score[span[0]].ql_start
            - score[previous.score_span[0]].ql_start
        )
        if score_ioi > 0:
            previous_ioi_ratio = (
                event.start - previous.start
            ) / max(score_ioi * local_tempo, 1e-3)
    if (
        following is not None
        and following.score_span is not None
        and span is not None
    ):
        next_score_interval = (
            score[following.score_span[0]].pitch - score[span[0]].pitch
        )
        score_ioi = (
            score[following.score_span[0]].ql_start
            - score[span[0]].ql_start
        )
        if score_ioi > 0:
            next_ioi_ratio = (
                following.start - event.start
            ) / max(score_ioi * local_tempo, 1e-3)
    operation = _semantic_operation(event, score)
    extras = (
        evidence.edge_score,
        evidence.margin,
        evidence.extra_probability,
        evidence.noise_probability,
        evidence.delete_probability,
        global_tempo,
        local_tempo,
        local_tempo / max(global_tempo, 1e-3),
        observed_duration,
        score_duration,
        max(-3.0, min(3.0, duration_ratio)),
        max(-2.0, min(2.0, previous_perf_interval / 12.0)),
        max(-2.0, min(2.0, next_perf_interval / 12.0)),
        max(-2.0, min(2.0, previous_score_interval / 12.0)),
        max(-2.0, min(2.0, next_score_interval / 12.0)),
        max(
            -2.0,
            min(2.0, (previous_perf_interval - previous_score_interval) / 12.0),
        ),
        max(
            -2.0,
            min(2.0, (next_perf_interval - next_score_interval) / 12.0),
        ),
        max(-3.0, min(3.0, previous_ioi_ratio)),
        max(-3.0, min(3.0, next_ioi_ratio)),
        float(span[1] - span[0]) if span is not None else 0.0,
        float(span is not None),
        float(span is None),
        0.0,
        float(event.is_copy),
        float(event.is_copy or evidence.edge[18] > 0.5),
        float(event.confidence),
        float(previous.confidence if previous is not None else 0.0),
        float(following.confidence if following is not None else 0.0),
        max(-2.0, min(2.0, event.start - previous.end))
        if previous is not None
        else 0.0,
        max(-2.0, min(2.0, following.start - event.end))
        if following is not None
        else 0.0,
        (span[0] / max(len(score) - 1, 1)) if span is not None else -1.0,
        index / max(len(events) - 1, 1),
        float(evidence.edge[16]),
        float(operation == "match"),
        float(operation == "wrong_note"),
        float(operation == "extra_note"),
        0.0,
    )
    result = np.concatenate(
        (evidence.edge, np.asarray(extras, dtype=np.float32))
    ).astype(np.float32, copy=False)
    if result.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Error-head feature shape mismatch: {result.shape}")
    return result


def _gap_feature(
    *,
    score_index: int,
    score: Sequence[ScoreEvent],
    events: Sequence[JointEvent],
    global_tempo: float,
) -> tuple[np.ndarray, float, float]:
    edge = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)
    edge[_OPERATIONS.index(JointOperation.DELETE)] = 1.0
    edge[len(_OPERATIONS) + 8] = 1.0 / 16.0
    before = [
        event
        for event in events
        if event.score_span is not None and event.score_span[1] <= score_index
    ]
    after = [
        event
        for event in events
        if event.score_span is not None and event.score_span[0] > score_index
    ]
    left = before[-1] if before else None
    right = after[0] if after else None
    start = (
        left.end
        if left is not None
        else (
            right.start
            - max(
                score[score_index].ql_end - score[score_index].ql_start,
                0.05,
            )
            * global_tempo
            if right is not None
            else score[score_index].ql_start * global_tempo
        )
    )
    end = (
        right.start
        if right is not None
        else start
        + max(score[score_index].ql_end - score[score_index].ql_start, 0.05)
        * global_tempo
    )
    score_duration = score[score_index].ql_end - score[score_index].ql_start
    extras = (
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        global_tempo,
        global_tempo,
        1.0,
        0.0,
        score_duration,
        0.0,
        0.0,
        0.0,
        (
            (score[score_index].pitch - score[left.score_span[0]].pitch) / 12.0
            if left is not None and left.score_span is not None
            else 0.0
        ),
        (
            (score[right.score_span[0]].pitch - score[score_index].pitch) / 12.0
            if right is not None and right.score_span is not None
            else 0.0
        ),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        float(left.confidence if left is not None else 0.0),
        float(right.confidence if right is not None else 0.0),
        0.0,
        0.0,
        score_index / max(len(score) - 1, 1),
        -1.0,
        1.0 / 16.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    result = np.concatenate((edge, np.asarray(extras, dtype=np.float32)))
    if result.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Error-head gap feature shape mismatch: {result.shape}")
    return result.astype(np.float32, copy=False), float(start), float(max(end, start))


def build_inference_rows(
    lattice: SparseJointLattice,
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    path: LatticePath,
) -> tuple[HeadRow, ...]:
    """Build actual predicted-upstream rows without consulting any gold."""

    evidence = _path_evidence(lattice, candidates, score, path)
    events = tuple(path.joint_events(candidates))
    kept_evidence = tuple(
        value
        for value, step in zip(evidence, path.steps)
        if step.operation != JointOperation.NOISE
    )
    if len(events) != len(kept_evidence):
        raise RuntimeError("Decoded event/evidence lengths diverged")
    global_tempo, local_tempos = _tempo_context(events, score)
    rows = [
        HeadRow(
            features=_event_feature(
                index=index,
                events=events,
                score=score,
                evidence=kept_evidence[index],
                global_tempo=global_tempo,
                local_tempos=local_tempos,
            ),
            kind="event",
            event_index=index,
            score_span=event.score_span,
            start_sec=float(event.start),
            end_sec=float(event.end),
            is_copy=event.is_copy,
        )
        for index, event in enumerate(events)
    ]
    deleted = set(path.trailing_deletions)
    for step in path.steps:
        deleted.update(step.deleted_events)
    for score_index in sorted(deleted):
        if not 0 <= score_index < len(score):
            continue
        features, start, end = _gap_feature(
            score_index=score_index,
            score=score,
            events=events,
            global_tempo=global_tempo,
        )
        rows.append(
            HeadRow(
                features=features,
                kind="gap",
                event_index=None,
                score_span=(score_index, score_index + 1),
                start_sec=start,
                end_sec=end,
                is_copy=False,
            )
        )
    return tuple(rows)


def _nearest_candidate(
    target: JointEvent,
    candidates: Sequence[JointCandidate],
) -> JointCandidate:
    if not candidates:
        return JointCandidate(target.pitch, target.start, target.end)
    best = min(
        candidates,
        key=lambda value: (
            abs(value.start - target.start)
            + 0.04 * abs(value.pitch - target.pitch),
            abs(value.end - target.end),
        ),
    )
    return JointCandidate(
        pitch=target.pitch,
        start=target.start,
        end=target.end,
        confidence=best.confidence,
        score_hints=best.score_hints,
        acoustic_features=best.acoustic_features,
    )


def build_oracle_rows(
    lattice: SparseJointLattice,
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    target_events: Sequence[JointEvent],
    target_deletions: Sequence[int],
) -> tuple[HeadRow, ...]:
    """Oracle-upstream ablation rows, used only for the downstream ceiling."""

    oracle_candidates = tuple(
        _nearest_candidate(event, candidates) for event in target_events
    )
    state = StructuralState()
    steps = []
    for index, (candidate, target) in enumerate(
        zip(oracle_candidates, target_events)
    ):
        if target.score_span is None:
            steps.append(
                LatticeStep(index, None, JointOperation.EXTRA, None, None)
            )
            continue
        transition = lattice._transition(
            state, target.score_span, allow_long_delete=True
        )
        if transition is None:
            raise RuntimeError("Oracle target has an invalid score transition")
        destination, structural, deleted = transition
        base = (
            JointOperation.MATCH
            if candidate.pitch == score[target.score_span[0]].pitch
            else JointOperation.SUBSTITUTE
        )
        steps.append(
            LatticeStep(
                index,
                target.score_span,
                base,
                structural,
                (
                    destination.resume_event
                    if destination.mode == "replay"
                    else None
                ),
                deleted,
            )
        )
        state = destination
    path = LatticePath(tuple(steps), (), 0.0)
    rows = list(
        build_inference_rows(lattice, oracle_candidates, score, path)
    )
    existing_gaps = {
        row.score_span[0]
        for row in rows
        if row.kind == "gap" and row.score_span is not None
    }
    events = tuple(path.joint_events(oracle_candidates))
    global_tempo, _local = _tempo_context(events, score)
    for score_index in sorted(set(int(value) for value in target_deletions)):
        if score_index in existing_gaps or not 0 <= score_index < len(score):
            continue
        features, start, end = _gap_feature(
            score_index=score_index,
            score=score,
            events=events,
            global_tempo=global_tempo,
        )
        rows.append(
            HeadRow(
                features,
                "gap",
                None,
                (score_index, score_index + 1),
                start,
                end,
                False,
            )
        )
    return tuple(rows)


def pair_events_sequence_consistent(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
) -> dict[int, int]:
    """Monotonic edit alignment that resynchronizes after missing notes."""

    n, m = len(predicted), len(target)
    costs = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    pointers = np.zeros((n + 1, m + 1), dtype=np.uint8)
    costs[:, 0] = np.arange(n + 1)
    costs[0, :] = np.arange(m + 1)
    pointers[1:, 0] = 1
    pointers[0, 1:] = 2
    for left in range(1, n + 1):
        pred = predicted[left - 1]
        for right in range(1, m + 1):
            gold = target[right - 1]
            onset_delta = abs(pred.start - gold.start)
            if onset_delta > 0.40:
                pair_cost = 3.0
            else:
                pair_cost = (
                    0.35 * min(onset_delta / 0.080, 3.0)
                    + 0.45 * float(pred.pitch != gold.pitch)
                    + 0.45 * float(pred.score_span != gold.score_span)
                    + 0.20 * float(pred.is_copy != gold.is_copy)
                )
            options = (
                costs[left - 1, right - 1] + pair_cost,
                costs[left - 1, right] + 1.0,
                costs[left, right - 1] + 1.0,
            )
            choice = int(np.argmin(options))
            costs[left, right] = options[choice]
            pointers[left, right] = choice
    pairs: dict[int, int] = {}
    left, right = n, m
    while left or right:
        pointer = int(pointers[left, right])
        if left and right and pointer == 0:
            pred = predicted[left - 1]
            gold = target[right - 1]
            if (
                abs(pred.start - gold.start) <= 0.40
                and costs[left, right]
                < costs[left - 1, right - 1] + 2.0
            ):
                pairs[left - 1] = right - 1
            left -= 1
            right -= 1
        elif left and (not right or pointer == 1):
            left -= 1
        else:
            right -= 1
    return pairs


def _layer2_for_target(event: JointEvent) -> int:
    if (
        event.relationship == "substitute"
        or event.origin_relationship == "substitute"
    ):
        return LAYER2_CLASSES.index("wrong_note")
    if event.is_extra:
        return LAYER2_CLASSES.index("extra_note")
    return LAYER2_CLASSES.index("match")


def attach_training_targets(
    rows: Sequence[HeadRow],
    *,
    predicted_events: Sequence[JointEvent],
    target_events: Sequence[JointEvent],
    target_deletions: Sequence[int],
    rhythm_rows: Sequence[Mapping[str, Any]],
    score: Sequence[ScoreEvent],
) -> LabeledHeadRows:
    """Attach exact audited labels after inference-safe features are frozen."""

    event_rows = [row for row in rows if row.kind == "event"]
    pairs = pair_events_sequence_consistent(predicted_events, target_events)
    rhythm_by_event = {
        int(row["rendered_event"]): bool(row.get("rhythm_error"))
        for row in rhythm_rows
        if row.get("rendered_event") is not None
        and bool(row.get("supervision_mask", True))
    }
    _global, target_tempos = _tempo_context(target_events, score)
    target_deletion_set = set(int(value) for value in target_deletions)
    layer2 = []
    rhythm = []
    rhythm_mask = []
    deviation = []
    deviation_mask = []
    subtype = []
    mapping_correct = []
    target_indices = []
    event_cursor = 0
    for row in rows:
        if row.kind == "gap":
            assert row.score_span is not None
            is_missed = row.score_span[0] in target_deletion_set
            layer2.append(
                LAYER2_CLASSES.index("missed_note")
                if is_missed
                else LAYER2_CLASSES.index("match")
            )
            rhythm.append(0)
            rhythm_mask.append(False)
            deviation.append(0.0)
            deviation_mask.append(False)
            subtype.append(0)
            mapping_correct.append(False)
            target_indices.append(-1)
            continue
        target_index = pairs.get(event_cursor)
        event_cursor += 1
        if target_index is None:
            # No audited performance error exists for an unmatched upstream
            # hallucination.  With no public "noise" class, suppress it rather
            # than exposing it as a false extra-note label.
            layer2.append(LAYER2_CLASSES.index("match"))
            rhythm.append(0)
            rhythm_mask.append(False)
            deviation.append(0.0)
            deviation_mask.append(False)
            subtype.append(0)
            mapping_correct.append(False)
            target_indices.append(-1)
            continue
        target = target_events[target_index]
        layer2.append(_layer2_for_target(target))
        exact_mapping = row.score_span == target.score_span
        mapping_correct.append(exact_mapping)
        target_indices.append(target_index)
        # Local rhythm evidence is meaningful only when the frozen upstream
        # selected the audited score event.  V1 trained rhythm targets through
        # mapping errors, so duration/IOI features described a different note.
        supervised = (
            target.score_span is not None
            and not target.is_copy
            and not row.is_copy
            and exact_mapping
        )
        is_rhythm = bool(rhythm_by_event.get(target_index, False)) if supervised else False
        rhythm.append(int(is_rhythm))
        rhythm_mask.append(supervised)
        if supervised:
            span = target.score_span
            assert span is not None
            score_duration = (
                score[span[1] - 1].ql_end - score[span[0]].ql_start
            )
            expected = score_duration * target_tempos[target_index]
            value = float(target.end - target.start - expected)
            deviation.append(value)
            deviation_mask.append(True)
            subtype.append(1 if is_rhythm and value < 0.0 else 2 if is_rhythm else 0)
        else:
            deviation.append(0.0)
            deviation_mask.append(False)
            subtype.append(0)
    if event_cursor != len(event_rows):
        raise RuntimeError("Event target attachment cursor diverged")
    result = LabeledHeadRows(
        rows=tuple(rows),
        layer2=np.asarray(layer2, dtype=np.int64),
        rhythm=np.asarray(rhythm, dtype=np.float32),
        rhythm_mask=np.asarray(rhythm_mask, dtype=np.bool_),
        deviation_sec=np.asarray(deviation, dtype=np.float32),
        deviation_mask=np.asarray(deviation_mask, dtype=np.bool_),
        rhythm_subtype=np.asarray(subtype, dtype=np.int64),
        mapping_correct=np.asarray(mapping_correct, dtype=np.bool_),
        target_event_index=np.asarray(target_indices, dtype=np.int64),
    )
    result.validate()
    return result


def stack_labeled_rows(
    values: Sequence[LabeledHeadRows],
) -> dict[str, np.ndarray]:
    if not values:
        raise ValueError("Cannot stack empty error-head examples")
    return {
        "features": np.concatenate(
            [
                np.stack([row.features for row in value.rows]).astype(np.float32)
                for value in values
            ]
        ),
        "layer2": np.concatenate([value.layer2 for value in values]),
        "rhythm": np.concatenate([value.rhythm for value in values]),
        "rhythm_mask": np.concatenate([value.rhythm_mask for value in values]),
        "deviation_sec": np.concatenate(
            [value.deviation_sec for value in values]
        ),
        "deviation_mask": np.concatenate(
            [value.deviation_mask for value in values]
        ),
        "rhythm_subtype": np.concatenate(
            [value.rhythm_subtype for value in values]
        ),
    }


def measured_class_weights(target: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(target.astype(np.int64), minlength=classes)
    present = counts > 0
    weights = np.ones(classes, dtype=np.float32)
    if np.any(present):
        weights[present] = np.sqrt(
            np.mean(counts[present]) / np.maximum(counts[present], 1)
        )
    return torch.from_numpy(np.clip(weights, 0.35, 6.0))


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    log_probability = F.log_softmax(logits, dim=-1)
    probability = torch.exp(log_probability)
    selected_log = log_probability.gather(1, target[:, None]).squeeze(1)
    selected_probability = probability.gather(1, target[:, None]).squeeze(1)
    weights = class_weights.to(logits)[target]
    return torch.mean(
        -weights * torch.pow(1.0 - selected_probability, gamma) * selected_log
    )


def error_head_loss(
    output: ErrorHeadOutput,
    targets: Mapping[str, torch.Tensor],
    *,
    layer2_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    layer2 = focal_cross_entropy(
        output.layer2_logits,
        targets["layer2"].long(),
        class_weights=layer2_weights,
    )
    rhythm_mask = targets["rhythm_mask"].bool()
    if torch.any(rhythm_mask):
        rhythm_target = targets["rhythm"][rhythm_mask].float()
        positive = torch.sum(rhythm_target)
        negative = rhythm_target.numel() - positive
        positive_weight = torch.clamp(
            negative / positive.clamp_min(1.0), min=1.0, max=12.0
        )
        binary = F.binary_cross_entropy_with_logits(
            output.rhythm_logit[rhythm_mask],
            rhythm_target,
            pos_weight=positive_weight,
            reduction="none",
        )
        rhythm_probability = torch.sigmoid(output.rhythm_logit[rhythm_mask])
        pt = torch.where(
            rhythm_target > 0.5,
            rhythm_probability,
            1.0 - rhythm_probability,
        )
        rhythm = torch.mean(binary * torch.pow(1.0 - pt, 1.5))
        subtype = F.cross_entropy(
            output.rhythm_subtype_logits[rhythm_mask],
            targets["rhythm_subtype"][rhythm_mask].long(),
        )
    else:
        rhythm = output.rhythm_logit.sum() * 0.0
        subtype = output.rhythm_subtype_logits.sum() * 0.0
    deviation_mask = targets["deviation_mask"].bool()
    deviation = (
        F.smooth_l1_loss(
            output.deviation_sec[deviation_mask],
            targets["deviation_sec"][deviation_mask].float(),
            beta=0.050,
        )
        if torch.any(deviation_mask)
        else output.deviation_sec.sum() * 0.0
    )
    components = {
        "layer2": layer2,
        "rhythm": rhythm,
        "rhythm_subtype": subtype,
        "deviation": deviation,
    }
    return (
        layer2 + 0.75 * rhythm + 0.25 * subtype + 0.35 * deviation,
        components,
    )


def _binary_prf(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    predicted = predicted.astype(bool)
    target = target.astype(bool)
    correct = int(np.sum(predicted & target))
    predicted_count = int(np.sum(predicted))
    target_count = int(np.sum(target))
    precision = correct / predicted_count if predicted_count else 0.0
    recall = correct / target_count if target_count else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "correct": correct,
        "predicted": predicted_count,
        "target": target_count,
    }


def calibrate_thresholds(
    layer2_probabilities: np.ndarray,
    rhythm_probabilities: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    minimum_rhythm_precision: float = 0.70,
    deviation_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    thresholds: dict[str, float] = {}
    details: dict[str, Any] = {}
    grid = np.linspace(0.10, 0.95, 86)
    for class_index, name in enumerate(LAYER2_CLASSES[1:], 1):
        target = targets["layer2"] == class_index
        scored = []
        for threshold in grid:
            metric = _binary_prf(
                layer2_probabilities[:, class_index] >= threshold, target
            )
            scored.append((float(metric["f1"]), float(metric["precision"]), threshold, metric))
        best = max(scored, key=lambda value: (value[0], value[1], value[2]))
        thresholds[name] = float(best[2])
        details[name] = dict(best[3])
    features = targets["features"]
    event_mask = features[:, FEATURE_NAMES.index("is_score_gap")] < 0.5
    path_wrong = features[:, FEATURE_NAMES.index("path_operation_wrong")] > 0.5
    mapped = features[:, FEATURE_NAMES.index("is_mapped")] > 0.5
    copy = features[:, FEATURE_NAMES.index("is_copy")] > 0.5
    wrong_context = path_wrong.astype(np.int64) * 2 + copy.astype(np.int64)
    extra_context = mapped.astype(np.int64) * 2 + copy.astype(np.int64)
    context_thresholds = np.asarray(
        [
            thresholds["wrong_note"],
        ]
        * 4
        + [thresholds["extra_note"]] * 4
        + [thresholds["missed_note"]],
        dtype=np.float64,
    )

    def context_predictions(values: np.ndarray) -> np.ndarray:
        wrong_threshold = values[wrong_context]
        extra_threshold = values[4 + extra_context]
        wrong_strength = layer2_probabilities[:, 1] / wrong_threshold
        extra_strength = layer2_probabilities[:, 2] / extra_threshold
        operation = np.where(wrong_strength > extra_strength, 1, 2)
        active = np.maximum(wrong_strength, extra_strength) >= 1.0
        output = np.zeros(len(features), dtype=np.int64)
        output[event_mask & active] = operation[event_mask & active]
        gap = ~event_mask
        output[gap & (layer2_probabilities[:, 3] >= values[8])] = 3
        return output

    def typed_score(values: np.ndarray) -> tuple[float, float]:
        output = context_predictions(values)
        target = targets["layer2"].astype(np.int64)
        correct = int(np.sum((output == target) & (target != 0)))
        predicted_count = int(np.sum(output != 0))
        target_count = int(np.sum(target != 0))
        precision = correct / max(predicted_count, 1)
        recall = correct / max(target_count, 1)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return f1, precision

    context_grid = np.linspace(0.10, 1.00, 91)
    for _iteration in range(4):
        changed = False
        for context_index in range(len(context_thresholds)):
            candidates = []
            for value in context_grid:
                trial = context_thresholds.copy()
                trial[context_index] = value
                f1, precision = typed_score(trial)
                candidates.append((f1, precision, float(value)))
            selected = max(candidates)
            if selected[2] != context_thresholds[context_index]:
                changed = True
            context_thresholds[context_index] = selected[2]
        if not changed:
            break
    context_f1, context_precision = typed_score(context_thresholds)
    layer2_context = {
        "wrong_note": {
            name: float(context_thresholds[index])
            for index, name in enumerate(
                (
                    "not_path_wrong_ordinary",
                    "not_path_wrong_copy",
                    "path_wrong_ordinary",
                    "path_wrong_copy",
                )
            )
        },
        "extra_note": {
            name: float(context_thresholds[4 + index])
            for index, name in enumerate(
                (
                    "unmapped_ordinary",
                    "unmapped_copy",
                    "mapped_ordinary",
                    "mapped_copy",
                )
            )
        },
        "missed_note": float(context_thresholds[8]),
        "validation_typed_f1": context_f1,
        "validation_precision": context_precision,
    }
    mask = targets["rhythm_mask"].astype(bool)
    rhythm_target = targets["rhythm"][mask].astype(bool)
    robust = robust_rhythm_probabilities(targets["features"])
    scored = []
    for learned_weight in np.linspace(0.0, 1.0, 5):
        combined = (
            learned_weight * rhythm_probabilities
            + (1.0 - learned_weight) * robust
        )
        for threshold in grid:
            metric = _binary_prf(combined[mask] >= threshold, rhythm_target)
            precision_ok = (
                float(metric["precision"]) >= minimum_rhythm_precision
                and int(metric["predicted"]) > 0
            )
            scored.append(
                (
                    precision_ok,
                    float(metric["f1"]),
                    float(metric["precision"]),
                    threshold,
                    float(learned_weight),
                    metric,
                )
            )
    eligible = [value for value in scored if value[0]]
    best_rhythm = (
        max(eligible, key=lambda value: (value[1], value[2], value[3]))
        if eligible
        else max(scored, key=lambda value: (value[1], value[2], value[3]))
    )
    deviation_weight = 1.0
    deviation_validation: dict[str, Any] = {"status": "not_calibrated"}
    if deviation_predictions is not None:
        deviation_mask = targets["deviation_mask"].astype(bool)
        baseline = robust_deviation_seconds(targets["features"])
        if np.any(deviation_mask):
            choices = []
            for model_weight in np.linspace(0.0, 1.0, 11):
                combined = (
                    model_weight * deviation_predictions
                    + (1.0 - model_weight) * baseline
                )
                mae = float(
                    np.mean(
                        np.abs(
                            combined[deviation_mask]
                            - targets["deviation_sec"][deviation_mask]
                        )
                    )
                )
                choices.append((mae, -float(model_weight), float(model_weight)))
            selected = min(choices)
            deviation_weight = selected[2]
            deviation_validation = {
                "mae_seconds": selected[0],
                "model_weight": deviation_weight,
                "baseline_weight": 1.0 - deviation_weight,
            }
    return {
        "layer2": thresholds,
        "layer2_context": layer2_context,
        "layer2_validation": details,
        "rhythm": float(best_rhythm[3]),
        "rhythm_learned_weight": float(best_rhythm[4]),
        "rhythm_validation": dict(best_rhythm[5]),
        "minimum_rhythm_precision": minimum_rhythm_precision,
        "rhythm_precision_floor_met": bool(best_rhythm[0]),
        "deviation_model_weight": deviation_weight,
        "deviation_validation": deviation_validation,
        "operation_decoder": "thresholded_sequence_consistent_v2",
        "calibrated_on": "validation_only",
    }


def robust_deviation_seconds(features: np.ndarray) -> np.ndarray:
    """Tempo-normalized duration residual available without learned state."""

    observed = features[:, FEATURE_NAMES.index("observed_duration_sec")]
    score_duration = features[:, FEATURE_NAMES.index("score_duration_ql")]
    local_tempo = features[:, FEATURE_NAMES.index("local_tempo_sec_per_ql")]
    return observed - score_duration * local_tempo


def robust_rhythm_probabilities(features: np.ndarray) -> np.ndarray:
    """Robust local duration/IOI evidence with confidence uncertainty gating."""

    duration = np.abs(
        features[:, FEATURE_NAMES.index("log_tempo_normalized_duration_ratio")]
    )
    previous_ratio = features[:, FEATURE_NAMES.index("previous_ioi_ratio")]
    next_ratio = features[:, FEATURE_NAMES.index("next_ioi_ratio")]

    def ratio_signal(values: np.ndarray) -> np.ndarray:
        present = values > 0.0
        signal = np.zeros_like(values, dtype=np.float32)
        signal[present] = np.abs(
            np.log(np.clip(values[present], 0.05, 20.0))
        )
        return signal

    local_signal = np.maximum(
        duration,
        0.65 * np.maximum(
            ratio_signal(previous_ratio), ratio_signal(next_ratio)
        ),
    )
    confidence = np.clip(
        (
            features[:, FEATURE_NAMES.index("candidate_confidence")]
            - 0.20
        )
        / 0.70,
        0.0,
        1.0,
    )
    margin = features[:, FEATURE_NAMES.index("upstream_path_margin")]
    mapping_reliability = 1.0 / (1.0 + np.exp(-np.clip(margin, -8.0, 8.0)))
    reliability = 0.50 + 0.25 * confidence + 0.25 * mapping_reliability
    probability = (
        1.0 / (1.0 + np.exp(-8.0 * (local_signal - 0.16)))
    ) * reliability
    excluded = (
        (features[:, FEATURE_NAMES.index("is_score_gap")] > 0.5)
        | (features[:, FEATURE_NAMES.index("is_copy")] > 0.5)
        | (features[:, FEATURE_NAMES.index("is_replay_state")] > 0.5)
        | (features[:, FEATURE_NAMES.index("is_mapped")] < 0.5)
    )
    probability[excluded] = 0.0
    return probability.astype(np.float32, copy=False)


def decode_probabilities(
    rows: Sequence[HeadRow],
    layer2_probabilities: np.ndarray,
    rhythm_probabilities: np.ndarray,
    deviation_sec: np.ndarray,
    subtype_logits: np.ndarray,
    thresholds: Mapping[str, Any],
) -> HeadPrediction:
    layer2 = []
    selected_strengths = []
    class_thresholds = thresholds["layer2"]

    def class_threshold(row: HeadRow, name: str) -> float:
        context = thresholds.get("layer2_context") or {}
        if name == "missed_note":
            return float(context.get("missed_note", class_thresholds[name]))
        group = context.get(name)
        if not isinstance(group, Mapping):
            return float(class_thresholds[name])
        if name == "wrong_note":
            path_wrong = (
                row.features[FEATURE_NAMES.index("path_operation_wrong")] > 0.5
            )
            key = (
                ("path_wrong_" if path_wrong else "not_path_wrong_")
                + ("copy" if row.is_copy else "ordinary")
            )
        else:
            key = (
                ("mapped_" if row.score_span is not None else "unmapped_")
                + ("copy" if row.is_copy else "ordinary")
            )
        return float(group.get(key, class_thresholds[name]))

    for row, probabilities in zip(rows, layer2_probabilities):
        allowed = (
            (LAYER2_CLASSES.index("missed_note"),)
            if row.kind == "gap"
            else (
                LAYER2_CLASSES.index("wrong_note"),
                LAYER2_CLASSES.index("extra_note"),
            )
        )
        candidates = [
            index
            for index in allowed
            if probabilities[index]
            >= class_threshold(row, LAYER2_CLASSES[index])
        ]
        selected = (
            max(
                candidates,
                key=lambda index: (
                    probabilities[index]
                    / max(
                        class_threshold(row, LAYER2_CLASSES[index]), 1e-6
                    )
                ),
            )
            if candidates
            else 0
        )
        layer2.append(LAYER2_CLASSES[selected])
        selected_strengths.append(
            (
                float(probabilities[selected])
                / max(
                        class_threshold(row, LAYER2_CLASSES[selected]),
                        1e-6,
                )
                if selected
                else 0.0
            )
        )
    # A clean-score event can support only one operation on a decoded pass.
    # Keep the strongest decision when upstream fragmentation maps multiple
    # transcription events onto the same core.
    strongest_by_span: dict[tuple[int, int, str], int] = {}
    for index, (row, name, strength) in enumerate(
        zip(rows, layer2, selected_strengths)
    ):
        if name == "match" or row.score_span is None or row.is_copy:
            continue
        key = (row.score_span[0], row.score_span[1], name)
        previous = strongest_by_span.get(key)
        if previous is None or strength > selected_strengths[previous]:
            if previous is not None:
                layer2[previous] = "match"
            strongest_by_span[key] = index
        else:
            layer2[index] = "match"
    rhythm_threshold = float(thresholds["rhythm"])
    learned_weight = float(thresholds.get("rhythm_learned_weight", 1.0))
    robust_rhythm = robust_rhythm_probabilities(
        np.stack([row.features for row in rows])
    )
    combined_rhythm = (
        learned_weight * rhythm_probabilities
        + (1.0 - learned_weight) * robust_rhythm
    )
    rhythm = tuple(
        bool(
            row.kind == "event"
            and row.score_span is not None
            and not row.is_copy
            and probability >= rhythm_threshold
        )
        for row, probability in zip(rows, combined_rhythm)
    )
    subtype_index = np.argmax(subtype_logits[:, 1:], axis=1) + 1
    baseline_deviation = robust_deviation_seconds(
        np.stack([row.features for row in rows])
    )
    deviation_weight = float(thresholds.get("deviation_model_weight", 1.0))
    combined_deviation = (
        deviation_weight * deviation_sec
        + (1.0 - deviation_weight) * baseline_deviation
    )
    subtypes = tuple(
        (
            ("short" if value < 0.0 else "long")
            if active and abs(float(value)) >= 1e-4
            else RHYTHM_SUBTYPES[int(index)] if active else "none"
        )
        for index, active, value in zip(
            subtype_index, rhythm, combined_deviation
        )
    )
    return HeadPrediction(
        layer2=tuple(layer2),
        rhythm=rhythm,
        deviation_sec=tuple(float(value) for value in combined_deviation),
        rhythm_subtype=subtypes,
        layer2_probabilities=tuple(
            tuple(float(value) for value in row)
            for row in layer2_probabilities
        ),
        rhythm_probabilities=tuple(
            float(value) for value in rhythm_probabilities
        ),
    )


@torch.inference_mode()
def infer_error_heads(
    model: FrozenUpstreamErrorHeads,
    rows: Sequence[HeadRow],
    thresholds: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> HeadPrediction:
    """Deployment inference; its signature cannot accept audited targets."""

    model.eval()
    if not rows:
        return HeadPrediction((), (), (), (), (), ())
    features = torch.from_numpy(
        np.stack([row.features for row in rows])
    ).to(device)
    output = model(features)
    return decode_probabilities(
        rows,
        torch.softmax(output.layer2_logits, dim=-1).cpu().numpy(),
        torch.sigmoid(output.rhythm_logit).cpu().numpy(),
        output.deviation_sec.cpu().numpy(),
        output.rhythm_subtype_logits.cpu().numpy(),
        thresholds,
    )


@torch.inference_mode()
def infer_frozen_error_pipeline(
    model: FrozenUpstreamErrorHeads,
    lattice: SparseJointLattice,
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    thresholds: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[LatticePath, tuple[HeadRow, ...], HeadPrediction]:
    """Run the frozen joint decoder and downstream heads end to end."""

    path = lattice.decode(candidates, score)
    rows = build_inference_rows(lattice, candidates, score, path)
    prediction = infer_error_heads(model, rows, thresholds, device=device)
    return path, rows, prediction


def heuristic_prediction(
    rows: Sequence[HeadRow],
    *,
    rhythm_log_ratio: float = 0.30,
) -> HeadPrediction:
    layer2 = []
    rhythm = []
    deviation = []
    subtype = []
    duration_feature = FEATURE_NAMES.index(
        "log_tempo_normalized_duration_ratio"
    )
    for row in rows:
        if row.kind == "gap":
            name = "missed_note"
        elif row.score_span is None:
            name = "extra_note"
        elif row.features[FEATURE_NAMES.index("path_operation_wrong")] > 0.5:
            name = "wrong_note"
        else:
            name = "match"
        layer2.append(name)
        ratio = float(row.features[duration_feature])
        active = (
            row.kind == "event"
            and row.score_span is not None
            and not row.is_copy
            and abs(ratio) >= rhythm_log_ratio
        )
        rhythm.append(active)
        expected = (
            float(row.features[FEATURE_NAMES.index("score_duration_ql")])
            * float(row.features[FEATURE_NAMES.index("local_tempo_sec_per_ql")])
        )
        value = float(row.end_sec - row.start_sec - expected)
        deviation.append(value)
        subtype.append("short" if active and value < 0 else "long" if active else "none")
    probabilities = []
    for name in layer2:
        row = [0.0] * len(LAYER2_CLASSES)
        row[LAYER2_CLASSES.index(name)] = 1.0
        probabilities.append(tuple(row))
    return HeadPrediction(
        tuple(layer2),
        tuple(rhythm),
        tuple(deviation),
        tuple(subtype),
        tuple(probabilities),
        tuple(float(value) for value in rhythm),
    )


def evaluate_predictions(
    clips: Sequence[tuple[str, LabeledHeadRows, HeadPrediction, Mapping[str, Any]]],
) -> dict[str, Any]:
    if not clips:
        raise ValueError("Error-head evaluation requires clips")
    predicted = np.concatenate(
        [
            np.asarray(
                [LAYER2_CLASSES.index(name) for name in prediction.layer2],
                dtype=np.int64,
            )
            for _sample, _rows, prediction, _meta in clips
        ]
    )
    target = np.concatenate([rows.layer2 for _sample, rows, _prediction, _meta in clips])
    per_class = {}
    for index, name in enumerate(LAYER2_CLASSES):
        per_class[name] = _binary_prf(predicted == index, target == index)
    error_predicted = predicted != 0
    error_target = target != 0
    typed_correct = int(np.sum((predicted == target) & error_target))
    error_predicted_count = int(np.sum(error_predicted))
    error_target_count = int(np.sum(error_target))
    error_precision = typed_correct / max(error_predicted_count, 1)
    error_recall = typed_correct / max(error_target_count, 1)
    false_labels = int(np.sum(error_predicted & ~error_target))
    mapping_correct = np.concatenate(
        [rows.mapping_correct for _sample, rows, _prediction, _meta in clips]
    )
    row_is_gap = np.concatenate(
        [
            np.asarray([row.kind == "gap" for row in rows.rows], dtype=bool)
            for _sample, rows, _prediction, _meta in clips
        ]
    )
    range_hit = mapping_correct | row_is_gap
    typed_range_correct = int(
        np.sum((predicted == target) & error_target & range_hit)
    )
    range_precision = typed_range_correct / max(error_predicted_count, 1)
    range_recall = typed_range_correct / max(error_target_count, 1)

    rhythm_predicted = np.concatenate(
        [
            np.asarray(prediction.rhythm, dtype=bool)
            for _sample, _rows, prediction, _meta in clips
        ]
    )
    rhythm_target = np.concatenate(
        [rows.rhythm.astype(bool) for _sample, rows, _prediction, _meta in clips]
    )
    rhythm_mask = np.concatenate(
        [rows.rhythm_mask for _sample, rows, _prediction, _meta in clips]
    )
    deviation_predicted = np.concatenate(
        [
            np.asarray(prediction.deviation_sec, dtype=np.float32)
            for _sample, _rows, prediction, _meta in clips
        ]
    )
    deviation_target = np.concatenate(
        [rows.deviation_sec for _sample, rows, _prediction, _meta in clips]
    )
    deviation_mask = np.concatenate(
        [rows.deviation_mask for _sample, rows, _prediction, _meta in clips]
    )
    subtype_predicted = np.concatenate(
        [
            np.asarray(
                [RHYTHM_SUBTYPES.index(name) for name in prediction.rhythm_subtype]
            )
            for _sample, _rows, prediction, _meta in clips
        ]
    )
    subtype_target = np.concatenate(
        [rows.rhythm_subtype for _sample, rows, _prediction, _meta in clips]
    )
    subtype = {
        name: _binary_prf(
            subtype_predicted[rhythm_mask] == index,
            subtype_target[rhythm_mask] == index,
        )
        for index, name in enumerate(RHYTHM_SUBTYPES[1:], 1)
    }
    breakdown: dict[str, Any] = {}
    for dimension in ("source", "repeats", "duration", "mapping_correct"):
        groups: dict[str, list[tuple[str, LabeledHeadRows, HeadPrediction, Mapping[str, Any]]]] = {}
        for clip in clips:
            sample, rows, prediction, metadata = clip
            if dimension == "mapping_correct":
                key = (
                    "majority_correct"
                    if float(np.mean(rows.mapping_correct)) >= 0.5
                    else "majority_incorrect"
                )
            else:
                key = str(metadata.get(dimension, "unknown"))
            groups.setdefault(key, []).append(clip)
        breakdown[dimension] = {
            key: _compact_evaluation(value)
            for key, value in groups.items()
        }
    return {
        "validation_rows": len(clips),
        "layer2": {
            "per_class": per_class,
            "macro_f1": float(np.mean([value["f1"] for value in per_class.values()])),
            "micro_accuracy": float(np.mean(predicted == target)),
            "error_only": {
                "precision": error_precision,
                "recall": error_recall,
                "f1": (
                    2 * error_precision * error_recall
                    / max(error_precision + error_recall, 1e-12)
                ),
                "typed_correct": typed_correct,
                "predicted": error_predicted_count,
                "target": error_target_count,
            },
            "false_labels_per_clip": false_labels / len(clips),
        },
        "layer3": {
            "rhythm": _binary_prf(
                rhythm_predicted[rhythm_mask],
                rhythm_target[rhythm_mask],
            ),
            "deviation_mae_seconds": (
                float(
                    np.mean(
                        np.abs(
                            deviation_predicted[deviation_mask]
                            - deviation_target[deviation_mask]
                        )
                    )
                )
                if np.any(deviation_mask)
                else None
            ),
            "per_subtype": subtype,
        },
        "full_typed_error_f1": (
            2 * error_precision * error_recall
            / max(error_precision + error_recall, 1e-12)
        ),
        "schema_1_2_score_ranges": {
            "typed_error": {
                "precision": range_precision,
                "recall": range_recall,
                "f1": (
                    2 * range_precision * range_recall
                    / max(range_precision + range_recall, 1e-12)
                ),
                "correct_type_and_range": typed_range_correct,
                "predicted": error_predicted_count,
                "target": error_target_count,
            },
            "rhythm": _binary_prf(
                rhythm_predicted[rhythm_mask] & mapping_correct[rhythm_mask],
                rhythm_target[rhythm_mask],
            ),
            "range_policy": (
                "diagnostic row-level mapping proxy only; official exclusive "
                "schema 1.2 pitch-list metrics are reported by the v2 runner"
            ),
        },
        "breakdown": {
            **breakdown,
            "timbre_render": {
                "status": "not_available_in_verified_packed_release",
                "unknown": _compact_evaluation(clips),
            },
        },
        "intonation_masked": True,
    }


def _compact_evaluation(
    clips: Sequence[tuple[str, LabeledHeadRows, HeadPrediction, Mapping[str, Any]]],
) -> dict[str, Any]:
    predicted = []
    target = []
    rhythm_predicted = []
    rhythm_target = []
    for _sample, rows, prediction, _metadata in clips:
        predicted.extend(LAYER2_CLASSES.index(name) for name in prediction.layer2)
        target.extend(rows.layer2.tolist())
        selected = rows.rhythm_mask.astype(bool)
        rhythm_predicted.extend(
            np.asarray(prediction.rhythm, dtype=bool)[selected].tolist()
        )
        rhythm_target.extend(rows.rhythm[selected].astype(bool).tolist())
    predicted_array = np.asarray(predicted)
    target_array = np.asarray(target)
    typed = (predicted_array == target_array) & (target_array != 0)
    precision = int(np.sum(typed)) / max(int(np.sum(predicted_array != 0)), 1)
    recall = int(np.sum(typed)) / max(int(np.sum(target_array != 0)), 1)
    return {
        "clips": len(clips),
        "typed_error_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "rhythm_f1": _binary_prf(
            np.asarray(rhythm_predicted), np.asarray(rhythm_target)
        )["f1"],
    }


def schema12_document(
    sample: str,
    rows: Sequence[HeadRow],
    prediction: HeadPrediction,
    score: Sequence[ScoreEvent],
    *,
    pad_notes: int = 1,
) -> dict[str, Any]:
    """Project event decisions to exclusive-metric schema 1.2 melodies.

    The audited release uses one clean-score context note on either side.
    Adjacent decisions of the same type are emitted as one contiguous core;
    copy-path rows become one repetition source range instead of duplicate
    per-pass labels.
    """

    if not score:
        return {
            "schema_version": "1.2",
            "audio_reference": "performance_audio.wav",
            "annotator_id": "frozen_error_heads_v2",
            "labels": [],
            "pipeline": {
                "sample_id": sample,
                "upstream_frozen": True,
                "intonation_masked": True,
            },
        }

    specs: list[dict[str, Any]] = []

    def core_for(index: int, row: HeadRow, label_type: str) -> tuple[int, int]:
        mapped_before = [
            candidate.score_span
            for candidate in rows[:index]
            if candidate.score_span is not None and not candidate.is_copy
        ]
        mapped_after = [
            candidate.score_span
            for candidate in rows[index + 1 :]
            if candidate.score_span is not None and not candidate.is_copy
        ]
        if label_type == "extra_note":
            # Extras are represented by the two clean neighbours around the
            # insertion anchor, then receive the normal one-note context.
            anchor = (
                row.score_span[0]
                if row.score_span is not None
                else mapped_before[-1][0]
                if mapped_before
                else mapped_after[0][0] if mapped_after else 0
            )
            return anchor, min(len(score), anchor + 2)
        if row.score_span is not None:
            return row.score_span
        anchor = (
            mapped_after[0][0]
            if mapped_after
            else mapped_before[-1][1] - 1 if mapped_before else 0
        )
        return anchor, min(len(score), anchor + 1)

    for index, (row, layer2, rhythm, deviation, subtype) in enumerate(
        zip(
            rows,
            prediction.layer2,
            prediction.rhythm,
            prediction.deviation_sec,
            prediction.rhythm_subtype,
        )
    ):
        emitted: list[tuple[str, float | None, str | None]] = []
        if layer2 != "match" and not row.is_copy:
            emitted.append((layer2, None, None))
        if rhythm and not row.is_copy:
            emitted.append(("rhythm_error", deviation * 1000.0, subtype))
        for label_type, deviation_ms, rhythm_name in emitted:
            specs.append(
                {
                    "type": label_type,
                    "core": core_for(index, row, label_type),
                    "row_index": index,
                    "start_time": row.start_sec,
                    "end_time": row.end_sec,
                    "deviation_ms": deviation_ms,
                    "rhythm_subtype": rhythm_name,
                }
            )

    copy_specs = [
        (index, row)
        for index, row in enumerate(rows)
        if row.kind == "event" and row.is_copy and row.score_span is not None
    ]
    if copy_specs:
        # The schema represents a replay as one source melody plus a copy
        # count.  Segmenting at each replay-state transition generated
        # thousands of duplicate labels in v1/v2 diagnostics.
        spans = [row.score_span for _index, row in copy_specs]
        assert all(span is not None for span in spans)
        start_index, start_row = min(
            copy_specs, key=lambda value: value[1].start_sec
        )
        end_row = max(copy_specs, key=lambda value: value[1].end_sec)[1]
        specs.append(
            {
            "type": "repetition",
            "core": (
                min(int(span[0]) for span in spans if span is not None),
                max(int(span[1]) for span in spans if span is not None),
            ),
            "row_index": start_index,
            "start_time": start_row.start_sec,
            "end_time": end_row.end_sec,
            "deviation_ms": None,
            "rhythm_subtype": None,
            "extra_copies": 1,
            }
        )

    specs.sort(key=lambda value: (int(value["row_index"]), str(value["type"])))
    merged: list[dict[str, Any]] = []
    for spec in specs:
        if merged:
            previous = merged[-1]
            previous_core = previous["core"]
            core = spec["core"]
            can_merge = (
                spec["type"] == previous["type"]
                and int(spec["row_index"]) <= int(previous["last_row_index"]) + 1
                and core[0] <= previous_core[1]
            )
            if can_merge:
                previous["core"] = (
                    min(previous_core[0], core[0]),
                    max(previous_core[1], core[1]),
                )
                previous["last_row_index"] = max(
                    int(previous["last_row_index"]), int(spec["row_index"])
                )
                previous["end_time"] = max(previous["end_time"], spec["end_time"])
                if spec.get("deviation_ms") is not None:
                    previous["deviation_ms"] = max(
                        (previous.get("deviation_ms"), spec["deviation_ms"]),
                        key=lambda value: abs(float(value or 0.0)),
                    )
                    previous["rhythm_subtype"] = spec.get("rhythm_subtype")
                continue
        merged.append({**spec, "last_row_index": int(spec["row_index"])})

    labels = []
    pad = max(0, int(pad_notes))
    for spec in merged:
        core_start = max(0, min(int(spec["core"][0]), len(score) - 1))
        core_end = max(core_start, min(int(spec["core"][1]) - 1, len(score) - 1))
        padded_start = max(0, core_start - pad)
        padded_end = min(len(score) - 1, core_end + pad)
        label = {
            "id": f"error_head_{len(labels):04d}",
            "source": "frozen_error_heads_v2",
            "type": spec["type"],
            "start_time": round(float(spec["start_time"]), 4),
            "end_time": round(float(spec["end_time"]), 4),
            "score_part": {
                "start_note_index": padded_start,
                "end_note_index": padded_end,
                "pad_notes": pad,
                "core_start_note_index": core_start,
                "core_end_note_index": core_end,
                "start_measure": score[padded_start].measure,
                "end_measure": score[padded_end].measure,
            },
            "pitches": [
                int(score[position].pitch)
                for position in range(padded_start, padded_end + 1)
            ],
            "note_ids": [
                f"note_{position:04d}"
                for position in range(padded_start, padded_end + 1)
            ],
        }
        if spec.get("deviation_ms") is not None:
            label["deviation_ms"] = round(float(spec["deviation_ms"]), 2)
            label["rhythm_subtype"] = spec.get("rhythm_subtype")
        if spec["type"] == "repetition":
            label["extra_copies"] = int(spec.get("extra_copies", 1))
            label["repeats_label_range"] = {
                "start_time": label["start_time"],
                "end_time": label["end_time"],
            }
        labels.append(label)
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": "frozen_error_heads_v2",
        "labels": labels,
        "pipeline": {
            "sample_id": sample,
            "upstream_frozen": True,
            "intonation_masked": True,
        },
    }


def schema12_document_v3(
    sample: str,
    rows: Sequence[HeadRow],
    prediction: HeadPrediction,
    score: Sequence[ScoreEvent],
    config: SchemaDecodeConfig,
    *,
    include_repetition: bool = True,
) -> dict[str, Any]:
    """Decode calibrated row probabilities into sparse score-range clusters.

    Wrong/rhythm labels remain centered on aligned score events. Extras require
    stable mapped events on both sides and use those immediate clean neighbors.
    Missed-note DELETE rows require local mapped resynchronization, and adjacent
    DELETE rows become one contiguous operation rather than a cascade.
    """

    if not score:
        return {
            "schema_version": "1.2",
            "audio_reference": "performance_audio.wav",
            "annotator_id": "frozen_error_heads_v3",
            "labels": [],
            "pipeline": {
                "sample_id": sample,
                "upstream_frozen": True,
                "intonation_masked": True,
                "range_decoder": "sequence_cluster_v3",
            },
        }
    if not (
        len(rows)
        == len(prediction.layer2_probabilities)
        == len(prediction.rhythm_probabilities)
    ):
        raise ValueError("V3 schema decoding inputs do not align")

    def mapped_neighbor(index: int, direction: int) -> HeadRow | None:
        position = index + direction
        while 0 <= position < len(rows):
            candidate = rows[position]
            if (
                candidate.kind == "event"
                and candidate.score_span is not None
                and not candidate.is_copy
            ):
                return candidate
            position += direction
        return None

    def score_neighbor(
        core: tuple[int, int], direction: int
    ) -> HeadRow | None:
        candidates = [
            candidate
            for candidate in rows
            if (
                candidate.kind == "event"
                and candidate.score_span is not None
                and not candidate.is_copy
                and (
                    candidate.score_span[1] <= core[0]
                    if direction < 0
                    else candidate.score_span[0] >= core[1]
                )
            )
        ]
        if not candidates:
            return None
        return (
            max(candidates, key=lambda value: value.score_span[1])
            if direction < 0
            else min(candidates, key=lambda value: value.score_span[0])
        )

    proposals: list[dict[str, Any]] = []
    error_indices = tuple(range(1, len(LAYER2_CLASSES)))
    for index, row in enumerate(rows):
        probabilities = prediction.layer2_probabilities[index]
        for kind in LAYER2_CLASSES[1:]:
            class_index = LAYER2_CLASSES.index(kind)
            confidence = float(probabilities[class_index])
            high = float(config.high_thresholds.get(kind, 1.01))
            low = high * float(config.low_ratios.get(kind, 1.0))
            if confidence < low:
                continue
            second = max(
                float(probabilities[value])
                for value in (0, *error_indices)
                if value != class_index
            )
            if confidence - second < float(
                config.uncertainty_margins.get(kind, 0.0)
            ):
                continue
            core: tuple[int, int] | None = None
            if kind == "wrong_note":
                if (
                    row.kind == "event"
                    and row.score_span is not None
                    and not row.is_copy
                ):
                    core = row.score_span
            elif kind == "extra_note":
                if (
                    row.kind != "event"
                    or row.score_span is not None
                    or row.is_copy
                ):
                    continue
                before = mapped_neighbor(index, -1)
                after = mapped_neighbor(index, 1)
                if before is None or after is None:
                    if config.require_extra_neighbors:
                        continue
                elif (
                    before.score_span is not None
                    and after.score_span is not None
                    # These must be the immediate clean-score neighbours.
                    # If the path skipped score events, that region belongs
                    # to a DELETE run and is not a stable insertion anchor.
                    and before.score_span[1] == after.score_span[0]
                ):
                    core = (
                        max(0, before.score_span[1] - 1),
                        min(len(score), after.score_span[0] + 1),
                    )
            elif kind == "missed_note":
                if row.kind != "gap" or row.score_span is None:
                    continue
                # Gap rows are appended after decoded event rows, so local
                # resynchronization must be established in clean-score order.
                before = score_neighbor(row.score_span, -1)
                after = score_neighbor(row.score_span, 1)
                resynchronized = bool(
                    before is not None
                    and after is not None
                    and before.score_span is not None
                    and after.score_span is not None
                    and before.score_span[1] <= row.score_span[0]
                    and after.score_span[0] >= row.score_span[1]
                )
                if config.require_missed_resynchronization and not resynchronized:
                    continue
                core = row.score_span
            if core is not None and core[1] > core[0]:
                proposals.append(
                    {
                        "type": kind,
                        "core": core,
                        "row_index": index,
                        "confidence": confidence,
                        "seed": confidence >= high,
                        "start_time": row.start_sec,
                        "end_time": row.end_sec,
                        "deviation_ms": None,
                        "rhythm_subtype": None,
                    }
                )

        rhythm_confidence = float(prediction.rhythm_probabilities[index])
        rhythm_high = float(
            config.high_thresholds.get("rhythm_error", 1.01)
        )
        rhythm_low = rhythm_high * float(
            config.low_ratios.get("rhythm_error", 1.0)
        )
        if (
            prediction.rhythm[index]
            and row.kind == "event"
            and row.score_span is not None
            and not row.is_copy
            and rhythm_confidence >= rhythm_low
            and abs(rhythm_confidence - rhythm_high)
            >= float(config.uncertainty_margins.get("rhythm_error", 0.0))
        ):
            proposals.append(
                {
                    "type": "rhythm_error",
                    "core": row.score_span,
                    "row_index": index,
                    "confidence": rhythm_confidence,
                    "seed": rhythm_confidence >= rhythm_high,
                    "start_time": row.start_sec,
                    "end_time": row.end_sec,
                    "deviation_ms": prediction.deviation_sec[index] * 1000.0,
                    "rhythm_subtype": prediction.rhythm_subtype[index],
                }
            )

    proposals.sort(
        key=lambda value: (
            str(value["type"]),
            int(value["core"][0]),
            int(value["row_index"]),
        )
    )
    clusters: list[dict[str, Any]] = []
    for proposal in proposals:
        previous = clusters[-1] if clusters else None
        can_merge = bool(
            previous is not None
            and previous["type"] == proposal["type"]
            and int(proposal["row_index"])
            <= int(previous["last_row_index"]) + int(config.max_row_gap)
            and int(proposal["core"][0])
            <= int(previous["core"][1])
            + int(config.merge_score_gap.get(str(proposal["type"]), 0))
        )
        if not can_merge:
            clusters.append(
                {
                    **proposal,
                    "last_row_index": int(proposal["row_index"]),
                    "support": 1,
                    "has_seed": bool(proposal["seed"]),
                }
            )
            continue
        assert previous is not None
        previous["core"] = (
            min(int(previous["core"][0]), int(proposal["core"][0])),
            max(int(previous["core"][1]), int(proposal["core"][1])),
        )
        previous["last_row_index"] = int(proposal["row_index"])
        previous["support"] = int(previous["support"]) + 1
        previous["has_seed"] = bool(previous["has_seed"] or proposal["seed"])
        previous["start_time"] = min(
            float(previous["start_time"]), float(proposal["start_time"])
        )
        previous["end_time"] = max(
            float(previous["end_time"]), float(proposal["end_time"])
        )
        if float(proposal["confidence"]) > float(previous["confidence"]):
            previous["confidence"] = float(proposal["confidence"])
            previous["deviation_ms"] = proposal.get("deviation_ms")
            previous["rhythm_subtype"] = proposal.get("rhythm_subtype")

    clusters = [
        value
        for value in clusters
        if bool(value["has_seed"])
        and int(value["support"])
        >= int(config.minimum_support.get(str(value["type"]), 1))
    ]
    if config.nms_overlap:
        selected: list[dict[str, Any]] = []
        for cluster in sorted(
            clusters,
            key=lambda value: (
                float(value["confidence"])
                / max(
                    float(
                        config.high_thresholds.get(
                            str(value["type"]), 1.0
                        )
                    ),
                    1e-6,
                ),
                int(value["support"]),
            ),
            reverse=True,
        ):
            start, end = cluster["core"]
            if any(
                start < kept["core"][1] and kept["core"][0] < end
                for kept in selected
            ):
                continue
            selected.append(cluster)
        clusters = selected

    labels: list[dict[str, Any]] = []
    for cluster in sorted(
        clusters,
        key=lambda value: (
            int(value["row_index"]),
            str(value["type"]),
        ),
    ):
        core_start = max(0, min(int(cluster["core"][0]), len(score) - 1))
        core_end = max(
            core_start,
            min(int(cluster["core"][1]) - 1, len(score) - 1),
        )
        pad = max(0, min(int(config.pad_notes), 2))
        padded_start = max(0, core_start - pad)
        padded_end = min(len(score) - 1, core_end + pad)
        label: dict[str, Any] = {
            "id": f"error_head_v3_{len(labels):04d}",
            "source": "frozen_error_heads_v3",
            "type": str(cluster["type"]),
            "start_time": round(float(cluster["start_time"]), 4),
            "end_time": round(float(cluster["end_time"]), 4),
            "score_part": {
                "start_note_index": padded_start,
                "end_note_index": padded_end,
                "pad_notes": pad,
                "core_start_note_index": core_start,
                "core_end_note_index": core_end,
                "start_measure": score[padded_start].measure,
                "end_measure": score[padded_end].measure,
            },
            "pitches": [
                int(score[position].pitch)
                for position in range(padded_start, padded_end + 1)
            ],
            "note_ids": [
                f"note_{position:04d}"
                for position in range(padded_start, padded_end + 1)
            ],
            "decoder": {
                "support": int(cluster["support"]),
                "confidence": round(float(cluster["confidence"]), 6),
            },
        }
        if cluster.get("deviation_ms") is not None:
            label["deviation_ms"] = round(
                float(cluster["deviation_ms"]), 2
            )
            label["rhythm_subtype"] = cluster.get("rhythm_subtype")
        labels.append(label)

    if include_repetition:
        neutral = HeadPrediction(
            layer2=tuple("match" for _ in rows),
            rhythm=tuple(False for _ in rows),
            deviation_sec=prediction.deviation_sec,
            rhythm_subtype=tuple("none" for _ in rows),
            layer2_probabilities=prediction.layer2_probabilities,
            rhythm_probabilities=prediction.rhythm_probabilities,
        )
        repetition = [
            dict(label)
            for label in schema12_document(
                sample,
                rows,
                neutral,
                score,
                pad_notes=config.pad_notes,
            )["labels"]
            if label.get("type") == "repetition"
        ]
        for label in repetition:
            label["id"] = f"error_head_v3_{len(labels):04d}"
            label["source"] = "frozen_error_heads_v3"
            labels.append(label)

    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": "frozen_error_heads_v3",
        "labels": labels,
        "pipeline": {
            "sample_id": sample,
            "upstream_frozen": True,
            "intonation_masked": True,
            "range_decoder": "sequence_cluster_v3",
        },
    }


def direct_operation_probabilities(
    rows: Sequence[HeadRow],
    events: Sequence[JointEvent] | None = None,
    score: Sequence[ScoreEvent] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inference-only direct operation reliability from frozen joint evidence.

    The upstream path margin is computed by an exact local n-best softmax over
    every legal operation/span at each candidate. This converts that margin,
    operation posterior mass, acoustic confidence, neighbor consistency, and
    short DELETE-run structure into conservative direct probabilities.
    """

    probabilities = np.zeros((len(rows), len(LAYER2_CLASSES)), dtype=np.float32)
    margin_i = FEATURE_NAMES.index("upstream_path_margin")
    extra_i = FEATURE_NAMES.index("upstream_extra_probability")
    delete_i = FEATURE_NAMES.index("upstream_delete_probability")
    confidence_i = FEATURE_NAMES.index("candidate_confidence")
    previous_confidence_i = FEATURE_NAMES.index("previous_confidence")
    next_confidence_i = FEATURE_NAMES.index("next_confidence")
    operation_wrong_i = FEATURE_NAMES.index("path_operation_wrong")
    operation_extra_i = FEATURE_NAMES.index("path_operation_extra")
    operation_missed_i = FEATURE_NAMES.index("path_operation_missed")

    def margin_posterior(value: float) -> float:
        clipped = max(-12.0, min(12.0, float(value)))
        return 1.0 / (1.0 + math.exp(-clipped))

    def certainty(value: float) -> float:
        probability = max(1e-6, min(1.0 - 1e-6, float(value)))
        entropy = -(
            probability * math.log(probability)
            + (1.0 - probability) * math.log(1.0 - probability)
        ) / math.log(2.0)
        return max(0.0, 1.0 - entropy)

    event_positions = [
        index
        for index, row in enumerate(rows)
        if row.kind == "event" and not row.is_copy
    ]

    def performance_neighbors(index: int) -> tuple[HeadRow | None, HeadRow | None]:
        before = next(
            (
                rows[position]
                for position in reversed(event_positions)
                if position < index and rows[position].score_span is not None
            ),
            None,
        )
        after = next(
            (
                rows[position]
                for position in event_positions
                if position > index and rows[position].score_span is not None
            ),
            None,
        )
        return before, after

    direct_counts: Counter[str] = Counter()
    ambiguity_counts: Counter[str] = Counter()
    gap_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if row.kind == "gap" and row.score_span is not None
    ]
    gap_runs: list[list[tuple[int, HeadRow]]] = []
    for index, row in sorted(gap_rows, key=lambda value: value[1].score_span[0]):
        if (
            gap_runs
            and gap_runs[-1][-1][1].score_span is not None
            and gap_runs[-1][-1][1].score_span[1] == row.score_span[0]
        ):
            gap_runs[-1].append((index, row))
        else:
            gap_runs.append([(index, row)])

    for index, row in enumerate(rows):
        if row.kind != "event" or row.is_copy or row.event_index is None:
            continue
        event = (
            events[row.event_index]
            if events is not None and row.event_index < len(events)
            else None
        )
        features = row.features
        selected_posterior = margin_posterior(features[margin_i])
        path_certainty = certainty(selected_posterior)
        acoustic = max(0.0, min(1.0, float(features[confidence_i])))
        before, after = performance_neighbors(index)
        monotonic = bool(
            before is not None
            and after is not None
            and before.score_span is not None
            and after.score_span is not None
            and (
                row.score_span is None
                and before.score_span[1] == after.score_span[0]
                or row.score_span is not None
                and before.score_span[1] <= row.score_span[0]
                and row.score_span[1] <= after.score_span[0]
            )
        )
        neighbor_factor = 1.0 if monotonic else 0.0
        if (
            row.score_span is not None
            and features[operation_wrong_i] > 0.5
            and (
                event is None
                or score is None
                or event.pitch != score[row.score_span[0]].pitch
            )
        ):
            ambiguity = any(
                gap.score_span is not None
                and gap.score_span[0] <= row.score_span[0] < gap.score_span[1]
                for _gap_index, gap in gap_rows
            )
            ambiguity_factor = 0.45 if ambiguity else 1.0
            value = (
                acoustic
                * selected_posterior
                * (0.35 + 0.65 * path_certainty)
                * neighbor_factor
                * ambiguity_factor
            )
            probabilities[index, LAYER2_CLASSES.index("wrong_note")] = value
            direct_counts["wrong_note"] += value > 0.0
            ambiguity_counts["wrong_delete_insert"] += ambiguity
        if row.score_span is None and features[operation_extra_i] > 0.5:
            extra_posterior = max(
                float(features[extra_i]), selected_posterior
            )
            value = (
                acoustic
                * extra_posterior
                * (0.35 + 0.65 * certainty(extra_posterior))
                * neighbor_factor
            )
            probabilities[index, LAYER2_CLASSES.index("extra_note")] = value
            direct_counts["extra_note"] += value > 0.0

    mapped = [
        row
        for row in rows
        if row.kind == "event"
        and row.score_span is not None
        and not row.is_copy
    ]
    for run in gap_runs:
        first_span = run[0][1].score_span
        last_span = run[-1][1].score_span
        assert first_span is not None and last_span is not None
        before = [
            row
            for row in mapped
            if row.score_span is not None and row.score_span[1] <= first_span[0]
        ]
        after = [
            row
            for row in mapped
            if row.score_span is not None and row.score_span[0] >= last_span[1]
        ]
        left = (
            max(before, key=lambda value: value.score_span[1])
            if before
            else None
        )
        right = (
            min(after, key=lambda value: value.score_span[0])
            if after
            else None
        )
        resynchronized = bool(
            left is not None
            and right is not None
            and left.score_span is not None
            and right.score_span is not None
            and left.score_span[1] == first_span[0]
            and right.score_span[0] == last_span[1]
        )
        run_factor = 1.0 if len(run) <= 3 else 0.35
        for index, row in run:
            features = row.features
            neighbor_confidence = min(
                float(features[previous_confidence_i]),
                float(features[next_confidence_i]),
            )
            value = (
                max(0.0, min(1.0, float(features[delete_i])))
                * max(0.0, min(1.0, neighbor_confidence))
                * run_factor
                * float(resynchronized)
                * float(features[operation_missed_i] > 0.5)
            )
            probabilities[index, LAYER2_CLASSES.index("missed_note")] = value
            direct_counts["missed_note"] += value > 0.0
        ambiguity_counts["long_delete_run"] += len(run) > 3
        ambiguity_counts["unresynchronized_delete_run"] += not resynchronized

    return probabilities, {
        "posterior": (
            "inference-only exact local n-best softmax over all legal "
            "operation/span alternatives"
        ),
        "direct_candidate_counts": dict(direct_counts),
        "ambiguity_counts": dict(ambiguity_counts),
        "repeat_rows_excluded": sum(row.is_copy for row in rows),
    }


def direct_rhythm_probabilities(rows: Sequence[HeadRow]) -> np.ndarray:
    """Conservative tempo-normalized rhythm-core evidence without gold."""

    duration_i = FEATURE_NAMES.index("log_tempo_normalized_duration_ratio")
    previous_ioi_i = FEATURE_NAMES.index("previous_ioi_ratio")
    next_ioi_i = FEATURE_NAMES.index("next_ioi_ratio")
    confidence_i = FEATURE_NAMES.index("candidate_confidence")
    previous_confidence_i = FEATURE_NAMES.index("previous_confidence")
    next_confidence_i = FEATURE_NAMES.index("next_confidence")
    margin_i = FEATURE_NAMES.index("upstream_path_margin")
    mapped_i = FEATURE_NAMES.index("is_mapped")
    replay_i = FEATURE_NAMES.index("is_replay_state")
    output = np.zeros(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        features = row.features
        if (
            row.kind != "event"
            or row.is_copy
            or features[mapped_i] <= 0.5
            or features[replay_i] > 0.5
        ):
            continue
        duration_error = abs(float(features[duration_i]))
        previous_ioi_error = abs(float(features[previous_ioi_i]) - 1.0)
        next_ioi_error = abs(float(features[next_ioi_i]) - 1.0)
        local_error = max(
            duration_error - 0.22,
            previous_ioi_error - 0.30,
            next_ioi_error - 0.30,
            0.0,
        )
        acoustic = max(0.0, min(1.0, float(features[confidence_i])))
        anchors = max(
            0.0,
            min(
                1.0,
                min(
                    float(features[previous_confidence_i]),
                    float(features[next_confidence_i]),
                ),
            ),
        )
        margin = max(-12.0, min(12.0, float(features[margin_i])))
        posterior = 1.0 / (1.0 + math.exp(-margin))
        output[index] = (
            (1.0 - math.exp(-1.7 * local_error))
            * acoustic
            * (0.4 + 0.6 * anchors)
            * (0.4 + 0.6 * posterior)
        )
    return output


def blend_direct_learned_prediction(
    prediction: HeadPrediction,
    direct_probabilities: np.ndarray,
    *,
    direct_weight: float,
) -> HeadPrediction:
    """Blend direct operation reliability with learned class probabilities."""

    learned = np.asarray(prediction.layer2_probabilities, dtype=np.float32)
    direct = np.asarray(direct_probabilities, dtype=np.float32)
    if learned.shape != direct.shape:
        raise ValueError("Direct and learned probabilities do not align")
    weight = max(0.0, min(1.0, float(direct_weight)))
    blended = weight * direct + (1.0 - weight) * learned
    layer2 = tuple(
        LAYER2_CLASSES[int(index)] for index in np.argmax(blended, axis=1)
    )
    return HeadPrediction(
        layer2=layer2,
        rhythm=prediction.rhythm,
        deviation_sec=prediction.deviation_sec,
        rhythm_subtype=prediction.rhythm_subtype,
        layer2_probabilities=tuple(
            tuple(float(item) for item in row) for row in blended
        ),
        rhythm_probabilities=prediction.rhythm_probabilities,
    )


def filter_prediction_for_schema(
    prediction: HeadPrediction,
    thresholds: Mapping[str, float],
) -> HeadPrediction:
    """Apply validation-calibrated conservative schema emission thresholds."""

    layer2 = tuple(
        (
            name
            if name == "match"
            or probabilities[LAYER2_CLASSES.index(name)]
            >= float(thresholds.get(name, 0.0))
            else "match"
        )
        for name, probabilities in zip(
            prediction.layer2, prediction.layer2_probabilities
        )
    )
    rhythm_threshold = float(thresholds.get("rhythm_error", 0.0))
    rhythm = tuple(
        bool(active and probability >= rhythm_threshold)
        for active, probability in zip(
            prediction.rhythm, prediction.rhythm_probabilities
        )
    )
    subtype = tuple(
        value if active else "none"
        for value, active in zip(prediction.rhythm_subtype, rhythm)
    )
    return HeadPrediction(
        layer2=layer2,
        rhythm=rhythm,
        deviation_sec=prediction.deviation_sec,
        rhythm_subtype=subtype,
        layer2_probabilities=prediction.layer2_probabilities,
        rhythm_probabilities=prediction.rhythm_probabilities,
    )


def save_checkpoint_atomic(
    path: Path,
    *,
    model: FrozenUpstreamErrorHeads,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    progress: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    data_fingerprint: str,
    upstream: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "progress": dict(progress),
        "history": list(history),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        "data_fingerprint": data_fingerprint,
        "upstream": dict(upstream),
        "thresholds": dict(thresholds or {}),
        "intonation_masked": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_error_heads(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    expected_data_fingerprint: str | None = None,
    expected_upstream: Mapping[str, Any] | None = None,
) -> tuple[FrozenUpstreamErrorHeads, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported frozen error-head checkpoint")
    if (
        expected_data_fingerprint is not None
        and payload.get("data_fingerprint") != expected_data_fingerprint
    ):
        raise ValueError("Error-head data fingerprint mismatch")
    for name, value in (expected_upstream or {}).items():
        if (payload.get("upstream") or {}).get(name) != value:
            raise ValueError(f"Error-head upstream mismatch for {name!r}")
    model = FrozenUpstreamErrorHeads(
        ErrorHeadsConfig(**payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def jsonable_row(row: HeadRow) -> dict[str, Any]:
    return {
        "features": row.features.tolist(),
        "kind": row.kind,
        "event_index": row.event_index,
        "score_span": list(row.score_span) if row.score_span else None,
        "start_sec": row.start_sec,
        "end_sec": row.end_sec,
        "is_copy": row.is_copy,
    }


def row_from_json(value: Mapping[str, Any]) -> HeadRow:
    return HeadRow(
        features=np.asarray(value["features"], dtype=np.float32),
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


def labeled_to_json(value: LabeledHeadRows) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "rows": [jsonable_row(row) for row in value.rows],
        "layer2": value.layer2.tolist(),
        "rhythm": value.rhythm.tolist(),
        "rhythm_mask": value.rhythm_mask.tolist(),
        "deviation_sec": value.deviation_sec.tolist(),
        "deviation_mask": value.deviation_mask.tolist(),
        "rhythm_subtype": value.rhythm_subtype.tolist(),
        "mapping_correct": value.mapping_correct.tolist(),
        "target_event_index": value.target_event_index.tolist(),
    }


def labeled_from_json(value: Mapping[str, Any]) -> LabeledHeadRows:
    if value.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported error-head example cache")
    output = LabeledHeadRows(
        rows=tuple(row_from_json(row) for row in value["rows"]),
        layer2=np.asarray(value["layer2"], dtype=np.int64),
        rhythm=np.asarray(value["rhythm"], dtype=np.float32),
        rhythm_mask=np.asarray(value["rhythm_mask"], dtype=np.bool_),
        deviation_sec=np.asarray(value["deviation_sec"], dtype=np.float32),
        deviation_mask=np.asarray(value["deviation_mask"], dtype=np.bool_),
        rhythm_subtype=np.asarray(value["rhythm_subtype"], dtype=np.int64),
        mapping_correct=np.asarray(value["mapping_correct"], dtype=np.bool_),
        target_event_index=np.asarray(value["target_event_index"], dtype=np.int64),
    )
    output.validate()
    return output
