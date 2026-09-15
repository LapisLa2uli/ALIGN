from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import torch
from torch import nn

from .index import JointEvent, ScoreEvent


class JointOperation(str, Enum):
    MATCH = "MATCH"
    SUBSTITUTE = "SUBSTITUTE"
    EXTRA = "EXTRA"
    DELETE = "DELETE"
    REPEAT_ENTER = "REPEAT_ENTER"
    REPLAY = "REPLAY"
    CONTINUE = "CONTINUE"
    NOISE = "NOISE"


_OPERATIONS = tuple(JointOperation)
LEGACY_FEATURE_DIM = len(_OPERATIONS) + 13
ACOUSTIC_FEATURE_DIM = 5
FEATURE_DIM = LEGACY_FEATURE_DIM + ACOUSTIC_FEATURE_DIM
CONTINUATION_FEATURE_INDEX = len(_OPERATIONS) + 11

_ACOUSTIC_INPUT_INDICES = (
    *range(len(_OPERATIONS)),
    11,
    12,
    20,
    *range(LEGACY_FEATURE_DIM, FEATURE_DIM),
)
_SCORE_INPUT_INDICES = (
    *range(len(_OPERATIONS)),
    8,
    9,
    10,
    13,
    14,
    15,
    17,
)
_STRUCTURAL_INPUT_INDICES = (
    *range(len(_OPERATIONS)),
    16,
    17,
    18,
    19,
)


@dataclass(frozen=True)
class JointCandidate:
    pitch: int
    start: float
    end: float
    confidence: float = 1.0
    score_hints: tuple[int, ...] = ()
    acoustic_features: tuple[float, ...] = ()


@dataclass(frozen=True)
class StructuralState:
    cursor: int = -1
    mode: str = "normal"
    resume_event: int = -1


@dataclass(frozen=True)
class LatticeStep:
    candidate_index: int
    score_span: tuple[int, int] | None
    operation: JointOperation
    structural_operation: JointOperation | None
    resume_event: int | None
    deleted_events: tuple[int, ...] = ()


@dataclass(frozen=True)
class LatticePath:
    steps: tuple[LatticeStep, ...]
    trailing_deletions: tuple[int, ...]
    score: float

    def joint_events(
        self, candidates: Sequence[JointCandidate]
    ) -> list[JointEvent]:
        events: list[JointEvent] = []
        for step in self.steps:
            if step.operation == JointOperation.NOISE:
                continue
            candidate = candidates[step.candidate_index]
            relationship = (
                "extra"
                if step.score_span is None
                else (
                    "copy"
                    if step.structural_operation
                    in {JointOperation.REPEAT_ENTER, JointOperation.REPLAY}
                    else (
                        "match"
                        if step.operation == JointOperation.MATCH
                        else "substitute"
                    )
                )
            )
            events.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=step.score_span,
                    relationship=relationship,
                    copy_pass=1 if relationship == "copy" else 0,
                    confidence=candidate.confidence,
                )
            )
        return events


class JointEdgeScorer(nn.Module):
    """Shared edge scorer with optional trainable whole-model component heads.

    The legacy network keeps old checkpoints bit-compatible.  End-to-end mode
    adds repository-owned acoustic, score-option, structural-transition, and
    path heads while treating cached Basic Pitch activations as immutable
    frontend inputs.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        component_mode: bool = False,
        component_dim: int = 32,
        residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.component_mode = bool(component_mode)
        self.component_dim = int(component_dim)
        self.residual_scale = float(residual_scale)
        self.network = nn.Sequential(
            nn.Linear(LEGACY_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if self.component_mode:
            self.acoustic_projection = self._projection(
                len(_ACOUSTIC_INPUT_INDICES), self.component_dim, dropout
            )
            self.score_projection = self._projection(
                len(_SCORE_INPUT_INDICES), self.component_dim, dropout
            )
            self.structural_projection = self._projection(
                len(_STRUCTURAL_INPUT_INDICES), self.component_dim, dropout
            )
            self.emission_head = nn.Linear(self.component_dim, 1)
            self.option_head = nn.Linear(self.component_dim * 2, 1)
            self.transition_head = nn.Linear(self.component_dim, 1)
            self.path_head = nn.Sequential(
                nn.Linear(self.component_dim * 3, self.component_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.component_dim, 1),
            )
            # Preserve the initialized legacy decoder exactly at step zero.
            # Output weights learn first; projection gradients begin flowing
            # on the next optimizer step.
            for head in (
                self.emission_head,
                self.option_head,
                self.transition_head,
                self.path_head[-1],
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    @staticmethod
    def _projection(
        input_dim: int, output_dim: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        legacy_score = self.network(
            features[..., :LEGACY_FEATURE_DIM]
        ).squeeze(-1)
        if not self.component_mode:
            return legacy_score
        acoustic = self.acoustic_projection(
            features[..., _ACOUSTIC_INPUT_INDICES]
        )
        score = self.score_projection(features[..., _SCORE_INPUT_INDICES])
        structural = self.structural_projection(
            features[..., _STRUCTURAL_INPUT_INDICES]
        )
        residual = (
            self.emission_head(acoustic)
            + self.option_head(torch.cat((acoustic, score), dim=-1))
            + self.transition_head(structural)
            + self.path_head(torch.cat((acoustic, score, structural), dim=-1))
        ).squeeze(-1)
        return legacy_score + self.residual_scale * residual


@dataclass(frozen=True)
class LatticeConfig:
    max_options_per_candidate: int = 32
    max_span_events: int = 4
    max_delete_events: int = 16
    max_states: int = 256
    timing_scale_floor_sec_per_ql: float = 0.08
    timing_scale_ceiling_sec_per_ql: float = 2.5
    noise_inference_bias: float = -3.0
    continuation_feature_enabled: bool = False
    continuation_lookahead_notes: int = 3
    continuation_candidate_skips: int = 1
    continuation_score_skips: int = 1
    continuation_pitch_tolerance: int = 0
    continuation_score_weight: float = 0.0
    continuation_hard_negative_copies: int = 0
    repeat_fragment_penalty: float = 0.0


def segment_logsumexp_fp32(
    groups: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Reduce ragged incoming paths in one FP32 segmented operation."""

    if not groups:
        raise ValueError("Segmented logsumexp requires at least one group")
    lengths = torch.tensor(
        [len(group) for group in groups],
        dtype=torch.long,
        device=groups[0].device,
    )
    values = torch.cat([group.float() for group in groups])
    group_ids = torch.repeat_interleave(
        torch.arange(len(groups), device=values.device),
        lengths,
    )
    maxima = torch.full(
        (len(groups),),
        -torch.inf,
        dtype=torch.float32,
        device=values.device,
    )
    maxima.scatter_reduce_(
        0,
        group_ids,
        values,
        reduce="amax",
        include_self=True,
    )
    sums = torch.zeros_like(maxima)
    sums.scatter_add_(0, group_ids, torch.exp(values - maxima[group_ids]))
    return maxima + torch.log(sums.clamp_min(1e-30))


def segment_argmax_first(groups: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return global first-max offsets for ragged groups on one device sync."""

    if not groups:
        raise ValueError("Segmented argmax requires at least one group")
    lengths = torch.tensor(
        [len(group) for group in groups],
        dtype=torch.long,
        device=groups[0].device,
    )
    values = torch.cat(groups)
    group_ids = torch.repeat_interleave(
        torch.arange(len(groups), device=values.device),
        lengths,
    )
    maxima = torch.full(
        (len(groups),),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    maxima.scatter_reduce_(
        0,
        group_ids,
        values,
        reduce="amax",
        include_self=True,
    )
    positions = torch.arange(len(values), device=values.device)
    sentinel = torch.full_like(positions, len(values))
    candidates = torch.where(
        values == maxima[group_ids],
        positions,
        sentinel,
    )
    winners = torch.full(
        (len(groups),),
        len(values),
        dtype=torch.long,
        device=values.device,
    )
    winners.scatter_reduce_(
        0,
        group_ids,
        candidates,
        reduce="amin",
        include_self=True,
    )
    return winners


def continuation_compatibility(
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    *,
    candidate_index: int,
    replay_span: tuple[int, int],
    resume_event: int,
    lookahead_notes: int = 3,
    candidate_skips: int = 1,
    score_skips: int = 1,
    pitch_tolerance: int = 0,
) -> float:
    """Soft replay/resume evidence in [-1, 1], neutral when unavailable."""

    lookahead = max(1, int(lookahead_notes))
    replay_length = max(1, int(resume_event) - int(replay_span[0]))
    observed_start = int(candidate_index) + replay_length
    if observed_start >= len(candidates) or resume_event >= len(score):
        return 0.0
    observed = [
        int(value.pitch)
        for value in candidates[
            observed_start : observed_start + lookahead + max(0, candidate_skips)
        ]
    ]
    expected = [
        int(value.pitch)
        for value in score[
            resume_event : resume_event + lookahead + max(0, score_skips)
        ]
    ]
    target = min(lookahead, len(expected))
    if target < 2 or len(observed) < 2:
        return 0.0

    # A bounded LCS permits one missing or wrong transcription note without
    # letting an unrelated later phrase satisfy the continuation.
    active: dict[tuple[int, int, int, int], int] = {(0, 0, 0, 0): 0}
    best = 0
    while active:
        next_active: dict[tuple[int, int, int, int], int] = {}
        for (obs_i, score_i, obs_used, score_used), matches in active.items():
            best = max(best, matches)
            if obs_i < len(observed) and score_i < len(expected):
                if (
                    abs(observed[obs_i] - expected[score_i])
                    <= pitch_tolerance
                ):
                    key = (obs_i + 1, score_i + 1, obs_used, score_used)
                    next_active[key] = max(next_active.get(key, -1), matches + 1)
                if obs_used < candidate_skips:
                    key = (obs_i + 1, score_i, obs_used + 1, score_used)
                    next_active[key] = max(next_active.get(key, -1), matches)
                if score_used < score_skips:
                    key = (obs_i, score_i + 1, obs_used, score_used + 1)
                    next_active[key] = max(next_active.get(key, -1), matches)
        active = next_active
    return max(-1.0, min(1.0, 2.0 * best / max(target, 1) - 1.0))


class SparseJointLattice:
    """Sparse semi-Markov/pair-HMM lattice with explicit replay/resume state."""

    def __init__(
        self,
        scorer: JointEdgeScorer,
        config: LatticeConfig = LatticeConfig(),
    ) -> None:
        self.scorer = scorer
        self.config = config
        self._continuation_cache: dict[
            tuple[int, int, int, tuple[int, int], int], float
        ] = {}

    def _tempo_scale(
        self,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
    ) -> float:
        if not candidates or not score:
            return 0.5
        audio_span = max(
            candidates[-1].end - candidates[0].start,
            0.05,
        )
        score_span = max(score[-1].ql_end - score[0].ql_start, 0.25)
        return min(
            self.config.timing_scale_ceiling_sec_per_ql,
            max(
                self.config.timing_scale_floor_sec_per_ql,
                audio_span / score_span,
            ),
        )

    def _span_options(
        self,
        candidate: JointCandidate,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
        *,
        forced: tuple[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        if not score:
            return []
        audio_total = max(candidates[-1].end, 0.05)
        expected = candidate.start / audio_total * len(score)
        ranked: list[tuple[tuple[float, ...], tuple[int, int]]] = []
        for start, event in enumerate(score):
            pitch_delta = abs(candidate.pitch - event.pitch)
            near_time = abs(start - expected)
            if pitch_delta > 2 and near_time > 6:
                continue
            for width in range(1, self.config.max_span_events + 1):
                end = start + width
                if end > len(score):
                    break
                rank = (
                    0.0 if pitch_delta == 0 else 1.0,
                    float(pitch_delta),
                    0.0 if width == 1 else 1.0,
                    0.0 if start in candidate.score_hints else 1.0,
                    near_time,
                    float(width),
                )
                ranked.append((rank, (start, end)))
        ranked.sort()
        options = [
            span
            for _rank, span in ranked[: self.config.max_options_per_candidate]
        ]
        if forced is not None and forced not in options:
            options.append(forced)
        return options

    def _transition(
        self,
        state: StructuralState,
        span: tuple[int, int],
        *,
        allow_long_delete: bool = False,
    ) -> tuple[StructuralState, JointOperation | None, tuple[int, ...]] | None:
        start, end = span
        next_cursor = end - 1
        if state.mode == "normal":
            if state.cursor < 0 or start > state.cursor:
                gap_start = state.cursor + 1
                deleted = tuple(range(gap_start, start))
                if (
                    len(deleted) > self.config.max_delete_events
                    and not allow_long_delete
                ):
                    return None
                return StructuralState(next_cursor), None, deleted
            resume = state.cursor + 1
            return (
                StructuralState(next_cursor, "replay", resume),
                JointOperation.REPEAT_ENTER,
                (),
            )

        resume = state.resume_event
        if start >= resume:
            deleted = tuple(range(resume, start))
            if (
                len(deleted) > self.config.max_delete_events
                and not allow_long_delete
            ):
                return None
            return (
                StructuralState(next_cursor),
                JointOperation.CONTINUE,
                deleted,
            )
        if start > state.cursor:
            deleted = tuple(range(state.cursor + 1, start))
            if (
                len(deleted) > self.config.max_delete_events
                and not allow_long_delete
            ):
                return None
            return (
                StructuralState(next_cursor, "replay", resume),
                JointOperation.REPLAY,
                deleted,
            )
        # A replay can itself restart before the outer resume point. Keep the
        # original resume anchor so continuation remains explicit.
        return (
            StructuralState(next_cursor, "replay", resume),
            JointOperation.REPEAT_ENTER,
            (),
        )

    def _edge_features(
        self,
        operation: JointOperation,
        candidate: JointCandidate | None,
        span: tuple[int, int] | None,
        state: StructuralState,
        destination: StructuralState,
        deleted_count: int,
        score: Sequence[ScoreEvent],
        tempo_scale: float,
        previous_candidate: JointCandidate | None,
        candidates: Sequence[JointCandidate] = (),
        candidate_index: int = -1,
    ) -> list[float]:
        one_hot = [float(operation == value) for value in _OPERATIONS]
        acoustic = (
            [0.0] * ACOUSTIC_FEATURE_DIM
            if candidate is None or not candidate.acoustic_features
            else [float(value) for value in candidate.acoustic_features]
        )
        if len(acoustic) != ACOUSTIC_FEATURE_DIM:
            raise ValueError(
                "Joint candidate acoustic feature dimension mismatch: "
                f"expected {ACOUSTIC_FEATURE_DIM}, got {len(acoustic)}"
            )
        if candidate is None:
            return one_hot + [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                min(deleted_count / 16.0, 2.0),
                0.0,
                float(state.mode == "replay"),
                0.0,
                0.0,
            ] + acoustic
        if span is None:
            candidate_duration = max(candidate.end - candidate.start, 1e-3)
            ioi = (
                candidate.start - previous_candidate.start
                if previous_candidate is not None
                else 0.0
            )
            return one_hot + [
                0.0,
                0.0,
                0.0,
                max(0.0, min(1.0, candidate.confidence)),
                min(candidate_duration / 2.0, 2.0),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(state.mode == "replay"),
                0.0,
                max(-2.0, min(2.0, ioi / max(tempo_scale, 1e-3))),
            ] + acoustic
        start, end = span
        first = score[start]
        last = score[end - 1]
        pitch_delta = candidate.pitch - first.pitch
        ql_duration = max(last.ql_end - first.ql_start, 1e-3)
        expected_duration = ql_duration * tempo_scale
        candidate_duration = max(candidate.end - candidate.start, 1e-3)
        ioi = (
            candidate.start - previous_candidate.start
            if previous_candidate is not None
            else 0.0
        )
        observed_interval = (
            candidate.pitch - previous_candidate.pitch
            if previous_candidate is not None
            else 0
        )
        score_interval = (
            first.pitch - score[state.cursor].pitch
            if 0 <= state.cursor < len(score)
            else 0
        )
        interval_error = observed_interval - score_interval
        score_advance = start - state.cursor if state.cursor >= 0 else start + 1
        resume_feature = (
            max(-2.0, min(2.0, (destination.resume_event - start) / 16.0))
            if destination.mode == "replay"
            else 0.0
        )
        resume_event = (
            destination.resume_event
            if destination.mode == "replay"
            else state.resume_event
        )
        if (
            self.config.continuation_feature_enabled
            and operation
            in {JointOperation.REPEAT_ENTER, JointOperation.REPLAY}
            and resume_event >= 0
            and candidate_index >= 0
        ):
            key = (
                id(candidates),
                id(score),
                candidate_index,
                span,
                resume_event,
            )
            cached = self._continuation_cache.get(key)
            if cached is None:
                cached = continuation_compatibility(
                    candidates,
                    score,
                    candidate_index=candidate_index,
                    replay_span=span,
                    resume_event=resume_event,
                    lookahead_notes=self.config.continuation_lookahead_notes,
                    candidate_skips=self.config.continuation_candidate_skips,
                    score_skips=self.config.continuation_score_skips,
                    pitch_tolerance=self.config.continuation_pitch_tolerance,
                )
                self._continuation_cache[key] = cached
            resume_feature = cached
        return one_hot + [
            max(-2.0, min(2.0, pitch_delta / 12.0)),
            float(pitch_delta == 0),
            float(abs(pitch_delta) == 1),
            max(0.0, min(1.0, candidate.confidence)),
            min(candidate_duration / max(expected_duration, 1e-3), 4.0) / 2.0,
            min(ql_duration / 4.0, 2.0),
            max(-2.0, min(2.0, interval_error / 12.0)),
            float(
                previous_candidate is not None
                and state.cursor >= 0
                and observed_interval == score_interval
            ),
            min(deleted_count / 16.0, 2.0),
            max(-2.0, min(2.0, score_advance / 16.0)),
            float(state.mode == "replay"),
            resume_feature,
            max(-2.0, min(2.0, ioi / max(tempo_scale, 1e-3))),
        ] + acoustic

    def _score_features(
        self, rows: list[list[float]], device: torch.device
    ) -> torch.Tensor:
        if any(len(row) != FEATURE_DIM for row in rows):
            raise RuntimeError("Internal joint edge feature dimension mismatch")
        return self.scorer(
            torch.tensor(rows, dtype=torch.float32, device=device)
        )

    def _expand_layer(
        self,
        previous: dict[StructuralState, torch.Tensor],
        candidate: JointCandidate,
        candidate_index: int,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
        tempo_scale: float,
        *,
        forced_span: tuple[int, int] | None,
        required_destination: StructuralState | None,
        viterbi: bool,
        backpointers: list[dict[StructuralState, tuple[StructuralState, LatticeStep]]],
    ) -> dict[StructuralState, torch.Tensor]:
        options = self._span_options(
            candidate, candidates, score, forced=forced_span
        )
        edge_rows: list[list[float]] = []
        edge_meta: list[
            tuple[StructuralState, StructuralState, LatticeStep]
        ] = []
        previous_candidate = (
            candidates[candidate_index - 1] if candidate_index else None
        )
        for state in previous:
            for unlinked_operation in (
                JointOperation.EXTRA,
                JointOperation.NOISE,
            ):
                extra = LatticeStep(
                    candidate_index,
                    None,
                    unlinked_operation,
                    None,
                    state.resume_event if state.mode == "replay" else None,
                )
                edge_rows.append(
                    self._edge_features(
                        unlinked_operation,
                        candidate,
                        None,
                        state,
                        state,
                        0,
                        score,
                        tempo_scale,
                        previous_candidate,
                        candidates,
                        candidate_index,
                    )
                )
                edge_meta.append((state, state, extra))
            for span in options:
                transition = self._transition(
                    state,
                    span,
                    allow_long_delete=forced_span == span,
                )
                if transition is None:
                    continue
                destination, structural, deleted = transition
                base_operation = (
                    JointOperation.MATCH
                    if candidate.pitch == score[span[0]].pitch
                    else JointOperation.SUBSTITUTE
                )
                scored_operation = structural or (
                    JointOperation.DELETE if deleted else base_operation
                )
                step = LatticeStep(
                    candidate_index,
                    span,
                    base_operation,
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
                edge_rows.append(
                    self._edge_features(
                        scored_operation,
                        candidate,
                        span,
                        state,
                        destination,
                        len(deleted),
                        score,
                        tempo_scale,
                        previous_candidate,
                        candidates,
                        candidate_index,
                    )
                )
                edge_meta.append((state, destination, step))

        if not edge_rows:
            raise ValueError(f"Joint lattice is empty at candidate {candidate_index}")
        device = next(self.scorer.parameters()).device
        edge_scores = self._score_features(edge_rows, device)
        if (
            (viterbi and self.config.noise_inference_bias)
            or (
                self.config.continuation_feature_enabled
                and self.config.continuation_score_weight
            )
            or self.config.repeat_fragment_penalty
        ):
            biases = []
            for index, (source, _destination, step) in enumerate(edge_meta):
                bias = (
                    self.config.noise_inference_bias
                    if viterbi
                    and step.operation == JointOperation.NOISE
                    else 0.0
                )
                if (
                    self.config.continuation_feature_enabled
                    and self.config.continuation_score_weight
                    and step.structural_operation
                    in {
                        JointOperation.REPEAT_ENTER,
                        JointOperation.REPLAY,
                    }
                ):
                    bias += (
                        self.config.continuation_score_weight
                        * edge_rows[index][CONTINUATION_FEATURE_INDEX]
                    )
                if (
                    self.config.repeat_fragment_penalty
                    and source.mode == "replay"
                    and step.structural_operation
                    == JointOperation.REPEAT_ENTER
                ):
                    bias -= self.config.repeat_fragment_penalty
                biases.append(bias)
            edge_scores = edge_scores + torch.tensor(
                biases, dtype=edge_scores.dtype, device=device
            )
        incoming: dict[
            StructuralState,
            list[tuple[torch.Tensor, StructuralState, LatticeStep]],
        ] = {}
        for edge_score, (source, destination, step) in zip(
            edge_scores, edge_meta
        ):
            incoming.setdefault(destination, []).append(
                (previous[source] + edge_score, source, step)
            )

        incoming_stacked = {
            destination: torch.stack([value[0] for value in values])
            for destination, values in incoming.items()
        }
        if len(incoming) > self.config.max_states:
            destinations = list(incoming)
            rank_values = torch.stack(
                [
                    torch.max(incoming_stacked[destination])
                    for destination in destinations
                ]
            ).detach().cpu().tolist()
            ranked = [
                destinations[index]
                for index in sorted(
                    range(len(destinations)),
                    key=lambda index: rank_values[index],
                    reverse=True,
                )
            ]
            kept = ranked[: self.config.max_states]
            if (
                required_destination is not None
                and required_destination in incoming
                and required_destination not in kept
            ):
                kept[-1] = required_destination
            incoming = {state: incoming[state] for state in kept}
            incoming_stacked = {
                state: incoming_stacked[state] for state in kept
            }
        output: dict[StructuralState, torch.Tensor] = {}
        pointers: dict[StructuralState, tuple[StructuralState, LatticeStep]] = {}
        destinations = list(incoming)
        stacked_groups = [
            incoming_stacked[destination] for destination in destinations
        ]
        if viterbi:
            winners = segment_argmax_first(stacked_groups)
            global_winners = winners.detach().cpu().tolist()
            offset = 0
            for destination, stacked, global_winner in zip(
                destinations,
                stacked_groups,
                global_winners,
            ):
                best = int(global_winner) - offset
                values = incoming[destination]
                output[destination] = stacked[best]
                pointers[destination] = (values[best][1], values[best][2])
                offset += len(stacked)
        else:
            reduced = segment_logsumexp_fp32(stacked_groups)
            output = {
                destination: reduced[index]
                for index, destination in enumerate(destinations)
            }
        if viterbi:
            backpointers.append(pointers)
        return output

    def log_partition(
        self,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
        *,
        forced_spans: Sequence[tuple[int, int] | None] | None = None,
    ) -> torch.Tensor:
        self._continuation_cache.clear()
        if forced_spans is not None and len(forced_spans) != len(candidates):
            raise ValueError("forced_spans must match candidate count")
        device = next(self.scorer.parameters()).device
        zero = torch.zeros((), device=device)
        active = {StructuralState(): zero}
        required_state = StructuralState()
        tempo_scale = self._tempo_scale(candidates, score)
        for index, candidate in enumerate(candidates):
            required_destination = None
            if forced_spans is not None:
                forced_span = forced_spans[index]
                if forced_span is None:
                    required_destination = required_state
                else:
                    transition = self._transition(
                        required_state,
                        forced_span,
                        allow_long_delete=True,
                    )
                    if transition is None:
                        raise ValueError(
                            f"Gold transition {required_state} -> "
                            f"{forced_span} is outside lattice"
                        )
                    required_destination = transition[0]
            active = self._expand_layer(
                active,
                candidate,
                index,
                candidates,
                score,
                tempo_scale,
                forced_span=forced_spans[index] if forced_spans else None,
                required_destination=required_destination,
                viterbi=False,
                backpointers=[],
            )
            if required_destination is not None:
                required_state = required_destination
        return torch.logsumexp(torch.stack(list(active.values())), dim=0)

    def gold_path_score(
        self,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
        spans: Sequence[tuple[int, int] | None],
        keep_unlinked: Sequence[bool] | None = None,
        *,
        batch_edge_scoring: bool = False,
    ) -> torch.Tensor:
        self._continuation_cache.clear()
        if len(candidates) != len(spans):
            raise ValueError("Gold spans must match candidate count")
        if keep_unlinked is not None and len(keep_unlinked) != len(candidates):
            raise ValueError("keep_unlinked must match candidate count")
        device = next(self.scorer.parameters()).device
        state = StructuralState()
        tempo_scale = self._tempo_scale(candidates, score)
        previous_candidate: JointCandidate | None = None
        rows: list[list[float]] = []
        biases: list[float] = []
        for candidate_index, (candidate, span) in enumerate(
            zip(candidates, spans)
        ):
            if span is None:
                operation = (
                    JointOperation.EXTRA
                    if keep_unlinked is None or keep_unlinked[candidate_index]
                    else JointOperation.NOISE
                )
                destination = state
                deleted: tuple[int, ...] = ()
                feature_span = None
            else:
                transition = self._transition(
                    state,
                    span,
                    allow_long_delete=True,
                )
                if transition is None:
                    raise ValueError(
                        f"Gold transition {state} -> {span} is outside lattice"
                    )
                destination, structural, deleted = transition
                base = (
                    JointOperation.MATCH
                    if candidate.pitch == score[span[0]].pitch
                    else JointOperation.SUBSTITUTE
                )
                operation = structural or (
                    JointOperation.DELETE if deleted else base
                )
                feature_span = span
            features = self._edge_features(
                operation,
                candidate,
                feature_span,
                state,
                destination,
                len(deleted),
                score,
                tempo_scale,
                previous_candidate,
                candidates,
                candidate_index,
            )
            bias = 0.0
            if (
                self.config.continuation_feature_enabled
                and self.config.continuation_score_weight
                and operation
                in {JointOperation.REPEAT_ENTER, JointOperation.REPLAY}
            ):
                bias += (
                    self.config.continuation_score_weight
                    * features[CONTINUATION_FEATURE_INDEX]
                )
            if (
                self.config.repeat_fragment_penalty
                and state.mode == "replay"
                and operation == JointOperation.REPEAT_ENTER
            ):
                bias -= self.config.repeat_fragment_penalty
            rows.append(features)
            biases.append(bias)
            state = destination
            previous_candidate = candidate
        if not rows:
            return torch.zeros((), dtype=torch.float32, device=device)
        edge_scores = (
            self._score_features(rows, device)
            if batch_edge_scoring
            else torch.stack(
                [self._score_features([row], device)[0] for row in rows]
            )
        )
        total = torch.zeros((), dtype=torch.float32, device=device)
        for edge_score, bias in zip(edge_scores, biases):
            total = total + edge_score.float() + bias
        return total

    def nll(
        self,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
        gold_spans: Sequence[tuple[int, int] | None],
        gold_keep_unlinked: Sequence[bool] | None = None,
    ) -> torch.Tensor:
        return self.log_partition(
            candidates, score, forced_spans=gold_spans
        ) - self.gold_path_score(
            candidates,
            score,
            gold_spans,
            keep_unlinked=gold_keep_unlinked,
        )

    def local_warmup_nll(
        self,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
        gold_spans: Sequence[tuple[int, int] | None],
        gold_keep_unlinked: Sequence[bool],
    ) -> torch.Tensor:
        """Vectorized local normalization used only before path-CRF training."""

        rows, groups = self.local_warmup_edges(
            candidates,
            score,
            gold_spans,
            gold_keep_unlinked,
        )
        device = next(self.scorer.parameters()).device
        scores = self._score_features(rows, device)
        losses = [
            torch.logsumexp(scores[start:end], dim=0)
            - scores[start + gold_offset]
            for start, end, gold_offset in groups
        ]
        return torch.stack(losses).mean()

    def local_warmup_edges(
        self,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
        gold_spans: Sequence[tuple[int, int] | None],
        gold_keep_unlinked: Sequence[bool],
    ) -> tuple[list[list[float]], list[tuple[int, int, int]]]:
        """Build local edge groups without moving tensors to a device."""

        self._continuation_cache.clear()
        if not (
            len(candidates) == len(gold_spans) == len(gold_keep_unlinked)
        ):
            raise ValueError("Warmup targets must match candidate count")
        tempo_scale = self._tempo_scale(candidates, score)
        state = StructuralState()
        rows: list[list[float]] = []
        groups: list[tuple[int, int, int]] = []
        previous_candidate: JointCandidate | None = None
        for candidate_index, (candidate, gold_span, keep_unlinked) in enumerate(
            zip(candidates, gold_spans, gold_keep_unlinked)
        ):
            start_offset = len(rows)
            gold_offset: int | None = None
            for operation in (JointOperation.EXTRA, JointOperation.NOISE):
                if gold_span is None and (
                    (operation == JointOperation.EXTRA and keep_unlinked)
                    or (operation == JointOperation.NOISE and not keep_unlinked)
                ):
                    gold_offset = len(rows) - start_offset
                rows.append(
                    self._edge_features(
                        operation,
                        candidate,
                        None,
                        state,
                        state,
                        0,
                        score,
                        tempo_scale,
                        previous_candidate,
                        candidates,
                        candidate_index,
                    )
                )
            options = self._span_options(
                candidate,
                candidates,
                score,
                forced=gold_span,
            )
            gold_destination = state
            for span in options:
                transition = self._transition(
                    state,
                    span,
                    allow_long_delete=span == gold_span,
                )
                if transition is None:
                    continue
                destination, structural, deleted = transition
                base = (
                    JointOperation.MATCH
                    if candidate.pitch == score[span[0]].pitch
                    else JointOperation.SUBSTITUTE
                )
                operation = structural or (
                    JointOperation.DELETE if deleted else base
                )
                if span == gold_span:
                    gold_offset = len(rows) - start_offset
                    gold_destination = destination
                row = self._edge_features(
                    operation,
                    candidate,
                    span,
                    state,
                    destination,
                    len(deleted),
                    score,
                    tempo_scale,
                    previous_candidate,
                    candidates,
                    candidate_index,
                )
                rows.append(row)
                if (
                    span != gold_span
                    and operation == JointOperation.REPEAT_ENTER
                    and row[CONTINUATION_FEATURE_INDEX] < 0.0
                ):
                    rows.extend(
                        [list(row)]
                        * max(
                            0,
                            int(
                                self.config.continuation_hard_negative_copies
                            ),
                        )
                    )
            if gold_offset is None:
                raise ValueError(
                    f"Gold warmup edge missing at candidate {candidate_index}"
                )
            groups.append((start_offset, len(rows), gold_offset))
            state = gold_destination
            previous_candidate = candidate
        return rows, groups

    @torch.no_grad()
    def decode(
        self,
        candidates: Sequence[JointCandidate],
        score: Sequence[ScoreEvent],
    ) -> LatticePath:
        self.scorer.eval()
        self._continuation_cache.clear()
        device = next(self.scorer.parameters()).device
        active = {StructuralState(): torch.zeros((), device=device)}
        backpointers: list[
            dict[StructuralState, tuple[StructuralState, LatticeStep]]
        ] = []
        tempo_scale = self._tempo_scale(candidates, score)
        for index, candidate in enumerate(candidates):
            active = self._expand_layer(
                active,
                candidate,
                index,
                candidates,
                score,
                tempo_scale,
                forced_span=None,
                required_destination=None,
                viterbi=True,
                backpointers=backpointers,
            )
        states = list(active)
        state = states[
            int(torch.argmax(torch.stack([active[key] for key in states])).item())
        ]
        score_value = float(active[state].cpu())
        trailing = tuple(range(state.cursor + 1, len(score)))
        steps: list[LatticeStep] = []
        for pointers in reversed(backpointers):
            previous, step = pointers[state]
            steps.append(step)
            state = previous
        steps.reverse()
        return LatticePath(tuple(steps), trailing, score_value)


def candidates_from_events(events: Iterable[object]) -> list[JointCandidate]:
    return sorted(
        [
            JointCandidate(
                pitch=int(getattr(value, "pitch")),
                start=float(getattr(value, "start")),
                end=float(getattr(value, "end")),
                confidence=float(getattr(value, "confidence", 1.0)),
                acoustic_features=tuple(
                    float(item)
                    for item in getattr(value, "acoustic_features", ())
                ),
            )
            for value in events
        ],
        key=lambda value: (value.start, value.pitch, value.end),
    )
