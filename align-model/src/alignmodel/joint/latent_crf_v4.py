from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .grammar_mapper_v2 import GrammarHypothesis
from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate


OPERATION_NAMES = ("match", "copy", "substitute", "extra", "delete")
FEATURE_DIM = 16


@dataclass(frozen=True)
class LatentTransition:
    source: tuple[int, int]
    destination: tuple[int, int]
    operation: str
    event_index: int | None
    unit_start: int
    unit_end: int
    score_span: tuple[int, int] | None
    copy_pass: int


class LatentSpanCRF(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.plan_copy_bias = nn.Parameter(torch.zeros(3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.edge(features).squeeze(-1)


def transitions(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    plan: GrammarHypothesis,
    *,
    max_span_units: int = 8,
) -> tuple[LatentTransition, ...]:
    output = []
    for event_count in range(len(events) + 1):
        for unit_count in range(len(plan.units) + 1):
            if event_count < len(events):
                copy_pass = (
                    plan.units[unit_count - 1][1] if unit_count else 0
                )
                output.append(
                    LatentTransition(
                        (event_count, unit_count),
                        (event_count + 1, unit_count),
                        "copy" if copy_pass else "extra",
                        event_count,
                        unit_count,
                        unit_count,
                        None,
                        copy_pass,
                    )
                )
            if unit_count < len(plan.units):
                score_index, copy_pass = plan.units[unit_count]
                output.append(
                    LatentTransition(
                        (event_count, unit_count),
                        (event_count, unit_count + 1),
                        "delete",
                        None,
                        unit_count,
                        unit_count + 1,
                        (score_index, score_index + 1),
                        copy_pass,
                    )
                )
            if event_count >= len(events):
                continue
            for width in range(1, max_span_units + 1):
                end = unit_count + width
                if end > len(plan.units):
                    break
                selected = plan.units[unit_count:end]
                copy_passes = {value[1] for value in selected}
                if len(copy_passes) != 1:
                    continue
                copy_pass = next(iter(copy_passes))
                score_indices = [value[0] for value in selected]
                span = (min(score_indices), max(score_indices) + 1)
                endpoint = score_indices[-1]
                operation = (
                    "copy"
                    if copy_pass
                    else "match"
                    if events[event_count].pitch == score[endpoint].pitch
                    else "substitute"
                )
                output.append(
                    LatentTransition(
                        (event_count, unit_count),
                        (event_count + 1, end),
                        operation,
                        event_count,
                        unit_count,
                        end,
                        span,
                        copy_pass,
                    )
                )
    return tuple(output)


def transition_features(
    transition: LatentTransition,
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    plan: GrammarHypothesis,
) -> torch.Tensor:
    one_hot = [
        float(transition.operation == name) for name in OPERATION_NAMES
    ]
    if transition.event_index is not None:
        event = events[transition.event_index]
        event_position = transition.event_index / max(len(events) - 1, 1)
        duration = min(event.end - event.start, 2.0)
    else:
        event = None
        event_position = 0.0
        duration = 0.0
    if transition.score_span is not None:
        endpoint = transition.score_span[1] - 1
        target = score[endpoint]
        pitch_delta = (
            (event.pitch - target.pitch) / 12.0 if event is not None else 0.0
        )
        score_position = endpoint / max(len(score) - 1, 1)
        score_duration = min(target.ql_end - target.ql_start, 2.0)
        span_length = (
            transition.score_span[1] - transition.score_span[0]
        ) / max(len(score), 1)
    else:
        pitch_delta = score_position = score_duration = span_length = 0.0
    values = [
        *one_hot,
        pitch_delta,
        float(event is not None and abs(pitch_delta) < 1e-9),
        span_length,
        float(transition.copy_pass == 0),
        float(transition.copy_pass == 1),
        float(transition.copy_pass == 2),
        event_position,
        score_position,
        duration,
        score_duration,
        abs(duration - score_duration),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _gold_consistent(
    transition: LatentTransition,
    gold: Sequence[JointEvent],
    target_deletions: frozenset[int],
) -> bool:
    if transition.operation == "delete":
        assert transition.score_span is not None
        return (
            transition.copy_pass == 0
            and transition.score_span[0] in target_deletions
        )
    assert transition.event_index is not None
    target = gold[transition.event_index]
    if transition.score_span is None:
        return (
            target.score_span is None
            and target.copy_pass == transition.copy_pass
        )
    target_type = "copy" if target.is_copy else target.relationship
    return (
        transition.score_span == target.score_span
        and transition.copy_pass == target.copy_pass
        and transition.operation == target_type
    )


def log_partition(
    edge_scores: torch.Tensor,
    lattice: Sequence[LatentTransition],
    shape: tuple[int, int],
    *,
    allowed: Sequence[bool] | None = None,
) -> torch.Tensor:
    incoming: dict[tuple[int, int], list[torch.Tensor]] = {(0, 0): []}
    values: dict[tuple[int, int], torch.Tensor] = {
        (0, 0): edge_scores.new_zeros(())
    }
    by_destination: dict[
        tuple[int, int], list[tuple[int, LatentTransition]]
    ] = {}
    for index, transition in enumerate(lattice):
        if allowed is not None and not allowed[index]:
            continue
        by_destination.setdefault(transition.destination, []).append(
            (index, transition)
        )
    for event_count in range(shape[0] + 1):
        for unit_count in range(shape[1] + 1):
            destination = (event_count, unit_count)
            if destination == (0, 0):
                continue
            candidates = [
                values[transition.source] + edge_scores[index]
                for index, transition in by_destination.get(destination, ())
                if transition.source in values
            ]
            if candidates:
                values[destination] = torch.logsumexp(
                    torch.stack(candidates).float(), dim=0
                )
    final = (shape[0], shape[1])
    return values.get(final, edge_scores.new_full((), -torch.inf))


def latent_crf_loss(
    model: LatentSpanCRF,
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    plan: GrammarHypothesis,
    gold: Sequence[JointEvent],
    target_deletions: frozenset[int] = frozenset(),
    *,
    max_span_units: int = 8,
) -> tuple[torch.Tensor, dict[str, object]]:
    lattice = transitions(
        events, score, plan, max_span_units=max_span_units
    )
    device = next(model.parameters()).device
    features = torch.stack(
        [
            transition_features(edge, events, score, plan)
            for edge in lattice
        ]
    ).to(device)
    scores = model(features)
    scores = scores + model.plan_copy_bias[min(plan.copies, 2)]
    allowed = [
        _gold_consistent(edge, gold, target_deletions) for edge in lattice
    ]
    shape = (len(events), len(plan.units))
    all_partition = log_partition(scores, lattice, shape)
    gold_partition = log_partition(
        scores, lattice, shape, allowed=allowed
    )
    if not torch.isfinite(gold_partition):
        raise ValueError("Latent lattice has no gold-consistent path")
    return all_partition - gold_partition, {
        "edges": len(lattice),
        "gold_edges": sum(allowed),
        "gold_coverage": True,
        "all_log_partition": float(all_partition.detach()),
        "gold_log_partition": float(gold_partition.detach()),
    }
