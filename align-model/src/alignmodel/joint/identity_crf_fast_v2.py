"""Packed NumPy/Numba runtime exactly matching identity_crf_v1 semantics.

The neural emission graph remains in PyTorch. The dynamic program itself uses
one custom autograd node whose backward pass analytically accumulates edge
marginals, avoiding a Python autograd graph per lattice edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Mapping

import numpy as np
import torch
from numba import njit
from torch import Tensor

from .identity_crf_v1 import (
    ACTION_COUNT,
    CANDIDATE_EXTRA,
    CANDIDATE_EXTRA_COPY,
    LINK,
    ORNAMENT_EXTRA,
    SCORE_DELETE,
    SKIP_ORNAMENT,
    IdentityLattice,
    LatticeHypothesis,
    OrnamentIdentityCRF,
    _action_allowed,
    _emitted_relationship,
    _link_span_options,
    _predecessor,
)
from .index import JointEvent


SCHEMA_VERSION = "align-ornament-identity-crf-fast-v2"
_NEG_INF = -np.inf


@dataclass(frozen=True)
class PackedEdges:
    rows: int
    columns: int
    src: np.ndarray
    dst: np.ndarray
    action: np.ndarray
    row: np.ndarray
    column: np.ndarray
    previous_column: np.ndarray
    template_is_ornament: np.ndarray


_PACK_CACHE: dict[tuple[int, int, bool], PackedEdges] = {}


def pack_hypothesis_edges(
    lattice: IdentityLattice,
    hypothesis: LatticeHypothesis,
    *,
    gold_only: bool,
) -> PackedEdges:
    key = (id(lattice), id(hypothesis), bool(gold_only))
    cached = _PACK_CACHE.get(key)
    if cached is not None:
        return cached
    if gold_only and lattice.targets is None:
        raise ValueError("Gold edge packing requires targets")
    rows, columns = len(lattice.candidates), len(hypothesis.template)
    records = []
    for diagonal in range(1, rows + columns + 1):
        first_row = max(0, diagonal - columns)
        last_row = min(rows, diagonal)
        for row in range(first_row, last_row + 1):
            column = diagonal - row
            if not 0 <= column <= columns:
                continue
            candidate = lattice.candidates[row - 1] if row else None
            target = (
                lattice.targets[row - 1]
                if gold_only and row and lattice.targets is not None
                else None
            )
            unit = hypothesis.template[column - 1] if column else None
            for action in range(ACTION_COUNT):
                if action == LINK and candidate is not None:
                    for previous_column, score_span, copy_pass in (
                        _link_span_options(hypothesis.template, column)
                    ):
                        if gold_only:
                            assert target is not None and unit is not None
                            if (
                                target.score_span != score_span
                                or target.copy_pass != copy_pass
                                or target.relationship
                                != _emitted_relationship(candidate, unit)
                            ):
                                continue
                        previous_row = row - 1
                        src = (
                            -1
                            if previous_row == 0 and previous_column == 0
                            else previous_row * (columns + 1)
                            + previous_column
                        )
                        records.append(
                            (
                                src,
                                row * (columns + 1) + column,
                                action,
                                row,
                                column,
                                previous_column,
                            )
                        )
                    continue
                predecessor = _predecessor(action, row, column)
                if predecessor is None or not _action_allowed(
                    action,
                    candidate,
                    unit,
                    target,
                    lattice,
                    gold_only=gold_only,
                ):
                    continue
                previous_row, previous_column = predecessor
                src = (
                    -1
                    if previous_row == 0 and previous_column == 0
                    else previous_row * (columns + 1) + previous_column
                )
                records.append(
                    (
                        src,
                        row * (columns + 1) + column,
                        action,
                        row,
                        column,
                        previous_column,
                    )
                )
    packed = PackedEdges(
        rows=rows,
        columns=columns,
        src=np.asarray([row[0] for row in records], dtype=np.int32),
        dst=np.asarray([row[1] for row in records], dtype=np.int32),
        action=np.asarray([row[2] for row in records], dtype=np.int8),
        row=np.asarray([row[3] for row in records], dtype=np.int32),
        column=np.asarray([row[4] for row in records], dtype=np.int32),
        previous_column=np.asarray(
            [row[5] for row in records], dtype=np.int32
        ),
        template_is_ornament=np.asarray(
            [unit.kind == "ornament_extra" for unit in hypothesis.template],
            dtype=np.bool_,
        ),
    )
    _PACK_CACHE[key] = packed
    return packed


@njit(cache=True)
def _logadd(left: float, right: float) -> float:
    if np.isneginf(left):
        return right
    if np.isneginf(right):
        return left
    maximum = max(left, right)
    return maximum + np.log(np.exp(left - maximum) + np.exp(right - maximum))


@njit(cache=True)
def _logsum(values: np.ndarray) -> float:
    maximum = np.max(values)
    if np.isneginf(maximum):
        return maximum
    return maximum + np.log(np.exp(values - maximum).sum())


@njit(cache=True)
def _edge_emission(
    edge: int,
    actions: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    previous_columns: np.ndarray,
    template_is_ornament: np.ndarray,
    link: np.ndarray,
    ornament: np.ndarray,
    extra: np.ndarray,
    delete: np.ndarray,
) -> float:
    action = int(actions[edge])
    row = int(rows[edge])
    column = int(columns[edge])
    if action == LINK:
        value = link[row - 1, column - 1]
        for template_index in range(
            int(previous_columns[edge]), column - 1
        ):
            value += delete[
                template_index,
                1 if template_is_ornament[template_index] else 0,
            ]
        return value
    if action == ORNAMENT_EXTRA:
        return ornament[row - 1, column - 1]
    if action == CANDIDATE_EXTRA:
        return extra[row - 1, 0]
    if action == CANDIDATE_EXTRA_COPY:
        return extra[row - 1, 1]
    if action == SCORE_DELETE:
        return delete[column - 1, 0]
    return delete[column - 1, 1]


@njit(cache=True)
def _forward(
    cell_count: int,
    final_cell: int,
    src: np.ndarray,
    dst: np.ndarray,
    actions: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    previous_columns: np.ndarray,
    template_is_ornament: np.ndarray,
    link: np.ndarray,
    ornament: np.ndarray,
    extra: np.ndarray,
    delete: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
) -> tuple[float, np.ndarray]:
    alpha = np.full((cell_count, ACTION_COUNT), _NEG_INF, dtype=np.float64)
    scratch = np.empty(ACTION_COUNT, dtype=np.float64)
    for edge in range(len(src)):
        action = int(actions[edge])
        emission = _edge_emission(
            edge,
            actions,
            rows,
            columns,
            previous_columns,
            template_is_ornament,
            link,
            ornament,
            extra,
            delete,
        )
        if src[edge] < 0:
            value = initial[action] + emission
        else:
            for previous_action in range(ACTION_COUNT):
                scratch[previous_action] = (
                    alpha[src[edge], previous_action]
                    + transitions[previous_action, action]
                )
            value = _logsum(scratch) + emission
        alpha[dst[edge], action] = _logadd(
            alpha[dst[edge], action], value
        )
    return _logsum(alpha[final_cell]), alpha


@njit(cache=True)
def _backward_marginals(
    cell_count: int,
    final_cell: int,
    src: np.ndarray,
    dst: np.ndarray,
    actions: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    previous_columns: np.ndarray,
    template_is_ornament: np.ndarray,
    link: np.ndarray,
    ornament: np.ndarray,
    extra: np.ndarray,
    delete: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    log_partition: float,
    alpha: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    beta = np.full((cell_count, ACTION_COUNT), _NEG_INF, dtype=np.float64)
    beta[final_cell, :] = 0.0
    for edge in range(len(src) - 1, -1, -1):
        if src[edge] < 0:
            continue
        action = int(actions[edge])
        emission = _edge_emission(
            edge,
            actions,
            rows,
            columns,
            previous_columns,
            template_is_ornament,
            link,
            ornament,
            extra,
            delete,
        )
        for previous_action in range(ACTION_COUNT):
            value = (
                transitions[previous_action, action]
                + emission
                + beta[dst[edge], action]
            )
            beta[src[edge], previous_action] = _logadd(
                beta[src[edge], previous_action], value
            )
    grad_link = np.zeros_like(link)
    grad_ornament = np.zeros_like(ornament)
    grad_extra = np.zeros_like(extra)
    grad_delete = np.zeros_like(delete)
    grad_initial = np.zeros_like(initial)
    grad_transitions = np.zeros_like(transitions)
    for edge in range(len(src)):
        action = int(actions[edge])
        emission = _edge_emission(
            edge,
            actions,
            rows,
            columns,
            previous_columns,
            template_is_ornament,
            link,
            ornament,
            extra,
            delete,
        )
        marginal = 0.0
        if src[edge] < 0:
            value = np.exp(
                initial[action]
                + emission
                + beta[dst[edge], action]
                - log_partition
            )
            grad_initial[action] += value
            marginal = value
        else:
            for previous_action in range(ACTION_COUNT):
                value = np.exp(
                    alpha[src[edge], previous_action]
                    + transitions[previous_action, action]
                    + emission
                    + beta[dst[edge], action]
                    - log_partition
                )
                grad_transitions[previous_action, action] += value
                marginal += value
        row = int(rows[edge])
        column = int(columns[edge])
        if action == LINK:
            grad_link[row - 1, column - 1] += marginal
            for template_index in range(
                int(previous_columns[edge]), column - 1
            ):
                grad_delete[
                    template_index,
                    1 if template_is_ornament[template_index] else 0,
                ] += marginal
        elif action == ORNAMENT_EXTRA:
            grad_ornament[row - 1, column - 1] += marginal
        elif action == CANDIDATE_EXTRA:
            grad_extra[row - 1, 0] += marginal
        elif action == CANDIDATE_EXTRA_COPY:
            grad_extra[row - 1, 1] += marginal
        elif action == SCORE_DELETE:
            grad_delete[column - 1, 0] += marginal
        else:
            grad_delete[column - 1, 1] += marginal
    return (
        grad_link,
        grad_ornament,
        grad_extra,
        grad_delete,
        grad_initial,
        grad_transitions,
    )


class _PackedPartition(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        packed: PackedEdges,
        link: Tensor,
        ornament: Tensor,
        extra: Tensor,
        delete: Tensor,
        hypothesis: Tensor,
        initial: Tensor,
        transitions: Tensor,
    ) -> Tensor:
        arrays = [
            value.detach().cpu().double().numpy()
            for value in (link, ornament, extra, delete, initial, transitions)
        ]
        cell_count = (packed.rows + 1) * (packed.columns + 1)
        final_cell = packed.rows * (packed.columns + 1) + packed.columns
        log_partition, alpha = _forward(
            cell_count,
            final_cell,
            packed.src,
            packed.dst,
            packed.action,
            packed.row,
            packed.column,
            packed.previous_column,
            packed.template_is_ornament,
            *arrays,
        )
        if not np.isfinite(log_partition):
            raise ValueError("Packed identity lattice has no complete path")
        ctx.packed = packed
        ctx.log_partition = log_partition
        ctx.alpha = alpha
        ctx.save_for_backward(
            link,
            ornament,
            extra,
            delete,
            initial,
            transitions,
        )
        return hypothesis + hypothesis.new_tensor(log_partition)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Any, ...]:
        link, ornament, extra, delete, initial, transitions = (
            ctx.saved_tensors
        )
        arrays = [
            value.detach().cpu().double().numpy()
            for value in (link, ornament, extra, delete, initial, transitions)
        ]
        packed = ctx.packed
        cell_count = (packed.rows + 1) * (packed.columns + 1)
        final_cell = packed.rows * (packed.columns + 1) + packed.columns
        gradients = _backward_marginals(
            cell_count,
            final_cell,
            packed.src,
            packed.dst,
            packed.action,
            packed.row,
            packed.column,
            packed.previous_column,
            packed.template_is_ornament,
            *arrays,
            ctx.log_partition,
            ctx.alpha,
        )
        converted = [
            torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device)
            * grad_output
            for value, tensor in zip(
                gradients,
                (link, ornament, extra, delete, initial, transitions),
            )
        ]
        return (
            None,
            converted[0],
            converted[1],
            converted[2],
            converted[3],
            grad_output,
            converted[4],
            converted[5],
        )


def fast_hypothesis_log_partition(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
    hypothesis: LatticeHypothesis,
    *,
    gold_only: bool = False,
) -> Tensor:
    scores = model.scores(lattice, hypothesis)
    packed = pack_hypothesis_edges(
        lattice, hypothesis, gold_only=gold_only
    )
    return _PackedPartition.apply(
        packed,
        scores["link"],
        scores["ornament"],
        scores["extra"],
        scores["delete"],
        scores["hypothesis"],
        model.initial,
        model.transitions,
    )


def fast_identity_crf_nll(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
    *,
    normalize: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    if lattice.targets is None:
        raise ValueError("CRF training requires exact identity targets")
    all_values = [
        fast_hypothesis_log_partition(model, lattice, hypothesis)
        for hypothesis in lattice.hypotheses
    ]
    gold_values = [
        fast_hypothesis_log_partition(
            model, lattice, hypothesis, gold_only=True
        )
        for hypothesis in lattice.hypotheses
        if hypothesis.is_gold_compatible
    ]
    if not gold_values:
        raise ValueError("Selected lattice contains no gold hypothesis")
    log_partition = torch.logsumexp(torch.stack(all_values), dim=0)
    gold_partition = torch.logsumexp(torch.stack(gold_values), dim=0)
    loss = log_partition - gold_partition
    if normalize:
        loss = loss / max(
            len(lattice.candidates) + len(lattice.score), 1
        )
    return loss, {
        "loss": loss.detach(),
        "log_partition": log_partition.detach(),
        "gold_log_partition": gold_partition.detach(),
    }


@njit(cache=True)
def _viterbi(
    cell_count: int,
    final_cell: int,
    src: np.ndarray,
    dst: np.ndarray,
    actions: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    previous_columns: np.ndarray,
    template_is_ornament: np.ndarray,
    link: np.ndarray,
    ornament: np.ndarray,
    extra: np.ndarray,
    delete: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    values = np.full((cell_count, ACTION_COUNT), _NEG_INF, dtype=np.float64)
    back_edge = np.full((cell_count, ACTION_COUNT), -1, dtype=np.int32)
    back_action = np.full((cell_count, ACTION_COUNT), -1, dtype=np.int8)
    for edge in range(len(src)):
        action = int(actions[edge])
        emission = _edge_emission(
            edge,
            actions,
            rows,
            columns,
            previous_columns,
            template_is_ornament,
            link,
            ornament,
            extra,
            delete,
        )
        previous_action = -1
        if src[edge] < 0:
            value = initial[action] + emission
        else:
            value = _NEG_INF
            for old_action in range(ACTION_COUNT):
                proposed = (
                    values[src[edge], old_action]
                    + transitions[old_action, action]
                    + emission
                )
                if proposed > value:
                    value = proposed
                    previous_action = old_action
        if value > values[dst[edge], action]:
            values[dst[edge], action] = value
            back_edge[dst[edge], action] = edge
            back_action[dst[edge], action] = previous_action
    final_action = 0
    best = values[final_cell, 0]
    for action in range(1, ACTION_COUNT):
        if values[final_cell, action] > best:
            best = values[final_cell, action]
            final_action = action
    return best, final_action, back_edge, back_action


def _fast_viterbi_hypothesis(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
    hypothesis: LatticeHypothesis,
) -> tuple[float, list[int]]:
    with torch.no_grad():
        scores = model.scores(lattice, hypothesis)
    packed = pack_hypothesis_edges(
        lattice, hypothesis, gold_only=False
    )
    arrays = [
        value.detach().cpu().double().numpy()
        for value in (
            scores["link"],
            scores["ornament"],
            scores["extra"],
            scores["delete"],
            model.initial,
            model.transitions,
        )
    ]
    cell_count = (packed.rows + 1) * (packed.columns + 1)
    final_cell = packed.rows * (packed.columns + 1) + packed.columns
    score, action, back_edge, back_action = _viterbi(
        cell_count,
        final_cell,
        packed.src,
        packed.dst,
        packed.action,
        packed.row,
        packed.column,
        packed.previous_column,
        packed.template_is_ornament,
        *arrays,
    )
    path = []
    cell = final_cell
    while cell:
        edge = int(back_edge[cell, action])
        if edge < 0:
            raise RuntimeError("Packed Viterbi backtrace is incomplete")
        path.append(edge)
        cell = int(packed.src[edge])
        if cell < 0:
            break
        action = int(back_action[int(packed.dst[edge]), action])
    path.reverse()
    return (
        float(score + float(scores["hypothesis"].detach().cpu())),
        path,
    )


def fast_decode_identity_crf(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
) -> tuple[tuple[JointEvent, ...], frozenset[int], dict[str, Any]]:
    ranked = [
        (*_fast_viterbi_hypothesis(model, lattice, hypothesis), hypothesis)
        for hypothesis in lattice.hypotheses
    ]
    score_value, path, hypothesis = max(
        ranked,
        key=lambda value: (
            value[0],
            -value[2].grammar.copies,
            value[2].grammar.source_span or (-1, -1),
        ),
    )
    packed = pack_hypothesis_edges(
        lattice, hypothesis, gold_only=False
    )
    events = []
    deletions = set()
    action_counts = np.zeros(ACTION_COUNT, dtype=np.int64)
    for edge in path:
        action = int(packed.action[edge])
        row = int(packed.row[edge])
        column = int(packed.column[edge])
        previous_column = int(packed.previous_column[edge])
        action_counts[action] += 1
        candidate = lattice.candidates[row - 1] if row else None
        unit = hypothesis.template[column - 1] if column else None
        if action == LINK:
            assert candidate is not None and unit is not None
            linked_units = [
                value
                for value in hypothesis.template[previous_column:column]
                if value.kind == "linked"
                and value.score_index is not None
                and value.copy_pass == unit.copy_pass
            ]
            span_start = min(
                int(value.score_index) for value in linked_units
            )
            span_end = max(
                int(value.score_index) for value in linked_units
            ) + 1
            relationship = _emitted_relationship(candidate, unit)
            events.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=(span_start, span_end),
                    relationship=relationship,
                    copy_pass=unit.copy_pass,
                    origin_relationship=(
                        "match"
                        if candidate.pitch == unit.pitch
                        else "substitute"
                    ),
                    rendered_index=row - 1,
                    confidence=candidate.confidence,
                )
            )
        elif action in {
            ORNAMENT_EXTRA,
            CANDIDATE_EXTRA,
            CANDIDATE_EXTRA_COPY,
        }:
            assert candidate is not None
            events.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=None,
                    relationship=(
                        "copy"
                        if action == CANDIDATE_EXTRA_COPY
                        else "extra"
                    ),
                    copy_pass=1 if action == CANDIDATE_EXTRA_COPY else 0,
                    rendered_index=row - 1,
                    confidence=candidate.confidence,
                )
            )
        elif (
            action == SCORE_DELETE
            and unit is not None
            and unit.copy_pass == 0
            and unit.score_index is not None
        ):
            deletions.add(int(unit.score_index))
    return tuple(events), frozenset(deletions), {
        "score": score_value,
        "copies": hypothesis.grammar.copies,
        "source_span": hypothesis.grammar.source_span,
        "hypotheses": len(lattice.hypotheses),
        "actions": {
            name: int(action_counts[index])
            for index, name in enumerate(
                (
                    "link",
                    "ornament_extra",
                    "candidate_extra",
                    "candidate_extra_copy",
                    "score_delete",
                    "skip_ornament",
                )
            )
        },
        "runtime": SCHEMA_VERSION,
    }
