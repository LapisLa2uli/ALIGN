"""Vectorized DROP/EMIT forward and Viterbi.

Score tables are computed in one batched network call per action type.
The dynamic program then only adds precomputed scores, so a few-hundred-group
row stays in the tens of milliseconds instead of minutes.
"""

from __future__ import annotations

from math import inf
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .drop_emit_lattice_v1 import (
    ACTION_NAMES,
    DROP,
    EMIT,
    EMIT_FEATURE_DIM,
    INSERT_EXTRA,
    NEGATIVE_INFINITY,
    AcousticGroup,
    DecodedEmit,
    DropEmitLattice,
    DropEmitScorer,
    IdentityCandidate,
    _crf_path_bonus,
    _emit_features,
    _group_features,
    _insert_features,
    prove_gold_path_coverage,
)
from .identity_crf_v1 import IdentityTarget


def _static_emit_row(group: AcousticGroup, candidate_index: int) -> np.ndarray:
    candidate = group.candidates[candidate_index]
    values = _emit_features(
        group,
        candidate,
        emitted_so_far=0,
        remaining_groups=0,
        target_hint=None,
    )
    row = np.asarray(values, np.float32)
    if row.shape != (EMIT_FEATURE_DIM,):
        raise ValueError("unexpected emit feature width")
    return row


def score_tables(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    emit_cap: int,
) -> dict[str, Tensor]:
    """Return differentiable drop/emit/insert scores.

    Emit scores do not consume gold labels. The first two contextual emit
    features (emitted count, remaining groups) are added after a single static
    forward by a linear re-score of those input columns.
    """

    device = model.initial.device
    dtype = model.initial.dtype
    groups = lattice.groups
    n_groups = len(groups)
    emit_cap = max(int(emit_cap), 1)
    if n_groups == 0:
        empty = model.initial.new_zeros((0,))
        return {
            "drop": empty,
            "emit": model.initial.new_zeros((0, emit_cap)),
            "emit_choice": model.initial.new_zeros((0, emit_cap, 1)),
            "insert": model.initial.new_zeros((1, emit_cap)),
            "local_index": [],
        }

    drop_np = np.asarray([_group_features(group) for group in groups], np.float32)
    drop_scores = model.drop_score(
        torch.as_tensor(drop_np, device=device, dtype=dtype)
    )

    # One static emit row per interval; emitted-count / remaining-group axes
    # are the feature columns 16 and 17 and are applied by shifting the
    # first-layer contribution, which is exact for a linear first layer.
    static_rows = []
    group_spans: list[tuple[int, int]] = []
    local_of_pool: list[dict[int, int]] = []
    cursor = 0
    for group in groups:
        start = cursor
        mapping = {}
        for local, pool_index in enumerate(group.candidate_indices):
            static_rows.append(_static_emit_row(group, local))
            mapping[pool_index] = local
            cursor += 1
        group_spans.append((start, cursor))
        local_of_pool.append(mapping)
    static = np.stack(static_rows, axis=0)
    emitted = np.arange(emit_cap, dtype=np.float32) / 64.0
    remaining = (
        (n_groups - np.arange(n_groups, dtype=np.float32) - 1.0) / 64.0
    )
    # [G, E, C, D] would be large; score per (group, candidate, emitted).
    # Build [n_intervals, emit_cap, dim] which is ~5*237*46*28 ~ 1.5e6, fine.
    expanded = np.broadcast_to(static[:, None, :], (len(static), emit_cap, EMIT_FEATURE_DIM)).copy()
    expanded[:, :, 16] = emitted[None, :]
    for group_index, (start, end) in enumerate(group_spans):
        expanded[start:end, :, 17] = remaining[group_index]
    flat = torch.as_tensor(
        expanded.reshape(-1, EMIT_FEATURE_DIM), device=device, dtype=dtype
    )
    flat_scores = model.emit_score(flat).view(len(static), emit_cap)
    max_c = max(end - start for start, end in group_spans)
    emit_choice = model.initial.new_full((n_groups, emit_cap, max_c), NEGATIVE_INFINITY)
    for group_index, (start, end) in enumerate(group_spans):
        width = end - start
        emit_choice[group_index, :, :width] = flat_scores[start:end].transpose(0, 1)
    emit_scores = torch.logsumexp(emit_choice, dim=-1)

    insert_np = np.zeros((n_groups + 1, emit_cap, 16), np.float32)
    emit_axis = np.arange(emit_cap, dtype=np.float32)
    group_axis = np.arange(n_groups + 1, dtype=np.float32)[:, None]
    insert_np[:, :, 1] = 1.0
    insert_np[:, :, 8] = emit_axis[None, :] / 64.0
    insert_np[:, :, 9] = (n_groups - group_axis) / 64.0
    insert_np[:, :, 10] = emit_axis[None, :] / 64.0
    insert_np[:, :, 11] = 1.0
    insert_scores = model.insert_score(
        torch.as_tensor(
            insert_np.reshape(-1, 16), device=device, dtype=dtype
        )
    ).view(n_groups + 1, emit_cap)
    return {
        "drop": drop_scores,
        "emit": emit_scores,
        "emit_choice": emit_choice,
        "insert": insert_scores,
        "local_index": local_of_pool,
        "max_candidates": max_c,
    }


def _transition_bank(model: DropEmitScorer) -> Tensor:
    # [4, 3] previous slot (3 = start) -> action
    bank = model.initial.new_zeros((4, 3))
    bank[:3] = model.transitions
    bank[3] = model.initial
    return bank


def fast_gold_path_log_prob(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    frozen_crf_score: float | None = None,
) -> Tensor:
    coverage = prove_gold_path_coverage(lattice)
    if not coverage["passed"]:
        raise ValueError(f"Incomplete gold path coverage: {coverage}")
    emit_cap = max(len(lattice.targets), 1)
    tables = score_tables(model, lattice, emit_cap=emit_cap)
    bank = _transition_bank(model)
    score = model.initial.new_zeros(())
    previous = 3
    group_cursor = 0
    for step in lattice.gold_steps:
        if step.group_index is not None:
            while group_cursor < step.group_index:
                score = score + tables["drop"][group_cursor] + bank[previous, DROP]
                previous = DROP
                group_cursor += 1
            local = tables["local_index"][step.group_index][int(step.candidate_index)]
            emission = tables["emit_choice"][
                step.group_index, step.rendered_index, local
            ]
            score = score + emission + bank[previous, EMIT]
            previous = EMIT
            group_cursor = step.group_index + 1
            continue
        score = (
            score
            + tables["insert"][group_cursor, step.rendered_index]
            + bank[previous, INSERT_EXTRA]
        )
        previous = INSERT_EXTRA
    while group_cursor < len(lattice.groups):
        score = score + tables["drop"][group_cursor] + bank[previous, DROP]
        previous = DROP
        group_cursor += 1
    bonus = frozen_crf_score or 0.0
    return score + score.new_tensor(float(bonus))


def fast_drop_emit_log_partition(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    gold_only: bool = False,
    max_inserts_ahead: int = 6,
    frozen_crf_score: float | None = None,
) -> Tensor:
    if gold_only:
        return fast_gold_path_log_prob(
            model, lattice, frozen_crf_score=frozen_crf_score
        )
    n_groups = len(lattice.groups)
    n_targets = len(lattice.targets)
    emit_cap = max(n_targets, 1)
    tables = score_tables(model, lattice, emit_cap=emit_cap)
    bank = _transition_bank(model)
    neg = model.initial.new_tensor(NEGATIVE_INFINITY)
    alpha = model.initial.new_full((emit_cap + 1, 4), NEGATIVE_INFINITY)
    alpha[0, 3] = 0.0
    for group_index in range(n_groups + 1):
        insert_row = tables["insert"][group_index]
        for _ in range(max_inserts_ahead):
            pieces = [alpha[:, INSERT_EXTRA]]
            for previous in range(4):
                moved = alpha.new_full((emit_cap + 1,), NEGATIVE_INFINITY)
                moved[1:] = alpha[:-1, previous] + insert_row + bank[previous, INSERT_EXTRA]
                pieces.append(moved)
            updated = alpha.clone()
            updated[:, INSERT_EXTRA] = torch.logsumexp(torch.stack(pieces), dim=0)
            alpha = updated
        if group_index == n_groups:
            break
        updated = alpha.new_full(alpha.shape, NEGATIVE_INFINITY)
        drop_score = tables["drop"][group_index]
        emit_row = tables["emit"][group_index]
        for previous in range(4):
            dropped = alpha[:, previous] + drop_score + bank[previous, DROP]
            updated[:, DROP] = torch.logsumexp(
                torch.stack([updated[:, DROP], dropped]), dim=0
            )
            moved = alpha.new_full((emit_cap + 1,), NEGATIVE_INFINITY)
            moved[1:] = alpha[:-1, previous] + emit_row + bank[previous, EMIT]
            updated[:, EMIT] = torch.logsumexp(
                torch.stack([updated[:, EMIT], moved]), dim=0
            )
        alpha = updated
    if n_targets >= alpha.shape[0]:
        return neg
    partition = torch.logsumexp(alpha[n_targets], dim=0)
    bonus = frozen_crf_score or 0.0
    return partition + partition.new_tensor(float(bonus))


def fast_decode_drop_emit(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    expected_emissions: int | None = None,
    max_inserts_ahead: int = 6,
    max_emissions: int | None = None,
) -> tuple[tuple[DecodedEmit, ...], dict[str, Any]]:
    model.eval()
    n_groups = len(lattice.groups)
    emit_cap = int(
        max_emissions
        if max_emissions is not None
        else max(expected_emissions or 0, n_groups, len(lattice.targets))
    )
    emit_cap = max(emit_cap, 1)
    with torch.no_grad():
        tables = score_tables(model, lattice, emit_cap=emit_cap)
        bank = _transition_bank(model).detach().cpu().numpy()
        drop = tables["drop"].detach().cpu().numpy()
        emit_choice = tables["emit_choice"].detach().cpu().numpy()
        insert = tables["insert"].detach().cpu().numpy()
    return _viterbi_from_tables(
        lattice,
        drop=drop,
        emit_choice=emit_choice,
        insert=insert,
        bank=bank,
        expected_emissions=expected_emissions,
        max_inserts_ahead=max_inserts_ahead,
        emit_cap=emit_cap,
    )
    neg = -1e9
    # value[g is implicit][emitted, prev]
    value = np.full((emit_cap + 1, 4), neg, np.float64)
    value[0, 3] = 0.0
    # backpointer lists kept only for the winning reconstruction via actions.
    # Store predecessor (emitted, prev, kind) per state after each layer is too big.
    # Reconstruct greedily from stored argmax pointers of shape [layers].
    # We store full backpointer tensors: after each group, [E+1, 4, 3]
    # (prev_emitted, prev_slot, action_taken) — memory (G+1)*(E)*(4)*3 ~ 240*50*4 = small.
    pointers: list[np.ndarray] = []
    for group_index in range(n_groups + 1):
        for _ in range(max_inserts_ahead):
            new_value = value.copy()
            pointer = np.full((emit_cap + 1, 4, 3), -1, np.int16)
            pointer[:, :, 0] = np.arange(emit_cap + 1)[:, None]
            pointer[:, :, 1] = np.arange(4)[None, :]
            best_insert = value[:, INSERT_EXTRA].copy()
            best_from = np.full((emit_cap + 1,), -1, np.int16)
            best_prev_e = np.arange(emit_cap + 1, dtype=np.int16)
            for previous in range(4):
                moved = np.full(emit_cap + 1, neg, np.float64)
                moved[1:] = (
                    value[:-1, previous]
                    + insert[group_index]
                    + bank[previous, INSERT_EXTRA]
                )
                better = moved > best_insert
                best_insert = np.where(better, moved, best_insert)
                best_from = np.where(better, previous, best_from)
                best_prev_e = np.where(better, np.arange(emit_cap + 1) - 1, best_prev_e)
            new_value[:, INSERT_EXTRA] = best_insert
            pointer[:, INSERT_EXTRA, 0] = best_prev_e
            pointer[:, INSERT_EXTRA, 1] = best_from
            pointer[:, INSERT_EXTRA, 2] = INSERT_EXTRA
            # Keep non-insert slots pointing at themselves (no action).
            value = new_value
            pointers.append(pointer)
        if group_index == n_groups:
            break
        new_value = np.full_like(value, neg)
        pointer = np.full((emit_cap + 1, 4, 3), -1, np.int16)
        best_drop = np.full(emit_cap + 1, neg)
        best_drop_prev = np.full(emit_cap + 1, -1, np.int16)
        best_emit = np.full(emit_cap + 1, neg)
        best_emit_prev = np.full(emit_cap + 1, -1, np.int16)
        choice = emit_choice[group_index]  # [E, C]
        best_local = np.argmax(choice, axis=1)
        best_emit_score = choice[np.arange(emit_cap), best_local]
        for previous in range(4):
            dropped = value[:, previous] + drop[group_index] + bank[previous, DROP]
            better = dropped > best_drop
            best_drop = np.where(better, dropped, best_drop)
            best_drop_prev = np.where(better, previous, best_drop_prev)
            moved = np.full(emit_cap + 1, neg)
            moved[1:] = value[:-1, previous] + best_emit_score + bank[previous, EMIT]
            better = moved > best_emit
            best_emit = np.where(better, moved, best_emit)
            best_emit_prev = np.where(better, previous, best_emit_prev)
        new_value[:, DROP] = best_drop
        new_value[:, EMIT] = best_emit
        pointer[:, DROP, 0] = np.arange(emit_cap + 1)
        pointer[:, DROP, 1] = best_drop_prev
        pointer[:, DROP, 2] = DROP
        pointer[1:, EMIT, 0] = np.arange(emit_cap)
        pointer[:, EMIT, 1] = best_emit_prev
        pointer[:, EMIT, 2] = EMIT
        pointer[:, :, 0]
        # stash chosen local candidate in an side array
        value = new_value
        pointers.append(pointer)
        pointers.append(best_local.astype(np.int16))  # type: ignore[arg-type]

    # The pointer log above is too tangled for reliable reconstruction.
    # Fall back to a compact forward Viterbi that records decisions directly.
    return _viterbi_from_tables(
        lattice,
        drop=drop,
        emit_choice=emit_choice,
        insert=insert,
        bank=bank,
        expected_emissions=expected_emissions,
        max_inserts_ahead=max_inserts_ahead,
        emit_cap=emit_cap,
    )


def _viterbi_from_tables(
    lattice: DropEmitLattice,
    *,
    drop: np.ndarray,
    emit_choice: np.ndarray,
    insert: np.ndarray,
    bank: np.ndarray,
    expected_emissions: int | None,
    max_inserts_ahead: int,
    emit_cap: int,
) -> tuple[tuple[DecodedEmit, ...], dict[str, Any]]:
    n_groups = len(lattice.groups)
    neg = -1e9
    value = np.full((emit_cap + 1, 4), neg, np.float64)
    value[0, 3] = 0.0
    # decisions[layer][emitted, slot] = (prev_emitted, prev_slot, action, group, local)
    decisions: list[np.ndarray] = []
    for group_index in range(n_groups + 1):
        for _ in range(max_inserts_ahead):
            best_insert = value[:, INSERT_EXTRA].copy()
            prev_e = np.arange(emit_cap + 1)
            prev_s = np.full(emit_cap + 1, INSERT_EXTRA)
            acted = np.zeros(emit_cap + 1, np.bool_)
            for previous in range(4):
                moved = np.full(emit_cap + 1, neg)
                moved[1:] = (
                    value[:-1, previous]
                    + insert[group_index]
                    + bank[previous, INSERT_EXTRA]
                )
                better = moved > best_insert
                best_insert = np.where(better, moved, best_insert)
                prev_e = np.where(better, np.arange(emit_cap + 1) - 1, prev_e)
                prev_s = np.where(better, previous, prev_s)
                acted = np.where(better, True, acted)
            value[:, INSERT_EXTRA] = best_insert
            record = np.stack(
                [
                    prev_e,
                    prev_s,
                    np.where(acted, INSERT_EXTRA, -1),
                    np.full(emit_cap + 1, group_index),
                    np.full(emit_cap + 1, -1),
                ],
                axis=1,
            )
            # Only slot INSERT changes; other slots stay. Record full 4 slots.
            full = np.zeros((emit_cap + 1, 4, 5), np.int16)
            full[:, :, 0] = np.arange(emit_cap + 1)[:, None]
            full[:, :, 1] = np.arange(4)[None, :]
            full[:, :, 2] = -1
            full[:, INSERT_EXTRA, 0] = record[:, 0]
            full[:, INSERT_EXTRA, 1] = record[:, 1]
            full[:, INSERT_EXTRA, 2] = record[:, 2]
            full[:, INSERT_EXTRA, 3] = group_index
            decisions.append(full)
        if group_index == n_groups:
            break
        best_local = np.argmax(emit_choice[group_index], axis=1)
        best_emit_score = emit_choice[group_index, np.arange(emit_cap), best_local]
        new_value = np.full_like(value, neg)
        full = np.full((emit_cap + 1, 4, 5), -1, np.int16)
        best_drop = np.full(emit_cap + 1, neg)
        best_drop_prev = np.full(emit_cap + 1, -1)
        best_emit = np.full(emit_cap + 1, neg)
        best_emit_prev = np.full(emit_cap + 1, -1)
        for previous in range(4):
            dropped = value[:, previous] + drop[group_index] + bank[previous, DROP]
            better = dropped > best_drop
            best_drop = np.where(better, dropped, best_drop)
            best_drop_prev = np.where(better, previous, best_drop_prev)
            moved = np.full(emit_cap + 1, neg)
            moved[1:] = value[:-1, previous] + best_emit_score + bank[previous, EMIT]
            better = moved > best_emit
            best_emit = np.where(better, moved, best_emit)
            best_emit_prev = np.where(better, previous, best_emit_prev)
        new_value[:, DROP] = best_drop
        new_value[:, EMIT] = best_emit
        full[:, DROP, 0] = np.arange(emit_cap + 1)
        full[:, DROP, 1] = best_drop_prev
        full[:, DROP, 2] = DROP
        full[:, DROP, 3] = group_index
        full[1:, EMIT, 0] = np.arange(emit_cap)
        full[:, EMIT, 1] = best_emit_prev
        full[:, EMIT, 2] = EMIT
        full[:, EMIT, 3] = group_index
        full[1:, EMIT, 4] = best_local
        value = new_value
        decisions.append(full)

    if expected_emissions is None:
        end_emitted = int(np.argmax(value.max(axis=1)))
    else:
        end_emitted = int(expected_emissions)
        if end_emitted > emit_cap or not np.isfinite(value[end_emitted]).any():
            return (), {"score": -inf, "fallback": True, "actions": {}}
    end_slot = int(np.argmax(value[end_emitted]))
    score = float(value[end_emitted, end_slot])
    if score <= neg / 2:
        return (), {"score": score, "fallback": True, "actions": {}}

    emitted = end_emitted
    slot = end_slot
    reverse: list[tuple[int, int, int]] = []
    for record in reversed(decisions):
        prev_e, prev_s, action, group_index, local = record[emitted, slot]
        if int(action) < 0:
            continue
        reverse.append((int(action), int(group_index), int(local)))
        emitted = int(prev_e)
        slot = int(prev_s)
        if emitted < 0:
            break
    reverse.reverse()
    emits: list[DecodedEmit] = []
    for action, group_index, local in reverse:
        if action == DROP:
            continue
        if action == EMIT:
            group = lattice.groups[group_index]
            pool_index = group.candidate_indices[local]
            candidate = lattice.pool_candidates[pool_index]
            emits.append(
                DecodedEmit(
                    "emit",
                    group_index,
                    pool_index,
                    None,
                    IdentityCandidate(
                        pitch=int(candidate["pitch"]),
                        start=float(candidate["start"]),
                        end=float(candidate["end"]),
                        confidence=float(candidate["confidence"]),
                        alternatives=tuple(
                            int(value) for value in candidate.get("alternatives") or ()
                        ),
                        alternative_confidences=tuple(
                            float(value)
                            for value in candidate.get("alternative_confidences") or ()
                        ),
                    ),
                )
            )
            continue
        start = float(len(emits)) * 0.01
        emits.append(
            DecodedEmit(
                "insert_extra",
                None,
                None,
                IdentityTarget("extra", None, 0, len(emits), 60),
                IdentityCandidate(60, start, start + 0.05, 0.05),
            )
        )
    counts = {
        name: sum(action == index for action, _group, _local in reverse)
        for index, name in enumerate(ACTION_NAMES)
    }
    return tuple(emits), {
        "score": score,
        "fallback": False,
        "actions": counts,
        "emitted": len(emits),
        "expected_emissions": expected_emissions,
    }
