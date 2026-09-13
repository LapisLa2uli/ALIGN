"""Alternative resilient note-sequence alignment strategies."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pitch(note: Any) -> int:
    return int(note["pitch"] if isinstance(note, dict) else note.pitch)


def _confidence(note: Any) -> float:
    return float(
        note.get("confidence", 1.0)
        if isinstance(note, dict)
        else getattr(note, "confidence", 1.0)
    )


def _pair_cost(observed: Any, score: Any) -> float:
    delta = abs(_pitch(observed) - _pitch(score))
    return 0.0 if delta == 0 else 1.05 + min(delta, 12) / 60.0


def _subsequence_mapping(observed: list[Any], score: list[Any]) -> list[int | None]:
    """Align one audio window to its best free-start/free-end score location."""

    n, m = len(observed), len(score)
    if not n:
        return []
    if not m:
        return [None] * n
    gap = 0.9
    dp = np.full((n + 1, m + 1), np.inf, np.float64)
    back = np.zeros((n + 1, m + 1), np.int8)
    dp[0, :] = 0.0  # each window may start anywhere in the score
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
        back[i, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                (dp[i - 1, j - 1] + _pair_cost(observed[i - 1], score[j - 1]), 0),
                (dp[i - 1, j] + gap + 0.1 * _confidence(observed[i - 1]), 1),
                (dp[i, j - 1] + gap, 2),
            )
            dp[i, j], back[i, j] = min(choices, key=lambda item: item[0])
    i = n
    j = int(np.argmin(dp[n]))
    mapping: list[int | None] = [None] * n
    while i:
        code = int(back[i, j])
        if j and code == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif not j or code == 1:
            i -= 1
        else:
            j -= 1
    return mapping


def _global_mapping(
    observed: list[Any],
    score: list[Any],
    *,
    pair_bonus: np.ndarray | None = None,
    score_i0: int = 0,
) -> list[int | None]:
    n, m = len(observed), len(score)
    if not n:
        return []
    if not m:
        return [None] * n
    gap = 0.9
    dp = np.zeros((n + 1, m + 1), np.float64)
    back = np.zeros((n + 1, m + 1), np.int8)
    dp[:, 0] = np.arange(n + 1) * gap
    dp[0, :] = np.arange(m + 1) * gap
    back[1:, 0] = 1
    back[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            bonus = (
                float(pair_bonus[i - 1, j - 1])
                if pair_bonus is not None
                else 0.0
            )
            choices = (
                (
                    dp[i - 1, j - 1]
                    + _pair_cost(observed[i - 1], score[j - 1])
                    - bonus,
                    0,
                ),
                (dp[i - 1, j] + gap + 0.1 * _confidence(observed[i - 1]), 1),
                (dp[i, j - 1] + gap, 2),
            )
            dp[i, j], back[i, j] = min(choices, key=lambda item: item[0])
    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i or j:
        code = int(back[i, j])
        if i and j and code == 0:
            mapping[i - 1] = score_i0 + j - 1
            i -= 1
            j -= 1
        elif i and (not j or code == 1):
            i -= 1
        else:
            j -= 1
    return mapping


def multi_start_mapping(
    observed: list[Any],
    score: list[Any],
    *,
    window_notes: int = 14,
    stride_notes: int = 6,
) -> list[int | None]:
    """Align overlapping audio windows independently, then combine their votes."""

    n, m = len(observed), len(score)
    if n <= window_notes:
        return _global_mapping(observed, score)
    starts = list(range(0, max(1, n - window_notes + 1), stride_notes))
    final_start = max(0, n - window_notes)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    votes = np.zeros((n, m), np.float32)
    for start in starts:
        end = min(n, start + window_notes)
        local = _subsequence_mapping(observed[start:end], score)
        for offset, score_index in enumerate(local):
            if score_index is None:
                continue
            exact_weight = (
                1.5
                if _pitch(observed[start + offset]) == _pitch(score[score_index])
                else 0.5
            )
            votes[start + offset, score_index] += exact_weight
    maximum = np.maximum(votes.max(axis=1, keepdims=True), 1.0)
    return _global_mapping(observed, score, pair_bonus=0.45 * votes / maximum)


def _mapping_error(
    observed: list[Any],
    score: list[Any],
    mapping: list[int | None],
) -> float:
    mapped = {value for value in mapping if value is not None}
    error = 0.0
    for note, target in zip(observed, mapping):
        if target is None:
            error += 0.9
        elif not (0 <= target < len(score)):
            error += 2.0
        else:
            error += _pair_cost(note, score[target])
    error += 0.9 * max(0, len(score) - len(mapped))
    return error


def dynamic_revision_mapping(
    observed: list[Any],
    score: list[Any],
    initial: list[int | None],
    *,
    error_window: int = 6,
    lookback_notes: int = 8,
    lookahead_notes: int = 12,
    max_passes: int = 3,
) -> list[int | None]:
    """Reopen past alignment when a recent window contains too many errors."""

    mapping = list(initial)
    if len(mapping) != len(observed):
        mapping = _global_mapping(observed, score)
    for _ in range(max_passes):
        changed = False
        for end in range(error_window, len(observed) + 1):
            begin = end - error_window
            errors = sum(
                mapping[index] is None
                or not (0 <= int(mapping[index]) < len(score))
                or _pitch(observed[index]) != _pitch(score[int(mapping[index])])
                for index in range(begin, end)
            )
            if errors < max(4, int(np.ceil(0.75 * error_window))):
                continue
            obs_i0 = max(0, begin - lookback_notes)
            obs_i1 = min(len(observed), end + lookahead_notes)
            before = next(
                (
                    mapping[index]
                    for index in range(obs_i0 - 1, -1, -1)
                    if mapping[index] is not None
                ),
                None,
            )
            after = next(
                (
                    mapping[index]
                    for index in range(obs_i1, len(mapping))
                    if mapping[index] is not None
                ),
                None,
            )
            score_i0 = int(before) + 1 if before is not None else 0
            score_i1 = int(after) if after is not None else len(score)
            if score_i1 <= score_i0:
                score_i0, score_i1 = 0, len(score)
            revised = _global_mapping(
                observed[obs_i0:obs_i1],
                score[score_i0:score_i1],
                score_i0=score_i0,
            )
            old_segment = mapping[obs_i0:obs_i1]
            if _mapping_error(
                observed[obs_i0:obs_i1], score, revised
            ) + 0.25 < _mapping_error(
                observed[obs_i0:obs_i1], score, old_segment
            ):
                mapping[obs_i0:obs_i1] = revised
                changed = True
                break
        if not changed:
            break
    return mapping
