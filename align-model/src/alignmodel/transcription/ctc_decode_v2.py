"""CTC pitch-sequence decoders for the mel CTC transcriber.

``greedy`` is the frozen round-2 decoder (blank probability scaled before the
per-frame argmax). ``structured`` adds three corrections that target the
observed errors:

* same-pitch re-emissions separated only by weak blank evidence are merged
  unless an onset/rearticulation head peak supports a new attack (splits);
* long same-pitch runs are split where a strong attack peak coincides with a
  dip in that pitch's CTC probability (merged repeats);
* short A-B-A flickers with low confidence and no attack evidence are removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numba
import numpy as np


BLANK = 0


@dataclass(frozen=True)
class StructuredDecodeConfig:
    blank_scale: float = 0.3
    split_attack_threshold: float = 0.35
    split_confirm_blank_scale: float = 1.0
    repeat_attack_threshold: float = 0.55
    repeat_dip_ratio: float = 0.75
    repeat_min_frames: int = 3
    flicker_max_frames: int = 3
    flicker_max_probability: float = 0.6
    flicker_attack_threshold: float = 0.35
    attack_window: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


def greedy_tokens(ctc: np.ndarray, blank_scale: float) -> np.ndarray:
    values = np.asarray(ctc, np.float32)
    if blank_scale != 1.0:
        values = values.copy()
        values[:, BLANK] *= blank_scale
    return values.argmax(axis=1)


def runs_from_tokens(tokens: np.ndarray) -> list[tuple[int, int, int]]:
    """Collapsed non-blank runs as (token, first_frame, last_frame_exclusive)."""

    runs: list[tuple[int, int, int]] = []
    previous = BLANK
    start = 0
    for frame, token in enumerate(tokens.tolist()):
        if token != previous:
            if previous != BLANK:
                runs.append((previous, start, frame))
            start = frame
        previous = token
    if previous != BLANK:
        runs.append((previous, start, len(tokens)))
    return runs


def greedy_decode_notes(ctc: np.ndarray, midi_min: int, blank_scale: float) -> list[tuple[int, int]]:
    return [(midi_min + token - 1, first) for token, first, _last in runs_from_tokens(greedy_tokens(ctc, blank_scale))]


def rich_decode(
    ctc: np.ndarray,
    midi_min: int,
    blank_scale: float = 0.3,
    candidate_threshold: float = 0.08,
    candidate_guard: int = 3,
) -> list[dict]:
    """Greedy notes with confidence and runner-up pitch, plus optional weak peaks.

    Optional candidates are local maxima of the best non-blank probability in
    frames the greedy path leaves blank, at least ``candidate_threshold`` and
    not within ``candidate_guard`` frames of an emitted run of the same pitch.
    """

    values = np.asarray(ctc, np.float32)
    runs = runs_from_tokens(greedy_tokens(values, blank_scale))
    notes: list[dict] = []
    covered = np.zeros(len(values), np.bool_)
    for token, first, last in runs:
        segment = values[first:last, token]
        peak = first + int(segment.argmax())
        row = values[peak].copy()
        row[BLANK] = -1.0
        row[token] = -1.0
        alternative = int(row.argmax())
        notes.append({
            "pitch": midi_min + token - 1, "frame": first, "confidence": float(segment.max()),
            "optional": False, "alternative_pitch": midi_min + alternative - 1,
            "alternative_confidence": float(row[alternative]),
        })
        covered[max(0, first - candidate_guard):min(len(values), last + candidate_guard)] = True
    nonblank = values[:, 1:]
    best_token = nonblank.argmax(axis=1) + 1
    best_probability = nonblank.max(axis=1)
    run_tokens = {(token, frame) for token, start, end in runs for frame in range(start - candidate_guard, end + candidate_guard)}
    for frame in range(1, len(values) - 1):
        probability = best_probability[frame]
        if probability < candidate_threshold or covered[frame]:
            continue
        if probability < best_probability[frame - 1] or probability < best_probability[frame + 1]:
            continue
        token = int(best_token[frame])
        if (token, frame) in run_tokens:
            continue
        notes.append({
            "pitch": midi_min + token - 1, "frame": frame, "confidence": float(probability),
            "optional": True, "alternative_pitch": -1, "alternative_confidence": 0.0,
        })
    notes.sort(key=lambda note: (note["frame"], note["optional"]))
    return notes


def _attack(heads: Mapping[str, np.ndarray], frame: int, window: int) -> float:
    lo = max(0, frame - window)
    hi = min(len(heads["onset"]), frame + window + 1)
    if hi <= lo:
        return 0.0
    return float(max(
        heads["onset"][lo:hi].max(),
        heads["rearticulation"][lo:hi].max(),
        heads["boundary"][lo:hi].max(),
    ))


def structured_decode_notes(
    outputs: Mapping[str, np.ndarray],
    midi_min: int,
    config: StructuredDecodeConfig = StructuredDecodeConfig(),
) -> list[tuple[int, int]]:
    ctc = np.asarray(outputs["ctc"], np.float32)
    heads = {name: np.asarray(outputs[name], np.float32) for name in ("onset", "rearticulation", "boundary")}
    runs = runs_from_tokens(greedy_tokens(ctc, config.blank_scale))
    if not runs:
        return []
    confirm_tokens = greedy_tokens(ctc, config.split_confirm_blank_scale)

    # 1. Flicker removal: A B A with a short, weak, unsupported B.
    kept: list[tuple[int, int, int]] = []
    for index, run in enumerate(runs):
        token, first, last = run
        if 0 < index < len(runs) - 1:
            previous, following = runs[index - 1], runs[index + 1]
            if (
                previous[0] == following[0] != token
                and last - first <= config.flicker_max_frames
                and float(ctc[first:last, token].max()) < config.flicker_max_probability
                and _attack(heads, first, config.attack_window) < config.flicker_attack_threshold
            ):
                continue
        kept.append(run)

    # 2. Same-pitch re-emissions: merge unless an attack or a confident blank supports a split.
    merged: list[tuple[int, int, int]] = []
    for run in kept:
        token, first, last = run
        if merged and merged[-1][0] == token:
            previous_token, previous_first, previous_last = merged[-1]
            gap = confirm_tokens[previous_last:first]
            confident_blank = gap.size > 0 and bool((gap == BLANK).any())
            if not confident_blank and _attack(heads, first, config.attack_window) < config.split_attack_threshold:
                merged[-1] = (token, previous_first, last)
                continue
        merged.append(run)

    # 3. Recover merged repeats inside a same-pitch region (until the next note).
    notes: list[tuple[int, int]] = []
    for index, (token, first, last) in enumerate(merged):
        notes.append((midi_min + token - 1, first))
        region_end = merged[index + 1][1] if index + 1 < len(merged) else len(ctc)
        cursor = first + config.repeat_min_frames
        attack = np.maximum(heads["onset"], heads["rearticulation"])
        reference = float(ctc[first:max(first + 1, last), token].max())
        while cursor < region_end - 1:
            if (
                attack[cursor] >= config.repeat_attack_threshold
                and attack[cursor] >= attack[max(0, cursor - 1)]
                and attack[cursor] >= attack[min(len(attack) - 1, cursor + 1)]
            ):
                lo = max(first, cursor - 4)
                dip = float(ctc[lo:cursor + 1, token].min())
                after = float(ctc[cursor:min(region_end, cursor + 4), token].max())
                if dip < config.repeat_dip_ratio * max(reference, 1e-6) and after >= 0.5 * reference:
                    notes.append((midi_min + token - 1, cursor))
                    cursor += config.repeat_min_frames
                    continue
            cursor += 1
    return notes


@numba.njit(cache=True)
def lcs_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    n, m = left.shape[0], right.shape[0]
    table = np.zeros((n + 1, m + 1), np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if left[i - 1] == right[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            elif table[i - 1, j] >= table[i, j - 1]:
                table[i, j] = table[i - 1, j]
            else:
                table[i, j] = table[i, j - 1]
    pairs = np.zeros((table[n, m], 2), np.int64)
    k = table[n, m] - 1
    i, j = n, m
    while i > 0 and j > 0:
        if left[i - 1] == right[j - 1]:
            pairs[k, 0] = i - 1
            pairs[k, 1] = j - 1
            k -= 1
            i -= 1
            j -= 1
        elif table[i - 1, j] >= table[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return pairs
