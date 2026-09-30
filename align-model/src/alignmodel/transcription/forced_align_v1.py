"""Forced Viterbi alignment of a known written-pitch sequence to mel posteriors.

Dataset 9.2 audio was rendered by Muse Sounds from the performance score, so
``note_map`` timestamps do not match the audio. The pitch order does, so each
gold note is placed on frames by a left-to-right Viterbi pass over note-entry,
note-body, and inter-note silence states. Notes the audio does not support can
be skipped at a fixed penalty per skipped note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


NEG = -1e18


@dataclass(frozen=True)
class ForcedAlignConfig:
    skip_penalty: float = 12.0
    max_skip: int = 4
    onset_weight: float = 0.6
    onset_floor: float = 0.03
    out_of_range_pitch_probability: float = 0.02
    epsilon: float = 1e-6
    use_boundary_heads: bool = False


@dataclass(frozen=True)
class AlignedNote:
    index: int
    pitch: int
    start_frame: int
    end_frame: int


def forced_align(
    probabilities: Mapping[str, np.ndarray],
    pitches: Sequence[int],
    *,
    midi_min: int,
    config: ForcedAlignConfig = ForcedAlignConfig(),
) -> tuple[list[AlignedNote], dict[str, float]]:
    voiced = np.asarray(probabilities["voiced"], np.float64)
    pitch_probability = np.asarray(probabilities["pitch"], np.float64)
    onset = np.asarray(probabilities["onset"], np.float64)
    if config.use_boundary_heads:
        onset = np.maximum.reduce((
            onset,
            np.asarray(probabilities["boundary"], np.float64),
            np.asarray(probabilities["rearticulation"], np.float64),
        ))
    frames = int(voiced.shape[0])
    count = len(pitches)
    if count == 0 or frames == 0:
        return [], {"frames": frames, "notes": count, "aligned": 0, "skipped": count}
    eps = config.epsilon
    columns = np.array([int(p) - midi_min for p in pitches])
    in_range = (columns >= 0) & (columns < pitch_probability.shape[1])
    note_prob = np.full((frames, count), config.out_of_range_pitch_probability)
    note_prob[:, in_range] = pitch_probability[:, columns[in_range]]
    emit_note = np.log(voiced[:, None] * note_prob + eps)
    emit_entry = config.onset_weight * np.log(
        np.maximum(onset, config.onset_floor)
    )
    emit_silence = np.log(1.0 - voiced + eps)

    k = count
    entry_offset, body_offset, silence_offset = 0, k, 2 * k
    states = 3 * k + 1
    back = np.empty((frames, states), np.int32)
    penalty = config.skip_penalty
    skips = range(1, config.max_skip + 1)

    entry = np.full(k, NEG)
    body = np.full(k, NEG)
    silence = np.full(k + 1, NEG)
    silence[0] = emit_silence[0]
    back[0, silence_offset] = -1
    entry[0] = emit_note[0, 0] + emit_entry[0]
    back[0, entry_offset] = -1
    for j in skips:
        if j < k:
            entry[j] = emit_note[0, j] + emit_entry[0] - j * penalty
            back[0, entry_offset + j] = -1
        if j <= k:
            silence[j] = emit_silence[0] - j * penalty
            back[0, silence_offset + j] = -1

    index_k = np.arange(k)
    index_s = np.arange(k + 1)
    for t in range(1, frames):
        # Entry of note n: from body n-1-j or silence n-j, j skipped notes.
        best_entry = silence[:k].copy()
        arg_entry = silence_offset + index_k
        from_body = np.full(k, NEG)
        from_body[1:] = body[:-1]
        better = from_body > best_entry
        best_entry = np.where(better, from_body, best_entry)
        arg_entry = np.where(better, body_offset + index_k - 1, arg_entry)
        for j in skips:
            candidate = np.full(k, NEG)
            if j + 1 < k + 1:
                candidate[j + 1:] = body[:k - j - 1] - j * penalty
            better = candidate > best_entry
            best_entry = np.where(better, candidate, best_entry)
            arg_entry = np.where(better, body_offset + index_k - j - 1, arg_entry)
            candidate = np.full(k, NEG)
            if j < k:
                candidate[j:] = silence[:k - j] - j * penalty
            better = candidate > best_entry
            best_entry = np.where(better, candidate, best_entry)
            arg_entry = np.where(better, silence_offset + index_k - j, arg_entry)

        stay_better = body >= entry
        best_body = np.where(stay_better, body, entry)
        arg_body = np.where(stay_better, body_offset + index_k, entry_offset + index_k)

        best_silence = silence.copy()
        arg_silence = silence_offset + index_s
        from_body = np.full(k + 1, NEG)
        from_body[1:] = body
        better = from_body > best_silence
        best_silence = np.where(better, from_body, best_silence)
        arg_silence = np.where(better, body_offset + index_s - 1, arg_silence)
        for j in skips:
            candidate = np.full(k + 1, NEG)
            if j + 1 <= k:
                candidate[j + 1:] = body[:k - j] - j * penalty
            better = candidate > best_silence
            best_silence = np.where(better, candidate, best_silence)
            arg_silence = np.where(better, body_offset + index_s - j - 1, arg_silence)

        entry = best_entry + emit_note[t] + emit_entry[t]
        body = best_body + emit_note[t]
        silence = best_silence + emit_silence[t]
        back[t, entry_offset:body_offset] = arg_entry
        back[t, body_offset:silence_offset] = arg_body
        back[t, silence_offset:] = arg_silence

    finals = [(silence[k], silence_offset + k), (body[k - 1], body_offset + k - 1)]
    for j in skips:
        if j <= k:
            finals.append((silence[k - j] - j * penalty, silence_offset + k - j))
        if j < k:
            finals.append((body[k - 1 - j] - j * penalty, body_offset + k - 1 - j))
    score, state = max(finals)

    path = np.empty(frames, np.int32)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state = int(back[t, state]) if t > 0 else -1

    aligned: list[AlignedNote] = []
    current = None
    start = 0
    for t, value in enumerate(path.tolist()):
        if value < body_offset:
            note = value - entry_offset
            if current is not None:
                aligned.append(AlignedNote(current, int(pitches[current]), start, t))
            current, start = note, t
        elif value < silence_offset:
            continue
        else:
            if current is not None:
                aligned.append(AlignedNote(current, int(pitches[current]), start, t))
                current = None
    if current is not None:
        aligned.append(AlignedNote(current, int(pitches[current]), start, frames))

    note_frames = sum(note.end_frame - note.start_frame for note in aligned)
    mean_note_log = (
        float(np.mean([
            emit_note[note.start_frame:note.end_frame, note.index].mean()
            for note in aligned
        ]))
        if aligned else float("nan")
    )
    stats = {
        "frames": frames,
        "notes": count,
        "aligned": len(aligned),
        "skipped": count - len(aligned),
        "voiced_frames": note_frames,
        "score": float(score),
        "mean_note_log_emission": mean_note_log,
    }
    return aligned, stats
