from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np

from alignmodel.audio import chroma_slice, dtw_normalized_cost
from alignmodel.types import (
    NoteRepetition,
    PipelineLabel,
    PipelineState,
    RepeatRange,
    UnfoldedSegment,
    next_label_id,
)


@dataclass(frozen=True)
class RepetitionCandidate:
    """A performance span retrieved from an earlier performance span."""

    start_time: float
    end_time: float
    source_start: float
    source_end: float
    confidence: float
    extra_copies: int = 1


def silence_intervals(
    audio: np.ndarray,
    sr: int,
    *,
    silence_db: float = -40.0,
    min_silence_sec: float = 0.12,
    hop_length: int = 512,
) -> list[tuple[float, float]]:
    """Return sustained silent runs as exact start/end restart anchors."""
    if audio.size == 0 or sr <= 0:
        return []
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    quiet = db < silence_db
    hop_sec = hop_length / float(sr)
    min_frames = max(2, int(round(min_silence_sec / hop_sec)))
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i + 1
        while j < len(quiet) and quiet[j]:
            j += 1
        if j - i >= min_frames:
            out.append((float(i * hop_sec), float(j * hop_sec)))
        i = j
    return out


def _resample_columns(feature: np.ndarray, n_frames: int = 64) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float64)
    if feature.ndim != 2 or feature.shape[1] == 0:
        return np.zeros((12, n_frames), dtype=np.float64)
    old = np.linspace(0.0, 1.0, feature.shape[1])
    new = np.linspace(0.0, 1.0, n_frames)
    out = np.vstack([np.interp(new, old, row) for row in feature])
    return out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-8)


def sequence_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tempo-tolerant framewise chroma similarity in [0, 1]."""
    ra = _resample_columns(a)
    rb = _resample_columns(b)
    return float(np.clip(np.mean(np.sum(ra * rb, axis=0)), 0.0, 1.0))


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ha = np.maximum(np.asarray(a, dtype=np.float64).mean(axis=1), 0.0)
    hb = np.maximum(np.asarray(b, dtype=np.float64).mean(axis=1), 0.0)
    denom = float(np.linalg.norm(ha) * np.linalg.norm(hb))
    return 0.0 if denom < 1e-8 else float(np.clip(np.dot(ha, hb) / denom, 0.0, 1.0))


def _retrieval_score(a: np.ndarray, b: np.ndarray, *, refine: bool) -> float:
    ra = _resample_columns(a)
    rb = _resample_columns(b)
    aligned = float(np.clip(np.mean(np.sum(ra * rb, axis=0)), 0.0, 1.0))
    hist = _histogram_similarity(ra, rb)
    if not refine:
        return 0.75 * aligned + 0.25 * hist
    # Bound DTW to 64x64 regardless of the original span duration. Long
    # (10-20 second) source bars otherwise dominate Stage 1 runtime.
    cost, _ = dtw_normalized_cost(ra, rb)
    dtw_score = float(np.exp(-5.0 * max(cost, 0.0))) if np.isfinite(cost) else 0.0
    return 0.55 * aligned + 0.20 * hist + 0.25 * dtw_score


def _slice(
    chroma: np.ndarray, start: float, end: float, hop_sec: float
) -> np.ndarray:
    return chroma_slice(chroma, max(0.0, start), max(start + 0.05, end), hop_sec)


def _best_source_start(
    chroma: np.ndarray,
    *,
    query_start: float,
    source_end: float,
    duration: float,
    hop_sec: float,
    max_lookback_sec: float,
    search_step_sec: float,
    probe_sec: float,
) -> tuple[float, float] | None:
    query_end = min(duration, query_start + probe_sec)
    if query_end - query_start < 0.8:
        return None
    query = _slice(chroma, query_start, query_end, hop_sec)
    lo = max(0.0, query_start - max_lookback_sec)
    hi = source_end - (query_end - query_start)
    if hi < lo:
        return None
    starts = np.arange(lo, hi + 0.5 * search_step_sec, search_step_sec)
    coarse: list[tuple[float, float]] = []
    width = query_end - query_start
    for source_start in starts:
        source = _slice(chroma, float(source_start), float(source_start + width), hop_sec)
        coarse.append((_retrieval_score(source, query, refine=False), float(source_start)))
    coarse.sort(reverse=True)
    best: tuple[float, float] | None = None
    for _coarse_score, source_start in coarse[:4]:
        source = _slice(chroma, source_start, source_start + width, hop_sec)
        score = _retrieval_score(source, query, refine=True)
        if best is None or score > best[1]:
            best = (source_start, score)
    return best


def find_past_repetitions(
    chroma: np.ndarray,
    hop_sec: float,
    duration: float,
    silences: list[tuple[float, float]],
    *,
    max_lookback_sec: float = 20.0,
    search_step_sec: float = 0.25,
    probe_sec: float = 3.0,
    min_confidence: float = 0.86,
    max_candidates: int = 2,
) -> list[RepetitionCandidate]:
    """Retrieve post-silence extra music from up to 20 seconds in the past."""
    proposals: list[RepetitionCandidate] = []
    for silence_start, silence_end in silences:
        query_start = float(silence_end)
        source_end = float(silence_start)
        if query_start >= duration - 0.8 or source_end < 0.8:
            continue
        best = _best_source_start(
            chroma,
            query_start=query_start,
            source_end=source_end,
            duration=duration,
            hop_sec=hop_sec,
            max_lookback_sec=max_lookback_sec,
            search_step_sec=search_step_sec,
            probe_sec=probe_sec,
        )
        if best is None:
            continue
        source_start, seed_score = best
        source_duration = source_end - source_start
        if source_duration < 0.8:
            continue
        first_end = min(duration, query_start + source_duration)
        source = _slice(chroma, source_start, source_end, hop_sec)
        first = _slice(chroma, query_start, first_end, hop_sec)
        full_score = _retrieval_score(source, first, refine=True)
        confidence = 0.65 * seed_score + 0.35 * full_score
        if confidence < min_confidence:
            continue

        copies = 1
        repetition_end = first_end
        second_start = query_start + source_duration
        second_end = min(duration, second_start + source_duration)
        if second_end - second_start >= 0.75 * source_duration:
            second = _slice(chroma, second_start, second_end, hop_sec)
            second_score = _retrieval_score(source, second, refine=True)
            if second_score >= min_confidence - 0.04:
                copies = 2
                repetition_end = second_end
                confidence = max(confidence, 0.5 * (confidence + second_score))
        proposals.append(
            RepetitionCandidate(
                start_time=query_start,
                end_time=repetition_end,
                source_start=source_start,
                source_end=source_end,
                confidence=float(confidence),
                extra_copies=copies,
            )
        )

    proposals.sort(key=lambda item: item.confidence, reverse=True)
    kept: list[RepetitionCandidate] = []
    for item in proposals:
        if any(
            item.start_time < other.end_time and other.start_time < item.end_time
            for other in kept
        ):
            continue
        kept.append(item)
        if len(kept) >= max_candidates:
            break
    return sorted(kept, key=lambda item: item.start_time)


def _note_value(note: Any, name: str, default: float = 0.0):
    if isinstance(note, dict):
        return note.get(name, default)
    return getattr(note, name, default)


def _note_sequence_score(
    notes: list[Any], source_i0: int, repeat_i0: int, length: int
) -> tuple[float, float]:
    source = notes[source_i0 : source_i0 + length]
    repeated = notes[repeat_i0 : repeat_i0 + length]
    exact = sum(
        int(_note_value(left, "pitch")) == int(_note_value(right, "pitch"))
        for left, right in zip(source, repeated)
    )
    exact_fraction = exact / max(length, 1)
    if length <= 1:
        return exact_fraction, 0.0
    source_ioi = np.diff(
        [float(_note_value(note, "start")) for note in source]
    )
    repeat_ioi = np.diff(
        [float(_note_value(note, "start")) for note in repeated]
    )
    valid = (source_ioi > 1e-4) & (repeat_ioi > 1e-4)
    if not np.any(valid):
        timing_score = 0.0
    else:
        ratios = repeat_ioi[valid] / source_ioi[valid]
        center = float(np.median(ratios))
        residual = float(
            np.median(np.abs(np.log(ratios / max(center, 1e-6))))
        )
        timing_score = float(np.exp(-4.0 * residual))
    return exact_fraction, timing_score


def consolidate_note_repetitions(
    repetitions: list[NoteRepetition],
    *,
    max_gap_notes: int = 1,
    diagonal_tolerance: int = 2,
) -> list[NoteRepetition]:
    """Join overlapping/adjacent fragments from one replay event."""

    merged: list[NoteRepetition] = []
    for item in sorted(
        repetitions,
        key=lambda value: (
            value.repeat_i0,
            value.source_i0,
            -value.repeat_i1,
        ),
    ):
        target = None
        for index, other in enumerate(merged):
            repeat_gap = max(
                0,
                item.repeat_i0 - other.repeat_i1,
                other.repeat_i0 - item.repeat_i1,
            )
            source_gap = max(
                0,
                item.source_i0 - other.source_i1,
                other.source_i0 - item.source_i1,
            )
            diagonal_delta = abs(
                (item.repeat_i0 - item.source_i0)
                - (other.repeat_i0 - other.source_i0)
            )
            source_end = max(item.source_i1, other.source_i1)
            repeat_start = min(item.repeat_i0, other.repeat_i0)
            if (
                repeat_gap <= max_gap_notes
                and source_gap <= max_gap_notes
                and diagonal_delta <= diagonal_tolerance
                and source_end <= repeat_start
            ):
                target = index
                break
        if target is None:
            merged.append(item)
            continue
        other = merged[target]
        merged[target] = NoteRepetition(
            source_i0=min(other.source_i0, item.source_i0),
            source_i1=max(other.source_i1, item.source_i1),
            repeat_i0=min(other.repeat_i0, item.repeat_i0),
            repeat_i1=max(other.repeat_i1, item.repeat_i1),
            source_start=min(other.source_start, item.source_start),
            source_end=max(other.source_end, item.source_end),
            repeat_start=min(other.repeat_start, item.repeat_start),
            repeat_end=max(other.repeat_end, item.repeat_end),
            confidence=max(other.confidence, item.confidence),
        )
    return sorted(merged, key=lambda value: value.repeat_i0)


def find_note_sequence_repetitions(
    notes: list[Any],
    *,
    min_notes: int = 6,
    max_notes: int = 64,
    min_confidence: float = 0.80,
) -> list[NoteRepetition]:
    """Find later repeated note phrases without consulting the written score."""

    ordered = sorted(notes, key=lambda note: float(_note_value(note, "start")))
    count = len(ordered)
    if count < 2 * min_notes:
        return []
    seeds: dict[tuple[int, ...], list[int]] = {}
    seed_width = min(4, min_notes)
    pitches = [int(_note_value(note, "pitch")) for note in ordered]
    for index in range(0, count - seed_width + 1):
        key = tuple(
            pitches[pos + 1] - pitches[pos]
            for pos in range(index, index + seed_width - 1)
        )
        seeds.setdefault(key, []).append(index)

    proposals: list[NoteRepetition] = []
    for positions in seeds.values():
        for later_position, repeat_i0 in enumerate(positions):
            for source_i0 in positions[:later_position]:
                if repeat_i0 - source_i0 < min_notes:
                    continue
                maximum = min(
                    max_notes,
                    repeat_i0 - source_i0,
                    count - repeat_i0,
                )
                mismatches = 0
                best_length = 0
                best_confidence = 0.0
                for length in range(1, maximum + 1):
                    if pitches[source_i0 + length - 1] != pitches[
                        repeat_i0 + length - 1
                    ]:
                        mismatches += 1
                    allowed = max(1, int(round(0.12 * length)))
                    if mismatches > allowed:
                        break
                    if length < min_notes:
                        continue
                    exact_fraction, timing_score = _note_sequence_score(
                        ordered, source_i0, repeat_i0, length
                    )
                    confidence = 0.82 * exact_fraction + 0.18 * timing_score
                    if confidence >= min_confidence:
                        best_length = length
                        best_confidence = confidence
                if best_length < min_notes:
                    continue
                source_end_note = ordered[source_i0 + best_length - 1]
                repeat_end_note = ordered[repeat_i0 + best_length - 1]
                proposals.append(
                    NoteRepetition(
                        source_i0=source_i0,
                        source_i1=source_i0 + best_length,
                        repeat_i0=repeat_i0,
                        repeat_i1=repeat_i0 + best_length,
                        source_start=float(
                            _note_value(ordered[source_i0], "start")
                        ),
                        source_end=float(_note_value(source_end_note, "end")),
                        repeat_start=float(
                            _note_value(ordered[repeat_i0], "start")
                        ),
                        repeat_end=float(_note_value(repeat_end_note, "end")),
                        confidence=best_confidence,
                    )
                )

    proposals = consolidate_note_repetitions(proposals)
    proposals.sort(
        key=lambda value: (
            -(value.repeat_i1 - value.repeat_i0),
            -value.confidence,
            value.repeat_i0,
        )
    )
    kept: list[NoteRepetition] = []
    for item in proposals:
        if any(
            item.repeat_i0 < other.repeat_i1
            and other.repeat_i0 < item.repeat_i1
            for other in kept
        ):
            continue
        kept.append(item)
    return sorted(kept, key=lambda value: value.repeat_i0)


def _model_note_repetitions(
    state: PipelineState, model
) -> list[NoteRepetition]:
    import torch

    from alignmodel.stages.dc_alignment import _monotonic_pitch_mapping
    from alignmodel.stages.note_repetition_model import (
        propose_note_repeat_candidates,
    )

    provisional = _monotonic_pitch_mapping(
        state.transcribed_notes, state.score.notes
    )
    extra_mask = [value is None for value in provisional]
    candidates = propose_note_repeat_candidates(
        state.transcribed_notes,
        extra_mask=extra_mask,
        max_notes=int(model.config.max_notes),
        max_sources_per_note=int(model.config.max_sources_per_note),
        continuation_lookahead_notes=int(
            model.config.continuation_lookahead_notes
        ),
        continuation_candidate_skips=int(
            model.config.continuation_candidate_skips
        ),
        continuation_source_skips=int(
            model.config.continuation_source_skips
        ),
    )
    # The learned scorer is used for the ambiguous one-note case. Longer
    # phrases remain substantially more reliable under deterministic sequence
    # matching, and replacing them with candidate-level scores fragments spans.
    candidates = [
        candidate
        for candidate in candidates
        if candidate.repeat_i1 - candidate.repeat_i0 == 1
    ]
    if not candidates:
        return []
    device = next(model.parameters()).device
    features = torch.from_numpy(
        np.stack([candidate.features for candidate in candidates])
    ).to(device)
    with torch.no_grad():
        probabilities = model(features).sigmoid().cpu().numpy()
    ranked = sorted(
        zip(candidates, probabilities),
        key=lambda item: (
            -(
                float(item[1])
                + 0.15
                * np.log1p(item[0].repeat_i1 - item[0].repeat_i0)
            ),
            -(item[0].repeat_i1 - item[0].repeat_i0),
        ),
    )
    kept: list[NoteRepetition] = []
    threshold = float(model.config.threshold)
    for candidate, probability in ranked:
        if float(probability) < threshold:
            continue
        length = candidate.repeat_i1 - candidate.repeat_i0
        if length == 1 and (
            float(probability) < max(0.90, threshold)
            or float(candidate.features[9]) < 0.5
        ):
            continue
        if any(
            candidate.repeat_i0 < other.repeat_i1
            and other.repeat_i0 < candidate.repeat_i1
            for other in kept
        ):
            continue
        source_first = state.transcribed_notes[candidate.source_i0]
        source_last = state.transcribed_notes[candidate.source_i1 - 1]
        repeat_first = state.transcribed_notes[candidate.repeat_i0]
        repeat_last = state.transcribed_notes[candidate.repeat_i1 - 1]
        kept.append(
            NoteRepetition(
                candidate.source_i0,
                candidate.source_i1,
                candidate.repeat_i0,
                candidate.repeat_i1,
                source_first.start,
                source_last.end,
                repeat_first.start,
                repeat_last.end,
                float(probability),
            )
        )
        if len(kept) >= int(model.config.max_repetitions):
            break
    return sorted(kept, key=lambda value: value.repeat_i0)


def filter_repetitions_by_score_continuation(
    notes: list[Any],
    score_notes: list[Any],
    repetitions: list[NoteRepetition],
    *,
    aligner=None,
    continuation_notes: int = 3,
    minimum_confirmed_notes: int = 2,
) -> list[NoteRepetition]:
    """Keep replays that resume immediately after their source score span.

    Each candidate is removed from the performance sequence, the remaining
    notes are aligned to the score, and the notes after the replay must map to
    the score continuation directly following the mapped source phrase.  This
    rejects isolated or naturally recurring motifs that do not behave like a
    stop/replay/resume event.
    """

    if not notes or not score_notes or not repetitions:
        return []
    continuation_notes = max(1, int(continuation_notes))
    minimum_confirmed_notes = max(1, int(minimum_confirmed_notes))
    kept: list[NoteRepetition] = []
    for repetition in repetitions:
        if (
            repetition.source_i0 < 0
            or repetition.source_i1 <= repetition.source_i0
            or repetition.repeat_i0 < repetition.source_i1
            or repetition.repeat_i1 <= repetition.repeat_i0
            or repetition.repeat_i1 >= len(notes)
        ):
            continue
        original_indices = [
            index
            for index in range(len(notes))
            if not (repetition.repeat_i0 <= index < repetition.repeat_i1)
        ]
        collapsed = [notes[index] for index in original_indices]
        if aligner is not None:
            collapsed_mapping = aligner.align(collapsed, score_notes)
        else:
            from alignmodel.stages.dc_alignment import _monotonic_pitch_mapping

            collapsed_mapping = _monotonic_pitch_mapping(
                collapsed, score_notes
            )
        mapping = {
            original_index: score_index
            for original_index, score_index in zip(
                original_indices, collapsed_mapping
            )
        }
        source_mapping = [
            mapping.get(index)
            for index in range(
                repetition.source_i0, repetition.source_i1
            )
        ]
        source_mapping = [
            int(value) for value in source_mapping if value is not None
        ]
        if not source_mapping or any(
            right <= left
            for left, right in zip(source_mapping, source_mapping[1:])
        ):
            continue
        expected = source_mapping[-1] + 1
        if expected >= len(score_notes):
            continue
        post_indices = range(
            repetition.repeat_i1,
            min(
                len(notes),
                repetition.repeat_i1 + continuation_notes,
            ),
        )
        post_mapping = [
            int(mapping[index])
            for index in post_indices
            if mapping.get(index) is not None
        ]
        if len(post_mapping) < minimum_confirmed_notes:
            continue
        confirmed = 0
        cursor = expected
        for score_index in post_mapping:
            if score_index < cursor:
                continue
            # Permit one missing transcribed score note, but no unrelated jump.
            if score_index > cursor + 1:
                break
            confirmed += 1
            cursor = score_index + 1
            if confirmed >= minimum_confirmed_notes:
                kept.append(repetition)
                break
    return kept


def apply_note_repetitions(state: PipelineState, learned=None) -> None:
    """Layer 1: detect and label repetitions from the shared transcription."""

    repetition_model = (
        getattr(learned, "note_repetition", None)
        if learned is not None
        and bool(getattr(state.config, "use_note_repetition_model", True))
        else None
    )
    repetitions = find_note_sequence_repetitions(
        state.transcribed_notes,
        min_notes=max(6, int(state.config.note_repetition_min_notes)),
        max_notes=int(state.config.note_repetition_max_notes),
        min_confidence=float(state.config.note_repetition_min_confidence),
    )
    if repetition_model is not None and not repetitions:
        learned_singletons = _model_note_repetitions(
            state, repetition_model
        )
        if learned_singletons:
            repetitions.append(learned_singletons[0])
    repetitions = consolidate_note_repetitions(repetitions)
    if str(
        getattr(state.config, "note_alignment_strategy", "contextual")
    ) == "contextual_continuation":
        repetitions = filter_repetitions_by_score_continuation(
            state.transcribed_notes,
            state.score.notes,
            repetitions,
            aligner=(
                getattr(learned, "contextual_note_aligner", None)
                if learned is not None
                else None
            ),
        )
    state.note_repetitions = repetitions
    state.segments = [
        UnfoldedSegment(
            0.0,
            state.duration_sec,
            0,
            len(state.score.notes),
            0.0,
        )
    ]
    for item in repetitions:
        state.labels.append(
            PipelineLabel(
                id=next_label_id(state),
                type="repetition",
                start_time=item.repeat_start,
                end_time=item.repeat_end,
                comment=(
                    "transcribed-note phrase repetition "
                    f"confidence={item.confidence:.3f}"
                ),
                repeats_label_range=RepeatRange(
                    item.source_start, item.source_end
                ),
                extra_copies=1,
            )
        )
