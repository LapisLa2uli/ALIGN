"""DataCreate DTW → stage-3 note/rest pairs.

Stage 2 keeps its own chroma DTW for edits. Stage 3 timing uses the same
alignment as DataCreate Stage 5: bounded/jump DTW, rest compression, phrase
DTW, and onset-refined score events (from ``alignment.npz`` when present).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from alignmodel.types import (
    GraphNote,
    PairedEvent,
    PipelineState,
    TranscribedNote,
)

logger = logging.getLogger(__name__)

_MATCH_DT_SEC = 0.45


def load_aligned_events(sample_dir: Path | str) -> list[dict]:
    """Return DataCreate-aligned score events, running DTW only if needed."""
    sample_dir = Path(sample_dir)
    align_path = sample_dir / "alignment.npz"
    if align_path.exists():
        try:
            from datacreate.note_alignment import build_note_alignment

            blob = build_note_alignment(sample_dir)
            return list(blob.get("events") or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not map %s: %s", align_path, exc)

    perf_wav = sample_dir / "performance_audio.wav"
    ref_wav = sample_dir / "reference_audio.wav"
    if not (perf_wav.exists() and ref_wav.exists()):
        return []

    try:
        from datacreate.config import PipelineConfig
        from datacreate.note_alignment import build_note_alignment
        from datacreate.stages.stage5_alignment import run_alignment

        cfg = PipelineConfig.load()
        run_alignment(
            perf_wav,
            ref_wav,
            sample_dir,
            cfg,
            logger,
            detect_candidates=False,
        )
        blob = build_note_alignment(sample_dir)
        return list(blob.get("events") or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("DataCreate alignment failed for %s: %s", sample_dir.name, exc)
        return []


def pairs_from_aligned_events(
    events: list[dict],
    notes: list[GraphNote],
) -> list[PairedEvent]:
    """Convert DataCreate events into stage-3 pairs (notes + merged rests)."""
    from datacreate.stages.stage5_alignment import _merge_consecutive_rests

    merged = _merge_consecutive_rests(list(events or []))
    used: set[int] = set()
    note_i = 0
    sequential = _can_zip_sounding(merged, notes)
    pairs: list[PairedEvent] = []
    for ev in merged:
        ref_s = float(ev.get("ref_start") or 0.0)
        ref_e = float(ev.get("ref_end") or ref_s + 0.001)
        perf_s = float(ev.get("perf_start") or 0.0)
        perf_e = float(ev.get("perf_end") or perf_s + 0.001)
        measure = ev.get("measure")
        if ev.get("is_rest"):
            pairs.append(
                PairedEvent(
                    score_index=-1,
                    pitch=-1,
                    ref_start=ref_s,
                    ref_end=ref_e,
                    perf_start=perf_s,
                    perf_end=perf_e,
                    kind="rest",
                    measure=int(measure) if measure is not None else None,
                )
            )
            continue
        midi = ev.get("midi")
        note: GraphNote | None = None
        if sequential and note_i < len(notes):
            note = notes[note_i]
            note_i += 1
            used.add(note.index)
        else:
            note = _match_graph_note(notes, used, midi, ref_s)
            if note is not None:
                used.add(note.index)
        pairs.append(
            PairedEvent(
                score_index=note.index if note is not None else -1,
                pitch=int(note.pitch if note is not None else (midi if midi is not None else -1)),
                ref_start=ref_s,
                ref_end=ref_e,
                perf_start=perf_s,
                perf_end=perf_e,
                kind="match",
                measure=(
                    int(measure)
                    if measure is not None
                    else (note.measure if note is not None else None)
                ),
            )
        )
    return pairs


def transcribe_pipeline_state(state: PipelineState, learned) -> list[TranscribedNote]:
    """Populate the one shared transcription used by all three layers."""

    if state.transcribed_notes:
        return state.transcribed_notes
    if learned is None or getattr(learned, "transcriber", None) is None:
        return []
    from alignmodel.transcription import infer_note_decoder

    values = infer_note_decoder(learned.transcriber, state.sample_dir)
    state.transcribed_notes = [
        TranscribedNote(
            pitch=int(note.pitch),
            start=float(note.start),
            end=float(note.end),
            confidence=float(note.confidence),
            cents=float(getattr(note, "cents", 0.0)),
            pitch_candidates=tuple(
                int(value)
                for value in (getattr(note, "pitch_candidates", ()) or ())
            ),
        )
        for note in values
    ]
    return state.transcribed_notes


def pairs_from_learned_alignment(
    state: PipelineState, mel, learned
) -> list[PairedEvent]:
    """Align one shared transcription, treating Layer-1 repeats explicitly."""
    if (
        learned is None
        or getattr(learned, "transcriber", None) is None
    ):
        return []
    from alignmodel.stages.note_align import normalize_notes
    transcribed = transcribe_pipeline_state(state, learned)
    observed = normalize_notes(transcribed)
    repeated_indices = {
        index
        for repetition in state.note_repetitions
        for index in range(repetition.repeat_i0, repetition.repeat_i1)
    }
    first_pass_indices = [
        index for index in range(len(observed)) if index not in repeated_indices
    ]
    first_pass = [observed[index] for index in first_pass_indices]
    mapping: list[int | None] = [None] * len(observed)
    contextual = getattr(learned, "contextual_note_aligner", None)
    strategy = str(
        getattr(state.config, "note_alignment_strategy", "contextual")
    )
    if strategy == "multi_start":
        from alignmodel.stages.alignment_strategies import multi_start_mapping

        first_mapping = multi_start_mapping(first_pass, state.score.notes)
    elif strategy == "deterministic":
        first_mapping = _monotonic_pitch_mapping(
            first_pass, state.score.notes
        )
    elif contextual is not None and bool(
        getattr(state.config, "use_contextual_note_aligner", True)
    ):
        first_mapping = contextual.align(first_pass, state.score.notes)
        if strategy == "revision":
            from alignmodel.stages.alignment_strategies import (
                dynamic_revision_mapping,
            )

            first_mapping = dynamic_revision_mapping(
                first_pass,
                state.score.notes,
                first_mapping,
            )
    else:
        first_mapping = _monotonic_pitch_mapping(
            first_pass, state.score.notes
        )
    for local_index, score_index in enumerate(first_mapping):
        mapping[first_pass_indices[local_index]] = score_index
    for repetition in state.note_repetitions:
        source_count = repetition.source_i1 - repetition.source_i0
        repeat_count = repetition.repeat_i1 - repetition.repeat_i0
        for offset in range(min(source_count, repeat_count)):
            source_index = repetition.source_i0 + offset
            repeat_index = repetition.repeat_i0 + offset
            if (
                0 <= source_index < len(mapping)
                and 0 <= repeat_index < len(mapping)
            ):
                mapping[repeat_index] = mapping[source_index]
    state.note_mapping = mapping
    pairs: list[PairedEvent] = []
    for performance_index, score_index in enumerate(mapping):
        if score_index is None:
            continue
        if not (0 <= score_index < len(state.score.notes)):
            continue
        played = observed[performance_index]
        written = state.score.notes[score_index]
        pairs.append(
            PairedEvent(
                score_index=written.index,
                pitch=written.pitch,
                ref_start=written.start,
                ref_end=written.end,
                perf_start=played.start,
                perf_end=played.end,
                kind="match" if played.pitch == written.pitch else "substitute",
                cents=float(getattr(played, "cents", 0.0)),
                measure=written.measure,
            )
        )
    return sorted(pairs, key=lambda pair: (pair.perf_start, pair.ref_start))


def _monotonic_pitch_mapping(observed, score_notes) -> list[int | None]:
    """Edit-distance map for the non-repeated first pass."""

    n, m = len(observed), len(score_notes)
    gap = 0.90
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[:, 0] = np.arange(n + 1, dtype=np.float32) * gap
    dp[0, :] = np.arange(m + 1, dtype=np.float32) * gap
    back[1:, 0] = 1
    back[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delta = abs(int(observed[i - 1].pitch) - int(score_notes[j - 1].pitch))
            pair_cost = 0.0 if delta == 0 else 1.05 + min(delta, 12) / 60.0
            choices = (
                (float(dp[i - 1, j - 1]) + pair_cost, 0),
                (
                    float(dp[i - 1, j])
                    + gap
                    + 0.1 * float(observed[i - 1].confidence),
                    1,
                ),
                (float(dp[i, j - 1]) + gap, 2),
            )
            dp[i, j], back[i, j] = min(choices, key=lambda item: item[0])
    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i or j:
        code = int(back[i, j])
        if i and j and code == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i and (not j or code == 1):
            i -= 1
        else:
            j -= 1
    return mapping


def ensure_rhythm_pairs(
    state: PipelineState, *, learned=None, mel=None
) -> list[PairedEvent]:
    """Prefer learned note alignment; fall back to DataCreate DTW, then stage 2."""
    if state.rhythm_pairs:
        return state.rhythm_pairs
    state.rhythm_pairs = pairs_from_learned_alignment(state, mel, learned)
    if state.rhythm_pairs:
        return state.rhythm_pairs
    if not getattr(state.config, "use_dc_rhythm_alignment", True):
        state.rhythm_pairs = list(state.pairs)
        return state.rhythm_pairs
    events = load_aligned_events(state.sample_dir)
    if events:
        state.rhythm_pairs = pairs_from_aligned_events(events, state.score.notes)
    if not state.rhythm_pairs:
        state.rhythm_pairs = list(state.pairs)
    return state.rhythm_pairs


def _can_zip_sounding(events: list[dict], notes: list[GraphNote]) -> bool:
    n_sounding = sum(1 for ev in events if not ev.get("is_rest"))
    return bool(notes) and n_sounding == len(notes)


def _match_graph_note(
    notes: list[GraphNote],
    used: set[int],
    midi: int | None,
    ref_start: float,
) -> GraphNote | None:
    best: GraphNote | None = None
    best_d = _MATCH_DT_SEC
    for note in notes:
        if note.index in used:
            continue
        if midi is not None and int(note.pitch) != int(midi):
            continue
        dist = abs(float(note.start) - float(ref_start))
        if dist <= best_d:
            best_d = dist
            best = note
    if best is not None:
        return best
    best_d = _MATCH_DT_SEC
    for note in notes:
        if note.index in used:
            continue
        dist = abs(float(note.start) - float(ref_start))
        if dist <= best_d:
            best_d = dist
            best = note
    return best
