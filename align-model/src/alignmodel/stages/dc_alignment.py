"""DataCreate DTW → stage-3 note/rest pairs.

Stage 2 keeps its own chroma DTW for edits. Stage 3 timing uses the same
alignment as DataCreate Stage 5: bounded/jump DTW, rest compression, phrase
DTW, and onset-refined score events (from ``alignment.npz`` when present).
"""

from __future__ import annotations

import logging
from pathlib import Path

from alignmodel.types import GraphNote, PairedEvent, PipelineState

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


def pairs_from_learned_alignment(
    state: PipelineState, mel, learned
) -> list[PairedEvent]:
    """Transcribe audio notes, align them to the score, and return timed pairs."""
    if (
        mel is None
        or learned is None
        or getattr(learned, "transcriber", None) is None
        or getattr(learned, "note_aligner", None) is None
    ):
        return []
    from alignmodel.stages.note_align import normalize_notes
    from alignmodel.transcription import infer_sample_notes

    transcribed = infer_sample_notes(
        learned.transcriber,
        state.sample_dir,
        learned.device or state.device,
        decode_config=learned.transcriber_decode,
    )
    observed = normalize_notes(transcribed)
    result = learned.note_aligner.align(observed, state.score)
    pairs: list[PairedEvent] = []
    for op in result.operations:
        if (
            op.performance_index is None
            or op.score_index is None
            or op.kind not in {"match", "substitute"}
        ):
            continue
        if not (0 <= op.performance_index < len(observed)):
            continue
        if not (0 <= op.score_index < len(state.score.notes)):
            continue
        played = observed[op.performance_index]
        written = state.score.notes[op.score_index]
        pairs.append(
            PairedEvent(
                score_index=written.index,
                pitch=written.pitch,
                ref_start=written.start,
                ref_end=written.end,
                perf_start=played.start,
                perf_end=played.end,
                kind=op.kind,
                measure=written.measure,
            )
        )
    return sorted(pairs, key=lambda pair: (pair.perf_start, pair.ref_start))


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
