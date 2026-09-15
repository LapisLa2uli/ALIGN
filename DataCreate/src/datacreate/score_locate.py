"""Locate the full-score measure span that matches a performance transcription."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from music21 import converter, meter, stream

from datacreate.melody import ScoreSoundingNote, parse_sounding_notes

_PITCH_SHIFTS = (0, 2, -2, 12, -12)
_GAP = -0.85
_MIN_CONFIDENCE = 0.28
_MIN_MAPPED = 6


@dataclass(frozen=True)
class LocatedScoreSpan:
    start_measure: int
    end_measure: int
    start_beat: int = 1
    end_beat: int | None = None
    score_i0: int = 0
    score_i1: int = 0
    confidence: float = 0.0
    mapped_notes: int = 0
    duration_ratio: float = 1.0
    pitch_shift: int = 0

    def as_segment(self) -> dict[str, Any]:
        payload = {
            "start_measure": int(self.start_measure),
            "end_measure": int(self.end_measure),
            "start_beat": int(self.start_beat),
        }
        if self.end_beat is not None:
            payload["end_beat"] = int(self.end_beat)
        return payload

    def same_as(self, segment: dict[str, Any] | None) -> bool:
        if not segment:
            return False
        return (
            int(segment.get("start_measure") or 0) == self.start_measure
            and int(segment.get("end_measure") or 0) == self.end_measure
            and int(segment.get("start_beat") or 1) == self.start_beat
            and segment.get("end_beat") == self.end_beat
        )


@dataclass(frozen=True)
class TranscribedNote:
    pitch: int
    start: float
    end: float
    confidence: float = 1.0


def _pair_score(observed: int, written: int) -> float:
    if observed == written:
        return 2.0
    if observed % 12 == written % 12:
        return 0.55
    if abs(observed - written) == 1:
        return -0.15
    return -1.15


def _smith_waterman(
    observed: Sequence[int], written: Sequence[int]
) -> tuple[float, int, int, int]:
    """Return (score, obs_i0, written_i0, written_i1_exclusive) for the best local match."""

    n = len(observed)
    m = len(written)
    if n == 0 or m == 0:
        return 0.0, 0, 0, 0
    h = np.zeros((n + 1, m + 1), dtype=np.float32)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)
    best = 0.0
    best_cell = (0, 0)
    for i, obs in enumerate(observed, start=1):
        row = h[i]
        prev = h[i - 1]
        prow = ptr[i]
        for j, wr in enumerate(written, start=1):
            diag = prev[j - 1] + _pair_score(obs, wr)
            up = prev[j] + _GAP
            left = row[j - 1] + _GAP
            score = max(0.0, diag, up, left)
            row[j] = score
            if score <= 0.0:
                continue
            if score == diag:
                prow[j] = 1
            elif score == up:
                prow[j] = 2
            else:
                prow[j] = 3
            if score > best:
                best = float(score)
                best_cell = (i, j)
    i, j = best_cell
    if best <= 0.0 or ptr[i, j] == 0:
        return 0.0, 0, 0, 0
    while i > 0 and j > 0 and ptr[i, j]:
        code = int(ptr[i, j])
        if code == 1:
            i -= 1
            j -= 1
        elif code == 2:
            i -= 1
        else:
            j -= 1
    written_i0 = j
    written_i1 = best_cell[1]
    obs_i0 = i
    return best, obs_i0, written_i0, written_i1


def _duration_ratio(
    observed: Sequence[TranscribedNote],
    written: Sequence[ScoreSoundingNote],
    i0: int,
    i1: int,
) -> float:
    if not observed or i1 <= i0 or i1 > len(written):
        return 0.0
    obs = max(observed[-1].end - observed[0].start, 0.05)
    ref = max(written[i1 - 1].end - written[i0].start, 0.05)
    return float(obs / ref)


def _measure_timeline(score_path: Path) -> dict[int, tuple[float, float]]:
    parsed = converter.parse(str(score_path))
    if not parsed.parts:
        return {}
    part = parsed.parts[0]
    rows: dict[int, tuple[float, float]] = {}
    for measure in part.getElementsByClass(stream.Measure):
        if measure.number is None:
            continue
        try:
            offset = float(measure.getOffsetInHierarchy(parsed))
        except Exception:  # noqa: BLE001
            offset = float(measure.offset)
        ts = measure.timeSignature
        if ts is None:
            for item in measure.getElementsByClass(meter.TimeSignature):
                ts = item
                break
        beat_ql = float(ts.beatDuration.quarterLength) if ts is not None else 1.0
        rows[int(measure.number)] = (offset, beat_ql)
    return rows


def _beat_for_note(
    note: ScoreSoundingNote, timeline: dict[int, tuple[float, float]]
) -> int:
    if note.measure is None:
        return 1
    offset, beat_ql = timeline.get(int(note.measure), (note.ql_start, 1.0))
    beat = 1 + int(np.floor((float(note.ql_start) - offset) / max(beat_ql, 1e-6)))
    return max(1, beat)


def locate_score_span(
    transcribed: Sequence[TranscribedNote],
    written: Sequence[ScoreSoundingNote],
    *,
    score_path: Path | None = None,
) -> LocatedScoreSpan | None:
    """Find the written span that best explains a transcribed take."""

    if not transcribed or not written:
        return None
    observed_pitches = [int(note.pitch) for note in transcribed]
    written_pitches = [int(note.pitch) for note in written]
    best: tuple[float, float, int, int, int, int] | None = None
    for shift in _PITCH_SHIFTS:
        shifted = [pitch + shift for pitch in observed_pitches]
        raw, _obs_i0, i0, i1 = _smith_waterman(shifted, written_pitches)
        if i1 <= i0:
            continue
        mapped = i1 - i0
        if mapped < min(_MIN_MAPPED, len(transcribed)):
            continue
        ratio = _duration_ratio(transcribed, written, i0, i1)
        duration_pen = 0.0
        if ratio > 0:
            duration_pen = abs(np.log(max(ratio, 1e-3)))
        rank = raw - 1.4 * duration_pen
        if best is None or rank > best[0]:
            best = (rank, raw, shift, i0, i1, mapped)
    if best is None:
        return None
    _rank, raw, shift, i0, i1, mapped = best
    first = written[i0]
    last = written[i1 - 1]
    start_measure = int(first.measure or 1)
    end_measure = int(last.measure or start_measure)
    if end_measure < start_measure:
        end_measure = start_measure
    timeline = _measure_timeline(score_path) if score_path is not None else {}
    start_beat = _beat_for_note(first, timeline)
    denom = 2.0 * min(len(transcribed), mapped)
    confidence = float(raw / denom) if denom else 0.0
    return LocatedScoreSpan(
        start_measure=start_measure,
        end_measure=end_measure,
        start_beat=start_beat,
        end_beat=None,
        score_i0=i0,
        score_i1=i1,
        confidence=min(1.0, max(0.0, confidence)),
        mapped_notes=mapped,
        duration_ratio=_duration_ratio(transcribed, written, i0, i1),
        pitch_shift=shift,
    )


def current_segment_is_valid(
    segment: dict[str, Any] | None, total_measures: int
) -> bool:
    if not segment or total_measures <= 0:
        return False
    start = int(segment.get("start_measure") or 0)
    end = int(segment.get("end_measure") or 0)
    return 1 <= start <= end <= total_measures


def should_apply_location(
    located: LocatedScoreSpan | None,
    *,
    current: dict[str, Any] | None,
    total_measures: int,
) -> bool:
    if located is None:
        return False
    if located.end_measure > total_measures or located.start_measure < 1:
        return False
    current_ok = current_segment_is_valid(current, total_measures)
    if located.same_as(current) and current_ok:
        return False
    if located.confidence >= _MIN_CONFIDENCE and located.mapped_notes >= _MIN_MAPPED:
        return True
    return not current_ok


def notes_from_payload(raw_notes: Sequence[dict[str, Any]]) -> list[TranscribedNote]:
    notes: list[TranscribedNote] = []
    for item in raw_notes:
        pitch = item.get("pitch")
        if pitch is None:
            pitch = item.get("midi")
        if pitch is None:
            continue
        start = float(item.get("start") or item.get("perf_start") or 0.0)
        end = float(item.get("end") or item.get("perf_end") or start + 0.05)
        if end <= start:
            end = start + 0.05
        notes.append(
            TranscribedNote(
                pitch=int(pitch),
                start=start,
                end=end,
                confidence=float(item.get("confidence") or 1.0),
            )
        )
    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes


def locate_from_transcription_file(
    sample_dir: Path,
    transcription_path: Path,
) -> LocatedScoreSpan | None:
    payload = json.loads(transcription_path.read_text(encoding="utf-8"))
    transcribed = notes_from_payload(payload.get("transcribed_notes") or payload.get("notes") or [])
    full_score = sample_dir / "full_score.musicxml"
    if not full_score.exists():
        full_score = sample_dir / "verified_score.musicxml"
    written = parse_sounding_notes(full_score)
    return locate_score_span(transcribed, written, score_path=full_score)


def locate_payload(located: LocatedScoreSpan | None) -> dict[str, Any]:
    if located is None:
        return {"found": False}
    return {"found": True, **asdict(located)}
