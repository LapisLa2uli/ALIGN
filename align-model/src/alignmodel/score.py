from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from music21 import converter, note, tempo


@dataclass(frozen=True)
class ScoreNote:
    pitch: int
    start: float
    end: float
    duration: float


def _bpm(score) -> float:
    for mark in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if mark.number:
            return float(mark.number)
    return 120.0


def parse_score_notes(path: Path) -> list[ScoreNote]:
    parsed = converter.parse(str(path))
    bpm = _bpm(parsed)
    notes: list[ScoreNote] = []
    for n in parsed.recurse().getElementsByClass(note.Note):
        if n.duration.isGrace:
            continue
        try:
            start_ql = float(n.getOffsetInHierarchy(parsed))
        except Exception:
            start_ql = float(n.offset)
        dur_ql = float(n.duration.quarterLength or 0.0)
        start = start_ql * 60.0 / bpm
        end = (start_ql + dur_ql) * 60.0 / bpm
        if end <= start:
            end = start + 0.05
        notes.append(
            ScoreNote(
                pitch=int(n.pitch.midi),
                start=start,
                end=end,
                duration=end - start,
            )
        )
    notes.sort(key=lambda x: (x.start, x.pitch))
    return notes
