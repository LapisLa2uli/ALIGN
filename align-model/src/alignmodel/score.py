from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alignmodel.stages.score_graph import build_score_graph


@dataclass(frozen=True)
class ScoreNote:
    pitch: int
    start: float
    end: float
    duration: float


def parse_score_notes(path: Path) -> list[ScoreNote]:
    graph = build_score_graph(path)
    return [
        ScoreNote(pitch=n.pitch, start=n.start, end=n.end, duration=n.duration)
        for n in graph.notes
    ]

