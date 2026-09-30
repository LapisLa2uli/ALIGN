"""Shared gold loading and official scoring for 9.2 perfect-transcription aligners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample


@dataclass(frozen=True)
class AlignerClip:
    name: str
    score_path: Path
    index: ScoreEventIndex
    lineage: dict[str, Any]

    @property
    def rendered(self) -> tuple[JointEvent, ...]:
        return self.index.rendered_events


def load_clip(root: Path, name: str, lineage_dir: Path | None = None) -> AlignerClip:
    """Load a clip with repaired lineage when ``lineage_dir`` is given."""

    sample = root / name
    path = (
        lineage_dir / f"{name}.json" if lineage_dir is not None
        else sample / "note_map.json"
    )
    lineage = json.loads(path.read_text(encoding="utf-8"))
    score_path = sample / "verified_score.musicxml"
    index = ScoreEventIndex.from_musicxml(score_path, lineage)
    return AlignerClip(name, score_path, index, lineage)


def perfect_transcription(clip: AlignerClip) -> list[tuple[int, float, float]]:
    """Performed audio notes as a perfect score-free transcriber would report them."""

    return [(event.pitch, event.start, event.end) for event in clip.rendered]


def clip_source(name: str) -> str:
    if name.startswith("synth_gen_"):
        return "procedural"
    return "rawdata:" + name.rsplit("_", 1)[0].removeprefix("synth_")


def metric_sample(
    clip: AlignerClip,
    predicted: Sequence[JointEvent],
    predicted_deletions: frozenset[int],
) -> JointMetricSample:
    return JointMetricSample(
        predicted=tuple(predicted),
        target=clip.rendered,
        source=clip_source(clip.name),
        predicted_deletions=frozenset(predicted_deletions),
        target_deletions=clip.index.deleted_event_indices,
        score_event_count=len(clip.index.events),
    )
