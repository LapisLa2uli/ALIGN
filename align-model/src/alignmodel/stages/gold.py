from __future__ import annotations

from typing import Any

import numpy as np

from alignmodel.score import ScoreNote
from datacreate.melody import is_repeated_pass

SCORED_TYPES = {
    "wrong_note",
    "missed_note",
    "extra_note",
    "intonation_error",
    "rhythm_error",
    "repetition",
}


def load_labels(sample_dir) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    path = Path(sample_dir) / "labels.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("labels", [])


def first_pass_labels(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Schema 1.2 gold: drop repeated-pass copies; keep first pass + repetition."""
    return [lab for lab in labels if not is_repeated_pass(lab)]


def load_first_pass_labels(sample_dir) -> list[dict[str, Any]]:
    return first_pass_labels(load_labels(sample_dir))


def extra_copies_of(lab: dict[str, Any]) -> int:
    raw = lab.get("extra_copies")
    if raw is None:
        return 1
    return max(1, min(2, int(raw)))


def replay_spans(lab: dict[str, Any]) -> list[tuple[float, float]]:
    """Split a repetition window into extra_copies sequential replays."""
    t0 = float(lab["start_time"])
    t1 = float(lab["end_time"])
    n = extra_copies_of(lab)
    if n <= 1 or t1 <= t0:
        return [(t0, t1)]
    width = (t1 - t0) / n
    return [(t0 + i * width, t0 + (i + 1) * width) for i in range(n)]


def gap_span(lab: dict[str, Any]) -> tuple[float, float] | None:
    """Silent rest between first pass and the replay, if present."""
    src = lab.get("repeats_label_range") or {}
    if "end_time" not in src:
        return None
    s1 = float(src["end_time"])
    t0 = float(lab["start_time"])
    if t0 - s1 < 0.15:
        return None
    return (s1, t0)


def repetition_labs(labels: list[dict]) -> list[dict]:
    return [lab for lab in first_pass_labels(labels) if lab.get("type") == "repetition"]


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def map_notes_to_perf(
    notes: list[ScoreNote],
    labels: list[dict],
    perf_duration: float,
) -> list[tuple[float, float]]:
    """First-pass score→performance times, shifting later notes by gold repeats."""
    mapping = [(float(n.start), float(n.end)) for n in notes]
    reps = sorted(
        (lab for lab in repetition_labs(labels) if lab.get("repeats_label_range")),
        key=lambda lab: float(lab["start_time"]),
    )
    extra = 0.0
    for lab in reps:
        src = lab["repeats_label_range"]
        insert_at = float(src["end_time"])
        insert_len = max(0.0, float(lab["end_time"]) - float(lab["start_time"]))
        extra += insert_len
        for i, note in enumerate(notes):
            if note.start >= insert_at - 1e-3:
                a, b = mapping[i]
                mapping[i] = (a + insert_len, b + insert_len)
    out = []
    for a, b in mapping:
        a = float(np.clip(a, 0.0, max(perf_duration, 0.05)))
        b = float(np.clip(max(b, a + 0.04), 0.04, max(perf_duration, a + 0.04)))
        out.append((a, b))
    return out
