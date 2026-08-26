from __future__ import annotations

from typing import Any

import numpy as np

from alignmodel.score import ScoreNote


def load_labels(sample_dir) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    path = Path(sample_dir) / "labels.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("labels", [])


def repetition_labs(labels: list[dict]) -> list[dict]:
    return [lab for lab in labels if lab.get("type") == "repetition"]


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
