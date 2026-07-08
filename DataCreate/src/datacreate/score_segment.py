from __future__ import annotations

import logging
from pathlib import Path

from music21 import converter, stream


def get_measure_count(score_path: Path) -> int:
    score = converter.parse(str(score_path))
    if not score.parts:
        return 0
    measures = list(score.parts[0].recurse().getElementsByClass(stream.Measure))
    if not measures:
        return 0
    numbers = [m.number for m in measures if m.number is not None]
    return int(max(numbers)) if numbers else len(measures)


def get_score_info(score_path: Path) -> dict:
    score = converter.parse(str(score_path))
    title = None
    if score.metadata and score.metadata.title:
        title = score.metadata.title
    total = get_measure_count(score_path)
    return {"total_measures": total, "title": title}


def extract_measure_range(
    source_path: Path,
    dest_path: Path,
    start_measure: int,
    end_measure: int,
    logger: logging.Logger,
) -> Path:
    if start_measure < 1 or end_measure < start_measure:
        raise ValueError(
            f"Invalid measure range {start_measure}-{end_measure}; start must be >= 1 and end >= start"
        )

    score = converter.parse(str(source_path))
    total = get_measure_count(source_path)
    if total == 0:
        raise ValueError(f"Score has no measures: {source_path}")
    if end_measure > total:
        raise ValueError(
            f"End measure {end_measure} exceeds score length ({total} measures)"
        )

    segment = score.measures(start_measure, end_measure)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    segment.write("musicxml", fp=str(dest_path))
    logger.info(
        "Extracted measures %d-%d from %s -> %s",
        start_measure,
        end_measure,
        source_path,
        dest_path,
    )
    return dest_path
