from __future__ import annotations

import copy
import logging
from pathlib import Path

from music21 import chord, converter, duration, note, stream


_PITCH_TYPES = (note.Note, note.Rest, note.Unpitched, chord.Chord)


def get_measure_count(score_path: Path) -> int:
    score = converter.parse(str(score_path))
    if not score.parts:
        return 0
    measures = list(score.parts[0].recurse().getElementsByClass(stream.Measure))
    if not measures:
        return 0
    numbers = [m.number for m in measures if m.number is not None]
    return int(max(numbers)) if numbers else len(measures)


def _get_time_signature(measure: stream.Measure):
    ts = measure.timeSignature
    if ts is None:
        for el in measure.getElementsByClass("TimeSignature"):
            ts = el
            break
    return ts


def beats_per_measure(measure: stream.Measure) -> int:
    ts = _get_time_signature(measure)
    if ts:
        return int(round(ts.barDuration.quarterLength / ts.beatDuration.quarterLength))
    return 4


def _beat_quarter_length(measure: stream.Measure) -> float:
    ts = _get_time_signature(measure)
    return ts.beatDuration.quarterLength if ts else 1.0


def _bar_quarter_length(measure: stream.Measure) -> float:
    ts = _get_time_signature(measure)
    return ts.barDuration.quarterLength if ts else 4.0


def get_score_info(score_path: Path) -> dict:
    score = converter.parse(str(score_path))
    title = None
    if score.metadata and score.metadata.title:
        title = score.metadata.title
    total = get_measure_count(score_path)
    default_beats = 4
    if score.parts:
        first_measures = list(score.parts[0].getElementsByClass(stream.Measure))
        if first_measures:
            default_beats = beats_per_measure(first_measures[0])
    return {
        "total_measures": total,
        "title": title,
        "beats_per_measure": default_beats,
    }


def _copy_measure_layout(source: stream.Measure, dest: stream.Measure) -> None:
    for el in source:
        if isinstance(el, _PITCH_TYPES):
            continue
        dest.insert(0, el)


def trim_measure_end(measure: stream.Measure, end_beat: int | None) -> stream.Measure:
    max_beats = beats_per_measure(measure)
    if end_beat is None or end_beat >= max_beats:
        return measure

    end_offset = end_beat * _beat_quarter_length(measure)
    bar_length = _bar_quarter_length(measure)
    trimmed = stream.Measure(number=measure.number)
    _copy_measure_layout(measure, trimmed)

    for el in measure.notesAndRests:
        el_end = el.offset + el.duration.quarterLength
        if el.offset >= end_offset - 1e-9:
            continue
        if el_end > end_offset + 1e-9:
            el_copy = copy.deepcopy(el)
            el_copy.duration = duration.Duration(end_offset - el.offset)
            trimmed.insert(el.offset, el_copy)
        else:
            trimmed.insert(el.offset, copy.deepcopy(el))

    if end_offset < bar_length - 1e-9:
        trimmed.insert(end_offset, note.Rest(quarterLength=bar_length - end_offset))
    return trimmed


def trim_measure_start(measure: stream.Measure, start_beat: int) -> stream.Measure:
    if start_beat <= 1:
        return measure

    start_offset = (start_beat - 1) * _beat_quarter_length(measure)
    trimmed = stream.Measure(number=measure.number)
    _copy_measure_layout(measure, trimmed)

    if start_offset > 1e-9:
        trimmed.insert(0, note.Rest(quarterLength=start_offset))

    for el in measure.notesAndRests:
        el_end = el.offset + el.duration.quarterLength
        if el_end <= start_offset + 1e-9:
            continue
        if el.offset < start_offset - 1e-9:
            el_copy = copy.deepcopy(el)
            el_copy.duration = duration.Duration(el_end - start_offset)
            trimmed.insert(start_offset, el_copy)
        else:
            trimmed.insert(el.offset, copy.deepcopy(el))
    return trimmed


def _replace_measure(part: stream.Part, old: stream.Measure, new: stream.Measure) -> None:
    idx = part.index(old)
    part.remove(old)
    part.insert(idx, new)


def _apply_beat_trims(
    segment: stream.Score,
    start_beat: int,
    end_beat: int | None,
) -> None:
    for part in segment.parts:
        measures = list(part.getElementsByClass(stream.Measure))
        if not measures:
            continue
        if start_beat > 1:
            _replace_measure(part, measures[0], trim_measure_start(measures[0], start_beat))
            measures = list(part.getElementsByClass(stream.Measure))
        if end_beat is not None:
            _replace_measure(part, measures[-1], trim_measure_end(measures[-1], end_beat))


def _validate_beat_range(
    score: stream.Score,
    start_measure: int,
    end_measure: int,
    start_beat: int,
    end_beat: int | None,
) -> None:
    if start_beat < 1:
        raise ValueError(f"Start beat must be >= 1, got {start_beat}")
    if end_beat is not None and end_beat < 1:
        raise ValueError(f"End beat must be >= 1, got {end_beat}")

    first_measure = score.parts[0].measure(start_measure)
    last_measure = score.parts[0].measure(end_measure)
    start_max = beats_per_measure(first_measure)
    end_max = beats_per_measure(last_measure)

    if start_beat > start_max:
        raise ValueError(
            f"Start beat {start_beat} exceeds {start_max} beats in measure {start_measure}"
        )
    if end_beat is not None and end_beat > end_max:
        raise ValueError(
            f"End beat {end_beat} exceeds {end_max} beats in measure {end_measure}"
        )
    if start_measure == end_measure and end_beat is not None and end_beat < start_beat:
        raise ValueError(
            f"End beat {end_beat} must be >= start beat {start_beat} "
            f"when both refer to measure {start_measure}"
        )


def extract_measure_range(
    source_path: Path,
    dest_path: Path,
    start_measure: int,
    end_measure: int,
    logger: logging.Logger,
    start_beat: int = 1,
    end_beat: int | None = None,
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

    _validate_beat_range(score, start_measure, end_measure, start_beat, end_beat)

    segment = score.measures(start_measure, end_measure)
    if start_beat > 1 or end_beat is not None:
        _apply_beat_trims(segment, start_beat, end_beat)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    segment.write("musicxml", fp=str(dest_path))

    if end_beat is not None:
        logger.info(
            "Extracted measures %d-%d (beats %d-%d) from %s -> %s",
            start_measure,
            end_measure,
            start_beat,
            end_beat,
            source_path,
            dest_path,
        )
    elif start_beat > 1:
        logger.info(
            "Extracted measures %d-%d (from beat %d) from %s -> %s",
            start_measure,
            end_measure,
            start_beat,
            source_path,
            dest_path,
        )
    else:
        logger.info(
            "Extracted measures %d-%d from %s -> %s",
            start_measure,
            end_measure,
            source_path,
            dest_path,
        )
    return dest_path
