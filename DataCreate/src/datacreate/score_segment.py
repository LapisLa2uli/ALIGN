from __future__ import annotations

import copy
import logging
from pathlib import Path

from music21 import chord, converter, duration, meter, note, stream


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
        for el in measure.getElementsByClass(meter.TimeSignature):
            ts = el
            break
    return ts


def _active_time_signature(
    part: stream.Part | None, measure: stream.Measure
) -> meter.TimeSignature | None:
    """Measure TS, or the last TimeSignature at/before this measure on the part.

    ``Score.measures()`` often leaves the collected meter on the Part rather than
    inside the first extracted Measure.
    """
    direct = _get_time_signature(measure)
    if direct is not None:
        return direct
    if part is None:
        return None
    try:
        m_offset = float(part.elementOffset(measure))
    except Exception:  # noqa: BLE001
        m_offset = None
    last: meter.TimeSignature | None = None
    for el in part.flatten().getElementsByClass(meter.TimeSignature):
        if m_offset is None:
            last = el
            continue
        try:
            el_offset = float(el.getOffsetInHierarchy(part))
        except Exception:  # noqa: BLE001
            el_offset = float(getattr(el, "offset", 0.0))
        if el_offset <= m_offset + 1e-9:
            last = el
    return last


def beats_per_measure(
    measure: stream.Measure, ts: meter.TimeSignature | None = None
) -> int:
    ts = ts if ts is not None else _get_time_signature(measure)
    if ts:
        return int(round(ts.barDuration.quarterLength / ts.beatDuration.quarterLength))
    return 4


def _beat_quarter_length(
    measure: stream.Measure, ts: meter.TimeSignature | None = None
) -> float:
    ts = ts if ts is not None else _get_time_signature(measure)
    return ts.beatDuration.quarterLength if ts else 1.0


def _bar_quarter_length(
    measure: stream.Measure, ts: meter.TimeSignature | None = None
) -> float:
    ts = ts if ts is not None else _get_time_signature(measure)
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


def _partial_time_signature(
    length_ql: float, active_ts: meter.TimeSignature | None
) -> meter.TimeSignature:
    """Time signature whose bar length matches a shortened measure.

    MusicXML export pads incomplete bars back to the old meter unless the
    written time signature matches the remaining duration.
    """
    denom = int(active_ts.denominator) if active_ts is not None else 4
    unit_ql = 4.0 / float(denom)
    numer = max(1, int(round(length_ql / unit_ql)))
    return meter.TimeSignature(f"{numer}/{denom}")


def trim_measure_beats(
    measure: stream.Measure,
    start_beat: int = 1,
    end_beat: int | None = None,
    active_ts: meter.TimeSignature | None = None,
) -> stream.Measure:
    """Keep only [start_beat, end_beat] and shift content to offset 0.

    Dropped leading/trailing beats are removed entirely (shortened / pickup bar),
    not replaced with rests. ``end_beat`` is inclusive; ``None`` keeps through
    the end of the bar. A matching partial time signature is written so MusicXML
    exporters do not re-pad the bar with rests.
    """
    ts = active_ts if active_ts is not None else _get_time_signature(measure)
    max_beats = beats_per_measure(measure, ts)
    beat_ql = _beat_quarter_length(measure, ts)
    bar_ql = _bar_quarter_length(measure, ts)

    start_offset = max(0.0, (start_beat - 1) * beat_ql) if start_beat > 1 else 0.0
    if end_beat is None or end_beat >= max_beats:
        end_offset = bar_ql
    else:
        end_offset = float(end_beat) * beat_ql

    if start_offset <= 1e-9 and end_offset >= bar_ql - 1e-9:
        return measure
    if end_offset <= start_offset + 1e-9:
        raise ValueError(
            f"Invalid beat trim in measure {measure.number}: "
            f"start_beat={start_beat}, end_beat={end_beat}"
        )

    trimmed = stream.Measure(number=measure.number)
    _copy_measure_layout(measure, trimmed)
    for ts_el in list(trimmed.getElementsByClass(meter.TimeSignature)):
        trimmed.remove(ts_el)

    length_ql = end_offset - start_offset
    trimmed.insert(0, _partial_time_signature(length_ql, ts))

    for el in measure.notesAndRests:
        el_start = float(el.offset)
        el_end = el_start + float(el.duration.quarterLength)
        clip_start = max(el_start, start_offset)
        clip_end = min(el_end, end_offset)
        if clip_end <= clip_start + 1e-9:
            continue
        el_copy = copy.deepcopy(el)
        new_ql = clip_end - clip_start
        if abs(new_ql - float(el.duration.quarterLength)) > 1e-9:
            el_copy.duration = duration.Duration(new_ql)
        trimmed.insert(clip_start - start_offset, el_copy)

    return trimmed


def trim_measure_end(measure: stream.Measure, end_beat: int | None) -> stream.Measure:
    return trim_measure_beats(measure, start_beat=1, end_beat=end_beat)


def trim_measure_start(measure: stream.Measure, start_beat: int) -> stream.Measure:
    return trim_measure_beats(measure, start_beat=start_beat, end_beat=None)


def _replace_measure(part: stream.Part, old: stream.Measure, new: stream.Measure) -> None:
    # music21 Stream.insert(x, el) treats x as a *musical offset*, not a list index.
    # Part.measures() often leaves Instrument/TimeSignature at the front, so
    # part.index(measure) != measure offset — using index placed the trimmed bar
    # mid-stream (extra empty bar before start, or last bar vanishing/out of order).
    offset = part.elementOffset(old)
    part.remove(old)
    part.insert(offset, new)


def _restore_meter_on_following_measure(
    part: stream.Part,
    after_index: int,
    original_ts: meter.TimeSignature | None,
) -> None:
    if original_ts is None:
        return
    measures = list(part.getElementsByClass(stream.Measure))
    if after_index + 1 >= len(measures):
        return
    nxt = measures[after_index + 1]
    if _get_time_signature(nxt) is None:
        nxt.insert(0, copy.deepcopy(original_ts))


def _strip_part_level_time_signatures(part: stream.Part) -> None:
    """Remove meter collected onto the Part (outside any Measure)."""
    for el in list(part.getElementsByClass(meter.TimeSignature)):
        part.remove(el)


def _apply_beat_trims(
    segment: stream.Score,
    start_beat: int,
    end_beat: int | None,
) -> None:
    for part in segment.parts:
        measures = list(part.getElementsByClass(stream.Measure))
        if not measures:
            continue
        if len(measures) == 1:
            if start_beat > 1 or end_beat is not None:
                active = _active_time_signature(part, measures[0])
                _replace_measure(
                    part,
                    measures[0],
                    trim_measure_beats(
                        measures[0], start_beat, end_beat, active_ts=active
                    ),
                )
            _strip_part_level_time_signatures(part)
            continue
        if start_beat > 1:
            first = measures[0]
            orig_ts = _active_time_signature(part, first)
            _replace_measure(
                part,
                first,
                trim_measure_beats(first, start_beat, None, active_ts=orig_ts),
            )
            _restore_meter_on_following_measure(part, 0, orig_ts)
            measures = list(part.getElementsByClass(stream.Measure))
        if end_beat is not None:
            last = measures[-1]
            active = _active_time_signature(part, last)
            _replace_measure(
                part,
                last,
                trim_measure_beats(last, 1, end_beat, active_ts=active),
            )
        _strip_part_level_time_signatures(part)


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

    part0 = score.parts[0]
    first_measure = part0.measure(start_measure)
    last_measure = part0.measure(end_measure)
    start_max = beats_per_measure(
        first_measure, _active_time_signature(part0, first_measure)
    )
    end_max = beats_per_measure(
        last_measure, _active_time_signature(part0, last_measure)
    )

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
