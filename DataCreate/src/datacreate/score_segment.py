from __future__ import annotations

import copy
import logging
from pathlib import Path

from music21 import chord, converter, duration, meter, note, stream, tempo


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


def _measure_length_ql(measure: stream.Measure) -> float:
    length = float(measure.barDuration.quarterLength)
    try:
        length = max(length, float(measure.highestTime))
    except Exception:  # noqa: BLE001
        pass
    return max(length, 0.25)


def pack_measure_offsets(part: stream.Part) -> None:
    """Place measures back-to-back in number order with no gaps.

    ``trim_measure_beats`` shortens the first bar but leaves later bars at
    their pre-trim offsets, so a 4/4 → pickup extract keeps a multi-beat hole.
    MusicXML exporters and MuseScore then invent empty bars or reorder
    leftover measure numbers (e.g. m21 sounding before the m20 pickup).
    """
    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        return

    nums = [m.number for m in measures if m.number is not None]
    if (
        len(nums) == len(measures)
        and len(set(int(n) for n in nums)) == len(nums)
        and [int(n) for n in nums] != sorted(int(n) for n in nums)
    ):
        measures = sorted(measures, key=lambda m: int(m.number))

    lengths = [_measure_length_ql(m) for m in measures]
    for measure in measures:
        part.remove(measure)
    cursor = 0.0
    for measure, length in zip(measures, lengths):
        part.insert(cursor, measure)
        cursor += length


def measures_in_written_order(score: stream.Score) -> bool:
    """True when each part's measures increase in number along the timeline."""
    for part in score.parts:
        nums = [
            int(m.number)
            for m in part.getElementsByClass(stream.Measure)
            if m.number is not None
        ]
        if nums != sorted(nums):
            return False
    return True


def _remove_from_score(segment: stream.Score, el) -> None:
    site = el.activeSite
    if site is not None:
        try:
            site.remove(el)
            return
        except Exception:  # noqa: BLE001
            pass
    for part in segment.parts:
        if el in part:
            part.remove(el)
            return
        for measure in part.getElementsByClass(stream.Measure):
            if el in measure:
                measure.remove(el)
                return


def prune_stale_opening_tempi(segment: stream.Score) -> None:
    """Drop inherited opening metronome marks that share t=0 with a later mark.

    ``Score.measures()`` copies the piece's first tempo onto the excerpt even
    when a closer mark (theme Andante, etc.) also lands at offset 0 after a
    pickup trim. Event timing then uses 60 BPM while MuseScore renders 72.
    """
    opening: list[tempo.MetronomeMark] = []
    for el in list(segment.recurse().getElementsByClass(tempo.MetronomeMark)):
        if not el.number:
            continue
        try:
            off = float(el.getOffsetInHierarchy(segment))
        except Exception:  # noqa: BLE001
            off = float(getattr(el, "offset", 0.0))
        if off < 1e-4:
            opening.append(el)
    if len(opening) < 2:
        return
    for el in opening[:-1]:
        _remove_from_score(segment, el)


def normalize_extracted_segment(segment: stream.Score) -> None:
    """Repair gaps, measure order, and duplicate opening tempi after extract."""
    for part in segment.parts:
        pack_measure_offsets(part)
    prune_stale_opening_tempi(segment)


def _excerpt_start_ql(score: stream.Score, start_measure: int, start_beat: int) -> float:
    if not score.parts:
        return 0.0
    part = score.parts[0]
    measure = part.measure(start_measure)
    try:
        m_off = float(part.elementOffset(measure))
    except Exception:  # noqa: BLE001
        m_off = 0.0
    ts = _active_time_signature(part, measure)
    beat_ql = _beat_quarter_length(measure, ts)
    return m_off + max(0, int(start_beat) - 1) * beat_ql


def _tempo_offset(score: stream.Score, el) -> float:
    try:
        return float(el.getOffsetInHierarchy(score))
    except Exception:  # noqa: BLE001
        return float(getattr(el, "offset", 0.0))


def _tempo_snapshot(score: stream.Score) -> list[tuple[float, float]]:
    """(offset_ql, bpm) before ``Score.measures()`` aliases the same mark objects."""
    marks: list[tuple[float, float]] = []
    for el in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if not el.number:
            continue
        marks.append((_tempo_offset(score, el), float(el.number)))
    marks.sort(key=lambda item: item[0])
    return marks


def _active_bpm(marks: list[tuple[float, float]], at_ql: float) -> float | None:
    last = None
    for off, bpm in marks:
        if off <= at_ql + 1e-6:
            last = bpm
    return last


def _active_metronome(score: stream.Score, at_ql: float):
    bpm = _active_bpm(_tempo_snapshot(score), at_ql)
    if bpm is None:
        return None
    return tempo.MetronomeMark(number=bpm, referent=note.Note(type="quarter"))


def _excerpt_length_ql(segment: stream.Score) -> float:
    """Packed excerpt length; ``Score.highestTime`` is 0 before core update."""
    length = 0.0
    for part in segment.parts:
        for measure in part.getElementsByClass(stream.Measure):
            try:
                m_off = float(part.elementOffset(measure))
            except Exception:  # noqa: BLE001
                m_off = float(getattr(measure, "offset", 0.0))
            length = max(length, m_off + _measure_length_ql(measure))
    if length > 0:
        return length
    return float(getattr(segment, "highestTime", 0.0) or 0.0)


def _new_metronome(bpm: float) -> tempo.MetronomeMark:
    return tempo.MetronomeMark(number=float(bpm), referent=note.Note(type="quarter"))


def stamp_excerpt_tempi(
    source: stream.Score,
    segment: stream.Score,
    start_measure: int,
    start_beat: int,
    tempo_marks: list[tuple[float, float]] | None = None,
) -> None:
    """Write the tempo that is actually in force at the excerpt start.

    ``Score.measures()`` aliases metronome objects from the source. Collect
    ``tempo_marks`` before that call; otherwise removing excerpt marks also
    erases the source list this function reads.
    """
    if not segment.parts:
        return
    start_ql = _excerpt_start_ql(source, start_measure, start_beat)
    marks = tempo_marks if tempo_marks is not None else _tempo_snapshot(source)

    for el in list(segment.recurse().getElementsByClass(tempo.MetronomeMark)):
        _remove_from_score(segment, el)

    part = segment.parts[0]
    active_bpm = _active_bpm(marks, start_ql)
    if active_bpm is not None:
        first_measures = list(part.getElementsByClass(stream.Measure))
        mark = _new_metronome(active_bpm)
        if first_measures:
            first_measures[0].insert(0, mark)
        else:
            part.insert(0, mark)

    excerpt_ql = _excerpt_length_ql(segment)
    for off, bpm in marks:
        if off <= start_ql + 1e-6:
            continue
        dest = off - start_ql
        if dest <= 1e-6 or dest > excerpt_ql + 0.25:
            continue
        placed = False
        for measure in part.getElementsByClass(stream.Measure):
            m_off = float(part.elementOffset(measure))
            m_len = _measure_length_ql(measure)
            if m_off - 1e-6 <= dest <= m_off + m_len + 1e-6:
                measure.insert(max(0.0, dest - m_off), _new_metronome(bpm))
                placed = True
                break
        if not placed:
            part.insert(dest, _new_metronome(bpm))


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
        pack_measure_offsets(part)


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

    # Snapshot before measures() — it aliases tempo objects into the excerpt.
    tempo_marks = _tempo_snapshot(score)
    segment = score.measures(start_measure, end_measure)
    if start_beat > 1 or end_beat is not None:
        _apply_beat_trims(segment, start_beat, end_beat)
    normalize_extracted_segment(segment)
    stamp_excerpt_tempi(
        score, segment, start_measure, start_beat, tempo_marks=tempo_marks
    )

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
