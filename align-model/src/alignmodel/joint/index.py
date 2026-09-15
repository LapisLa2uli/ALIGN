from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from music21 import converter, note, stream


_TIE_FROM = frozenset({"start", "continue"})
_TIE_TO = frozenset({"continue", "stop"})
_RELATIONSHIPS = frozenset({"match", "substitute", "extra", "copy"})
_TIE_GAP_QL = 0.05


@dataclass(frozen=True)
class ScoreEvent:
    """One canonical score event, possibly spanning a notated tie chain."""

    index: int
    pitch: int
    ql_start: float
    ql_end: float
    source_indices: tuple[int, ...]
    measure: int | None = None


@dataclass(frozen=True)
class JointEvent:
    """One audible event projected into the canonical score-event space."""

    pitch: int
    start: float
    end: float
    score_span: tuple[int, int] | None
    relationship: str = "match"
    copy_pass: int = 0
    origin_relationship: str | None = None
    rendered_index: int | None = None
    source_indices: tuple[int, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("JointEvent end must be greater than start")
        if self.score_span is not None:
            start, end = self.score_span
            if start < 0 or end <= start:
                raise ValueError(f"Invalid canonical score span {self.score_span}")
        if self.relationship not in _RELATIONSHIPS:
            raise ValueError(f"Unknown lineage relationship {self.relationship!r}")
        if self.copy_pass < 0:
            raise ValueError("copy_pass must be non-negative")
        if self.relationship in {"match", "substitute"} and self.score_span is None:
            raise ValueError(f"{self.relationship} event requires a score span")

    @property
    def is_extra(self) -> bool:
        return self.score_span is None

    @property
    def is_copy(self) -> bool:
        return self.relationship == "copy" or self.copy_pass > 0


@dataclass(frozen=True)
class ScoreEventIndex:
    """Projection between MusicXML notes, rendered notes, and canonical events."""

    events: tuple[ScoreEvent, ...]
    source_to_event: tuple[int, ...]
    rendered_events: tuple[JointEvent, ...] = ()
    deleted_source_indices: frozenset[int] = frozenset()
    deleted_event_indices: frozenset[int] = frozenset()

    @classmethod
    def from_lineage(
        cls,
        lineage: Mapping[str, Any],
    ) -> "ScoreEventIndex":
        """Reconstruct training score events when source MusicXML is offline.

        This is intentionally a training/oracle API. Deployable inference must
        continue to call :meth:`from_musicxml`.
        """

        if lineage.get("kind") != "synth_note_lineage":
            raise ValueError("Expected kind='synth_note_lineage'")
        clean_rows = sorted(
            lineage.get("clean_notes") or [],
            key=lambda row: int(row["clean_index"]),
        )
        clean_indices = [int(row["clean_index"]) for row in clean_rows]
        if clean_indices != list(range(len(clean_rows))):
            raise ValueError("Lineage clean indices must be contiguous and zero-based")
        co_rendered: set[tuple[int, int]] = set()
        for rendered in lineage.get("rendered_notes") or []:
            sources = sorted(
                {int(value) for value in rendered.get("clean_indices") or []}
            )
            co_rendered.update(zip(sources, sources[1:]))

        events: list[ScoreEvent] = []
        source_to_event = [-1] * len(clean_rows)
        for row in clean_rows:
            source_index = int(row["clean_index"])
            pitch = int(row["pitch_midi"])
            ql_start = float(row["onset_ql"])
            ql_end = ql_start + max(float(row["duration_ql"]), 0.001)
            previous = events[-1] if events else None
            previous_source = (
                previous.source_indices[-1] if previous is not None else -1
            )
            inferred_tie = (
                previous is not None
                and source_index == previous_source + 1
                and previous.pitch == pitch
                and _abuts(previous.ql_end, ql_start)
                and (previous_source, source_index) in co_rendered
            )
            if inferred_tie:
                events[-1] = replace(
                    previous,
                    ql_end=max(previous.ql_end, ql_end),
                    source_indices=(*previous.source_indices, source_index),
                )
                source_to_event[source_index] = previous.index
                continue
            event_index = len(events)
            events.append(
                ScoreEvent(
                    index=event_index,
                    pitch=pitch,
                    ql_start=ql_start,
                    ql_end=ql_end,
                    source_indices=(source_index,),
                    measure=(
                        int(row["measure"])
                        if row.get("measure") is not None
                        else None
                    ),
                )
            )
            source_to_event[source_index] = event_index
        return cls(tuple(events), tuple(source_to_event)).project_lineage(lineage)

    @classmethod
    def from_musicxml(
        cls,
        score_path: Path | str,
        lineage: Mapping[str, Any] | None = None,
    ) -> "ScoreEventIndex":
        parsed = converter.parse(str(Path(score_path)))
        source_notes = _ordered_source_notes(parsed)
        elements_by_source = {
            source_index: element
            for source_index, element, *_rest in source_notes
        }
        events: list[ScoreEvent] = []
        source_to_event = [-1] * len(source_notes)
        for source_index, element, ql_start, ql_end, measure in source_notes:
            tie_type = _tie_type(element)
            previous = events[-1] if events else None
            previous_element = (
                elements_by_source[previous.source_indices[-1]]
                if previous is not None
                else None
            )
            joins_previous = (
                previous is not None
                and previous.pitch == int(element.pitch.midi)
                and _abuts(previous.ql_end, ql_start)
                and _tie_channel(previous_element) == _tie_channel(element)
                and _tie_type(previous_element) in _TIE_FROM
                and tie_type in _TIE_TO
            )
            if joins_previous:
                events[-1] = replace(
                    previous,
                    ql_end=max(previous.ql_end, ql_end),
                    source_indices=(*previous.source_indices, source_index),
                )
                source_to_event[source_index] = previous.index
                continue
            event_index = len(events)
            events.append(
                ScoreEvent(
                    index=event_index,
                    pitch=int(element.pitch.midi),
                    ql_start=ql_start,
                    ql_end=ql_end,
                    source_indices=(source_index,),
                    measure=measure,
                )
            )
            source_to_event[source_index] = event_index
        index = cls(tuple(events), tuple(source_to_event))
        return index.project_lineage(lineage) if lineage is not None else index

    @property
    def score_event_count(self) -> int:
        return len(self.events)

    def event_for_source(self, source_index: int) -> int:
        if not 0 <= int(source_index) < len(self.source_to_event):
            raise ValueError(f"Source note index {source_index} is out of range")
        event_index = self.source_to_event[int(source_index)]
        if not 0 <= event_index < len(self.events):
            raise ValueError(f"Source note index {source_index} is not projected")
        return event_index

    def validate_span(
        self, score_span: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        if score_span is None:
            return None
        start, end = (int(score_span[0]), int(score_span[1]))
        if start < 0 or end <= start or end > len(self.events):
            raise ValueError(
                f"Canonical score span {(start, end)} is outside "
                f"[0, {len(self.events)})"
            )
        return start, end

    def event_span_for_sources(
        self, source_indices: Iterable[int]
    ) -> tuple[int, int] | None:
        sources = tuple(dict.fromkeys(int(value) for value in source_indices))
        if not sources:
            return None
        event_indices = sorted({self.event_for_source(value) for value in sources})
        # A single renderer event can sustain across intervening score events
        # (for example, repeated same-pitch MIDI notes whose note-offs merge).
        # Canonical spans therefore consume the full half-open interval; source
        # membership remains available separately on JointEvent.
        return self.validate_span((event_indices[0], event_indices[-1] + 1))

    def project_lineage(
        self, lineage: Mapping[str, Any] | None
    ) -> "ScoreEventIndex":
        if lineage is None:
            return self
        if lineage.get("kind") != "synth_note_lineage":
            raise ValueError("Expected kind='synth_note_lineage'")
        clean_rows = list(lineage.get("clean_notes") or [])
        clean_indices = sorted(int(row["clean_index"]) for row in clean_rows)
        expected_sources = list(range(len(self.source_to_event)))
        if clean_indices != expected_sources:
            raise ValueError(
                "Lineage clean indices do not match MusicXML source indices: "
                f"expected {expected_sources}, got {clean_indices}"
            )

        performed = {
            int(row["performed_index"]): row
            for row in lineage.get("performed_notes") or []
        }
        rendered_rows = sorted(
            lineage.get("rendered_notes") or [],
            key=lambda row: int(row["rendered_index"]),
        )
        if not rendered_rows:
            rendered_rows = [
                {
                    "rendered_index": int(row["performed_index"]),
                    "performed_indices": [int(row["performed_index"])],
                    "clean_indices": (
                        [int(row["clean_index"])]
                        if row.get("clean_index") is not None
                        else []
                    ),
                    "primary_clean_index": row.get("clean_index"),
                    "relationship": row.get("relationship") or "extra",
                    "pitch_midi_written": row["pitch_midi"],
                    "start_sec": row["onset_ql"],
                    "end_sec": float(row["onset_ql"])
                    + max(float(row["duration_ql"]), 0.001),
                }
                for row in sorted(
                    lineage.get("performed_notes") or [],
                    key=lambda value: int(value["performed_index"]),
                )
            ]
        rendered_indices = [int(row["rendered_index"]) for row in rendered_rows]
        if rendered_indices != list(range(len(rendered_rows))):
            raise ValueError("Rendered indices must be contiguous and zero-based")

        rendered_events: list[JointEvent] = []
        for row in rendered_rows:
            relationship = str(row.get("relationship") or "extra")
            if relationship not in _RELATIONSHIPS:
                raise ValueError(f"Unknown lineage relationship {relationship!r}")
            sources = tuple(
                dict.fromkeys(int(value) for value in row.get("clean_indices") or [])
            )
            primary = row.get("primary_clean_index")
            if not sources and primary is not None:
                sources = (int(primary),)
            span = self.event_span_for_sources(sources)

            performed_rows = [
                performed[int(value)]
                for value in row.get("performed_indices") or []
                if int(value) in performed
            ]
            copy_pass = max(
                (int(value.get("copy_pass") or 0) for value in performed_rows),
                default=(1 if relationship == "copy" else 0),
            )
            origins = [
                str(value.get("origin_relationship"))
                for value in performed_rows
                if value.get("origin_relationship")
            ]
            origin = (
                origins[0]
                if origins
                else ("extra" if span is None else relationship)
            )
            rendered_events.append(
                JointEvent(
                    pitch=int(row["pitch_midi_written"]),
                    start=float(row["start_sec"]),
                    end=max(
                        float(row["end_sec"]),
                        float(row["start_sec"]) + 0.001,
                    ),
                    score_span=span,
                    relationship=relationship,
                    copy_pass=copy_pass,
                    origin_relationship=origin,
                    rendered_index=int(row["rendered_index"]),
                    source_indices=sources,
                )
            )

        deleted_sources = {
            int(value) for value in lineage.get("deleted_clean_notes") or []
        }
        deleted_sources.update(
            int(row["clean_index"])
            for row in clean_rows
            if bool(row.get("deleted"))
        )
        for source_index in deleted_sources:
            self.event_for_source(source_index)
        deleted_events = {
            event.index
            for event in self.events
            if event.source_indices
            and all(value in deleted_sources for value in event.source_indices)
        }
        return replace(
            self,
            rendered_events=tuple(rendered_events),
            deleted_source_indices=frozenset(deleted_sources),
            deleted_event_indices=frozenset(deleted_events),
        )


def _tie_type(element: note.Note | None) -> str | None:
    tie = getattr(element, "tie", None)
    value = getattr(tie, "type", None) if tie is not None else None
    return str(value).lower() if value else None


def _abuts(left_end: float, right_start: float) -> bool:
    return abs(float(right_start) - float(left_end)) <= _TIE_GAP_QL


def _tie_channel(element: note.Note | None) -> tuple[str | None, str | None]:
    if element is None:
        return None, None
    part = element.getContextByClass(stream.Part)
    voice = element.getContextByClass(stream.Voice)
    return (
        str(part.id) if part is not None and part.id is not None else None,
        str(voice.id) if voice is not None and voice.id is not None else None,
    )


def _ordered_source_notes(
    score: stream.Score,
) -> list[tuple[int, note.Note, float, float, int | None]]:
    rows: list[tuple[int, note.Note, float, float, int | None]] = []
    for source_index, element in enumerate(
        item
        for item in score.recurse().getElementsByClass(note.Note)
        if not item.duration.isGrace
    ):
        try:
            ql_start = float(element.getOffsetInHierarchy(score))
        except Exception:
            ql_start = float(element.offset)
        duration = float(element.duration.quarterLength or 0.0)
        if duration <= 0:
            raise ValueError(
                f"Source note {source_index} has non-positive duration"
            )
        measure = element.getContextByClass(stream.Measure)
        measure_number = (
            int(measure.number)
            if measure is not None and measure.number is not None
            else None
        )
        rows.append(
            (
                source_index,
                element,
                ql_start,
                ql_start + duration,
                measure_number,
            )
        )
    rows.sort(key=lambda value: (value[2], int(value[1].pitch.midi), value[0]))
    return rows
