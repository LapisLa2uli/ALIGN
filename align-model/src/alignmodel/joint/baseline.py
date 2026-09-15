from __future__ import annotations

from typing import Sequence

from alignmodel.stages.note_align import NoteAligner
from alignmodel.types import GraphNote, ScoreGraph

from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate


def current_note_aligner_baseline(
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    *,
    aligner: NoteAligner | None = None,
) -> tuple[list[JointEvent], frozenset[int]]:
    """Run the existing deployable symbolic aligner in canonical index space."""

    graph_notes = [
        GraphNote(
            index=event.index,
            pitch=event.pitch,
            start=event.ql_start,
            end=event.ql_end,
            duration=event.ql_end - event.ql_start,
            ql_start=event.ql_start,
            ql_end=event.ql_end,
            measure=event.measure,
            source_note_indices=list(event.source_indices),
        )
        for event in score
    ]
    graph = ScoreGraph(
        notes=graph_notes,
        duration_sec=max((event.ql_end for event in score), default=0.0),
    )
    result = (aligner or NoteAligner()).align(
        [
            {
                "pitch": candidate.pitch,
                "start": candidate.start,
                "end": candidate.end,
                "confidence": candidate.confidence,
            }
            for candidate in candidates
        ],
        graph,
    )
    copy_indices = {
        int(operation.performance_index)
        for operation in result.operations
        if operation.performance_index is not None and operation.is_copy
    }
    events = []
    for index, (candidate, mapped) in enumerate(
        zip(candidates, result.mapping)
    ):
        if mapped is None:
            relationship = "extra"
            span = None
        else:
            relationship = (
                "copy"
                if index in copy_indices
                else (
                    "match"
                    if candidate.pitch == score[mapped].pitch
                    else "substitute"
                )
            )
            span = (int(mapped), int(mapped) + 1)
        events.append(
            JointEvent(
                pitch=candidate.pitch,
                start=candidate.start,
                end=candidate.end,
                score_span=span,
                relationship=relationship,
                copy_pass=1 if relationship == "copy" else 0,
                confidence=candidate.confidence,
            )
        )
    return events, frozenset(int(value) for value in result.deletions)
