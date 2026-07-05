from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RepeatRange(BaseModel):
    start_time: float
    end_time: float


class Label(BaseModel):
    id: str
    source: Literal[
        "auto",
        "auto_confirmed",
        "auto_edited",
        "auto_rejected",
        "manual",
        "synthetic",
    ]
    start_time: float
    end_time: float
    type: str
    severity: int | None = None
    deviation_cents: float | None = None
    deviation_ms: float | None = None
    measure_number: int | None = None
    note_id: str | None = None
    comment: str | None = None
    repeats_label_range: RepeatRange | None = None


class SelfReportedMark(BaseModel):
    start_time: float
    end_time: float
    comment: str | None = None


class LabelsDocument(BaseModel):
    schema_version: str
    audio_reference: str = "performance_audio.wav"
    annotator_id: str | None = None
    self_reported: list[SelfReportedMark] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)

    @field_validator("labels")
    @classmethod
    def non_zero_length(cls, labels: list[Label]) -> list[Label]:
        for label in labels:
            if label.end_time <= label.start_time:
                raise ValueError(f"Label {label.id} has zero or negative duration")
        return labels


class CandidatesDocument(BaseModel):
    schema_version: str
    labels: list[Label] = Field(default_factory=list)


def label_to_dict(label: Label) -> dict[str, Any]:
    return label.model_dump(exclude_none=True)
