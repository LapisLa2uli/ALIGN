from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from datacreate.config import PipelineConfig
from datacreate.models import CandidatesDocument, LabelsDocument


LABELS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "labels"],
    "properties": {
        "schema_version": {"type": "string"},
        "audio_reference": {"type": "string"},
        "annotator_id": {"type": ["string", "null"]},
        "self_reported": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start_time", "end_time"],
                "properties": {
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "comment": {"type": ["string", "null"]},
                },
            },
        },
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "source", "start_time", "end_time", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string"},
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "type": {"type": "string"},
                    "severity": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
                    "deviation_cents": {"type": ["number", "null"]},
                    "deviation_ms": {"type": ["number", "null"]},
                    "measure_number": {"type": ["integer", "null"]},
                    "note_id": {"type": ["string", "null"]},
                    "comment": {"type": ["string", "null"]},
                    "repeats_label_range": {
                        "type": ["object", "null"],
                        "required": ["start_time", "end_time"],
                        "properties": {
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
}


def validate_labels_file(path: Path, config: PipelineConfig | None = None) -> list[str]:
    config = config or PipelineConfig.load()
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    validator = Draft202012Validator(LABELS_SCHEMA)
    for err in sorted(validator.iter_errors(data), key=lambda e: e.path):
        errors.append(f"{path}: {err.message}")

    try:
        doc = LabelsDocument.model_validate(data)
    except Exception as exc:
        errors.append(f"{path}: pydantic validation failed: {exc}")
        return errors

    allowed = config.taxonomy_set()
    for label in doc.labels:
        if label.type not in allowed:
            errors.append(
                f"{path}: label {label.id} type '{label.type}' not in taxonomy {sorted(allowed)}"
            )
        if label.type == "repetition" and label.repeats_label_range is None:
            errors.append(f"{path}: repetition label {label.id} missing repeats_label_range")

    if doc.schema_version != config.schema_version:
        errors.append(
            f"{path}: schema_version {doc.schema_version} != config {config.schema_version}"
        )
    return errors


def validate_candidates_file(path: Path, config: PipelineConfig | None = None) -> list[str]:
    config = config or PipelineConfig.load()
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        doc = CandidatesDocument.model_validate(data)
    except Exception as exc:
        errors.append(f"{path}: {exc}")
        return errors
    allowed = config.taxonomy_set()
    for label in doc.labels:
        if label.type not in allowed:
            errors.append(f"{path}: candidate {label.id} has invalid type {label.type}")
    return errors


def validate_corpus(root: Path, config: PipelineConfig | None = None) -> list[str]:
    errors: list[str] = []
    for labels_path in root.rglob("labels.json"):
        errors.extend(validate_labels_file(labels_path, config))
    for cand_path in root.rglob("candidates.json"):
        errors.extend(validate_candidates_file(cand_path, config))
    return errors
