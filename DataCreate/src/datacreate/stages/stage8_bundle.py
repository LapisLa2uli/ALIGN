from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datacreate.config import PipelineConfig
from datacreate.models import LabelsDocument
from datacreate.utils import utc_now_iso, write_json


def write_metadata(
    sample_dir: Path,
    config: PipelineConfig,
    extra: dict[str, Any] | None,
    logger: logging.Logger,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": config.schema_version,
        "sample_rate": config.sample_rate(),
        "mel_params": dict(config.mel),
        "alignment_params": dict(config.alignment),
        "taxonomy": list(config.taxonomy),
        "musescore": dict(config.musescore),
        "soundfont": config.paths.get("soundfont"),
        "pipeline_run_at": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    path = sample_dir / "metadata.json"
    write_json(path, payload)
    logger.info("Wrote metadata to %s", path)
    return path


def write_labels_template(
    sample_dir: Path,
    config: PipelineConfig,
    annotator_id: str | None = None,
) -> Path:
    doc = LabelsDocument(
        schema_version=config.schema_version,
        annotator_id=annotator_id,
        labels=[],
        self_reported=[],
    )
    path = sample_dir / "labels.json"
    write_json(path, doc.model_dump())
    return path


def bundle_complete(sample_dir: Path) -> bool:
    required = [
        "verified_score.musicxml",
        "performance_audio.wav",
        "reference_audio.wav",
        "performance_mel.npy",
        "reference_mel.npy",
        "candidates.json",
        "labels.json",
        "metadata.json",
    ]
    return all((sample_dir / name).exists() for name in required)
