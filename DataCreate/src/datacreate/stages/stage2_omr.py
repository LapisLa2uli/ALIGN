from __future__ import annotations

import logging
from pathlib import Path

from datacreate.config import PipelineConfig
from datacreate.stages.stage1_ingest import validate_score
from datacreate.tools.audiveris import open_for_manual_correction, run_omr


def process_pdf(
    pdf_path: Path,
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
    verified_path: Path | None = None,
) -> Path:
    omr_dir = sample_dir / "omr"
    draft = run_omr(config, pdf_path, omr_dir, logger)
    open_for_manual_correction(draft, config, logger)
    if verified_path and verified_path.exists():
        logger.info("Using provided verified score: %s", verified_path)
        return validate_score(verified_path, logger)
    logger.warning(
        "Awaiting manual correction. Save verified MusicXML to %s and rerun.",
        sample_dir / "verified_score.musicxml",
    )
    return draft
