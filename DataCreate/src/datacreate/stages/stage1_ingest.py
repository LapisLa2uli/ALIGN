from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

from music21 import converter


def validate_score(score_path: Path, logger: logging.Logger) -> Path:
    suffix = score_path.suffix.lower()
    if suffix == ".mxl":
        return _validate_mxl(score_path, logger)
    if suffix in {".musicxml", ".xml"}:
        return _validate_musicxml(score_path, logger)
    raise ValueError(f"Unsupported score format: {score_path.suffix}")


def _validate_musicxml(score_path: Path, logger: logging.Logger) -> Path:
    try:
        score = converter.parse(str(score_path))
        logger.info(
            "Parsed MusicXML: %s parts, duration %.2fs",
            len(score.parts),
            float(score.duration.quarterLength),
        )
    except Exception as exc:
        raise ValueError(f"Invalid MusicXML: {score_path}") from exc
    return score_path


def _validate_mxl(score_path: Path, logger: logging.Logger) -> Path:
    try:
        with zipfile.ZipFile(score_path) as archive:
            if not any(name.endswith(".xml") or name.endswith(".musicxml") for name in archive.namelist()):
                raise ValueError("MXL archive contains no MusicXML")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Corrupt MXL: {score_path}") from exc
    return _validate_musicxml(score_path, logger)


def copy_verified_score(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == dest.resolve():
        return dest
    shutil.copy2(source, dest)
    return dest


def ingest_verified_score(source: Path, sample_dir: Path) -> tuple[Path, Path]:
    """Store full score archive and working verified copy."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    full = copy_verified_score(source, sample_dir / "full_score.musicxml")
    verified = copy_verified_score(source, sample_dir / "verified_score.musicxml")
    return full, verified
