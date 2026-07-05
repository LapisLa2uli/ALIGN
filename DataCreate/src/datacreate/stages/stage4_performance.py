from __future__ import annotations

import logging
import shutil
from pathlib import Path

from datacreate.audio_utils import load_audio, save_wav
from datacreate.config import PipelineConfig


def ingest_performance(
    audio_path: Path,
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> Path:
    sr = config.sample_rate()
    mono = bool(config.audio.get("mono", True))
    audio, _ = load_audio(audio_path, sample_rate=sr, mono=mono)
    dest = sample_dir / "performance_audio.wav"
    save_wav(dest, audio, sr)
    logger.info("Stored performance audio at %s (%d samples @ %d Hz)", dest, len(audio), sr)
    raw_copy = sample_dir / "performance_audio_original" / audio_path.name
    raw_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_path, raw_copy)
    return dest
