from __future__ import annotations

import logging
from pathlib import Path

from datacreate.audio_utils import audio_stats, is_silent
from datacreate.config import PipelineConfig
from datacreate.stages.stage1_ingest import copy_verified_score
from datacreate.tools.musescore import check_musescore_version, render_score_to_wav


def synthesize_reference(
    score_path: Path,
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> Path:
    verified = copy_verified_score(score_path, sample_dir / "verified_score.musicxml")
    version = check_musescore_version(config, logger)
    output_wav = sample_dir / "reference_audio.wav"
    render_score_to_wav(config, verified, output_wav, logger)
    import librosa

    audio, _ = librosa.load(output_wav, sr=None, mono=True)
    stats = audio_stats(audio)
    logger.info("Reference audio stats: %s", stats)
    if is_silent(audio):
        raise RuntimeError(
            "Reference audio is silent. Check MuseScore version (>=4.2) and soundfont."
        )
    return output_wav
