from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from datacreate.audio_utils import load_audio, save_wav
from datacreate.config import PipelineConfig
from datacreate.score_segment import extract_measure_range, get_score_info
from datacreate.stages.stage3_reference import synthesize_reference
from datacreate.stages.stage5_alignment import run_alignment, write_candidates
from datacreate.stages.stage7_features import extract_mels
from datacreate.stages.stage8_bundle import write_labels_template, write_metadata
from datacreate.utils import read_json, write_json


def ensure_full_score(sample_dir: Path) -> Path:
    full = sample_dir / "full_score.musicxml"
    verified = sample_dir / "verified_score.musicxml"
    if full.exists():
        return full
    if verified.exists():
        shutil.copy2(verified, full)
        return full
    raise FileNotFoundError(f"No score found in {sample_dir}")


def get_prep_state(sample_dir: Path, config: PipelineConfig | None = None) -> dict[str, Any]:
    config = config or PipelineConfig.load()
    full = sample_dir / "full_score.musicxml"
    if not full.exists():
        try:
            ensure_full_score(sample_dir)
        except FileNotFoundError:
            return {"total_measures": 0}

    info = get_score_info(full)
    metadata: dict[str, Any] = {}
    meta_path = sample_dir / "metadata.json"
    if meta_path.exists():
        metadata = read_json(meta_path)

    perf_duration = None
    perf_path = sample_dir / "performance_audio.wav"
    if perf_path.exists():
        audio, sr = load_audio(perf_path, config.sample_rate())
        perf_duration = len(audio) / sr

    reference_duration = None
    ref_path = sample_dir / "reference_audio.wav"
    if ref_path.exists():
        ref_audio, sr = load_audio(ref_path, config.sample_rate())
        reference_duration = len(ref_audio) / sr

    return {
        **info,
        "score_segment": metadata.get("score_segment"),
        "performance_trim": metadata.get("performance_trim"),
        "performance_duration": perf_duration,
        "reference_duration_seconds": reference_duration,
    }


def apply_score_segment(
    sample_dir: Path,
    start_measure: int,
    end_measure: int,
    config: PipelineConfig,
    logger: logging.Logger,
    start_beat: int = 1,
    end_beat: int | None = None,
) -> dict[str, Any]:
    full_score = ensure_full_score(sample_dir)
    verified = sample_dir / "verified_score.musicxml"
    extract_measure_range(
        full_score,
        verified,
        start_measure,
        end_measure,
        logger,
        start_beat=start_beat,
        end_beat=end_beat,
    )
    synthesize_reference(verified, sample_dir, config, logger)
    _reprocess_alignment_and_features(sample_dir, config, logger)

    segment_info: dict[str, Any] = {
        "start_measure": start_measure,
        "end_measure": end_measure,
        "start_beat": start_beat,
        "source": "full_score.musicxml",
    }
    if end_beat is not None:
        segment_info["end_beat"] = end_beat
    _update_metadata(sample_dir, config, {"score_segment": segment_info}, logger)
    return segment_info


def apply_performance_trim(
    sample_dir: Path,
    trim_start: float,
    trim_end: float | None,
    config: PipelineConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    perf_path = sample_dir / "performance_audio.wav"
    if not perf_path.exists():
        raise FileNotFoundError("performance_audio.wav missing")

    sr = config.sample_rate()
    audio, _ = load_audio(perf_path, sr)
    duration = len(audio) / sr
    trim_end = duration if trim_end is None else trim_end

    if trim_start < 0 or trim_end > duration + 1e-6 or trim_end <= trim_start:
        raise ValueError(
            f"Invalid trim range {trim_start:.3f}-{trim_end:.3f}s (duration {duration:.3f}s)"
        )

    full_backup = sample_dir / "performance_audio_full.wav"
    if not full_backup.exists():
        shutil.copy2(perf_path, full_backup)

    start_idx = int(trim_start * sr)
    end_idx = int(trim_end * sr)
    trimmed = audio[start_idx:end_idx]
    save_wav(perf_path, trimmed, sr)
    logger.info(
        "Trimmed performance %s: %.3f-%.3f s (%d samples)",
        perf_path,
        trim_start,
        trim_end,
        len(trimmed),
    )

    _reprocess_alignment_and_features(sample_dir, config, logger)

    trim_info = {
        "trim_start": round(trim_start, 4),
        "trim_end": round(trim_end, 4),
        "original_duration": round(duration, 4),
        "trimmed_duration": round(len(trimmed) / sr, 4),
    }
    _update_metadata(sample_dir, config, {"performance_trim": trim_info}, logger)
    return trim_info


def reprocess_alignment(
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    perf = sample_dir / "performance_audio.wav"
    ref = sample_dir / "reference_audio.wav"
    if not perf.exists() or not ref.exists():
        raise FileNotFoundError("performance_audio.wav or reference_audio.wav missing")
    result = run_alignment(perf, ref, sample_dir, config, logger)
    write_candidates(result.candidates, sample_dir, config.schema_version)
    extract_mels(perf, ref, sample_dir, config, logger)
    return {
        "candidate_count": len(result.candidates),
        "alignment_path": str(result.alignment_path),
    }


def _reprocess_alignment_and_features(
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> None:
    reprocess_alignment(sample_dir, config, logger)
    if not (sample_dir / "labels.json").exists():
        write_labels_template(sample_dir, config)


def _update_metadata(
    sample_dir: Path,
    config: PipelineConfig,
    updates: dict[str, Any],
    logger: logging.Logger,
) -> None:
    meta_path = sample_dir / "metadata.json"
    existing = read_json(meta_path) if meta_path.exists() else {}
    existing.update(updates)
    write_metadata(sample_dir, config, existing, logger)
