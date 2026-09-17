"""Versioned, non-destructive preprocessing views for Transcriber Track A."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import wavfile

from .basic_pitch import (
    BASIC_PITCH_VERSION,
    FRONTEND_VERSION,
    BasicPitchFeatures,
    _run_official_inference,
    effective_pitch_policy,
    model_frame_times,
    remap_sounding_to_written,
    sha256_file,
)

TRACK_A_PREPROCESS_VERSION = "align-track-a-preprocess-v1"


@dataclass(frozen=True)
class TrackAPreprocessConfig:
    pre_emphasis: float = 0.85
    target_rms_dbfs: float = -22.0
    peak_limit: float = 0.97

    def validate(self) -> None:
        if not 0.0 <= self.pre_emphasis < 1.0:
            raise ValueError("pre_emphasis must be in [0, 1)")
        if not -60.0 <= self.target_rms_dbfs <= -3.0:
            raise ValueError("target_rms_dbfs is outside the gentle range")
        if not 0.1 <= self.peak_limit <= 1.0:
            raise ValueError("peak_limit must be in [0.1, 1.0]")


def preprocessing_identity(
    wav_path: Path | str,
    config: TrackAPreprocessConfig,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config.validate()
    config_payload = asdict(config)
    config_sha256 = hashlib.sha256(
        json.dumps(
            config_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "schema_version": TRACK_A_PREPROCESS_VERSION,
        "source_wav_sha256": sha256_file(wav_path),
        "config": config_payload,
        "config_sha256": config_sha256,
        "frontend_version": FRONTEND_VERSION,
        "basic_pitch_version": BASIC_PITCH_VERSION,
        "pitch_policy": effective_pitch_policy(source_metadata),
        "source_wav_mutated": False,
    }


def preprocess_audio(
    audio: np.ndarray,
    config: TrackAPreprocessConfig,
) -> np.ndarray:
    """Apply gentle attack emphasis and loudness normalization to a copy."""

    config.validate()
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = np.mean(values, axis=1)
    if values.ndim != 1:
        raise ValueError("Track A preprocessing expects mono or channel-last audio")
    output = values.copy()
    if len(output) > 1 and config.pre_emphasis:
        output[1:] = (
            output[1:] - float(config.pre_emphasis) * output[:-1]
        )
    rms = float(np.sqrt(np.mean(output * output))) if len(output) else 0.0
    target = 10.0 ** (float(config.target_rms_dbfs) / 20.0)
    if rms > 1e-8:
        output *= min(target / rms, 4.0)
    peak = float(np.max(np.abs(output))) if len(output) else 0.0
    if peak > config.peak_limit:
        output *= float(config.peak_limit) / peak
    return output.astype(np.float32, copy=False)


def extract_preprocessed_view(
    wav_path: Path | str,
    *,
    source_metadata: Mapping[str, Any] | None,
    cache_path: Path | str,
    config: TrackAPreprocessConfig = TrackAPreprocessConfig(),
) -> BasicPitchFeatures:
    """Run Basic Pitch on a versioned temporary view and cache activation maps."""

    wav = Path(wav_path)
    cache = Path(cache_path)
    identity = preprocessing_identity(wav, config, source_metadata)
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                actual = json.loads(str(np.asarray(saved["metadata"]).item()))
                if actual == identity:
                    return BasicPitchFeatures(
                        np.asarray(saved["note"], dtype=np.float32),
                        np.asarray(saved["onset"], dtype=np.float32),
                        np.asarray(saved["contour"], dtype=np.float32),
                        np.asarray(saved["frame_times"], dtype=np.float64),
                        actual,
                    )
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    rate, raw_audio = wavfile.read(wav)
    if np.issubdtype(raw_audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(raw_audio.dtype).min), np.iinfo(raw_audio.dtype).max))
        raw_audio = raw_audio.astype(np.float32) / scale
    processed = preprocess_audio(raw_audio, config)
    with tempfile.TemporaryDirectory() as temporary:
        temporary_wav = Path(temporary) / "track-a-view.wav"
        wavfile.write(temporary_wav, int(rate), processed)
        raw = _run_official_inference(temporary_wav)
    shift = int(identity["pitch_policy"]["audio_to_written_shift"])
    written = remap_sounding_to_written(raw, shift)
    features = BasicPitchFeatures(
        np.asarray(written["note"], dtype=np.float32),
        np.asarray(written["onset"], dtype=np.float32),
        np.asarray(written["contour"], dtype=np.float32),
        model_frame_times(len(written["note"])),
        identity,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".npz", dir=cache.parent, delete=False
    ) as stream:
        temporary_cache = Path(stream.name)
        np.savez_compressed(
            stream,
            note=features.note,
            onset=features.onset,
            contour=features.contour,
            frame_times=features.frame_times,
            metadata=np.asarray(json.dumps(identity, sort_keys=True)),
        )
    temporary_cache.replace(cache)
    return features
