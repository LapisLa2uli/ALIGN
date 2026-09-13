"""Cacheable PESTO fine-pitch features aligned to Basic Pitch frames."""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .basic_pitch import BasicPitchFeatures, sha256_file

PESTO_VERSION = "2.0.1"
FINE_PITCH_SCHEMA_VERSION = 1
_MODELS: dict[tuple[int, str], Any] = {}


def _cache_metadata(
    wav: Path,
    source_metadata: Mapping[str, Any],
    frame_times: np.ndarray,
) -> dict[str, Any]:
    from synthpipeline.pitch_convention import effective_audio_transpose

    return {
        "schema_version": FINE_PITCH_SCHEMA_VERSION,
        "pesto_version": PESTO_VERSION,
        "wav_sha256": sha256_file(wav),
        "effective_audio_transpose": effective_audio_transpose(
            dict(source_metadata)
        ),
        "frame_count": int(len(frame_times)),
        "frame_time_end": float(frame_times[-1]) if len(frame_times) else 0.0,
        "pitch_space": "written_midi",
    }


def _metadata_text(value: np.ndarray) -> str:
    scalar = np.asarray(value)
    if scalar.shape != ():
        raise ValueError("Fine-pitch metadata must be scalar JSON")
    return str(scalar.item())


def load_pesto_cache(
    path: Path | str,
    wav: Path | str,
    source_metadata: Mapping[str, Any],
    frame_times: np.ndarray,
) -> np.ndarray | None:
    cache = Path(path)
    if not cache.is_file():
        return None
    expected = _cache_metadata(Path(wav), source_metadata, frame_times)
    try:
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(_metadata_text(saved["metadata"])) != expected:
                return None
            feature = np.asarray(saved["pesto"], dtype=np.float32)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if feature.shape != (len(frame_times), 2):
        return None
    return feature


def save_pesto_cache(
    path: Path | str,
    feature: np.ndarray,
    metadata: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    feature = np.asarray(feature, dtype=np.float32)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                pesto=feature,
                metadata=np.asarray(
                    json.dumps(
                        dict(metadata),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            )
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return destination


def _model(sample_rate: int, device: str):
    key = (int(sample_rate), str(device))
    if key not in _MODELS:
        from pesto.loader import load_model

        _MODELS[key] = load_model(
            "mir-1k_g7",
            step_size=10.0,
            sampling_rate=int(sample_rate),
        ).to(device)
    return _MODELS[key]


def _run_pesto(wav: Path, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torchaudio
    from pesto.core import _predict

    installed = importlib.metadata.version("pesto-pitch")
    if installed != PESTO_VERSION:
        raise RuntimeError(
            f"PESTO {PESTO_VERSION} is required, found {installed}"
        )
    audio, sample_rate = torchaudio.load(str(wav))
    audio = audio.mean(dim=0).to(device)
    with torch.inference_mode():
        _times, frequency, confidence, _activations = _predict(
            audio,
            int(sample_rate),
            _model(int(sample_rate), device),
            num_chunks=2,
        )
    return (
        frequency.detach().cpu().numpy().reshape(-1),
        confidence.detach().cpu().numpy().reshape(-1),
    )


def extract_pesto_features(
    wav_path: Path | str,
    basic_features: BasicPitchFeatures,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    cache_path: Path | str | None = None,
    device: str = "cuda",
    force: bool = False,
) -> np.ndarray:
    """Return ``[T,2]`` written MIDI pitch and confidence."""

    from synthpipeline.pitch_convention import effective_audio_transpose

    wav = Path(wav_path)
    metadata = dict(source_metadata or {})
    frame_times = np.asarray(basic_features.frame_times, dtype=np.float64)
    if cache_path is not None and not force:
        cached = load_pesto_cache(cache_path, wav, metadata, frame_times)
        if cached is not None:
            return cached
    frequency, confidence = _run_pesto(wav, device)
    source_times = np.arange(len(frequency), dtype=np.float64) * 0.01
    if len(source_times):
        frequency = np.interp(
            frame_times, source_times, frequency, left=0.0, right=0.0
        )
        confidence = np.interp(
            frame_times, source_times, confidence, left=0.0, right=0.0
        )
    else:
        frequency = np.zeros(len(frame_times), dtype=np.float32)
        confidence = np.zeros(len(frame_times), dtype=np.float32)
    written_midi = np.zeros(len(frame_times), dtype=np.float32)
    voiced = frequency > 0
    written_midi[voiced] = (
        69.0 + 12.0 * np.log2(frequency[voiced] / 440.0)
        + effective_audio_transpose(metadata)
    )
    feature = np.stack(
        [written_midi, np.clip(confidence, 0.0, 1.0)], axis=-1
    ).astype(np.float32)
    if cache_path is not None:
        save_pesto_cache(
            cache_path,
            feature,
            _cache_metadata(wav, metadata, frame_times),
        )
    return feature


def pesto_cache_path(
    cache_root: Path | str, sample_dir: Path | str, corpus: str
) -> Path:
    return Path(cache_root) / str(corpus) / f"{Path(sample_dir).name}.npz"


def apply_pesto_cents(
    notes: list,
    basic_features: BasicPitchFeatures,
    pesto: np.ndarray,
    *,
    minimum_confidence: float = 0.80,
) -> list:
    """Estimate signed cents relative to each Basic Pitch intended note."""

    from .decode import TransNote

    frame_times = np.asarray(basic_features.frame_times, dtype=np.float64)
    feature = np.asarray(pesto, dtype=np.float32)
    output = []
    for note in notes:
        mask = (
            (frame_times >= float(note.start))
            & (frame_times < float(note.end))
            & (feature[:, 1] >= minimum_confidence)
            & (feature[:, 0] > 0)
        )
        if np.any(mask):
            deviation = 100.0 * float(
                np.median(feature[mask, 0] - int(note.pitch))
            )
            cents = float(np.clip(deviation, -100.0, 100.0))
        else:
            cents = float(note.cents)
        candidates = [int(note.pitch)]
        if abs(cents) >= 35.0:
            direction = 1 if cents > 0 else -1
            candidates.append(int(note.pitch) + direction)
        candidates.extend(int(value) for value in note.pitch_candidates)
        unique = tuple(dict.fromkeys(candidates))[:3]
        output.append(
            TransNote(
                pitch=int(note.pitch),
                start=float(note.start),
                end=float(note.end),
                confidence=float(note.confidence),
                cents=round(cents, 2),
                pitch_candidates=unique,
            )
        )
    return output
