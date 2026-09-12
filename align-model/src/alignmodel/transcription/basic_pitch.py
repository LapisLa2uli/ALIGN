"""Lazy, cacheable Basic Pitch 0.4.0 frontend for written-pitch inference.

Basic Pitch and its optional TensorFlow runtime are imported only when model
inference or official decoding is requested.  Cached arrays are raw frontend
activations (rather than decoded notes), with their pitch axes translated from
the WAV's sounding pitch to ALIGN's written-pitch convention.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

BASIC_PITCH_VERSION = "0.4.0"
FRONTEND_VERSION = "align-basic-pitch-0.4.0-v1"
CACHE_SCHEMA_VERSION = 1

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOT_N_FRAMES = ANNOTATIONS_FPS * 2
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * 2 - FFT_HOP
MIDI_OFFSET = 21
N_NOTE_BINS = 88
CONTOUR_BINS_PER_SEMITONE = 3
N_CONTOUR_BINS = N_NOTE_BINS * CONTOUR_BINS_PER_SEMITONE


@dataclass(frozen=True)
class BasicPitchDecodeConfig:
    """Frozen, calibrated baseline settings for ALIGN clarinet audio."""

    onset_threshold: float = 0.50
    frame_threshold: float = 0.30
    minimum_note_length_ms: float = 55.0
    minimum_frequency: float = 45.0
    maximum_frequency: float = 2600.0
    infer_onsets: bool = True
    melodia_trick: bool = True
    multiple_pitch_bends: bool = True


FROZEN_DECODE_CONFIG = BasicPitchDecodeConfig()


@dataclass(frozen=True)
class BasicPitchFeatures:
    """Written-space Basic Pitch activations and their official time grid."""

    note: np.ndarray
    onset: np.ndarray
    contour: np.ndarray
    frame_times: np.ndarray
    metadata: Mapping[str, Any]

    def model_output(self) -> dict[str, np.ndarray]:
        # Official postprocessing masks frequency ranges in place.
        return {
            "note": self.note.copy(),
            "onset": self.onset.copy(),
            "contour": self.contour.copy(),
        }


_MODEL: Any = None
_MODEL_PID: int | None = None
_MODEL_LOCK = threading.Lock()


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_audio_metadata(sample_dir: Path | str) -> dict[str, Any]:
    path = Path(sample_dir) / "metadata.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def effective_audio_transpose(metadata: Mapping[str, Any] | None) -> int:
    """Return semitones added to detected WAV pitch to recover written pitch."""

    values = metadata or {}
    raw = values.get("effective_audio_transpose")
    if raw is not None:
        return int(raw)
    if values.get("audio_pitch_space") == "written":
        return 0
    sounding_transpose = values.get("sounding_transpose", -2)
    return -int(-2 if sounding_transpose is None else sounding_transpose)


def effective_pitch_policy(
    metadata: Mapping[str, Any] | None,
) -> dict[str, int | str]:
    """Canonical policy included in cache validation metadata."""

    transpose = effective_audio_transpose(metadata)
    return {
        "policy": "effective_audio_transpose_to_written_axes_v1",
        "effective_audio_transpose": transpose,
        "audio_to_written_shift": transpose,
        "contour_bins_per_semitone": CONTOUR_BINS_PER_SEMITONE,
    }


def _shift_last_axis(values: np.ndarray, bins: int) -> np.ndarray:
    array = np.asarray(values)
    output = np.zeros_like(array)
    width = array.shape[-1]
    source_start = max(0, -bins)
    destination_start = max(0, bins)
    count = min(width - source_start, width - destination_start)
    if count > 0:
        output[
            ..., destination_start : destination_start + count
        ] = array[..., source_start : source_start + count]
    return output


def remap_sounding_to_written(
    model_output: Mapping[str, np.ndarray],
    audio_to_written_shift: int,
) -> dict[str, np.ndarray]:
    """Translate note/onset and 3-bin/semitone contour axes without resizing."""

    shift = int(audio_to_written_shift)
    note = np.asarray(model_output["note"])
    onset = np.asarray(model_output["onset"])
    contour = np.asarray(model_output["contour"])
    return {
        "note": _shift_last_axis(note, shift),
        "onset": _shift_last_axis(onset, shift),
        "contour": _shift_last_axis(
            contour, shift * CONTOUR_BINS_PER_SEMITONE
        ),
    }


def model_frame_times(n_frames: int) -> np.ndarray:
    """Reproduce Basic Pitch 0.4.0's corrected model-frame timestamps."""

    frame_numbers = np.arange(max(0, int(n_frames)), dtype=np.float64)
    original_times = frame_numbers * FFT_HOP / AUDIO_SAMPLE_RATE
    window_numbers = np.floor(frame_numbers / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + 0.0018
    return original_times - window_offset * window_numbers


def _installed_basic_pitch_version() -> str | None:
    try:
        return importlib.metadata.version("basic-pitch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _official_modules() -> tuple[Any, Any, Any]:
    """Import Basic Pitch only across an explicit inference/decode boundary."""

    package = importlib.import_module("basic_pitch")
    inference = importlib.import_module("basic_pitch.inference")
    note_creation = importlib.import_module("basic_pitch.note_creation")
    installed = _installed_basic_pitch_version()
    if installed is not None and installed != BASIC_PITCH_VERSION:
        raise RuntimeError(
            f"Basic Pitch {BASIC_PITCH_VERSION} is required, found {installed}"
        )
    return package, inference, note_creation


def get_basic_pitch_model() -> Any:
    """Return the process-local official model, loading it exactly once."""

    global _MODEL, _MODEL_PID
    process_id = os.getpid()
    if _MODEL is not None and _MODEL_PID == process_id:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_PID != process_id:
            package, inference, _note_creation = _official_modules()
            _MODEL = inference.Model(package.ICASSP_2022_MODEL_PATH)
            _MODEL_PID = process_id
    return _MODEL


def _run_official_inference(wav_path: Path) -> Mapping[str, np.ndarray]:
    _package, inference, _note_creation = _official_modules()
    return inference.run_inference(wav_path, get_basic_pitch_model())


def _validate_shapes(
    note: np.ndarray,
    onset: np.ndarray,
    contour: np.ndarray,
    frame_times: np.ndarray,
) -> None:
    if note.ndim != 2 or note.shape[1] != N_NOTE_BINS:
        raise ValueError(f"Expected Basic Pitch note shape [T, 88], got {note.shape}")
    if onset.shape != note.shape:
        raise ValueError(
            f"Expected onset shape {note.shape}, got {onset.shape}"
        )
    if contour.ndim != 2 or contour.shape != (note.shape[0], N_CONTOUR_BINS):
        raise ValueError(
            f"Expected contour shape [{note.shape[0]}, {N_CONTOUR_BINS}], "
            f"got {contour.shape}"
        )
    if frame_times.shape != (note.shape[0],):
        raise ValueError(
            f"Expected {note.shape[0]} frame times, got {frame_times.shape}"
        )


def _expected_cache_metadata(
    wav_path: Path,
    source_metadata: Mapping[str, Any] | None,
    *,
    wav_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "frontend_version": FRONTEND_VERSION,
        "basic_pitch_version": BASIC_PITCH_VERSION,
        "wav_sha256": wav_sha256 or sha256_file(wav_path),
        "pitch_policy": effective_pitch_policy(source_metadata),
        "pitch_space": "written",
    }


def _metadata_text(value: np.ndarray) -> str:
    scalar = np.asarray(value)
    if scalar.shape != ():
        raise ValueError("Cache metadata must be a scalar JSON string")
    return str(scalar.item())


def load_basic_pitch_cache(
    cache_path: Path | str,
    wav_path: Path | str,
    source_metadata: Mapping[str, Any] | None = None,
    *,
    wav_sha256: str | None = None,
) -> BasicPitchFeatures | None:
    """Load a valid cache, returning ``None`` for stale or malformed data."""

    path = Path(cache_path)
    if not path.is_file():
        return None
    expected = _expected_cache_metadata(
        Path(wav_path), source_metadata, wav_sha256=wav_sha256
    )
    try:
        with np.load(path, allow_pickle=False) as saved:
            actual = json.loads(_metadata_text(saved["metadata"]))
            if actual != expected:
                return None
            note = np.asarray(saved["note"], dtype=np.float32)
            onset = np.asarray(saved["onset"], dtype=np.float32)
            contour = np.asarray(saved["contour"], dtype=np.float32)
            frame_times = np.asarray(saved["frame_times"], dtype=np.float64)
        _validate_shapes(note, onset, contour, frame_times)
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return BasicPitchFeatures(note, onset, contour, frame_times, actual)


def save_basic_pitch_cache(
    cache_path: Path | str,
    features: BasicPitchFeatures,
) -> Path:
    """Atomically write a worker-safe NPZ cache."""

    path = Path(cache_path)
    _validate_shapes(
        features.note, features.onset, features.contour, features.frame_times
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{path.name}.", dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                note=np.asarray(features.note, dtype=np.float32),
                onset=np.asarray(features.onset, dtype=np.float32),
                contour=np.asarray(features.contour, dtype=np.float32),
                frame_times=np.asarray(features.frame_times, dtype=np.float64),
                metadata=np.asarray(
                    json.dumps(
                        dict(features.metadata),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            )
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return path


def extract_basic_pitch_features(
    wav_path: Path | str,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    cache_path: Path | str | None = None,
    force: bool = False,
) -> BasicPitchFeatures:
    """Run or reuse the official frontend and return written-space maps."""

    wav = Path(wav_path)
    if not wav.is_file():
        raise FileNotFoundError(wav)
    wav_digest = sha256_file(wav)
    if cache_path is not None and not force:
        cached = load_basic_pitch_cache(
            cache_path,
            wav,
            source_metadata,
            wav_sha256=wav_digest,
        )
        if cached is not None:
            return cached

    raw = _run_official_inference(wav)
    shift = int(effective_pitch_policy(source_metadata)["audio_to_written_shift"])
    written = remap_sounding_to_written(raw, shift)
    note = np.asarray(written["note"], dtype=np.float32)
    onset = np.asarray(written["onset"], dtype=np.float32)
    contour = np.asarray(written["contour"], dtype=np.float32)
    frame_times = model_frame_times(note.shape[0])
    cache_metadata = _expected_cache_metadata(
        wav, source_metadata, wav_sha256=wav_digest
    )
    features = BasicPitchFeatures(
        note, onset, contour, frame_times, cache_metadata
    )
    _validate_shapes(note, onset, contour, frame_times)
    if cache_path is not None:
        save_basic_pitch_cache(cache_path, features)
    return features


def extract_sample_basic_pitch_features(
    sample_dir: Path | str,
    *,
    cache_path: Path | str | None = None,
    force: bool = False,
) -> BasicPitchFeatures:
    sample = Path(sample_dir)
    return extract_basic_pitch_features(
        sample / "performance_audio.wav",
        source_metadata=load_audio_metadata(sample),
        cache_path=cache_path,
        force=force,
    )


def basic_pitch_cache_path(
    cache_root: Path | str,
    sample_dir: Path | str,
    corpus: str = "unknown",
) -> Path:
    component = str(corpus).strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if component in {"", ".", ".."}:
        component = "unknown"
    return Path(cache_root) / component / f"{Path(sample_dir).name}.npz"


def decode_basic_pitch_features(
    features: BasicPitchFeatures,
    config: BasicPitchDecodeConfig = FROZEN_DECODE_CONFIG,
) -> list[Any]:
    """Decode written-space maps to ``TransNote`` using official postprocessing."""

    from .decode import TransNote

    _package, _inference, note_creation = _official_modules()
    minimum_frames = int(
        round(
            config.minimum_note_length_ms
            / 1000.0
            * (AUDIO_SAMPLE_RATE / FFT_HOP)
        )
    )
    _midi, events = note_creation.model_output_to_notes(
        features.model_output(),
        onset_thresh=config.onset_threshold,
        frame_thresh=config.frame_threshold,
        infer_onsets=config.infer_onsets,
        min_note_len=minimum_frames,
        min_freq=config.minimum_frequency,
        max_freq=config.maximum_frequency,
        include_pitch_bends=True,
        multiple_pitch_bends=config.multiple_pitch_bends,
        melodia_trick=config.melodia_trick,
    )
    notes = []
    for start, end, pitch, amplitude, bends in events:
        cents = 0.0
        if bends is not None and len(bends):
            cents = float(np.median(np.asarray(bends, dtype=np.float32)))
            cents *= 100.0 / CONTOUR_BINS_PER_SEMITONE
        notes.append(
            TransNote(
                pitch=int(pitch),
                start=float(start),
                end=float(end),
                confidence=float(amplitude),
                cents=round(cents, 2),
                pitch_candidates=(int(pitch),),
            )
        )
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end))


def decode_frozen_basic_pitch(
    features: BasicPitchFeatures,
) -> list[Any]:
    """Decode the immutable calibrated baseline used for comparisons."""

    return decode_basic_pitch_features(features, FROZEN_DECODE_CONFIG)


def transcribe_with_basic_pitch(
    sample_dir: Path | str,
    *,
    cache_path: Path | str | None = None,
    force: bool = False,
) -> list[Any]:
    return decode_frozen_basic_pitch(
        extract_sample_basic_pitch_features(
            sample_dir, cache_path=cache_path, force=force
        )
    )
