"""Preserve MuScriptor's original five-second chunk grid across GUI trims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REVISION = 'muscriptor-full-context-v1'


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def context_signature(sample: Path) -> dict:
    """Inputs affecting context selection, including a missing/added backup."""
    metadata = sample / 'metadata.json'
    trim = json.loads(metadata.read_text(encoding='utf-8')).get('performance_trim') if metadata.is_file() else None
    backup = sample / 'performance_audio_full.wav'
    return {
        'context_revision': REVISION,
        'trim_sha256': hashlib.sha256(json.dumps(trim, sort_keys=True).encode()).hexdigest(),
        'full_audio_sha256': file_hash(backup) if backup.is_file() else None,
    }


def transcription_source(sample: Path) -> tuple[Path, dict]:
    """Use full audio only after verifying that the current WAV is its slice.

    Older trim metadata is rounded to 0.1 ms. Try adjacent samples, then confirm
    the entire slice within PCM rounding tolerance. A stale backup must never
    silently substitute different audio. Unverifiable context falls back to the
    current recording, with the reason retained in provenance.
    """
    current = sample / 'performance_audio.wav'
    full = sample / 'performance_audio_full.wav'
    info = {**context_signature(sample), 'source_audio': current.name,
            'source_audio_sha256': file_hash(current), 'mode': 'current_audio',
            'crop_start': 0.0, 'crop_end': None}
    if not full.is_file():
        info['reason'] = 'no_full_recording'
        return current, info
    metadata_path = sample / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    trim = metadata.get('performance_trim') or {}
    if trim.get('trim_start') is None:
        info['reason'] = 'no_trim_mapping'
        return current, info
    try:
        from scipy.io import wavfile
        read_wav = wavfile.read
    except ImportError:
        # MuScriptor's isolated runtime intentionally does not depend on SciPy.
        # DataCreate WAVs are PCM, so keep context verification available there
        # without changing the normal SciPy-backed path.
        import wave

        def read_wav(path):
            with wave.open(str(path), 'rb') as stream:
                channels = stream.getnchannels()
                rate = stream.getframerate()
                width = stream.getsampwidth()
                raw = stream.readframes(stream.getnframes())
            if width == 1:
                values = np.frombuffer(raw, dtype=np.uint8)
            elif width == 2:
                values = np.frombuffer(raw, dtype='<i2')
            elif width == 3:
                packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                values = (packed[:, 0].astype(np.int32)
                          | (packed[:, 1].astype(np.int32) << 8)
                          | (packed[:, 2].astype(np.int32) << 16))
                values = np.where(values >= (1 << 23), values - (1 << 24), values)
            elif width == 4:
                values = np.frombuffer(raw, dtype='<i4')
            else:
                raise ValueError(f'Unsupported WAV sample width: {width}')
            if channels > 1:
                values = values.reshape(-1, channels)
            return rate, values
    try:
        sr, original = read_wav(full)
        current_sr, cropped = read_wav(current)
    except (OSError, ValueError) as error:
        info['reason'] = 'unreadable_context: ' + str(error)
        return current, info
    if sr != current_sr or original.shape[1:] != cropped.shape[1:] or not len(cropped):
        info['reason'] = 'incompatible_audio_shape'
        return current, info
    start_seconds = float(trim.get('source_start', trim['trim_start']))
    if not np.isfinite(start_seconds) or start_seconds < 0:
        info['reason'] = 'invalid_trim_mapping'
        return current, info
    center = int(start_seconds * sr)
    radius = max(2, int(np.ceil(sr * .0001)))

    def normalized(array):
        if array.dtype == np.uint8:
            return (array.astype(np.float32) - 128) / 128
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else 1
        return array.astype(np.float32) / scale

    target = normalized(cropped)
    best = None
    for start in range(max(0, center-radius), center+radius+1):
        end = start+len(cropped)
        if end > len(original):
            continue
        delta = float(np.max(np.abs(normalized(original[start:end])-target)))
        if np.isfinite(delta) and (best is None or delta < best[0]):
            best = delta, start, end
    if best is None or best[0] > 2 / 32768:
        info['reason'] = 'full_recording_does_not_match_trim'
        return current, info
    delta, start, end = best
    info.update(source_audio=full.name, source_audio_sha256=info['full_audio_sha256'],
                mode='verified_full_recording', crop_start=start/sr, crop_end=end/sr,
                source_start_sample=start, source_end_sample=end, sample_rate=sr,
                max_slice_difference=delta, reason='verified_waveform_slice')
    return full, info

