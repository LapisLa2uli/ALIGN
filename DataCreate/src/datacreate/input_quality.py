"""Conservative original-WAV quality check, before resampling or inference.

Passing means this specific stationary noise pattern was not detected; it does
not establish the presence of music, valid alignment, or a healthy microphone.
"""
from pathlib import Path
import wave

import numpy as np
from scipy.signal import welch

REVISION = 'stationary-broadband-v1'
RECORD_AGAIN = ('This recording contains mostly steady broadband noise. '
                'Check the microphone connection and record again.')
THRESHOLDS = {
    'minimum_sample_rate': 32000,
    'minimum_seconds': 5,
    'minimum_rms_pcm16': 1.0,
    'maximum_rms_spread_db': 1.0,
    'minimum_broadband_flatness': 0.90,
    'minimum_high_frequency_fraction': 0.40,
    'minimum_band_flatness': 0.95,
    'maximum_tonal_peak_ratio': 5.0,
}


def _flatness(power):
    return float(np.exp(np.mean(np.log(np.maximum(power, 1e-30))))
                 / max(float(np.mean(power)), 1e-30))


def _block_metrics(audio, sr):
    frequencies, power = welch(audio, fs=sr, nperseg=2048, noverlap=1024)
    band = power[(frequencies >= 300) & (frequencies <= 8000)]
    return (float(np.sqrt(np.mean(audio ** 2))),
            _flatness(power[frequencies >= 80]),
            float(power[frequencies >= 10000].sum() / max(float(power.sum()), 1e-30)),
            _flatness(band),
            float(band.max() / max(float(np.median(band)), 1e-30)))


def assess_recording(path: Path) -> dict:
    """Read bounded one-second blocks of the original mono PCM16 WAV.

    Include a final overlapping full-second block when needed so music at the
    end is never silently discarded. Use that block to veto rejection, not to
    bias the distribution of full-block RMS values.
    """
    with wave.open(str(path), 'rb') as recording:
        sr, frames = recording.getframerate(), recording.getnframes()
        if recording.getnchannels() != 1 or recording.getsampwidth() != 2 or recording.getcomptype() != 'NONE':
            raise ValueError('Input quality requires a mono PCM16 WAV.')
        result = {'revision': REVISION, 'status': 'not_assessed',
                  'reason': 'unsupported_sample_rate' if sr < THRESHOLDS['minimum_sample_rate'] else 'too_short',
                  'sample_rate': sr, 'duration_seconds': frames / sr,
                  'thresholds': dict(THRESHOLDS), 'metrics': {},
                  'message': 'Outside the scope of the stationary broadband noise check.'}
        if sr < THRESHOLDS['minimum_sample_rate'] or frames < sr * THRESHOLDS['minimum_seconds']:
            return result
        rows, sum_squares = [], 0.0
        for start in range(0, frames, sr):
            count = min(sr, frames - start)
            raw = recording.readframes(count)
            if len(raw) != count * 2:
                raise ValueError('Incomplete WAV samples during input quality check.')
            audio = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
            sum_squares += float(np.dot(audio, audio))
            if count == sr:
                rows.append(_block_metrics(audio, sr))
        tail = None
        if frames % sr:
            recording.setpos(frames - sr)
            raw = recording.readframes(sr)
            if len(raw) != sr * 2:
                raise ValueError('Incomplete WAV tail during input quality check.')
            tail = _block_metrics(np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0, sr)
    values = np.array(rows)
    spread = float(20 * np.log10(max(float(np.percentile(values[:, 0], 95)), 1e-15)
                                / max(float(np.percentile(values[:, 0], 5)), 1e-15)))
    rms = float(np.sqrt(sum_squares / frames) * 32768)
    flat, high, band = (float(np.percentile(values[:, i], 10)) for i in (1, 2, 3))
    peak = float(values[:, 4].max())
    tail_consistent = True
    if tail is not None:
        # A changed ending (tone, reduced noise or level change) is enough to
        # spare the recording from this deliberately narrow rejection rule.
        tail_db = abs(float(20 * np.log10(max(tail[0], 1e-15)
                                         / max(float(np.median(values[:, 0])), 1e-15))))
        tail_consistent = (tail_db < THRESHOLDS['maximum_rms_spread_db']
                           and tail[1] > THRESHOLDS['minimum_broadband_flatness']
                           and tail[2] > THRESHOLDS['minimum_high_frequency_fraction']
                           and tail[3] > THRESHOLDS['minimum_band_flatness'])
        peak = max(peak, tail[4])
    rejected = (rms > THRESHOLDS['minimum_rms_pcm16']
                and spread < THRESHOLDS['maximum_rms_spread_db']
                and flat > THRESHOLDS['minimum_broadband_flatness']
                and high > THRESHOLDS['minimum_high_frequency_fraction']
                and band > THRESHOLDS['minimum_band_flatness']
                and peak < THRESHOLDS['maximum_tonal_peak_ratio'] and tail_consistent)
    result.update(status='rejected' if rejected else 'passed',
                  reason='stationary_broadband_noise' if rejected else 'noise_pattern_not_detected',
                  message=RECORD_AGAIN if rejected else 'Stationary broadband noise pattern not detected.',
                  metrics={'rms_pcm16': rms, 'rms_spread_db_p95_p05': spread,
                           'broadband_flatness_p10': flat, 'high_frequency_fraction_p10': high,
                           'band_flatness_p10': band, 'tonal_peak_ratio_max': peak,
                           'full_blocks': len(rows), 'tail_checked': tail is not None,
                           'tail_consistent': bool(tail_consistent)})
    return result
