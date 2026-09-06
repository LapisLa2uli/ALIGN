import numpy as np

from synthpipeline.transpose_audio import pitch_shift_duration_preserving


def _peak_hz(audio: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sr)
    return float(freqs[int(np.argmax(spec))])


def test_pitch_shift_preserves_duration_and_drops_two_semitones() -> None:
    sr = 22050
    seconds = 1.5
    freq = 440.0
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    tone = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    shifted = pitch_shift_duration_preserving(tone, sr, -2)
    assert shifted.shape == tone.shape
    expected = freq * (2.0 ** (-2.0 / 12.0))
    assert abs(_peak_hz(shifted, sr) - expected) < 8.0
