import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from synthpipeline.config import PACKAGE_ROOT, SynthConfig
from synthpipeline.degrade import best_lag, degrade_audio, micro_variation, tonguing

SR = 22050


def _notes(seconds: float = 6.0, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Clarinet-like tones with gaps; returns audio and onset samples."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(seconds * SR))
    onsets = []
    t = 0.25
    while t < seconds - 0.6:
        f0 = 147.0 * 2 ** (rng.integers(0, 24) / 12.0)
        n = int(0.35 * SR)
        k = np.arange(n) / SR
        tone = sum((0.6 ** h) * np.sin(2 * np.pi * f0 * (2 * h + 1) * k) for h in range(5))
        env = np.minimum(1.0, k / 0.02) * np.minimum(1.0, (0.35 - k) / 0.05)
        start = int(t * SR)
        x[start : start + n] += 0.3 * tone * env
        onsets.append(start)
        t += 0.5
    return x, onsets


def _realistic_params() -> dict:
    return SynthConfig.load(PACKAGE_ROOT / "config" / "realistic_10k_random.yaml").degrade


def test_degrade_keeps_length_and_timing():
    if shutil.which("ffmpeg") is None:
        from synthpipeline.musesounds import find_ffmpeg

        try:
            find_ffmpeg()
        except FileNotFoundError:
            pytest.skip("ffmpeg not available for the codec stage")
    x, _ = _notes()
    y, info = degrade_audio(x, SR, _realistic_params(), np.random.default_rng(3))
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert abs(best_lag(x, y.astype(np.float64), max_lag=int(0.1 * SR))) <= 2
    assert "codec" in info and abs(info["codec"]["lag_samples"]) < int(0.1 * SR)


def test_degrade_is_deterministic_for_a_seed():
    x, _ = _notes(3.0)
    params = dict(_realistic_params())
    params.pop("codec", None)
    a, ia = degrade_audio(x, SR, params, np.random.default_rng(11))
    b, ib = degrade_audio(x, SR, params, np.random.default_rng(11))
    assert np.array_equal(a, b)
    assert ia == ib


def test_micro_variation_clock_drift_is_tiny():
    x, _ = _notes(20.0)
    stage = {"pitch_cents": 8, "pitch_rate_hz": 3, "shimmer_db": 0, "shimmer_rate_hz": 8}
    y, info = micro_variation(np.random.default_rng(0), x, SR, stage)
    assert y.shape == x.shape
    # Far below one 23 ms mel frame, and bounded regardless of clip length.
    assert info["max_drift_ms"] < 2.5


def test_tonguing_leaves_onsets_untouched():
    x, onsets = _notes()
    stage = {"depth_db": 18, "width_ms": 30, "share": 1.0}
    y, info = tonguing(np.random.default_rng(0), x, SR, stage)
    assert info["dips"] > 0
    ratio = np.ones_like(x)
    nz = np.abs(x) > 1e-6
    ratio[nz] = y[nz] / x[nz]
    # Attack region after each onset keeps its level.
    for onset in onsets:
        seg = slice(onset + 256, onset + 1024)
        assert np.allclose(ratio[seg], 1.0, atol=1e-6)
    assert ratio.min() < 10 ** (-6 / 20)


def _tiny_bundle(tmp_path: Path, render: str) -> Path:
    from datacreate.audio_utils import save_wav

    d = tmp_path / "synth_gen_0001"
    d.mkdir()
    x, _ = _notes(2.0)
    save_wav(d / "performance_audio.wav", x, SR)
    save_wav(d / "reference_audio.wav", x, SR)
    (d / "metadata.json").write_text(json.dumps({"audio_render": render}), encoding="utf-8")
    return d


def test_bundle_skips_unrendered_and_done(tmp_path: Path):
    from synthpipeline.degrade_bundles import degrade_bundle

    d = _tiny_bundle(tmp_path, "soundfont_v1")
    params = {"version": "realistic_v1"}
    assert degrade_bundle(d, params) == "skip_render"
    meta = {"audio_render": "musesounds_v1", "audio_degrade": "realistic_v1"}
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    assert degrade_bundle(d, params) == "skip_done"
    assert not (d / "performance_audio_clean.wav").exists()
