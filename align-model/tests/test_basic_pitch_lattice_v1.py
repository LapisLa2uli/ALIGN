from __future__ import annotations

import numpy as np

from alignmodel.transcription.basic_pitch import BasicPitchFeatures
from alignmodel.transcription.basic_pitch_lattice_v1 import (
    generate_activation_lattice,
)


def _features() -> BasicPitchFeatures:
    frames = 30
    note = np.zeros((frames, 88), np.float32)
    onset = np.zeros_like(note)
    contour = np.zeros((frames, 264), np.float32)
    axis = 60 - 21
    note[5:10, axis] = 0.12
    onset[5, axis] = 0.09
    contour[5:10, axis * 3 : axis * 3 + 3] = 0.11
    return BasicPitchFeatures(
        note,
        onset,
        contour,
        np.arange(frames, dtype=np.float64) * (256 / 22050),
        {},
    )


def test_lattice_recovers_weak_short_activation() -> None:
    candidates = generate_activation_lattice(_features(), max_candidates=128)
    weak = [value for value in candidates if value.pitch == 60]
    assert weak
    assert any(value.duration_frames in {2, 4, 5, 8} for value in weak)
    assert min(value.start for value in weak) <= 5 * 256 / 22050 + 1e-9


def test_lattice_is_deterministic_and_score_free() -> None:
    first = generate_activation_lattice(_features(), max_candidates=64)
    second = generate_activation_lattice(_features(), max_candidates=64)
    assert first == second
    assert len(first) <= 64
