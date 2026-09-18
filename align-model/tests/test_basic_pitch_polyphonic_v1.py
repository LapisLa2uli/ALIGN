from __future__ import annotations

import numpy as np
import torch

from alignmodel.transcription.basic_pitch_polyphonic_v1 import (
    BasicPitchPolyphonicConfig,
    BasicPitchPolyphonicRefiner,
    basic_pitch_refiner_loss,
    make_basic_pitch_targets,
)


def test_zero_initialized_refiner_preserves_basic_maps() -> None:
    model = BasicPitchPolyphonicRefiner(
        BasicPitchPolyphonicConfig(hidden=16, blocks=1)
    )
    note = torch.rand(2, 20, 88).clamp(0.01, 0.99)
    onset = torch.rand(2, 20, 88).clamp(0.01, 0.99)
    contour = torch.rand(2, 20, 264)
    output = model(note, onset, contour)
    start = model.config.axis_start
    stop = start + model.config.n_pitches
    assert torch.allclose(
        output["activity_logits"].sigmoid(), note[:, :, start:stop]
    )
    assert torch.allclose(
        output["onset_logits"].sigmoid(), onset[:, :, start:stop]
    )


def test_targets_preserve_overlapping_pitches_and_short_weight() -> None:
    times = np.arange(30, dtype=np.float64) * 0.01
    target = make_basic_pitch_targets(
        [
            {"pitch": 60, "start_sec": 0.02, "end_sec": 0.18},
            {"pitch": 64, "start_sec": 0.05, "end_sec": 0.10},
        ],
        times,
        midi_min=52,
        midi_max=100,
    )
    assert target["activity"][6, 60 - 52] == 1.0
    assert target["activity"][6, 64 - 52] == 1.0
    assert target["duration_weight"][6, 64 - 52] == 5.0


def test_refiner_loss_is_finite_and_differentiable() -> None:
    model = BasicPitchPolyphonicRefiner(
        BasicPitchPolyphonicConfig(hidden=16, blocks=1)
    )
    note = torch.rand(2, 20, 88).clamp(0.01, 0.99)
    onset = torch.rand(2, 20, 88).clamp(0.01, 0.99)
    contour = torch.rand(2, 20, 264)
    output = model(note, onset, contour)
    target = make_basic_pitch_targets(
        [{"pitch": 60, "start_sec": 0.03, "end_sec": 0.14}],
        np.arange(20, dtype=np.float64) * 0.01,
        midi_min=52,
        midi_max=100,
    )
    batch = {
        key: torch.from_numpy(value).repeat(2, 1, 1)
        for key, value in target.items()
    }
    batch["frame_mask"] = torch.ones(2, 20, dtype=torch.bool)
    loss, _parts = basic_pitch_refiner_loss(output, batch)
    assert torch.isfinite(loss)
    loss.backward()
