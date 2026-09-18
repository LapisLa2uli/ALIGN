from __future__ import annotations

import numpy as np
import torch

from alignmodel.transcription.mel_v1 import (
    MelNoteTranscriber,
    MelTranscriberConfig,
)
from alignmodel.transcription.ornament_multipitch_v1 import (
    MultiPitchConfig,
    OrnamentMultiPitchTranscriber,
    decode_multipitch_notes,
    make_multipitch_targets,
    multipitch_loss,
)


def test_overlapping_targets_retain_independent_pitch_activity() -> None:
    target = make_multipitch_targets(
        [
            {"pitch": 60, "start_sec": 0.0, "end_sec": 0.20},
            {"pitch": 64, "start_sec": 0.05, "end_sec": 0.15},
        ],
        frames=24,
        hop_sec=0.01,
        midi_min=52,
        midi_max=100,
    )
    assert target["activity"][6, 60 - 52] == 1.0
    assert target["activity"][6, 64 - 52] == 1.0
    assert target["polyphony"][6] == 2


def test_track_b_initialization_and_multipitch_loss() -> None:
    backbone = MelTranscriberConfig(
        conv_channels=8,
        temporal_dim=16,
        spectral_blocks=2,
        temporal_blocks=1,
    )
    track_b = MelNoteTranscriber(backbone)
    model = OrnamentMultiPitchTranscriber(
        MultiPitchConfig(backbone=backbone, max_polyphony=4)
    )
    model.initialize_track_b(track_b)
    mel = torch.randn(2, 128, 32)
    output = model(mel)
    assert output["activity_logits"].shape == (2, 32, 49)
    target = make_multipitch_targets(
        [{"pitch": 60, "start_sec": 0.03, "end_sec": 0.14}],
        frames=32,
        hop_sec=0.01,
        midi_min=52,
        midi_max=100,
    )
    batch = {
        key: torch.from_numpy(value).repeat(
            (2, 1, 1) if value.ndim == 2 else (2, 1)
        )
        for key, value in target.items()
    }
    batch["frame_mask"] = torch.ones(2, 32, dtype=torch.bool)
    loss, parts = multipitch_loss(output, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    loss.backward()


def test_decoder_emits_overlap_and_same_pitch_rearticulation() -> None:
    frames = 30
    pitches = 49
    probability = {
        "activity": np.full((frames, pitches), 0.01, np.float32),
        "onset": np.full((frames, pitches), 0.01, np.float32),
        "offset": np.full((frames, pitches), 0.01, np.float32),
        "polyphony": np.zeros((frames, 4), np.float32),
    }
    probability["polyphony"][:, 0] = 1.0
    c4 = 60 - 52
    e4 = 64 - 52
    probability["activity"][2:24, c4] = 0.95
    probability["activity"][5:15, e4] = 0.90
    probability["onset"][2, c4] = 0.95
    probability["onset"][12, c4] = 0.90
    probability["onset"][5, e4] = 0.90
    probability["offset"][15, e4] = 0.90
    probability["offset"][24, c4] = 0.90
    probability["polyphony"][2:5] = [0.0, 1.0, 0.0, 0.0]
    probability["polyphony"][5:15] = [0.0, 0.0, 1.0, 0.0]
    probability["polyphony"][15:24] = [0.0, 1.0, 0.0, 0.0]
    notes = decode_multipitch_notes(
        probability, midi_min=52, hop_sec=0.01
    )
    assert [value.pitch for value in notes].count(60) == 2
    assert [value.pitch for value in notes].count(64) == 1
    assert any(
        left.start < right.end and right.start < left.end
        for left in notes
        for right in notes
        if left.pitch != right.pitch
    )
