from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from alignmodel.transcription.decode import TransNote
from alignmodel.transcription.refiner import (
    NoteRefiner,
    NoteRefinerConfig,
    decode_refined_notes,
    load_note_refiner,
    note_refiner_loss,
    save_note_refiner,
)
from alignmodel.transcription.semi_crf import (
    IntervalCandidates,
    semi_crf_log_partition,
    semi_crf_nll,
    weighted_interval_decode,
)
from alignmodel.transcription.refiner_data import (
    RefinerAugmentConfig,
    augment_cached_refiner_maps,
)


class NoteRefinerModelTests(unittest.TestCase):
    def _config(self) -> NoteRefinerConfig:
        return NoteRefinerConfig(
            midi_min=58,
            midi_max=64,
            channels=16,
            temporal_blocks=2,
            top_pitch_candidates=3,
            min_note_frames=4,
            max_note_frames=30,
            max_boundaries=32,
        )

    def _inputs(self, batch: int = 2, frames: int = 24):
        note = torch.rand(batch, frames, 88)
        onset = torch.rand(batch, frames, 88)
        contour = torch.rand(batch, frames, 264)
        pesto = torch.zeros(batch, frames, 2)
        pesto[..., 0] = 233.08  # sounding Bb3; +2 is written C4
        pesto[..., 1] = 0.8
        return note, onset, contour, pesto

    def test_output_shapes_and_finite_loss(self) -> None:
        cfg = self._config()
        model = NoteRefiner(cfg)
        outputs = model(*self._inputs())
        self.assertEqual(tuple(outputs["voiced_logits"].shape), (2, 24))
        self.assertEqual(tuple(outputs["onset_logits"].shape), (2, 24))
        self.assertEqual(tuple(outputs["offset_logits"].shape), (2, 24))
        self.assertEqual(tuple(outputs["pitch_logits"].shape), (2, 24, 7))
        self.assertEqual(tuple(outputs["boundary_correction"].shape), (2, 24, 2))
        self.assertEqual(tuple(outputs["confidence"].shape), (2, 24))
        self.assertEqual(tuple(outputs["cents"].shape), (2, 24))
        self.assertEqual(tuple(outputs["pitch_candidates"].shape), (2, 24, 3))

        targets = {
            "voiced": torch.zeros(2, 24),
            "onset": torch.zeros(2, 24),
            "offset": torch.zeros(2, 24),
            "pitch": torch.full((2, 24), -1, dtype=torch.long),
            "cents": torch.zeros(2, 24),
            "frame_mask": torch.ones(2, 24, dtype=torch.bool),
            "frame_weight": torch.ones(2, 24),
            "onset_weight": torch.ones(2, 24),
            "offset_weight": torch.ones(2, 24),
            "intervals": [[(4, 16, 60)], [(4, 16, 60)]],
        }
        targets["voiced"][:, 4:16] = 1
        targets["onset"][:, 4] = 1
        targets["offset"][:, 16] = 1
        targets["pitch"][:, 4:16] = 60  # written MIDI, not a local class
        targets["frame_weight"][:, 4:16] = 2.0
        targets["onset_weight"][:, 4] = 2.0
        loss, parts = note_refiner_loss(outputs, targets, cfg)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
        self.assertGreaterEqual(float(parts["interval"]), 0.0)
        loss.backward()
        self.assertIsNotNone(model.pitch_residual_head.weight.grad)

    def test_signed_cents_and_top_candidates(self) -> None:
        cfg = self._config()
        model = NoteRefiner(cfg)
        note, onset, contour, pesto = self._inputs(batch=1, frames=8)
        note.zero_()
        onset.zero_()
        contour.zero_()
        axis = 60 - cfg.basic_midi_min
        note[..., axis] = 0.95
        onset[:, 1, axis] = 0.95
        # Lower contour sub-bin gives a negative base cents estimate.
        contour[..., axis * 3] = 1.0
        pesto[..., 1] = 0.0
        outputs = model(note, onset, contour, pesto)
        self.assertTrue(bool((outputs["cents"] < 0).all()))
        self.assertEqual(int(outputs["pitch_candidates"][0, 3, 0]), 60)
        self.assertTrue(
            bool(
                (outputs["pitch_candidate_scores"] >= 0).all()
                & (outputs["pitch_candidate_scores"] <= 1).all()
            )
        )

    def test_checkpoint_config_roundtrip(self) -> None:
        cfg = self._config()
        cfg.pesto_written_shift = 2.5
        model = NoteRefiner(cfg)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "refiner.pt"
            save_note_refiner(path, model, extra={"epoch": 3})
            loaded, extra = load_note_refiner(path)
        self.assertEqual(loaded.config.to_dict(), cfg.to_dict())
        self.assertEqual(extra["epoch"], 3)
        self.assertEqual(
            set(loaded.state_dict()),
            set(model.state_dict()),
        )

    def test_difficult_timbre_augmentation_balances_short_notes(self) -> None:
        frames = 32
        note = np.full((frames, 88), 0.40, np.float32)
        onset = np.zeros((frames, 88), np.float32)
        contour = np.full((frames, 264), 0.30, np.float32)
        pesto = np.ones((frames, 2), np.float32)
        voiced = np.zeros(frames, np.float32)
        voiced[2:8] = 1.0
        voiced[10:26] = 1.0
        config = RefinerAugmentConfig(
            probability=1.0,
            band_attenuation_probability=1.0,
            filtered_timbre_probability=0.0,
            short_note_attenuation_probability=1.0,
            same_pitch_split_probability=1.0,
            hard_negative_ratio=1.0,
        )
        _note, _onset, _contour, _pesto, metadata = (
            augment_cached_refiner_maps(
                note,
                onset,
                contour,
                pesto,
                intervals=[(2, 8, 60), (10, 26, 60)],
                voiced_target=voiced,
                config=config,
                rng=np.random.default_rng(7),
            )
        )
        self.assertEqual(metadata["short_positive_count"], 1)
        self.assertEqual(metadata["hard_negative_count"], 1)
        self.assertEqual(metadata["artificial_split_count"], 1)
        self.assertGreater(float(np.max(metadata["frame_weight"])), 1.0)
        self.assertEqual(
            RefinerAugmentConfig(**asdict(config)),
            config,
        )


class SparseSemiCRFTests(unittest.TestCase):
    def test_log_partition_nll_has_gradients(self) -> None:
        scores = torch.tensor([1.2, 0.1, 1.4], requires_grad=True)
        graph = IntervalCandidates(
            starts=torch.tensor([0, 0, 4]),
            ends=torch.tensor([4, 8, 8]),
            pitches=torch.tensor([2, 3, 2]),
            scores=scores,
            num_frames=8,
        )
        partition = semi_crf_log_partition(graph)
        loss = semi_crf_nll(graph, [0, 2])
        self.assertTrue(torch.isfinite(partition))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)

    def test_repeated_same_pitch_edges_are_preserved(self) -> None:
        graph = IntervalCandidates(
            starts=torch.tensor([0, 4, 0]),
            ends=torch.tensor([4, 8, 8]),
            pitches=torch.tensor([2, 2, 2]),
            scores=torch.tensor([3.0, 3.0, 5.0]),
            num_frames=8,
        )
        self.assertEqual(weighted_interval_decode(graph), [0, 1])

    def test_decode_is_non_overlapping_and_respects_minimum_duration(self) -> None:
        cfg = NoteRefinerConfig(
            midi_min=58,
            midi_max=62,
            channels=16,
            temporal_blocks=1,
            min_note_frames=4,
            max_note_frames=20,
            boundary_threshold=0.5,
            max_boundaries=16,
            hop_sec=0.01,
        )
        frames, pitches = 20, cfg.n_pitches
        outputs = {
            "voiced_logits": torch.full((1, frames), 8.0),
            "onset_logits": torch.full((1, frames), -8.0),
            "offset_logits": torch.full((1, frames), -8.0),
            "pitch_logits": torch.full((1, frames, pitches), -8.0),
            "boundary_correction": torch.zeros(1, frames, 2),
            "confidence": torch.full((1, frames), 0.9),
            "cents": torch.full((1, frames), 17.0),
        }
        outputs["onset_logits"][0, 2] = 8.0
        outputs["onset_logits"][0, 10] = 8.0
        outputs["offset_logits"][0, 10] = 8.0
        outputs["offset_logits"][0, 18] = 8.0
        outputs["pitch_logits"][0, :, 2] = 8.0
        notes = decode_refined_notes(outputs, cfg)
        self.assertEqual(len(notes), 2)
        self.assertTrue(all(isinstance(note, TransNote) for note in notes))
        self.assertEqual([note.pitch for note in notes], [60, 60])
        self.assertLessEqual(notes[0].end, notes[1].start)
        self.assertTrue(all(note.end - note.start >= 0.04 for note in notes))
        self.assertTrue(all(note.cents == 17.0 for note in notes))
        self.assertTrue(all(note.pitch_candidates[0] == 60 for note in notes))


if __name__ == "__main__":
    unittest.main()
