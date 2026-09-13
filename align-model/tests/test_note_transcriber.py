from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from alignmodel.transcription import (
    DecodeConfig,
    NoteCropDataset,
    NoteFrameNet,
    NoteFrameNetConfig,
    NoteTrainConfig,
    TransNote,
    decode_notes,
    evaluate_note_lists,
    infer_full_clip,
    infer_sample_notes,
    load_split,
    load_written_notes,
    load_written_notes_with_cents,
    match_notes,
    note_frame_loss,
    train_note_transcriber,
)
from alignmodel.transcription.data import (
    TranscriptionExample,
    _load_written_notes_cached,
    make_crop_targets,
)
from alignmodel.transcription.decode import (
    apply_spectral_pitch,
    infer_probabilities,
    load_note_transcriber,
)


class NoteTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        _load_written_notes_cached.cache_clear()

    def _bundle(self, root: Path, name: str = "clip") -> Path:
        sample = root / name
        sample.mkdir()
        np.save(sample / "performance_mel.npy", np.full((128, 20), -40, np.float32))
        (sample / "performance_audio.mid").write_bytes(b"MThd")
        return sample

    def test_midi_sounding_transpose_recovers_written_plus_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = self._bundle(Path(temp))
            (sample / "metadata.json").write_text(
                json.dumps({"sounding_transpose": -2}), encoding="utf-8"
            )
            with patch(
                "synthpipeline.timing.midi_note_times",
                return_value=[(58, 0.10, 0.30)],
            ):
                self.assertEqual(
                    load_written_notes(sample, allow_legacy_midi=True),
                    [(60, 0.1, 0.3)],
                )

    def test_legacy_soundfont_midi_is_already_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = self._bundle(Path(temp))
            (sample / "metadata.json").write_text(
                json.dumps(
                    {
                        "sounding_transpose": -2,
                        "audio_render": "soundfont_rerender",
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "synthpipeline.timing.midi_note_times",
                return_value=[(60, 0.10, 0.30)],
            ):
                self.assertEqual(
                    load_written_notes(sample, allow_legacy_midi=True),
                    [(60, 0.1, 0.3)],
                )

    def test_intonation_keeps_written_pitch_and_records_cents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = self._bundle(Path(temp))
            (sample / "metadata.json").write_text(
                json.dumps({"sounding_transpose": -2, "midi_pitch_space": "sounding"}),
                encoding="utf-8",
            )
            (sample / "labels.json").write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "type": "intonation_error",
                                "start_time": 0.08,
                                "end_time": 0.35,
                                "deviation_cents": -55.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "synthpipeline.timing.midi_note_times",
                return_value=[(58, 0.10, 0.30)],
            ):
                self.assertEqual(
                    load_written_notes_with_cents(
                        sample, allow_legacy_midi=True
                    ),
                    [(60, 0.1, 0.3, -55.0)],
                )

    def test_manifest_resolves_multiple_roots_and_dataset_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root_a, root_b = base / "E_a", base / "E_b"
            root_a.mkdir()
            root_b.mkdir()
            train = self._bundle(root_a, "same_id")
            self._bundle(root_b, "same_id")
            val = self._bundle(root_b, "val_id")
            manifest = base / "split.json"
            manifest.write_text(
                json.dumps(
                    {
                        "train": [{"sample": "same_id", "root": 0}],
                        "val": [{"sample": "val_id", "root": 1}],
                    }
                ),
                encoding="utf-8",
            )
            train_rows = load_split([root_a, root_b], manifest, "train")
            val_rows = load_split([root_a, root_b], manifest, "val")
            self.assertEqual(train_rows[0].sample_dir, train)
            self.assertEqual(val_rows[0].sample_dir, val)
            (train / "note_map.json").write_text(
                json.dumps(
                    {
                        "rendered_notes": [
                            {
                                "pitch_midi_written": 60,
                                "start_sec": 0.05,
                                "end_sec": 0.20,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "synthpipeline.timing.midi_note_times",
                return_value=[(58, 0.05, 0.20)],
            ):
                dataset = NoteCropDataset(
                    [TranscriptionExample(train, "train")],
                    crop_frames=32,
                    training=False,
                    augment=False,
                )
                item = dataset[0]
            self.assertEqual(tuple(item["mel"].shape), (128, 32))
            self.assertEqual(int(item["frame_mask"].sum()), 20)
            self.assertGreater(int(item["voiced"].sum()), 0)
            self.assertIn(60 - 36, item["pitch"].tolist())
            self.assertIn("cents", item)
            self.assertEqual(tuple(item["f0"].shape), (2, 32))

    def test_crop_targets_keep_written_pitch_and_cents(self) -> None:
        targets = make_crop_targets(
            [(60, 0.10, 0.30, -40.0)],
            total_frames=20,
            crop_start=0,
            crop_frames=20,
        )
        voiced = np.flatnonzero(targets["voiced"] > 0)
        self.assertGreater(voiced.size, 0)
        self.assertTrue(np.all(targets["pitch"][voiced] == 60 - 36))
        self.assertTrue(np.allclose(targets["cents"][voiced], -40.0))


class NoteModelTests(unittest.TestCase):
    def test_heads_and_multitask_loss(self) -> None:
        cfg = NoteFrameNetConfig(
            channels=16, temporal_blocks=2, midi_min=58, midi_max=62
        )
        model = NoteFrameNet(cfg)
        mel = torch.randn(2, 128, 40)
        outputs = model(mel)
        self.assertEqual(tuple(outputs["voiced_logits"].shape), (2, 40))
        self.assertEqual(tuple(outputs["pitch_logits"].shape), (2, 40, 5))
        self.assertEqual(tuple(outputs["cents"].shape), (2, 40))
        batch = {
            "voiced": torch.zeros(2, 40),
            "pitch": torch.full((2, 40), -1, dtype=torch.long),
            "onset": torch.zeros(2, 40),
            "offset": torch.zeros(2, 40),
            "cents": torch.zeros(2, 40),
            "frame_mask": torch.ones(2, 40, dtype=torch.bool),
        }
        batch["voiced"][:, 5:15] = 1
        batch["pitch"][:, 5:15] = 2
        batch["onset"][:, 5] = 1
        batch["offset"][:, 15] = 1
        loss, parts = note_frame_loss(outputs, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["pitch"], 0)
        self.assertIn("cents", parts)
        self.assertLess(float(parts["cents"]), 2.0)
        loss.backward()

    def test_v2_keeps_sixteen_spectral_positions(self) -> None:
        model = NoteFrameNet(NoteFrameNetConfig())
        self.assertEqual(model.reduced_mels, 16)
        self.assertEqual(model.config.spectral_blocks, 3)
        self.assertEqual(len(model.temporal), 10)
        outputs = model(torch.randn(1, 128, 64), torch.zeros(1, 2, 64))
        self.assertEqual(tuple(outputs["cents"].shape), (1, 64))

    def test_sliding_inference_preserves_full_clip_length(self) -> None:
        model = NoteFrameNet(
            NoteFrameNetConfig(channels=16, temporal_blocks=1, midi_min=58, midi_max=62)
        )
        mel = np.full((128, 73), -40.0, dtype=np.float32)
        result = infer_probabilities(
            model,
            mel,
            "cpu",
            window_frames=32,
            overlap_frames=8,
            batch_size=2,
        )
        self.assertEqual(result["voiced"].shape, (73,))
        self.assertEqual(result["pitch"].shape, (73, 5))
        np.testing.assert_allclose(result["pitch"].sum(axis=1), 1.0, atol=1e-5)

    def test_checkpoint_round_trip_includes_calibration(self) -> None:
        model_cfg = NoteFrameNetConfig(
            channels=16, temporal_blocks=1, midi_min=58, midi_max=62
        )
        model = NoteFrameNet(model_cfg)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model_cfg.to_dict(),
                    "calibration": DecodeConfig(onset_threshold=0.6).to_dict(),
                },
                path,
            )
            loaded, calibration = load_note_transcriber(path)
        self.assertIsInstance(loaded, NoteFrameNet)
        self.assertAlmostEqual(calibration.onset_threshold, 0.6)

    def test_one_epoch_training_saves_checkpoint_history_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "root"
            root.mkdir()
            for name in ("train_clip", "val_clip"):
                sample = root / name
                sample.mkdir()
                np.save(
                    sample / "performance_mel.npy",
                    np.full((128, 24), -40, np.float32),
                )
                (sample / "performance_audio.mid").write_bytes(b"MThd")
                (sample / "metadata.json").write_text(
                    json.dumps({"sounding_transpose": -2}), encoding="utf-8"
                )
                (sample / "note_map.json").write_text(
                    json.dumps(
                        {
                            "rendered_notes": [
                                {
                                    "pitch_midi_written": 60,
                                    "start_sec": 0.05,
                                    "end_sec": 0.30,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            manifest = base / "split.json"
            manifest.write_text(
                json.dumps({"train": ["train_clip"], "val": ["val_clip"]}),
                encoding="utf-8",
            )
            out = base / "run"
            config = NoteTrainConfig(
                output_dir=out,
                epochs=1,
                batch_size=1,
                crop_frames=24,
                crops_per_clip=1,
                device="cpu",
                max_val_clips=1,
                infer_window_frames=24,
                infer_overlap_frames=6,
                infer_batch_size=1,
                model=NoteFrameNetConfig(
                    channels=16,
                    temporal_blocks=1,
                    midi_min=58,
                    midi_max=62,
                ),
            )
            _load_written_notes_cached.cache_clear()
            with patch(
                "synthpipeline.timing.midi_note_times",
                return_value=[(58, 0.05, 0.30)],
            ):
                checkpoint = train_note_transcriber([root], manifest, config)
            history = json.loads((out / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint, out / "best.pt")
            self.assertTrue((out / "last.pt").exists())
            self.assertEqual(len(history["history"]), 1)
            self.assertIn("calibration", history)


class NoteDecodeAndEvaluationTests(unittest.TestCase):
    def test_decode_emits_transnote_written_pitch(self) -> None:
        frames, classes = 16, 61
        probabilities = {
            "voiced": np.full(frames, 0.05, np.float32),
            "pitch": np.full((frames, classes), 1e-4, np.float32),
            "onset": np.full(frames, 0.05, np.float32),
            "offset": np.full(frames, 0.05, np.float32),
        }
        probabilities["voiced"][2:11] = 0.95
        probabilities["pitch"][2:11, 60 - 36] = 0.99
        probabilities["onset"][2] = 0.9
        probabilities["offset"][11] = 0.9
        notes = decode_notes(probabilities, midi_min=36)
        self.assertEqual(len(notes), 1)
        self.assertIsInstance(notes[0], TransNote)
        self.assertEqual(notes[0].pitch, 60)
        self.assertGreater(notes[0].end, notes[0].start)
        self.assertGreater(notes[0].confidence, 0.5)
        self.assertTrue(notes[0].end > notes[0].start)
        self.assertGreaterEqual(notes[0].end - notes[0].start, 0.05)
        self.assertEqual(notes[0].cents, 0.0)

    def test_decode_emits_valid_non_overlapping_sequence(self) -> None:
        frames, classes = 40, 10
        probabilities = {
            "voiced": np.full(frames, 0.95, np.float32),
            "pitch": np.full((frames, classes), 1e-4, np.float32),
            "onset": np.full(frames, 0.05, np.float32),
            "offset": np.full(frames, 0.05, np.float32),
            "cents": np.full(frames, 12.0, np.float32),
        }
        probabilities["pitch"][:20, 2] = 0.99
        probabilities["pitch"][20:, 4] = 0.99
        probabilities["onset"][0] = 0.95
        probabilities["onset"][20] = 0.95
        notes = decode_notes(probabilities, midi_min=36)
        self.assertGreaterEqual(len(notes), 2)
        for previous, current in zip(notes, notes[1:]):
            self.assertLessEqual(previous.end, current.start + 1e-6)
            self.assertGreater(previous.end, previous.start)
        self.assertTrue(any(abs(note.cents - 12.0) < 1e-3 for note in notes))

    def test_infer_sample_notes_matches_full_clip_without_wav(self) -> None:
        model = NoteFrameNet(
            NoteFrameNetConfig(channels=16, temporal_blocks=1, midi_min=58, midi_max=62)
        )
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp)
            np.save(sample / "performance_mel.npy", np.full((128, 40), -40.0, np.float32))
            neural = infer_full_clip(model, sample / "performance_mel.npy", "cpu")
            same = infer_sample_notes(model, sample, "cpu")
            fused = infer_sample_notes(model, sample, "cpu", fusion="conservative")
        self.assertEqual(
            [(n.pitch, n.start, n.end) for n in neural],
            [(n.pitch, n.start, n.end) for n in same],
        )
        self.assertEqual(
            [(n.pitch, n.start, n.end) for n in neural],
            [(n.pitch, n.start, n.end) for n in fused],
        )

    def test_spectral_fusion_leaves_confident_neural_pitch(self) -> None:
        notes = [TransNote(60, 0.10, 0.40, 0.90, cents=-15.0)]
        frame_pitch = np.full(20, 62, dtype=np.int16)
        frame_strength = np.ones(20, dtype=np.float32)
        fused = apply_spectral_pitch(
            notes, frame_pitch, frame_strength, only_low_confidence=True
        )
        self.assertEqual(fused[0].pitch, 60)
        self.assertEqual(fused[0].cents, -15.0)

    def test_v2_config_round_trip_and_legacy_defaults(self) -> None:
        v2 = NoteFrameNetConfig()
        restored = NoteFrameNetConfig.from_dict(v2.to_dict())
        self.assertEqual(restored.spectral_blocks, 3)
        self.assertEqual(restored.temporal_channels, 128)
        legacy = NoteFrameNetConfig.from_dict(
            {"channels": 48, "temporal_blocks": 5, "midi_min": 36, "midi_max": 108}
        )
        self.assertEqual(legacy.spectral_blocks, 4)
        self.assertFalse(legacy.use_multiscale)
        self.assertFalse(legacy.predict_cents)

    def test_exact_semitone_and_fifty_ms_matching(self) -> None:
        gold = [
            TransNote(60, 1.00, 1.40, 1.0),
            TransNote(62, 2.00, 2.30, 1.0),
        ]
        pred = [
            TransNote(60, 1.050, 1.45, 0.9),
            TransNote(63, 2.00, 2.31, 0.9),
        ]
        self.assertEqual(match_notes(pred, gold), [(0, 0)])
        metrics = evaluate_note_lists(pred, gold)
        self.assertEqual(metrics["n_matched"], 1)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["onset_mae_sec"], 0.05)
        self.assertAlmostEqual(metrics["offset_mae_sec"], 0.05)
        off_pitch = [
            TransNote(61, 1.00, 1.40, 0.9),
            TransNote(74, 2.00, 2.30, 0.9),
        ]
        pitch_metrics = evaluate_note_lists(off_pitch, gold)
        self.assertEqual(pitch_metrics["n_matched"], 0)
        self.assertEqual(pitch_metrics["n_semitone_errors"], 1)
        self.assertEqual(pitch_metrics["n_octave_errors"], 1)
        cents_metrics = evaluate_note_lists(
            [TransNote(60, 1.00, 1.40, 1.0, cents=-10.0)],
            [(60, 1.00, 1.40, -40.0)],
        )
        self.assertAlmostEqual(cents_metrics["cents_mae"], 30.0)


if __name__ == "__main__":
    unittest.main()
