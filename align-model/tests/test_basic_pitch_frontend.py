from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from alignmodel.transcription import basic_pitch as frontend


def _raw_features(frames: int = 6) -> dict[str, np.ndarray]:
    return {
        "note": np.zeros((frames, 88), dtype=np.float32),
        "onset": np.zeros((frames, 88), dtype=np.float32),
        "contour": np.zeros((frames, 264), dtype=np.float32),
    }


class BasicPitchFrontendTests(unittest.TestCase):
    def tearDown(self) -> None:
        frontend._MODEL = None
        frontend._MODEL_PID = None

    def test_import_does_not_import_basic_pitch_or_tensorflow(self) -> None:
        code = (
            "import sys; "
            "import alignmodel.transcription.basic_pitch; "
            "assert 'basic_pitch' not in sys.modules; "
            "assert 'tensorflow' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_hash_and_pitch_policy_invalidate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav = root / "performance_audio.wav"
            cache = root / "cache.npz"
            wav.write_bytes(b"first wav contents")
            calls = 0

            def infer(_path: Path) -> dict[str, np.ndarray]:
                nonlocal calls
                calls += 1
                output = _raw_features()
                output["note"][:, 10] = float(calls)
                return output

            with patch.object(frontend, "_run_official_inference", side_effect=infer):
                first = frontend.extract_basic_pitch_features(
                    wav,
                    source_metadata={"effective_audio_transpose": 0},
                    cache_path=cache,
                )
                reused = frontend.extract_basic_pitch_features(
                    wav,
                    source_metadata={"effective_audio_transpose": 0},
                    cache_path=cache,
                )
                self.assertEqual(calls, 1)
                np.testing.assert_array_equal(first.note, reused.note)

                wav.write_bytes(b"changed wav contents")
                frontend.extract_basic_pitch_features(
                    wav,
                    source_metadata={"effective_audio_transpose": 0},
                    cache_path=cache,
                )
                self.assertEqual(calls, 2)

                shifted = frontend.extract_basic_pitch_features(
                    wav,
                    source_metadata={"effective_audio_transpose": 2},
                    cache_path=cache,
                )
                self.assertEqual(calls, 3)
                self.assertEqual(float(shifted.note[0, 12]), 3.0)
                self.assertEqual(
                    shifted.metadata["wav_sha256"], frontend.sha256_file(wav)
                )
                self.assertEqual(
                    shifted.metadata["frontend_version"], frontend.FRONTEND_VERSION
                )

    def test_axis_shift_preserves_shapes_and_moves_three_contour_bins(self) -> None:
        raw = _raw_features(frames=3)
        raw["note"][1, 10] = 0.7
        raw["onset"][1, 10] = 0.8
        raw["contour"][1, 30:33] = (0.1, 0.2, 0.3)
        written = frontend.remap_sounding_to_written(raw, 2)
        self.assertEqual(written["note"].shape, raw["note"].shape)
        self.assertEqual(written["onset"].shape, raw["onset"].shape)
        self.assertEqual(written["contour"].shape, raw["contour"].shape)
        self.assertAlmostEqual(float(written["note"][1, 12]), 0.7)
        self.assertAlmostEqual(float(written["onset"][1, 12]), 0.8)
        np.testing.assert_array_equal(
            written["contour"][1, 36:39], raw["contour"][1, 30:33]
        )

    def test_cache_round_trip_preserves_shapes_and_frame_times(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav = root / "clip.wav"
            cache = root / "clip.npz"
            wav.write_bytes(b"wav")
            raw = _raw_features(frames=175)
            with patch.object(
                frontend, "_run_official_inference", return_value=raw
            ):
                generated = frontend.extract_basic_pitch_features(
                    wav,
                    source_metadata={"effective_audio_transpose": 2},
                    cache_path=cache,
                )
            loaded = frontend.load_basic_pitch_cache(
                cache,
                wav,
                {"effective_audio_transpose": 2},
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.note.shape, (175, 88))
            self.assertEqual(loaded.onset.shape, (175, 88))
            self.assertEqual(loaded.contour.shape, (175, 264))
            np.testing.assert_array_equal(loaded.frame_times, generated.frame_times)
            self.assertEqual(loaded.frame_times.shape, (175,))
            self.assertGreater(loaded.frame_times[-1], loaded.frame_times[0])

    def test_one_official_model_is_created_per_process(self) -> None:
        models: list[object] = []

        class FakeModel:
            def __init__(self, path: object) -> None:
                self.path = path
                models.append(self)

        package = types.ModuleType("basic_pitch")
        package.ICASSP_2022_MODEL_PATH = "official-model"
        inference = types.ModuleType("basic_pitch.inference")
        inference.Model = FakeModel
        note_creation = types.ModuleType("basic_pitch.note_creation")
        with patch.dict(
            sys.modules,
            {
                "basic_pitch": package,
                "basic_pitch.inference": inference,
                "basic_pitch.note_creation": note_creation,
            },
        ):
            one = frontend.get_basic_pitch_model()
            two = frontend.get_basic_pitch_model()
        self.assertIs(one, two)
        self.assertEqual(len(models), 1)

    def test_frozen_decode_returns_transnotes_with_pitch_bend_cents(self) -> None:
        captured: dict[str, object] = {}

        def model_output_to_notes(
            output: dict[str, np.ndarray], **kwargs: object
        ) -> tuple[object, list[tuple[float, float, int, float, list[int]]]]:
            captured["output"] = output
            captured.update(kwargs)
            return object(), [(0.1, 0.5, 62, 0.8, [-1, 1, 2])]

        package = types.ModuleType("basic_pitch")
        inference = types.ModuleType("basic_pitch.inference")
        note_creation = types.ModuleType("basic_pitch.note_creation")
        note_creation.model_output_to_notes = model_output_to_notes
        raw = _raw_features(frames=8)
        features = frontend.BasicPitchFeatures(
            raw["note"],
            raw["onset"],
            raw["contour"],
            frontend.model_frame_times(8),
            {},
        )
        with patch.dict(
            sys.modules,
            {
                "basic_pitch": package,
                "basic_pitch.inference": inference,
                "basic_pitch.note_creation": note_creation,
            },
        ):
            notes = frontend.decode_frozen_basic_pitch(features)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].pitch, 62)
        self.assertAlmostEqual(notes[0].cents, 33.33, places=2)
        self.assertEqual(captured["onset_thresh"], 0.5)
        self.assertEqual(captured["frame_thresh"], 0.3)
        self.assertEqual(captured["min_note_len"], 5)
        self.assertTrue(captured["multiple_pitch_bends"])


if __name__ == "__main__":
    unittest.main()
