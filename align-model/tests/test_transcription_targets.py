from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from alignmodel.transcription.data import (
    _load_written_notes_cached,
    load_written_notes,
    load_written_notes_with_cents,
    written_pitch_shift,
)


REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO / "align-model" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CanonicalTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        _load_written_notes_cached.cache_clear()

    def test_rendered_notes_are_preferred_and_receive_cents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp)
            (sample / "note_map.json").write_text(
                json.dumps(
                    {
                        "rendered_notes": [
                            {
                                "pitch_midi_written": 65,
                                "start_sec": 0.1,
                                "end_sec": 0.4,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (sample / "labels.json").write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "type": "intonation_error",
                                "start_time": 0.15,
                                "end_time": 0.35,
                                "deviation_cents": -47,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "synthpipeline.timing.midi_note_times",
                side_effect=AssertionError("canonical targets must not read MIDI"),
            ):
                self.assertEqual(load_written_notes(sample), [(65, 0.1, 0.4)])
                self.assertEqual(
                    load_written_notes_with_cents(sample),
                    [(65, 0.1, 0.4, -47.0)],
                )

    def test_midi_fallback_must_be_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp)
            (sample / "metadata.json").write_text(
                json.dumps({"sounding_transpose": -2}),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_written_notes(sample)
            with patch(
                "synthpipeline.timing.midi_note_times",
                return_value=[(58, 0.1, 0.3)],
            ):
                self.assertEqual(
                    load_written_notes(sample, allow_legacy_midi=True),
                    [(60, 0.1, 0.3)],
                )

    def test_f0_shift_uses_effective_audio_transpose(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp)
            (sample / "metadata.json").write_text(
                json.dumps(
                    {
                        "audio_pitch_space": "transposed",
                        "effective_audio_transpose": 4,
                        "sounding_transpose": -2,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(written_pitch_shift(sample), 4)


class CanonicalStagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage = _load_script("stage_transcriber_data")

    def _source(self, root: Path, *, note_map: bool) -> Path:
        source = root / "source" / "clip"
        source.mkdir(parents=True)
        np.save(source / "performance_mel.npy", np.zeros((128, 2), np.float32))
        (source / "performance_audio.wav").write_bytes(b"wav")
        if note_map:
            (source / "note_map.json").write_text(
                json.dumps({"rendered_notes": []}),
                encoding="utf-8",
            )
        return source

    def test_stages_wav_and_allows_missing_f0(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root, note_map=True)
            staged = self.stage._stage_row(
                {"sample_dir": str(source), "corpus": "procedural12k"},
                root / "cache",
            )
            destination = Path(staged["sample_dir"])
            self.assertTrue((destination / "performance_audio.wav").is_file())
            self.assertTrue((destination / "note_map.json").is_file())
            self.assertFalse((destination / "performance_f0.npy").exists())

    def test_missing_note_map_is_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root, note_map=False)
            with self.assertRaises(FileNotFoundError):
                self.stage._stage_row(
                    {"sample_dir": str(source), "corpus": "procedural12k"},
                    root / "cache",
                )


class ProceduralSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.split = _load_script("prepare_note_alignment_split")

    def test_procedural_only_never_requires_or_emits_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            procedural = root / "procedural"
            procedural.mkdir()
            for index in range(10):
                sample = procedural / f"sample_{index:02d}"
                sample.mkdir()
                np.save(
                    sample / "performance_mel.npy",
                    np.zeros((128, 20), np.float32),
                )
                for name in (
                    "performance_audio.mid",
                    "verified_score.musicxml",
                    "performance_score.musicxml",
                ):
                    (sample / name).write_bytes(b"x")
                (sample / "labels.json").write_text(
                    json.dumps({"labels": []}), encoding="utf-8"
                )
                (sample / "metadata.json").write_text(
                    json.dumps({"source": "procedural"}), encoding="utf-8"
                )
            output = root / "split.json"
            missing_raw = root / "must-not-be-read"
            argv = [
                "prepare_note_alignment_split.py",
                "--procedural-root",
                str(procedural),
                "--raw-root",
                str(missing_raw),
                "--out",
                str(output),
                "--val-fraction",
                "0.2",
                "--test-fraction",
                "0.2",
                "--procedural-only",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                self.split.main()
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["roots"]), {"procedural12k"})
            self.assertTrue(manifest["train"])
            self.assertTrue(manifest["val"])
            self.assertTrue(manifest["test_id"])
            self.assertEqual(manifest["test_ood"], [])
            self.assertFalse(missing_raw.exists())
            for split in ("train", "val", "test_id"):
                self.assertTrue(
                    all(row["corpus"] == "procedural12k" for row in manifest[split])
                )


if __name__ == "__main__":
    unittest.main()
