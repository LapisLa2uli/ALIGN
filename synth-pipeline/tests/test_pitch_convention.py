from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from synthpipeline.pitch_convention import (
    annotate_pitch_metadata,
    audio_pitch_space,
    audio_to_written_shift,
    effective_audio_transpose,
    midi_pitch_space,
    midi_to_written_shift,
    sounding_transpose,
)


class PitchConventionTests(unittest.TestCase):
    def test_current_generation_midi_is_sounding(self) -> None:
        meta = annotate_pitch_metadata(
            {"sounding_transpose": -2},
            midi_space="sounding",
            audio_render="oscillator_v1",
        )
        self.assertEqual(midi_pitch_space(meta), "sounding")
        self.assertEqual(midi_to_written_shift(meta), 2)
        self.assertEqual(sounding_transpose(meta), -2)

    def test_legacy_soundfont_midi_is_written(self) -> None:
        meta = {"sounding_transpose": -2, "audio_render": "soundfont_rerender"}
        self.assertEqual(midi_pitch_space(meta), "written")
        self.assertEqual(midi_to_written_shift(meta), 0)

    def test_explicit_space_wins_over_render_mark(self) -> None:
        meta = {
            "sounding_transpose": -2,
            "audio_render": "oscillator_v1",
            "midi_pitch_space": "written",
        }
        self.assertEqual(midi_pitch_space(meta), "written")
        self.assertEqual(midi_to_written_shift(meta), 0)

    def test_missing_metadata_defaults_to_sounding_plus_two(self) -> None:
        self.assertEqual(midi_to_written_shift({}), 2)
        self.assertEqual(midi_to_written_shift(None), 2)

    def test_annotate_writes_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.json"
            payload = annotate_pitch_metadata({"seed": 1}, midi_space="sounding")
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["midi_pitch_space"], "sounding")
        self.assertEqual(loaded["audio_render"], "oscillator_v1")
        self.assertEqual(loaded["sounding_transpose"], -2)
        self.assertEqual(loaded["audio_pitch_space"], "sounding")
        self.assertEqual(loaded["effective_audio_transpose"], 2)

    def test_written_sounding_round_trip(self) -> None:
        written = 72
        sounding = written + sounding_transpose({"sounding_transpose": -2})
        self.assertEqual(sounding, 70)
        restored = sounding + audio_to_written_shift({"sounding_transpose": -2})
        self.assertEqual(restored, written)
        legacy = written + midi_to_written_shift(
            {"audio_render": "soundfont_rerender", "sounding_transpose": -2}
        )
        self.assertEqual(legacy, written)

    def test_audio_shift_is_independent_of_midi_space(self) -> None:
        legacy = {
            "audio_render": "soundfont_rerender",
            "midi_pitch_space": "written",
            "sounding_transpose": -2,
        }
        current = {
            "audio_render": "oscillator_v1",
            "midi_pitch_space": "sounding",
            "sounding_transpose": -2,
        }
        self.assertEqual(audio_to_written_shift(legacy), 2)
        self.assertEqual(audio_to_written_shift(current), 2)

    def test_explicit_effective_audio_transpose_wins(self) -> None:
        procedural = {
            "sounding_transpose": -2,
            "audio_pitch_space": "transposed",
            "effective_audio_transpose": 4,
        }
        self.assertEqual(effective_audio_transpose(procedural), 4)
        self.assertEqual(audio_to_written_shift(procedural), 4)
        self.assertEqual(audio_pitch_space(procedural), "transposed")

    def test_audio_pitch_metadata_can_declare_written_audio(self) -> None:
        meta = annotate_pitch_metadata(
            {"sounding_transpose": -2},
            midi_space="written",
            audio_space="written",
            effective_audio_shift=0,
        )
        self.assertEqual(meta["audio_pitch_space"], "written")
        self.assertEqual(meta["effective_audio_transpose"], 0)
        self.assertEqual(effective_audio_transpose(meta), 0)

    def test_transpose_bundle_records_cumulative_audio_shift(self) -> None:
        from synthpipeline.transpose_audio import transpose_bundle

        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp)
            for name in ("performance_audio.wav", "reference_audio.wav"):
                (sample / name).write_bytes(b"wav")
            (sample / "metadata.json").write_text(
                json.dumps(
                    {
                        "sounding_transpose": 0,
                        "audio_pitch_space": "sounding",
                        "effective_audio_transpose": 2,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "synthpipeline.pitch_convention.infer_midi_pitch_space",
                    return_value=("written", 0),
                ),
                patch(
                    "synthpipeline.transpose_audio.load_audio",
                    return_value=(np.zeros(32, np.float32), 22050),
                ),
                patch(
                    "synthpipeline.transpose_audio.pitch_shift_duration_preserving",
                    return_value=np.zeros(32, np.float32),
                ),
                patch("synthpipeline.transpose_audio.save_wav"),
                patch("synthpipeline.transpose_audio._rewrite_mel"),
            ):
                self.assertEqual(transpose_bundle(sample, -2), "converted")
            metadata = json.loads(
                (sample / "metadata.json").read_text(encoding="utf-8")
            )
        self.assertEqual(metadata["effective_audio_transpose"], 4)
        self.assertEqual(metadata["audio_pitch_space"], "transposed")

    def test_regenerate_bundle_records_rendered_audio_shift(self) -> None:
        from synthpipeline.regenerate_audio import regenerate_bundle

        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp)
            for name in ("performance_audio.mid", "reference_audio.mid"):
                (sample / name).write_bytes(b"midi")
            (sample / "metadata.json").write_text(
                json.dumps({"audio_render": "legacy"}), encoding="utf-8"
            )
            with (
                patch(
                    "synthpipeline.regenerate_audio.resolve_bundle_soundfont",
                    return_value=(Path("unused.sf2"), 0),
                ),
                patch("synthpipeline.regenerate_audio._dc_config"),
                patch("synthpipeline.regenerate_audio.midi_note_transpose", return_value=0),
                patch("synthpipeline.regenerate_audio.render_midi_clarinet"),
                patch(
                    "synthpipeline.regenerate_audio._load_mono",
                    return_value=(np.zeros(32, np.float32), 22050),
                ),
                patch("synthpipeline.regenerate_audio._rewrite_mel"),
                patch(
                    "synthpipeline.pitch_convention.infer_midi_pitch_space",
                    return_value=("written", 0),
                ),
            ):
                self.assertEqual(regenerate_bundle(sample, -2), "converted")
            metadata = json.loads(
                (sample / "metadata.json").read_text(encoding="utf-8")
            )
        self.assertEqual(metadata["effective_audio_transpose"], 2)
        self.assertEqual(metadata["audio_pitch_space"], "sounding")


if __name__ == "__main__":
    unittest.main()
