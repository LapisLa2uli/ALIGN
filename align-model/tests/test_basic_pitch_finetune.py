from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "finetune_basic_pitch.py"
SPEC = importlib.util.spec_from_file_location("finetune_basic_pitch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bpft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bpft
SPEC.loader.exec_module(bpft)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_required_by_cli(self) -> None:
        with self.assertRaises(SystemExit):
            bpft.build_parser().parse_args(["--out", "unused"])

    def test_procedural_only_rejects_raw_before_resolving_row_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "frozen.json"
            manifest.write_text(
                json.dumps(
                    {
                        "train": [
                            {
                                "sample_dir": "E:/output_2k_rawdata/do-not-open",
                                "corpus": "raw2k",
                            }
                        ],
                        "val": [{"sample": "procedural-validation"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "rejected raw2k"):
                bpft.load_frozen_manifest(manifest, procedural_only=True)

    def test_deterministic_limit_is_order_independent(self) -> None:
        rows = [{"sample": f"clip-{i}"} for i in range(10)]
        first = bpft.deterministic_limit(rows, 3, 365)
        second = bpft.deterministic_limit(list(reversed(rows)), 3, 365)
        self.assertEqual(first, second)


class ExactTargetTests(unittest.TestCase):
    def _write_bundle(
        self,
        root: Path,
        *,
        metadata: dict,
        sounding_pitch: int = 58,
    ) -> Path:
        sample = root / "sample"
        sample.mkdir()
        (sample / "note_map.json").write_text(
            json.dumps(
                {
                    "kind": "synth_note_lineage",
                    "rendered_notes": [
                        {
                            "rendered_index": 0,
                            "pitch_midi_written": 60,
                            "pitch_midi_sounding": sounding_pitch,
                            "start_sec": 0.1,
                            "end_sec": 0.4,
                        }
                    ],
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
                            "start_time": 0.2,
                            "end_time": 0.35,
                            "deviation_cents": 34.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (sample / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return sample

    def test_exact_documents_require_explicit_effective_transpose(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = self._write_bundle(
                Path(temp), metadata={"sounding_transpose": -2}
            )
            with self.assertRaisesRegex(ValueError, "effective_audio_transpose"):
                bpft.load_target_documents(sample)

    def test_targets_have_basic_pitch_shapes_and_sounding_axis(self) -> None:
        notes = [
            {
                "pitch_midi_written": 60,
                "pitch_midi_sounding": 58,
                "start_sec": 0.1,
                "end_sec": 0.4,
            }
        ]
        labels = [
            {
                "type": "intonation_error",
                "start_time": 0.2,
                "end_time": 0.35,
                "deviation_cents": 34.0,
            }
        ]
        targets = bpft.build_window_targets(notes, labels, 2)
        self.assertEqual(targets["note"].shape, (172, 88))
        self.assertEqual(targets["onset"].shape, (172, 88))
        self.assertEqual(targets["contour"].shape, (172, 264))
        note_bin = 58 - 21
        self.assertEqual(targets["onset"][round(0.1 * 86), note_bin], 1.0)
        self.assertGreater(targets["note"][:, note_bin].sum(), 0)
        plain_frame = round(0.15 * 86)
        bent_frame = round(0.25 * 86)
        self.assertEqual(targets["contour"][plain_frame, note_bin * 3], 1.0)
        self.assertEqual(targets["contour"][bent_frame, note_bin * 3 + 1], 1.0)

    def test_rendered_sounding_pitch_must_match_explicit_metadata(self) -> None:
        note = {
            "pitch_midi_written": 60,
            "pitch_midi_sounding": 59,
            "start_sec": 0.1,
            "end_sec": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "pitch-space mismatch"):
            bpft.build_window_targets([note], [], 2)

    def test_window_is_exact_length_and_zero_padded(self) -> None:
        audio = np.arange(100, dtype=np.float32)
        window = bpft.slice_audio_window(audio, 25)
        self.assertEqual(window.shape, (43844, 1))
        np.testing.assert_array_equal(window[:75, 0], audio[25:])
        self.assertTrue(np.all(window[75:, 0] == 0))


class _FakeTensor:
    def __init__(self, layer: "_FakeLayer") -> None:
        self._keras_history = (layer, 0, 0)


class _FakeLayer:
    def __init__(
        self,
        name: str,
        inputs: "_FakeTensor | list[_FakeTensor] | None",
        *,
        weighted: bool = True,
    ) -> None:
        self.name = name
        self.input = inputs
        self.weights = [object()] if weighted else []
        self.trainable = True
        self.output = _FakeTensor(self)


class _FakeModel:
    def __init__(self) -> None:
        source = _FakeLayer("audio", None, weighted=False)
        cqt = _FakeLayer("cqt", source.output)
        harmonic = _FakeLayer("harmonic_stacking", cqt.output)
        contour_trunk = _FakeLayer("contour_trunk", harmonic.output)
        contour_head = _FakeLayer("contour_head", contour_trunk.output)
        note_branch = _FakeLayer("note_branch", contour_head.output)
        note_head = _FakeLayer("note_head", note_branch.output)
        onset_branch = _FakeLayer("onset_branch", harmonic.output)
        onset_head = _FakeLayer(
            "onset_head", [note_branch.output, onset_branch.output]
        )
        self.layers = [
            source,
            cqt,
            harmonic,
            contour_trunk,
            contour_head,
            note_branch,
            note_head,
            onset_branch,
            onset_head,
        ]
        self.output = {
            "contour": contour_head.output,
            "note": note_head.output,
            "onset": onset_head.output,
        }


class StagedFreezeTests(unittest.TestCase):
    def test_mocked_keras_graph_uses_three_progressive_phases(self) -> None:
        model = _FakeModel()
        groups = bpft.set_trainable_phase(model, "heads")
        trainable = {layer.name for layer in model.layers if layer.trainable}
        self.assertEqual(trainable, {"contour_head", "note_head", "onset_head"})
        self.assertIn("contour_trunk", groups["contour_trunk"])
        self.assertIn("cqt", groups["frontend"])
        self.assertIn("harmonic_stacking", groups["frontend"])

        bpft.set_trainable_phase(model, "branches")
        trainable = {layer.name for layer in model.layers if layer.trainable}
        self.assertEqual(
            trainable,
            {
                "contour_head",
                "note_branch",
                "note_head",
                "onset_branch",
                "onset_head",
            },
        )

        bpft.set_trainable_phase(model, "trunk")
        trainable = {layer.name for layer in model.layers if layer.trainable}
        self.assertIn("contour_trunk", trainable)
        self.assertNotIn("cqt", trainable)
        self.assertNotIn("harmonic_stacking", trainable)


if __name__ == "__main__":
    unittest.main()
