from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from alignmodel.contextual_align_train import load_alignment_sequence
from alignmodel.stages.contextual_note_aligner import (
    ContextualAlignerConfig,
    ContextualNoteAligner,
    sequence_features,
)
from alignmodel.types import GraphNote, TranscribedNote


class ContextualAlignerTests(unittest.TestCase):
    def test_feature_and_forward_shapes(self) -> None:
        features = sequence_features(
            [
                TranscribedNote(60, 0.0, 0.5),
                TranscribedNote(62, 0.5, 1.0),
            ]
        )
        self.assertEqual(features.shape, (2, 9))
        model = ContextualNoteAligner(
            ContextualAlignerConfig(hidden_dim=8, layers=1)
        )
        logits = model(
            torch.from_numpy(features)[None],
            torch.from_numpy(features)[None],
        )
        self.assertEqual(tuple(logits.shape), (1, 2, 3))

    def test_alignment_is_monotonic(self) -> None:
        model = ContextualNoteAligner(
            ContextualAlignerConfig(hidden_dim=8, layers=1)
        )
        observed = [
            TranscribedNote(60, 0.0, 0.4),
            TranscribedNote(62, 0.5, 0.9),
        ]
        score = [
            GraphNote(0, 60, 0.0, 1.0, 1.0, 0.0, 1.0),
            GraphNote(1, 62, 1.0, 2.0, 1.0, 1.0, 2.0),
        ]
        mapping = model.align(observed, score)
        selected = [value for value in mapping if value is not None]
        self.assertEqual(selected, sorted(selected))

    def test_training_sequence_removes_copy_and_uses_clean_pitch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "note_map.json"
            path.write_text(
                json.dumps(
                    {
                        "clean_notes": [
                            {
                                "clean_index": 0,
                                "pitch_midi": 60,
                                "onset_ql": 0.0,
                                "duration_ql": 1.0,
                            }
                        ],
                        "rendered_notes": [
                            {
                                "rendered_index": 0,
                                "pitch_midi_written": 62,
                                "start_sec": 0.0,
                                "end_sec": 0.5,
                                "primary_clean_index": 0,
                                "relationship": "match",
                            },
                            {
                                "rendered_index": 1,
                                "pitch_midi_written": 62,
                                "start_sec": 0.5,
                                "end_sec": 1.0,
                                "primary_clean_index": 0,
                                "relationship": "copy",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sequence = load_alignment_sequence(path)
        self.assertEqual(len(sequence.observed), 1)
        self.assertEqual(sequence.targets.tolist(), [0])
        self.assertAlmostEqual(float(sequence.observed[0, 0] * 128), 60.0)


if __name__ == "__main__":
    unittest.main()
