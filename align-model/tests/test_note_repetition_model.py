from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from alignmodel.note_repetition_train import candidate_rows
from alignmodel.stages.note_repetition_model import (
    NoteRepetitionScorer,
    load_note_repetition_model,
    propose_note_repeat_candidates,
    save_note_repetition_model,
)
from alignmodel.types import TranscribedNote


class NoteRepetitionModelTests(unittest.TestCase):
    def test_candidates_include_one_note_repetition(self) -> None:
        notes = [
            TranscribedNote(60, 0.0, 0.5),
            TranscribedNote(62, 0.5, 1.0),
            TranscribedNote(60, 1.0, 1.5),
        ]
        candidates = propose_note_repeat_candidates(
            notes, extra_mask=[False, False, True]
        )
        self.assertTrue(
            any(
                candidate.source_i0 == 0
                and candidate.repeat_i0 == 2
                and candidate.repeat_i1 - candidate.repeat_i0 == 1
                for candidate in candidates
            )
        )

    def test_copy_lineage_makes_positive_and_extra_feature(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "note_map.json"
            path.write_text(
                json.dumps(
                    {
                        "clean_notes": [
                            {"clean_index": 0, "pitch_midi": 60},
                            {"clean_index": 1, "pitch_midi": 62},
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
                                "pitch_midi_written": 64,
                                "start_sec": 0.5,
                                "end_sec": 1.0,
                                "primary_clean_index": 1,
                                "relationship": "match",
                            },
                            {
                                "rendered_index": 2,
                                "pitch_midi_written": 62,
                                "start_sec": 1.0,
                                "end_sec": 1.5,
                                "primary_clean_index": 0,
                                "relationship": "copy",
                            },
                            {
                                "rendered_index": 3,
                                "pitch_midi_written": 64,
                                "start_sec": 1.5,
                                "end_sec": 2.0,
                                "primary_clean_index": 1,
                                "relationship": "copy",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            features, labels = candidate_rows(path, seed=1)
        positives = [
            feature for feature, label in zip(features, labels) if label == 1.0
        ]
        self.assertTrue(positives)
        self.assertTrue(any(float(feature[9]) == 1.0 for feature in positives))

    def test_checkpoint_round_trip(self) -> None:
        model = NoteRepetitionScorer()
        model.config.threshold = 0.72
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "repeat.pt"
            save_note_repetition_model(path, model, extra={"epoch": 2})
            loaded, extra = load_note_repetition_model(path)
        self.assertEqual(loaded.config.threshold, 0.72)
        self.assertEqual(extra["epoch"], 2)
        features = torch.zeros(2, 12)
        self.assertEqual(tuple(loaded(features).shape), (2,))


if __name__ == "__main__":
    unittest.main()
