from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from alignmodel.note_repetition_train import candidate_rows
from alignmodel.stages.note_repetition_model import (
    NoteRepetitionModelConfig,
    NoteRepetitionScorer,
    continuation_similarity,
    load_note_repetition_model,
    propose_note_repeat_candidates,
    save_note_repetition_model,
)
from alignmodel.stages.repetition import consolidate_note_repetitions
from alignmodel.types import NoteRepetition, TranscribedNote


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
        self.assertEqual(loaded.config.feature_version, 2)
        self.assertEqual(loaded.net[0].in_features, 15)
        self.assertEqual(extra["epoch"], 2)
        features = torch.zeros(2, 12)
        self.assertEqual(tuple(loaded(features).shape), (2,))

    def test_soft_continuation_tolerates_one_error(self) -> None:
        pitches = (
            60, 62, 64, 65, 67, 69,
            60, 62, 64, 65, 80, 69,
        )
        notes = [
            TranscribedNote(pitch, index * 0.4, index * 0.4 + 0.3)
            for index, pitch in enumerate(pitches)
        ]
        fraction, score = continuation_similarity(notes, 0, 6, 3)
        self.assertGreaterEqual(fraction, 2 / 3)
        self.assertGreater(score, 0.0)
        divergent = list(notes)
        for index, pitch in enumerate((80, 81, 82), 9):
            divergent[index] = TranscribedNote(
                pitch, index * 0.4, index * 0.4 + 0.3
            )
        _fraction, divergent_score = continuation_similarity(
            divergent, 0, 6, 3
        )
        self.assertLess(divergent_score, 0.0)

    def test_v1_checkpoint_keeps_legacy_feature_width(self) -> None:
        model = NoteRepetitionScorer(
            NoteRepetitionModelConfig(feature_version=1)
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "repeat-v1.pt"
            save_note_repetition_model(path, model)
            loaded, _extra = load_note_repetition_model(path)
        self.assertEqual(loaded.net[0].in_features, 12)
        self.assertEqual(tuple(loaded(torch.zeros(2, 15)).shape), (2,))

    def test_consolidates_fragmented_repeat_spans(self) -> None:
        fragments = [
            NoteRepetition(0, 3, 8, 11, 0.0, 1.2, 3.2, 4.4, 0.90),
            NoteRepetition(3, 6, 11, 14, 1.2, 2.4, 4.4, 5.6, 0.85),
        ]
        merged = consolidate_note_repetitions(fragments)
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            (
                merged[0].source_i0,
                merged[0].source_i1,
                merged[0].repeat_i0,
                merged[0].repeat_i1,
            ),
            (0, 6, 8, 14),
        )


if __name__ == "__main__":
    unittest.main()
