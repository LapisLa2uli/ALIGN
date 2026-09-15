from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from music21 import duration, note, spanner, stream, tie

from alignmodel.stages.dc_alignment import _monotonic_pitch_mapping
from alignmodel.stages.score_graph import build_score_graph
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    BasicPitchFeatures,
    sanitize_basic_pitch_notes,
)
from alignmodel.transcription.decode import TransNote
from alignmodel.types import GraphNote, TranscribedNote


class TranscriptionCleanupTests(unittest.TestCase):
    def _features(self, frames: int = 40):
        return BasicPitchFeatures(
            note=np.zeros((frames, 88), np.float32),
            onset=np.zeros((frames, 88), np.float32),
            contour=np.zeros((frames, 264), np.float32),
            frame_times=np.arange(frames) * 0.01,
            metadata={},
        )

    def test_drops_overlapping_upper_harmonic(self) -> None:
        features = self._features()
        notes = [
            TransNote(60, 0.0, 0.3, 0.8),
            TransNote(84, 0.0, 0.3, 0.7),
        ]
        cleaned = sanitize_basic_pitch_notes(notes, features)
        self.assertEqual([value.pitch for value in cleaned], [60])

    def test_merges_weak_same_pitch_split(self) -> None:
        features = self._features()
        notes = [
            TransNote(60, 0.0, 0.18, 0.8),
            TransNote(60, 0.20, 0.38, 0.75),
        ]
        cleaned = sanitize_basic_pitch_notes(
            notes,
            features,
            BasicPitchDecodeConfig(merge_onset_threshold=0.6),
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual((cleaned[0].start, cleaned[0].end), (0.0, 0.38))

    def test_rescues_short_note_with_joint_activation_evidence(self) -> None:
        features = self._features()
        axis = 60 - 21
        features.onset[10, axis] = 0.40
        features.note[10:14, axis] = 0.30
        features.contour[10:14, axis * 3 : axis * 3 + 3] = 0.30
        cleaned = sanitize_basic_pitch_notes(
            [],
            features,
            BasicPitchDecodeConfig(adaptive_short_note_rescue=True),
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].pitch, 60)
        self.assertAlmostEqual(cleaned[0].start, 0.10)
        self.assertAlmostEqual(cleaned[0].end, 0.14)

    def test_does_not_rescue_isolated_onset_noise(self) -> None:
        features = self._features()
        axis = 60 - 21
        features.onset[10, axis] = 0.90
        features.note[10, axis] = 0.05
        cleaned = sanitize_basic_pitch_notes(
            [],
            features,
            BasicPitchDecodeConfig(adaptive_short_note_rescue=True),
        )
        self.assertEqual(cleaned, [])

    def test_merges_contour_continuous_weak_boundary(self) -> None:
        features = self._features(frames=60)
        axis = 60 - 21
        features.note[16:26, axis] = 0.30
        features.contour[16:26, axis * 3 : axis * 3 + 3] = 0.35
        notes = [
            TransNote(60, 0.0, 0.18, 0.8),
            TransNote(60, 0.23, 0.38, 0.75),
        ]
        cleaned = sanitize_basic_pitch_notes(notes, features)
        self.assertEqual(len(cleaned), 1)

    def test_preserves_strong_same_pitch_rearticulation(self) -> None:
        features = self._features(frames=60)
        axis = 60 - 21
        features.note[16:26, axis] = 0.40
        features.contour[16:26, axis * 3 : axis * 3 + 3] = 0.40
        features.onset[23, axis] = 0.90
        notes = [
            TransNote(60, 0.0, 0.18, 0.8),
            TransNote(60, 0.23, 0.38, 0.75),
        ]
        cleaned = sanitize_basic_pitch_notes(notes, features)
        self.assertEqual(len(cleaned), 2)

    def test_missing_note_resynchronizes_next_notes(self) -> None:
        score = [
            GraphNote(i, pitch, float(i), float(i + 1), 1.0, float(i), float(i + 1))
            for i, pitch in enumerate([60, 62, 64, 65])
        ]
        observed = [
            TranscribedNote(60, 0.0, 0.8),
            TranscribedNote(64, 1.0, 1.8),
            TranscribedNote(65, 2.0, 2.8),
        ]
        self.assertEqual(
            _monotonic_pitch_mapping(observed, score),
            [0, 2, 3],
        )

    def test_score_graph_collapses_tied_notes(self) -> None:
        score = stream.Score()
        part = stream.Part()
        first = note.Note("C4", quarterLength=1.0)
        first.tie = tie.Tie("start")
        second = note.Note("C4", quarterLength=1.0)
        second.tie = tie.Tie("stop")
        part.append(first)
        part.append(second)
        score.insert(0, part)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "score.musicxml"
            score.write("musicxml", fp=str(path))
            graph = build_score_graph(path)
        self.assertEqual(len(graph.notes), 1)
        self.assertAlmostEqual(graph.notes[0].duration, 1.0)
        self.assertEqual(graph.notes[0].source_note_indices, [0, 1])

    def test_score_graph_skips_grace_notes(self) -> None:
        score = stream.Score()
        part = stream.Part()
        grace = note.Note("D5")
        grace.duration = duration.GraceDuration(0.25)
        part.append(grace)
        part.append(note.Note("C4", quarterLength=1.0))
        score.insert(0, part)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "grace.musicxml"
            score.write("musicxml", fp=str(path))
            graph = build_score_graph(path)
        self.assertEqual([item.pitch for item in graph.notes], [60])

    def test_score_graph_folds_slurred_same_pitch(self) -> None:
        score = stream.Score()
        part = stream.Part()
        first = note.Note("C4", quarterLength=1.0)
        second = note.Note("C4", quarterLength=1.0)
        part.append(first)
        part.append(second)
        part.insert(0, spanner.Slur(first, second))
        score.insert(0, part)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "slur.musicxml"
            score.write("musicxml", fp=str(path))
            graph = build_score_graph(path)
        self.assertEqual(len(graph.notes), 1)
        self.assertAlmostEqual(graph.notes[0].duration, 1.0)


if __name__ == "__main__":
    unittest.main()
