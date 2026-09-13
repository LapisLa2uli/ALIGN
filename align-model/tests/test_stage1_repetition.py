import unittest

import numpy as np

from alignmodel.stages.repetition import (
    RepetitionCandidate,
    find_past_repetitions,
    sequence_similarity,
)
from alignmodel.stages.restart import _segments_from_retrieval
from alignmodel.types import GraphNote, PipelineConfig, PipelineState, ScoreGraph


def _motif(seconds: float, hop_sec: float, seed: int = 7) -> np.ndarray:
    frames = int(round(seconds / hop_sec))
    rng = np.random.default_rng(seed)
    pitches = rng.integers(0, 12, size=frames)
    chroma = np.zeros((12, frames), dtype=np.float64)
    chroma[pitches, np.arange(frames)] = 1.0
    return chroma


class Stage1RetrievalTests(unittest.TestCase):
    def test_retrieval_builds_explicit_replay_segment(self):
        notes = [
            GraphNote(i, 60 + i, float(i), float(i + 1), 1.0, float(i), float(i + 1))
            for i in range(10)
        ]
        state = PipelineState(
            sample_id="segments",
            sample_dir=".",
            sr=22050,
            duration_sec=14.3,
            hop_sec=0.05,
            config=PipelineConfig(weights_dir=None),
            score=ScoreGraph(notes=notes, duration_sec=10.0),
        )
        repetition = RepetitionCandidate(10.3, 14.3, 2.0, 6.0, 0.95)
        segments = _segments_from_retrieval(state, [repetition])
        self.assertEqual(len(segments), 2)
        self.assertFalse(segments[0].is_repetition)
        self.assertTrue(segments[1].is_repetition)
        self.assertEqual((segments[1].score_i0, segments[1].score_i1), (2, 6))

    def test_sequence_similarity_is_tempo_tolerant(self):
        motif = _motif(4.0, 0.05)
        stretched = np.vstack(
            [
                np.interp(
                    np.linspace(0.0, 1.0, 96),
                    np.linspace(0.0, 1.0, motif.shape[1]),
                    row,
                )
                for row in motif
            ]
        )
        self.assertGreater(sequence_similarity(motif, stretched), 0.90)

    def test_retrieves_repetition_from_long_ago(self):
        hop = 0.05
        source = _motif(18.0, hop)
        silence = np.full((12, 8), 1.0 / np.sqrt(12.0))
        performance = np.concatenate([source, silence, source], axis=1)
        results = find_past_repetitions(
            performance,
            hop,
            performance.shape[1] * hop,
            [(18.0, 18.4)],
            max_lookback_sec=20.0,
            min_confidence=0.82,
        )
        self.assertEqual(len(results), 1)
        hit = results[0]
        self.assertAlmostEqual(hit.start_time, 18.4, places=3)
        self.assertLess(abs(hit.source_start), 0.3)
        self.assertAlmostEqual(hit.source_end, 18.0, places=3)
        self.assertGreater(hit.confidence, 0.82)

    def test_rejects_unrelated_post_silence_music(self):
        hop = 0.05
        source = _motif(6.0, hop, seed=3)
        silence = np.full((12, 8), 1.0 / np.sqrt(12.0))
        unrelated = _motif(6.0, hop, seed=99)
        performance = np.concatenate([source, silence, unrelated], axis=1)
        results = find_past_repetitions(
            performance,
            hop,
            performance.shape[1] * hop,
            [(6.0, 6.4)],
            min_confidence=0.86,
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
