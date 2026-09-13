from __future__ import annotations

import unittest

from alignmodel.stages.alignment_strategies import (
    dynamic_revision_mapping,
    multi_start_mapping,
)
from alignmodel.types import GraphNote, TranscribedNote


def _score(pitches):
    return [
        GraphNote(i, pitch, float(i), float(i + 1), 1.0, float(i), float(i + 1))
        for i, pitch in enumerate(pitches)
    ]


class AlignmentStrategyTests(unittest.TestCase):
    def test_multi_start_recovers_after_missing_note(self) -> None:
        observed = [
            TranscribedNote(pitch, float(i), float(i) + 0.8)
            for i, pitch in enumerate([60, 64, 65, 67, 69, 71, 72])
        ]
        mapping = multi_start_mapping(
            observed,
            _score([60, 62, 64, 65, 67, 69, 71, 72]),
            window_notes=4,
            stride_notes=2,
        )
        self.assertEqual(mapping, [0, 2, 3, 4, 5, 6, 7])

    def test_revision_fixes_shifted_chain(self) -> None:
        observed = [
            TranscribedNote(pitch, float(i), float(i) + 0.8)
            for i, pitch in enumerate([60, 64, 65, 67, 69, 71, 72])
        ]
        score = _score([60, 62, 64, 65, 67, 69, 71, 72])
        revised = dynamic_revision_mapping(
            observed,
            score,
            [0, 1, 2, 3, 4, 5, 6],
            error_window=4,
            lookback_notes=2,
            lookahead_notes=4,
        )
        self.assertEqual(revised, [0, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
