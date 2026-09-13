import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from alignmodel.stages.dc_alignment import (
    pairs_from_aligned_events,
    pairs_from_learned_alignment,
)
from alignmodel.stages.note_align import AlignmentOperation, AlignmentResult
from alignmodel.stages.learned import rhythm_windows_from_pairs
from alignmodel.stages.rhythm import (
    flagged_rhythm_hits,
    merge_rhythm_labels,
    merge_time_spans,
)
from alignmodel.types import (
    GraphNote,
    PairedEvent,
    PipelineConfig,
    PipelineLabel,
    PipelineState,
    ScoreGraph,
)


def _note(index: int, pitch: int, start: float, dur: float = 1.0) -> GraphNote:
    return GraphNote(
        index=index,
        pitch=pitch,
        start=start,
        end=start + dur,
        duration=dur,
        ql_start=start,
        ql_end=start + dur,
        measure=1,
    )


def _pair(i: int, ref_dur: float, perf_dur: float, kind: str = "match") -> PairedEvent:
    t = float(i)
    return PairedEvent(
        score_index=i,
        pitch=60,
        ref_start=t,
        ref_end=t + ref_dur,
        perf_start=t,
        perf_end=t + perf_dur,
        kind=kind,
    )


class Stage3DcTests(unittest.TestCase):
    def test_pairs_from_dc_events_zip_sounding_notes(self):
        notes = [_note(0, 60, 0.0), _note(1, 62, 1.0), _note(2, 64, 2.0)]
        events = [
            {
                "is_rest": True,
                "ref_start": 0.0,
                "ref_end": 0.5,
                "perf_start": 0.0,
                "perf_end": 0.4,
                "measure": 1,
            },
            {
                "is_rest": True,
                "ref_start": 0.5,
                "ref_end": 0.8,
                "perf_start": 0.4,
                "perf_end": 0.55,
                "measure": 1,
            },
            {
                "is_rest": False,
                "midi": 60,
                "ref_start": 0.8,
                "ref_end": 1.8,
                "perf_start": 0.55,
                "perf_end": 1.7,
                "measure": 1,
            },
            {
                "is_rest": False,
                "midi": 62,
                "ref_start": 1.8,
                "ref_end": 2.8,
                "perf_start": 1.7,
                "perf_end": 2.9,
                "measure": 1,
            },
            {
                "is_rest": False,
                "midi": 64,
                "ref_start": 2.8,
                "ref_end": 3.8,
                "perf_start": 2.9,
                "perf_end": 4.0,
                "measure": 1,
            },
        ]
        pairs = pairs_from_aligned_events(events, notes)
        kinds = [p.kind for p in pairs]
        self.assertEqual(kinds[0], "rest")
        self.assertEqual(pairs[0].ref_end, 0.8)
        self.assertEqual(pairs[0].perf_end, 0.55)
        sounding = [p for p in pairs if p.kind == "match"]
        self.assertEqual([p.score_index for p in sounding], [0, 1, 2])
        self.assertEqual(sounding[0].perf_start, 0.55)
        self.assertEqual(sounding[0].ref_start, 0.8)

    def test_ewma_flags_duration_jump(self):
        cfg = PipelineConfig()
        pairs = [_pair(i, 1.0, 1.0) for i in range(4)]
        pairs.append(_pair(4, 1.0, 2.0))
        pairs.extend(_pair(i, 1.0, 1.0) for i in range(5, 8))
        hits = flagged_rhythm_hits(pairs, cfg)
        flagged = {h[0].score_index for h in hits}
        self.assertIn(4, flagged)
        self.assertNotIn(0, flagged)
        self.assertNotIn(1, flagged)

    def test_merge_adjacent_rhythm_labels(self):
        state = PipelineState(
            sample_id="t",
            sample_dir=".",
            sr=22050,
            duration_sec=4.0,
            hop_sec=0.023,
            config=PipelineConfig(),
            score=ScoreGraph(),
            labels=[
                PipelineLabel("a", "wrong_note", 0.2, 0.4),
                PipelineLabel("b", "rhythm_error", 1.0, 1.2, comment="ewma"),
                PipelineLabel("c", "rhythm_error", 1.22, 1.5, comment="far"),
                PipelineLabel("d", "rhythm_error", 3.0, 3.2, comment="later"),
            ],
        )
        merge_rhythm_labels(state, gap=0.05)
        rhythm = [lab for lab in state.labels if lab.type == "rhythm_error"]
        other = [lab for lab in state.labels if lab.type != "rhythm_error"]
        self.assertEqual(len(other), 1)
        self.assertEqual(len(rhythm), 2)
        self.assertAlmostEqual(rhythm[0].start_time, 1.0)
        self.assertAlmostEqual(rhythm[0].end_time, 1.5)
        self.assertAlmostEqual(rhythm[1].start_time, 3.0)

    def test_ungated_windows_cover_successive_pairs(self):
        pairs = [_pair(i, 0.4, 0.4) for i in range(6)]
        windows = rhythm_windows_from_pairs(pairs, gap=0.25, max_width=1.35, min_width=0.28)
        self.assertTrue(windows)
        self.assertEqual(windows[0][0], 0.0)
        self.assertGreater(windows[-1][1], windows[0][0])

    def test_merge_time_spans_joins_neighbors_and_expands_short(self):
        spans = merge_time_spans(
            [(1.0, 1.1), (1.12, 1.3), (3.0, 3.4)], gap=0.05, min_dur=0.15
        )
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][0], 1.0)
        self.assertAlmostEqual(spans[0][1], 1.3)
        self.assertAlmostEqual(spans[1][0], 3.0)

    def test_load_aligned_events_on_2k_sample_if_present(self):
        sample = Path(r"E:\output_2k_rawdata\synth_001_0042")
        if not (sample / "alignment.npz").exists():
            self.skipTest("2k sample not mounted")
        from alignmodel.stages.dc_alignment import load_aligned_events

        events = load_aligned_events(sample)
        self.assertTrue(events)
        self.assertTrue(any(not ev.get("is_rest") for ev in events))
        sounding = [ev for ev in events if not ev.get("is_rest")]
        self.assertTrue(all("perf_start" in ev and "ref_start" in ev for ev in sounding))

    def test_learned_alignment_becomes_timed_pairs(self):
        class FakeAligner:
            def align(self, notes, score):
                return AlignmentResult(
                    operations=[
                        AlignmentOperation("match", 0, 1, 0.9),
                        AlignmentOperation("extra", 1, None, 0.8),
                    ],
                    n_performance_notes=2,
                    n_score_notes=2,
                    total_cost=0.1,
                )

        state = PipelineState(
            sample_id="t",
            sample_dir=".",
            sr=22050,
            duration_sec=2.0,
            hop_sec=0.023,
            config=PipelineConfig(),
            score=ScoreGraph(notes=[_note(0, 60, 0.0), _note(1, 62, 1.0)]),
        )
        learned = SimpleNamespace(
            transcriber=object(),
            transcriber_decode=object(),
            note_aligner=FakeAligner(),
            device="cpu",
        )
        transcribed = [
            SimpleNamespace(pitch=62, start=0.8, end=1.25, confidence=0.9),
            SimpleNamespace(pitch=65, start=1.3, end=1.5, confidence=0.8),
        ]
        with patch(
            "alignmodel.transcription.infer_note_decoder",
            return_value=transcribed,
        ):
            pairs = pairs_from_learned_alignment(
                state, np.zeros((128, 10), np.float32), learned
            )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].score_index, 1)
        self.assertEqual(pairs[0].pitch, 62)
        self.assertAlmostEqual(pairs[0].perf_start, 0.8)
        self.assertAlmostEqual(pairs[0].ref_start, 1.0)
