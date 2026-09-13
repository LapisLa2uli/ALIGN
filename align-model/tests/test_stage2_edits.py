import unittest

import numpy as np
import torch
from torch import nn

from alignmodel.melody import extra_neighbor_core, pred_melodies_from_labels
from alignmodel.stage_train import (
    EDIT_CLASSES,
    EDIT_IOU_POSITIVE,
    _limit_edit_training_items,
    heuristic_edit_class,
    span_iou,
)
from alignmodel.stages.edits import (
    _f0_intonation_spans,
    _fine_cents_for_event,
    run_stage2,
)
from alignmodel.stages.learned import (
    StageModels,
    apply_learned_edits,
    edit_error_probability,
    gate_edit_prediction,
)
from alignmodel.types import (
    GraphNote,
    PipelineConfig,
    PipelineLabel,
    PipelineState,
    ScoreGraph,
    UnfoldedSegment,
)
from datacreate.melody import ScoreSoundingNote


def _notes(n: int) -> list[ScoreSoundingNote]:
    return [
        ScoreSoundingNote(
            index=i,
            note_id=f"note_{i:04d}",
            pitch=60 + i,
            start=float(i),
            end=float(i) + 0.8,
            ql_start=float(i),
            ql_end=float(i) + 1.0,
            measure=1,
        )
        for i in range(n)
    ]


def _state(labels: list[PipelineLabel]) -> PipelineState:
    return PipelineState(
        sample_id="clip",
        sample_dir=".",
        sr=22050,
        duration_sec=4.0,
        hop_sec=0.023,
        config=PipelineConfig(),
        score=ScoreGraph(notes=[], duration_sec=4.0),
        labels=labels,
    )


class _FixedEditNet(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self._logits.expand(mel.size(0), -1)


class Stage2EditTests(unittest.TestCase):
    def test_fine_pitch_comparison_keeps_sub_semitone_cents(self):
        strength = np.ones(20, dtype=np.float32)
        ref = (np.full(20, 60.0, dtype=np.float32), strength)
        perf = (np.full(20, 60.55, dtype=np.float32), strength)
        cents = _fine_cents_for_event(perf, ref, 0.0, 1.0, 0.0, 1.0, 0.05)
        self.assertAlmostEqual(float(cents or 0.0), 55.0, places=2)

    def test_aligned_f0_groups_sustained_intonation(self):
        strength = np.ones(12, dtype=np.float32)
        ref = (np.full(12, 60.0, dtype=np.float32), strength)
        perf_midi = np.full(12, 60.0, dtype=np.float32)
        perf_midi[3:9] += 0.55
        perf = (perf_midi, strength)
        path = np.asarray([[i, i] for i in range(12)], dtype=np.int32)
        spans = _f0_intonation_spans(
            path,
            perf,
            ref,
            perf_origin=0.0,
            ref_origin=0.0,
            hop_sec=0.05,
            tolerance=20.0,
        )
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][0], 0.15)
        self.assertAlmostEqual(spans[0][1], 0.45)
        self.assertAlmostEqual(spans[0][2], 55.0, places=2)

    def test_training_cap_preserves_intonation_examples(self):
        items = []
        for class_i in range(len(EDIT_CLASSES)):
            for i in range(100):
                items.append({"dir": f"sample-{class_i}-{i}", "y": class_i})
        limited = _limit_edit_training_items(items, 100, seed=365)
        counts = {
            EDIT_CLASSES[class_i]: sum(int(item["y"]) == class_i for item in limited)
            for class_i in range(len(EDIT_CLASSES))
        }
        self.assertEqual(len(limited), 100)
        self.assertEqual(counts["match"], 40)
        self.assertEqual(counts["intonation_error"], 15)

    def test_stage2_proposes_intonation_error(self):
        frames = 48
        ref_chroma = np.zeros((12, frames), dtype=np.float32)
        perf_chroma = np.zeros((12, frames), dtype=np.float32)
        ref_chroma[0, :] = 1.0
        perf_chroma[0, :] = 1.0
        perf_chroma[1, :] = 0.2
        note = GraphNote(
            index=0,
            pitch=60,
            start=0.0,
            end=1.0,
            duration=1.0,
            ql_start=0.0,
            ql_end=1.0,
            measure=1,
        )
        state = PipelineState(
            sample_id="intonation",
            sample_dir=".",
            sr=22050,
            duration_sec=1.0,
            hop_sec=1.0 / frames,
            config=PipelineConfig(
                weights_dir=None,
                cents_tolerance=20.0,
                detect_intonation=True,
            ),
            score=ScoreGraph(notes=[note], duration_sec=1.0),
            segments=[UnfoldedSegment(0.0, 1.0, 0, 1, 0.0)],
        )
        run_stage2(
            state,
            np.zeros(0, dtype=np.float32),
            perf_chroma,
            ref_chroma,
        )
        intonation = [lab for lab in state.labels if lab.type == "intonation_error"]
        self.assertEqual(len(intonation), 1)
        self.assertGreater(float(intonation[0].deviation_cents or 0.0), 20.0)

    def test_full_semitone_stays_wrong_note(self):
        frames = 48
        ref_chroma = np.zeros((12, frames), dtype=np.float32)
        perf_chroma = np.zeros((12, frames), dtype=np.float32)
        ref_chroma[0, :] = 1.0
        perf_chroma[1, :] = 1.0
        note = GraphNote(
            index=0,
            pitch=60,
            start=0.0,
            end=1.0,
            duration=1.0,
            ql_start=0.0,
            ql_end=1.0,
            measure=1,
        )
        state = PipelineState(
            sample_id="wrong-note",
            sample_dir=".",
            sr=22050,
            duration_sec=1.0,
            hop_sec=1.0 / frames,
            config=PipelineConfig(weights_dir=None, cents_tolerance=20.0),
            score=ScoreGraph(notes=[note], duration_sec=1.0),
            segments=[UnfoldedSegment(0.0, 1.0, 0, 1, 0.0)],
        )
        run_stage2(
            state,
            np.zeros(0, dtype=np.float32),
            perf_chroma,
            ref_chroma,
        )
        self.assertEqual([lab.type for lab in state.labels], ["wrong_note"])

    def test_binary_gate_uses_error_probability_not_argmax_confidence(self):
        # P(match)=0.45, P(missed)=0.35, rest spread. Old gate drops (0.35 < 0.40).
        logits = torch.tensor([0.45, 0.35, 0.08, 0.07, 0.05]).log()
        error_p = float(edit_error_probability(logits).item())
        self.assertGreater(error_p, 0.50)
        self.assertEqual(gate_edit_prediction(logits, 0.40), 1)
        self.assertEqual(gate_edit_prediction(logits, 0.70), 0)

    def test_high_match_probability_drops_heuristic_wrong_note(self):
        logits = torch.tensor([2.5, 0.2, 0.1, 0.1, 0.1])
        self.assertEqual(gate_edit_prediction(logits, 0.35), 0)

    def test_span_iou_and_hard_negative_matching(self):
        self.assertAlmostEqual(span_iou(0.0, 1.0, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(span_iou(0.0, 1.0, 2.0, 3.0), 0.0)
        gold = [("wrong_note", 1.0, 2.0), ("extra_note", 3.0, 3.4)]
        self.assertEqual(
            heuristic_edit_class(1.05, 1.95, "wrong_note", gold),
            EDIT_CLASSES.index("wrong_note"),
        )
        self.assertEqual(
            heuristic_edit_class(5.0, 5.4, "wrong_note", gold),
            EDIT_CLASSES.index("match"),
        )
        self.assertGreaterEqual(EDIT_IOU_POSITIVE, 0.25)

    def test_apply_learned_edits_drops_low_error_probability(self):
        labels = [
            PipelineLabel("e1", "wrong_note", 0.2, 0.6),
            PipelineLabel("r1", "repetition", 1.0, 2.0),
        ]
        state = _state(labels)
        mel = np.full((128, 64), -40.0, dtype=np.float32)
        logits = torch.tensor([[3.0, 0.1, 0.1, 0.1, 0.1]])
        models = StageModels(
            edits=_FixedEditNet(logits),
            device=torch.device("cpu"),
            edits_threshold=0.35,
        )
        apply_learned_edits(state, mel, models)
        self.assertEqual([lab.type for lab in state.labels], ["repetition"])

    def test_apply_learned_edits_keeps_heuristic_type_when_emitting(self):
        labels = [PipelineLabel("e1", "wrong_note", 0.2, 0.6)]
        state = _state(labels)
        mel = np.full((128, 64), -40.0, dtype=np.float32)
        logits = torch.tensor([[0.2, 1.5, 0.3, 0.2, 0.1]])
        models = StageModels(
            edits=_FixedEditNet(logits),
            device=torch.device("cpu"),
            edits_threshold=0.35,
        )
        apply_learned_edits(state, mel, models)
        self.assertEqual(len(state.labels), 1)
        self.assertEqual(state.labels[0].type, "wrong_note")
        self.assertIn("stage2-net", state.labels[0].comment or "")

    def test_extra_note_maps_to_neighbor_pitch_window(self):
        notes = _notes(8)
        i0, i1 = extra_neighbor_core(notes, 3)
        self.assertEqual((i0, i1), (3, 5))
        pred = pred_melodies_from_labels(
            [
                {
                    "type": "extra_note",
                    "start_time": 3.0,
                    "end_time": 3.6,
                    "pitches": [63, 64, 65],
                    "note_ids": ["note_0003", "note_0004", "note_0005"],
                }
            ],
            notes,
            pad_notes=2,
        )
        self.assertEqual(pred[0].type, "extra_note")
        self.assertEqual(pred[0].pitches, [63, 64, 65])


if __name__ == "__main__":
    unittest.main()
