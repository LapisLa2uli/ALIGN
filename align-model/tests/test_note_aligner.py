from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from alignmodel.note_align_train import NoteAlignTrainConfig, train_note_aligner
from alignmodel.stages.note_align import (
    NoteAlignConfig,
    NoteAligner,
    alignment_metrics,
)
from alignmodel.types import GraphNote, LegalEdge, ScoreGraph


def make_graph(pitches: list[int]) -> ScoreGraph:
    notes = [
        GraphNote(
            index=i,
            pitch=pitch,
            start=i * 0.5,
            end=i * 0.5 + 0.4,
            duration=0.4,
            ql_start=float(i),
            ql_end=float(i + 1),
            measure=1 + i // 4,
        )
        for i, pitch in enumerate(pitches)
    ]
    return ScoreGraph(
        notes=notes,
        legal_edges=[
            LegalEdge(i, i + 1, "next") for i in range(len(notes) - 1)
        ],
        duration_sec=notes[-1].end if notes else 0.0,
    )


def played(pitches: list[int], confidences: list[float] | None = None) -> list[dict]:
    confidences = confidences or [1.0] * len(pitches)
    return [
        {
            "written_pitch": pitch,
            "start": i * 0.46,
            "end": i * 0.46 + 0.36,
            "confidence": confidences[i],
        }
        for i, pitch in enumerate(pitches)
    ]


class NoteAlignerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aligner = NoteAligner(config=NoteAlignConfig(min_copy_notes=2))

    def test_mid_piece_repeat_maps_to_original_score(self) -> None:
        graph = make_graph([60, 62, 64, 65, 67, 69])
        notes = played([60, 62, 64, 62, 64, 65, 67, 69])
        result = self.aligner.align(notes, graph)

        self.assertEqual(result.mapping, [0, 1, 2, 1, 2, 3, 4, 5])
        self.assertEqual(len(result.copy_blocks), 1)
        self.assertEqual(result.copy_blocks[0].score_indices, [1, 2])
        copy_ops = [op for op in result.operations if op.is_copy]
        self.assertEqual([op.score_index for op in copy_ops], [1, 2])

        metrics = alignment_metrics(
            result,
            [0, 1, 2, 1, 2, 3, 4, 5],
            target_is_copy=[False, False, False, True, True, False, False, False],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["copy_accuracy"], 1.0)
        self.assertEqual(metrics["copy_f1"], 1.0)

    def test_spurious_note_is_extra(self) -> None:
        graph = make_graph([60, 62, 64, 65])
        result = self.aligner.align(played([60, 62, 91, 64, 65]), graph)
        self.assertEqual(result.mapping, [0, 1, None, 2, 3])
        self.assertEqual(result.extras, [2])
        self.assertFalse(result.copy_blocks)

    def test_missing_score_note_is_deletion(self) -> None:
        graph = make_graph([60, 62, 64, 65, 67])
        result = self.aligner.align(played([60, 62, 65, 67]), graph)
        self.assertEqual(result.mapping, [0, 1, 3, 4])
        self.assertIn(2, result.deletions)

    def test_noisy_sequence_keeps_context(self) -> None:
        graph = make_graph([60, 62, 64, 65, 67, 69, 71])
        # Pitch error at index 1, a dropped 65, and an unrelated inserted note.
        result = self.aligner.align(played([60, 63, 64, 90, 67, 69, 71]), graph)
        self.assertEqual(result.mapping, [0, 1, 2, None, 4, 5, 6])
        substitute = next(op for op in result.operations if op.performance_index == 1)
        self.assertEqual(substitute.kind, "substitute")
        self.assertIn(3, result.deletions)

    def test_low_confidence_note_is_unattached(self) -> None:
        graph = make_graph([60, 62, 64])
        result = self.aligner.align(
            played([60, 62, 64], confidences=[1.0, 0.03, 1.0]), graph
        )
        self.assertEqual(result.mapping, [0, None, 2])
        self.assertEqual(result.unattached, [1])
        uncertain = next(op for op in result.operations if op.performance_index == 1)
        self.assertLess(uncertain.confidence, 0.18)

    def test_tiny_training_writes_history_and_calibration(self) -> None:
        def cache_payload() -> dict:
            clean = [
                {
                    "clean_index": i,
                    "deleted": False,
                    "pitch_midi": pitch,
                    "pitch": str(pitch),
                    "onset_ql": float(i),
                    "duration_ql": 0.8,
                    "measure": 1,
                }
                for i, pitch in enumerate((60, 62, 64, 65))
            ]
            performed = [
                {
                    "performed_index": i,
                    "clean_index": i,
                    "relationship": "match",
                    "origin_relationship": "match",
                    "copy_pass": 0,
                    "pitch_midi": row["pitch_midi"],
                    "pitch": row["pitch"],
                    "onset_ql": row["onset_ql"],
                    "duration_ql": row["duration_ql"],
                    "measure": 1,
                }
                for i, row in enumerate(clean)
            ]
            return {
                "schema_version": "1.0",
                "kind": "synth_note_lineage",
                "clean_notes": clean,
                "performed_notes": performed,
                "deleted_clean_notes": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sample in ("a", "b"):
                directory = root / sample
                directory.mkdir()
                (directory / "note_map.json").write_text(
                    json.dumps(cache_payload()), encoding="utf-8"
                )
            output = root / "out"
            checkpoint = train_note_aligner(
                NoteAlignTrainConfig(
                    data_root=root,
                    output_dir=output,
                    epochs=1,
                    batch_size=32,
                    hidden_dim=8,
                    augmentations_per_map=0,
                    device="cpu",
                    calibration_maps=1,
                )
            )
            blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("calibration", blob)
            self.assertTrue(blob["history"])
            self.assertTrue((output / "note_aligner_history.json").exists())


if __name__ == "__main__":
    unittest.main()
