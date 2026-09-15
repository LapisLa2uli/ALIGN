from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from music21 import note, spanner, stream, tie

from alignmodel.joint import (
    JointEvent,
    JointMetricSample,
    OracleHarnessConfig,
    ScoreEventIndex,
    evaluate_joint_dataset,
    evaluate_joint_events,
    run_oracle_harness,
)


def _write_score(path: Path) -> None:
    score = stream.Score()
    part = stream.Part()
    first = note.Note("C4", quarterLength=1)
    first.tie = tie.Tie("start")
    second = note.Note("C4", quarterLength=1)
    second.tie = tie.Tie("stop")
    third = note.Note("D4", quarterLength=1)
    fourth = note.Note("D4", quarterLength=1)
    part.append([first, second, third, fourth])
    # A same-pitch slur is deliberately not a tie.
    part.insert(0, spanner.Slur(third, fourth))
    score.insert(0, part)
    score.write("musicxml", fp=str(path))


def _lineage() -> dict:
    clean = [
        {
            "clean_index": index,
            "deleted": index == 3,
            "pitch_midi": pitch,
            "onset_ql": float(index),
            "duration_ql": 1.0,
        }
        for index, pitch in enumerate((60, 60, 62, 62))
    ]
    performed = [
        {
            "performed_index": 0,
            "clean_index": 0,
            "relationship": "match",
            "origin_relationship": "match",
            "copy_pass": 0,
            "pitch_midi": 60,
            "onset_ql": 0.0,
            "duration_ql": 2.0,
        },
        {
            "performed_index": 1,
            "clean_index": 2,
            "relationship": "substitute",
            "origin_relationship": "substitute",
            "copy_pass": 0,
            "pitch_midi": 63,
            "onset_ql": 2.0,
            "duration_ql": 1.0,
        },
        {
            "performed_index": 2,
            "clean_index": None,
            "relationship": "extra",
            "origin_relationship": "extra",
            "copy_pass": 0,
            "pitch_midi": 70,
            "onset_ql": 3.0,
            "duration_ql": 0.5,
        },
        {
            "performed_index": 3,
            "clean_index": 0,
            "relationship": "copy",
            "origin_relationship": "match",
            "copy_pass": 1,
            "pitch_midi": 60,
            "onset_ql": 4.0,
            "duration_ql": 2.0,
        },
    ]
    rendered = [
        {
            "rendered_index": 0,
            "performed_indices": [0],
            "clean_indices": [0, 1],
            "primary_clean_index": 0,
            "relationship": "match",
            "pitch_midi_written": 60,
            "start_sec": 0.0,
            "end_sec": 2.0,
        },
        {
            "rendered_index": 1,
            "performed_indices": [1],
            "clean_indices": [2],
            "primary_clean_index": 2,
            "relationship": "substitute",
            "pitch_midi_written": 63,
            "start_sec": 2.0,
            "end_sec": 3.0,
        },
        {
            "rendered_index": 2,
            "performed_indices": [2],
            "clean_indices": [],
            "primary_clean_index": None,
            "relationship": "extra",
            "pitch_midi_written": 70,
            "start_sec": 3.0,
            "end_sec": 3.5,
        },
        {
            "rendered_index": 3,
            "performed_indices": [3],
            "clean_indices": [0, 1],
            "primary_clean_index": 0,
            "relationship": "copy",
            "pitch_midi_written": 60,
            "start_sec": 4.0,
            "end_sec": 6.0,
        },
    ]
    return {
        "schema_version": "1.1",
        "kind": "synth_note_lineage",
        "clean_notes": clean,
        "performed_notes": performed,
        "rendered_notes": rendered,
        "deleted_clean_notes": [3],
    }


def _event(
    pitch: int,
    start: float,
    span: tuple[int, int] | None,
    relationship: str = "match",
) -> JointEvent:
    return JointEvent(
        pitch=pitch,
        start=start,
        end=start + 0.4,
        score_span=span,
        relationship=relationship,
        copy_pass=1 if relationship == "copy" else 0,
        rendered_index=int(start * 1000) if relationship == "extra" else None,
    )


class ScoreEventIndexTests(unittest.TestCase):
    def test_ties_only_lineage_projection_and_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "score.musicxml"
            _write_score(score_path)
            index = ScoreEventIndex.from_musicxml(score_path, _lineage())

        self.assertEqual(len(index.events), 3)
        self.assertEqual(index.events[0].source_indices, (0, 1))
        self.assertEqual(index.events[1].source_indices, (2,))
        self.assertEqual(index.events[2].source_indices, (3,))
        self.assertEqual(index.source_to_event, (0, 0, 1, 2))
        self.assertEqual(index.deleted_event_indices, frozenset({2}))
        self.assertEqual(index.event_span_for_sources([2, 3]), (1, 3))

        rendered = index.rendered_events
        self.assertEqual(rendered[0].score_span, (0, 1))
        self.assertEqual(rendered[1].relationship, "substitute")
        self.assertEqual(rendered[1].score_span, (1, 2))
        self.assertTrue(rendered[2].is_extra)
        self.assertTrue(rendered[3].is_copy)
        self.assertEqual(rendered[3].score_span, (0, 1))

        reconstructed = ScoreEventIndex.from_lineage(_lineage())
        self.assertEqual(
            [
                (
                    event.pitch,
                    event.ql_start,
                    event.ql_end,
                    event.source_indices,
                )
                for event in reconstructed.events
            ],
            [
                (
                    event.pitch,
                    event.ql_start,
                    event.ql_end,
                    event.source_indices,
                )
                for event in index.events
            ],
        )
        self.assertEqual(
            reconstructed.rendered_events,
            index.rendered_events,
        )

    def test_spanning_render_event_and_invalid_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "score.musicxml"
            _write_score(score_path)
            spanning = _lineage()
            spanning["rendered_notes"][0]["clean_indices"] = [0, 3]
            index = ScoreEventIndex.from_musicxml(score_path, spanning)
            self.assertEqual(index.rendered_events[0].score_span, (0, 3))

            bad = _lineage()
            bad["clean_notes"][3]["clean_index"] = 99
            with self.assertRaisesRegex(ValueError, "do not match"):
                ScoreEventIndex.from_musicxml(score_path, bad)


class JointMetricTests(unittest.TestCase):
    def test_strict_tolerances_mapping_copy_extras_and_deletions(self) -> None:
        target = [
            _event(60, 0.00, (0, 1)),
            _event(62, 1.00, (1, 2), "copy"),
            _event(64, 2.00, None, "extra"),
            _event(65, 3.00, (2, 3), "substitute"),
        ]
        predicted = [
            _event(60, 0.03, (0, 1)),
            _event(62, 1.00, (0, 1), "copy"),
            _event(64, 2.00, None, "extra"),
            _event(65, 3.00, (2, 3), "substitute"),
        ]
        report = evaluate_joint_events(
            predicted,
            target,
            predicted_deletions={2, 3},
            target_deletions={2},
            score_event_count=4,
        )
        at_20 = report["tolerances"]["20ms"]
        at_50 = report["tolerances"]["50ms"]
        self.assertEqual(at_20["counts"]["paired"], 3)
        self.assertEqual(at_50["note"]["f1"], 1.0)
        self.assertEqual(at_50["joint"]["f1"], 0.75)
        self.assertAlmostEqual(
            at_50["conditional_mapping_accuracy"], 2 / 3
        )
        self.assertEqual(at_50["copy"]["recall"], 0.0)
        self.assertEqual(at_50["extras"]["f1"], 1.0)
        self.assertEqual(at_50["substitution_mapping_accuracy"], 1.0)
        self.assertAlmostEqual(
            report["current_metric_50ms"]["f1"], 2 / 3
        )
        self.assertEqual(report["deletions"]["precision"], 0.5)
        self.assertEqual(report["deletions"]["recall"], 1.0)

    def test_repeated_score_event_revisit_and_source_aggregation(self) -> None:
        target = [
            _event(60, 0.0, (0, 1)),
            _event(60, 1.0, (0, 1), "copy"),
        ]
        samples = [
            JointMetricSample(target, target, source="source-a", score_event_count=1),
            JointMetricSample(target, target, source="source-b", score_event_count=1),
        ]
        report = evaluate_joint_dataset(samples)
        self.assertEqual(report["aggregate"]["n_samples"], 2)
        self.assertEqual(report["per_source"]["source-a"]["n_samples"], 1)
        self.assertEqual(
            report["aggregate"]["tolerances"]["50ms"]["copy"]["recall"],
            1.0,
        )

    def test_invalid_metric_index_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            evaluate_joint_events(
                [_event(60, 0.0, (2, 3))],
                [_event(60, 0.0, (0, 1))],
                score_event_count=2,
            )


class OracleHarnessTests(unittest.TestCase):
    def test_non_test_harness_writes_json_and_optional_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample-a"
            sample.mkdir()
            _write_score(sample / "verified_score.musicxml")
            (sample / "note_map.json").write_text(
                json.dumps(_lineage()), encoding="utf-8"
            )
            candidate_root = root / "candidates"
            (candidate_root / "corpus-a").mkdir(parents=True)
            np.savez_compressed(
                candidate_root / "corpus-a" / "sample-a.npz",
                pitch=np.asarray([60, 63, 70, 60]),
                start=np.asarray([0.0, 2.0, 3.0, 4.0]),
                end=np.asarray([2.0, 3.0, 3.5, 6.0]),
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "val": [
                            {
                                "sample": "sample-a",
                                "sample_dir": str(sample),
                                "corpus": "corpus-a",
                                "source": "source-a",
                                "split": "val",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "oracle.json"
            result = run_oracle_harness(
                OracleHarnessConfig(
                    manifest=manifest,
                    split="val",
                    output=output,
                    candidate_root=candidate_root,
                )
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(saved["schema_version"], "align-joint-oracle-v1")
        self.assertEqual(result["n_samples"], 1)
        self.assertIn("acoustic_candidates_oracle_repeats", result["stages"])
        oracle = result["stages"]["oracle_notes_oracle_repeats"]["aggregate"]
        self.assertEqual(oracle["tolerances"]["50ms"]["joint"]["f1"], 1.0)

    def test_protected_test_split_is_rejected_before_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"test_id": [{}]}), encoding="utf-8")
            output = root / "must-not-exist.json"
            with self.assertRaisesRegex(ValueError, "protected"):
                run_oracle_harness(
                    OracleHarnessConfig(
                        manifest=manifest,
                        split="test_id",
                        output=output,
                    )
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
