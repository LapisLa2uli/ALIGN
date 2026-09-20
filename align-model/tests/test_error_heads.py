from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from alignmodel.joint.error_heads import (
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYER2_CLASSES,
    ErrorHeadsConfig,
    FrozenUpstreamErrorHeads,
    HeadPrediction,
    HeadRow,
    SchemaDecodeConfig,
    attach_training_targets,
    build_inference_rows,
    build_oracle_rows,
    blend_direct_learned_prediction,
    direct_operation_probabilities,
    error_head_loss,
    heuristic_prediction,
    infer_error_heads,
    load_error_heads,
    measured_class_weights,
    pair_events_sequence_consistent,
    save_checkpoint_atomic,
    schema12_document,
    schema12_document_v3,
    stack_labeled_rows,
)
from alignmodel.joint.index import JointEvent, ScoreEvent
from alignmodel.joint.lattice import (
    JointCandidate,
    JointEdgeScorer,
    JointOperation,
    LatticeConfig,
    LatticePath,
    LatticeStep,
    SparseJointLattice,
)
from alignmodel.melody import WeakMelody, match_melodies_detail, pred_melodies_from_labels


def _score() -> tuple[ScoreEvent, ...]:
    return (
        ScoreEvent(0, 60, 0.0, 1.0, (0, 1), measure=1),
        ScoreEvent(1, 62, 1.0, 2.0, (2,), measure=1),
        ScoreEvent(2, 64, 2.0, 3.0, (3,), measure=2),
    )


def _candidates() -> tuple[JointCandidate, ...]:
    acoustic = (0.8, 0.7, 0.6, -0.05, 0.5)
    return (
        JointCandidate(60, 0.0, 0.45, 0.9, acoustic_features=acoustic),
        JointCandidate(65, 1.0, 1.45, 0.8, acoustic_features=acoustic),
        JointCandidate(70, 1.6, 1.85, 0.7, acoustic_features=acoustic),
    )


def _path() -> LatticePath:
    return LatticePath(
        (
            LatticeStep(0, (0, 1), JointOperation.MATCH, None, None, ()),
            LatticeStep(
                1,
                (2, 3),
                JointOperation.SUBSTITUTE,
                None,
                None,
                (1,),
            ),
            LatticeStep(2, None, JointOperation.EXTRA, None, None, ()),
        ),
        (),
        0.0,
    )


def _lattice() -> SparseJointLattice:
    torch.manual_seed(10)
    scorer = JointEdgeScorer(hidden_dim=12, dropout=0.0)
    return SparseJointLattice(
        scorer,
        LatticeConfig(max_options_per_candidate=12, max_states=24),
    )


def _targets() -> tuple[JointEvent, ...]:
    return (
        JointEvent(60, 0.0, 0.45, (0, 1), "match"),
        JointEvent(65, 1.0, 1.45, (2, 3), "substitute"),
        JointEvent(70, 1.6, 1.85, None, "extra"),
    )


def _labeled():
    lattice = _lattice()
    candidates = _candidates()
    score = _score()
    path = _path()
    events = tuple(path.joint_events(candidates))
    rows = build_inference_rows(lattice, candidates, score, path)
    labeled = attach_training_targets(
        rows,
        predicted_events=events,
        target_events=_targets(),
        target_deletions=(1,),
        rhythm_rows=(
            {
                "rendered_event": 1,
                "rhythm_error": True,
                "supervision_mask": True,
            },
        ),
        score=score,
    )
    return lattice, rows, labeled


class ErrorTargetTests(unittest.TestCase):
    def test_missing_extra_wrong_and_tied_score_targets(self) -> None:
        _lattice_value, rows, labeled = _labeled()
        names = [LAYER2_CLASSES[index] for index in labeled.layer2]
        self.assertEqual(names[:3], ["match", "wrong_note", "extra_note"])
        self.assertIn("missed_note", names)
        tied = rows[0]
        self.assertEqual(tied.score_span, (0, 1))
        self.assertEqual(_score()[0].source_indices, (0, 1))
        self.assertTrue(all(row.features.shape == (FEATURE_DIM,) for row in rows))

    def test_sequence_pairing_resynchronizes_after_missing_note(self) -> None:
        predicted = (
            JointEvent(60, 0.0, 0.4, (0, 1), "match"),
            JointEvent(64, 1.0, 1.4, (2, 3), "match"),
        )
        target = (
            JointEvent(60, 0.0, 0.4, (0, 1), "match"),
            JointEvent(62, 0.5, 0.9, (1, 2), "match"),
            JointEvent(64, 1.0, 1.4, (2, 3), "match"),
        )
        self.assertEqual(
            pair_events_sequence_consistent(predicted, target),
            {0: 0, 1: 2},
        )

    def test_replay_is_excluded_from_rhythm_supervision(self) -> None:
        score = _score()
        target = (
            JointEvent(60, 0.0, 0.5, (0, 1), "match"),
            JointEvent(60, 0.7, 1.2, (0, 1), "copy", copy_pass=1),
            JointEvent(62, 1.3, 1.8, (1, 2), "match"),
        )
        candidates = tuple(
            JointCandidate(
                value.pitch,
                value.start,
                value.end,
                0.9,
                acoustic_features=(0.8, 0.7, 0.6, -0.05, 0.5),
            )
            for value in target
        )
        lattice = _lattice()
        rows = build_oracle_rows(lattice, candidates, score, target, ())
        labeled = attach_training_targets(
            rows,
            predicted_events=target,
            target_events=target,
            target_deletions=(),
            rhythm_rows=(
                {
                    "rendered_event": 1,
                    "rhythm_error": True,
                    "supervision_mask": True,
                },
            ),
            score=score,
        )
        event_rows = [index for index, row in enumerate(rows) if row.kind == "event"]
        self.assertFalse(bool(labeled.rhythm_mask[event_rows[1]]))
        self.assertFalse(bool(labeled.rhythm[event_rows[1]]))

    def test_predicted_replay_state_excludes_rhythm_even_if_target_is_ordinary(
        self,
    ) -> None:
        score = _score()
        row = HeadRow(
            np.zeros(FEATURE_DIM, dtype=np.float32),
            "event",
            0,
            (0, 1),
            0.0,
            0.5,
            True,
        )
        predicted = (JointEvent(60, 0.0, 0.5, (0, 1), "copy", copy_pass=1),)
        target = (JointEvent(60, 0.0, 0.5, (0, 1), "match"),)
        labeled = attach_training_targets(
            (row,),
            predicted_events=predicted,
            target_events=target,
            target_deletions=(),
            rhythm_rows=(
                {
                    "rendered_event": 0,
                    "rhythm_error": True,
                    "supervision_mask": True,
                },
            ),
            score=score,
        )
        self.assertFalse(bool(labeled.rhythm_mask[0]))
        self.assertFalse(bool(labeled.rhythm[0]))

    def test_tempo_normalization_removes_global_expression(self) -> None:
        score = _score()
        target = (
            JointEvent(60, 0.0, 0.5, (0, 1), "match"),
            JointEvent(62, 0.5, 1.0, (1, 2), "match"),
            JointEvent(64, 1.0, 1.5, (2, 3), "match"),
        )
        candidates = tuple(
            JointCandidate(
                event.pitch,
                event.start,
                event.end,
                acoustic_features=(0.8, 0.7, 0.6, -0.05, 0.5),
            )
            for event in target
        )
        lattice = _lattice()
        rows = build_oracle_rows(lattice, candidates, score, target, ())
        labeled = attach_training_targets(
            rows,
            predicted_events=target,
            target_events=target,
            target_deletions=(),
            rhythm_rows=(),
            score=score,
        )
        selected = labeled.deviation_sec[labeled.deviation_mask]
        self.assertTrue(np.all(np.abs(selected) < 1e-6), selected)


class ErrorHeadTrainingTests(unittest.TestCase):
    def test_all_heads_receive_gradient(self) -> None:
        _lattice_value, _rows, labeled = _labeled()
        arrays = stack_labeled_rows((labeled, labeled))
        model = FrozenUpstreamErrorHeads(
            ErrorHeadsConfig(hidden_dim=24, dropout=0.0)
        )
        output = model(torch.from_numpy(arrays["features"]))
        targets = {
            name: torch.from_numpy(value)
            for name, value in arrays.items()
            if name != "features"
        }
        loss, _detail = error_head_loss(
            output,
            targets,
            layer2_weights=measured_class_weights(
                arrays["layer2"], len(LAYER2_CLASSES)
            ),
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.all(torch.isfinite(parameter.grad)), name)

    def test_checkpoint_resume_preserves_optimizer_and_sampler_cursor(self) -> None:
        _lattice_value, _rows, labeled = _labeled()
        arrays = stack_labeled_rows((labeled,))
        config = ErrorHeadsConfig(hidden_dim=20, dropout=0.0)
        model = FrozenUpstreamErrorHeads(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        output = model(torch.from_numpy(arrays["features"]))
        targets = {
            name: torch.from_numpy(value)
            for name, value in arrays.items()
            if name != "features"
        }
        loss, _detail = error_head_loss(
            output,
            targets,
            layer2_weights=measured_class_weights(
                arrays["layer2"], len(LAYER2_CLASSES)
            ),
        )
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                progress={
                    "epoch": 1,
                    "sampler": {"seed": 11, "position": 7},
                    "rows_per_sec": 123.0,
                    "eta_seconds": 4.0,
                },
                history=(),
                data_fingerprint="pack",
                upstream={"decoder": {"sha256": "upstream"}},
            )
            loaded, payload = load_error_heads(
                checkpoint,
                expected_data_fingerprint="pack",
                expected_upstream={"decoder": {"sha256": "upstream"}},
            )
        self.assertEqual(payload["progress"]["sampler"]["position"], 7)
        self.assertIn("optimizer_state_dict", payload)
        self.assertIn("scheduler_state_dict", payload)
        self.assertIn("scaler_state_dict", payload)
        self.assertIn("rng_state", payload)
        for expected, actual in zip(model.parameters(), loaded.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_inference_has_no_gold_dependency(self) -> None:
        _lattice_value, rows, _labeled_value = _labeled()
        model = FrozenUpstreamErrorHeads(
            ErrorHeadsConfig(hidden_dim=20, dropout=0.0)
        )
        thresholds = {
            "layer2": {
                "wrong_note": 0.5,
                "extra_note": 0.5,
                "missed_note": 0.5,
            },
            "rhythm": 0.9,
        }
        with mock.patch(
            "alignmodel.joint.error_heads.attach_training_targets",
            side_effect=AssertionError("gold accessed"),
        ):
            prediction = infer_error_heads(model, rows, thresholds)
        self.assertEqual(len(prediction.layer2), len(rows))

    def test_schema_12_output_has_score_ranges(self) -> None:
        _lattice_value, rows, _labeled_value = _labeled()
        prediction = heuristic_prediction(rows)
        document = schema12_document("fixture", rows, prediction, _score())
        self.assertEqual(document["schema_version"], "1.2")
        self.assertTrue(document["labels"])
        for label in document["labels"]:
            self.assertIn("score_part", label)
            self.assertTrue(label["pitches"])
            self.assertEqual(len(label["pitches"]), len(label["note_ids"]))

    def test_schema_12_projection_uses_official_padding_and_merges_runs(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(5)
        )
        rows = tuple(
            HeadRow(
                np.zeros(FEATURE_DIM, dtype=np.float32),
                "event",
                i,
                (i, i + 1),
                float(i),
                float(i + 1),
                False,
            )
            for i in range(5)
        )
        prediction = HeadPrediction(
            ("match", "wrong_note", "wrong_note", "match", "match"),
            (False,) * 5,
            (0.0,) * 5,
            ("none",) * 5,
            ((1.0, 0.0, 0.0, 0.0),) * 5,
            (0.0,) * 5,
        )
        document = schema12_document("fixture", rows, prediction, score)
        self.assertEqual(len(document["labels"]), 1)
        label = document["labels"][0]
        self.assertEqual(label["pitches"], [60, 61, 62, 63])
        self.assertEqual(label["score_part"]["pad_notes"], 1)
        self.assertEqual(label["score_part"]["core_start_note_index"], 1)
        self.assertEqual(label["score_part"]["core_end_note_index"], 2)

    def test_schema_12_projection_passes_official_exclusive_metric(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(5)
        )
        row = HeadRow(
            np.zeros(FEATURE_DIM, dtype=np.float32),
            "event",
            0,
            (2, 3),
            2.0,
            3.0,
            False,
        )
        prediction = HeadPrediction(
            ("wrong_note",),
            (False,),
            (0.0,),
            ("none",),
            ((0.0, 1.0, 0.0, 0.0),),
            (0.0,),
        )
        document = schema12_document("fixture", (row,), prediction, score)
        predicted = pred_melodies_from_labels(document["labels"], [], pad_notes=1)
        gold = [WeakMelody([61, 62, 63], type="wrong_note")]
        metric = match_melodies_detail(gold, predicted)
        self.assertEqual(metric["f1"], 1.0)

    def test_schema_12_extra_uses_neighbors_and_copy_is_one_repetition(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(6)
        )
        rows = (
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 0, (2, 3), 2.0, 2.5, False),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 1, (2, 3), 2.5, 3.0, False),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 2, (0, 1), 4.0, 4.5, True),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 3, (1, 2), 4.5, 5.0, True),
        )
        prediction = HeadPrediction(
            ("extra_note", "extra_note", "match", "match"),
            (False,) * 4,
            (0.0,) * 4,
            ("none",) * 4,
            ((0.0, 0.0, 1.0, 0.0),) * 4,
            (0.0,) * 4,
        )
        labels = schema12_document("fixture", rows, prediction, score)["labels"]
        self.assertEqual([label["type"] for label in labels], ["extra_note", "repetition"])
        self.assertEqual(labels[0]["pitches"], [61, 62, 63, 64])
        self.assertEqual(labels[1]["pitches"], [60, 61, 62])
        self.assertEqual(labels[1]["extra_copies"], 1)

    def test_unpaired_transcription_is_suppressed_without_audited_error(self) -> None:
        score = _score()
        row = HeadRow(
            np.zeros(FEATURE_DIM, dtype=np.float32),
            "event",
            0,
            None,
            3.0,
            3.2,
            False,
        )
        predicted = (JointEvent(70, 3.0, 3.2, None, "extra"),)
        labeled = attach_training_targets(
            (row,),
            predicted_events=predicted,
            target_events=(),
            target_deletions=(),
            rhythm_rows=(),
            score=score,
        )
        self.assertEqual(
            LAYER2_CLASSES[int(labeled.layer2[0])],
            "match",
        )

    def test_v3_extra_abstains_without_two_stable_neighbors(self) -> None:
        score = _score()
        rows = (
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 0, (0, 1), 0.0, 0.5, False),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 1, None, 0.5, 0.7, False),
        )
        prediction = HeadPrediction(
            ("match", "extra_note"),
            (False, False),
            (0.0, 0.0),
            ("none", "none"),
            ((0.9, 0.02, 0.04, 0.04), (0.1, 0.05, 0.8, 0.05)),
            (0.0, 0.0),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"extra_note": 0.5},
            low_ratios={"extra_note": 1.0},
            minimum_support={"extra_note": 1},
            uncertainty_margins={"extra_note": 0.0},
            merge_score_gap={"extra_note": 0},
        )
        document = schema12_document_v3(
            "fixture", rows, prediction, score, config, include_repetition=False
        )
        self.assertEqual(document["labels"], [])

    def test_v3_consecutive_delete_run_merges_after_resynchronization(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(5)
        )
        rows = (
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 0, (0, 1), 0.0, 0.5, False),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 1, (3, 4), 0.7, 1.2, False),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "gap", None, (1, 2), 0.5, 0.6, False),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "gap", None, (2, 3), 0.6, 0.7, False),
        )
        probabilities = (
            (0.9, 0.03, 0.03, 0.04),
            (0.9, 0.03, 0.03, 0.04),
            (0.2, 0.05, 0.05, 0.7),
            (0.2, 0.05, 0.05, 0.7),
        )
        prediction = HeadPrediction(
            ("match", "match", "match", "match"),
            (False,) * 4,
            (0.0,) * 4,
            ("none",) * 4,
            probabilities,
            (0.0,) * 4,
        )
        config = SchemaDecodeConfig(
            high_thresholds={"missed_note": 0.6},
            low_ratios={"missed_note": 1.0},
            minimum_support={"missed_note": 1},
            uncertainty_margins={"missed_note": 0.0},
            merge_score_gap={"missed_note": 0},
        )
        labels = schema12_document_v3(
            "fixture", rows, prediction, score, config, include_repetition=False
        )["labels"]
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["type"], "missed_note")
        self.assertEqual(labels[0]["score_part"]["core_start_note_index"], 1)
        self.assertEqual(labels[0]["score_part"]["core_end_note_index"], 2)

    def test_v3_nms_keeps_one_competing_type_on_same_score_core(self) -> None:
        score = _score()
        row = HeadRow(
            np.zeros(FEATURE_DIM, np.float32), "event", 0, (1, 2), 0.0, 0.5, False
        )
        prediction = HeadPrediction(
            ("wrong_note",),
            (True,),
            (0.1,),
            ("long",),
            ((0.1, 0.8, 0.05, 0.05),),
            (0.6,),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"wrong_note": 0.5, "rhythm_error": 0.5},
            low_ratios={"wrong_note": 1.0, "rhythm_error": 1.0},
            minimum_support={"wrong_note": 1, "rhythm_error": 1},
            uncertainty_margins={"wrong_note": 0.0, "rhythm_error": 0.0},
            merge_score_gap={"wrong_note": 0, "rhythm_error": 0},
        )
        labels = schema12_document_v3(
            "fixture", (row,), prediction, score, config, include_repetition=False
        )["labels"]
        self.assertEqual([label["type"] for label in labels], ["wrong_note"])

    def test_v3_pad_notes_2_emits_annotator_padded_identity(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(8)
        )
        row = HeadRow(
            np.zeros(FEATURE_DIM, np.float32), "event", 0, (3, 4), 3.0, 3.5, False
        )
        prediction = HeadPrediction(
            ("wrong_note",),
            (False,),
            (0.0,),
            ("none",),
            ((0.1, 0.8, 0.05, 0.05),),
            (0.0,),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"wrong_note": 0.5},
            low_ratios={"wrong_note": 1.0},
            minimum_support={"wrong_note": 1},
            uncertainty_margins={"wrong_note": 0.0},
            merge_score_gap={"wrong_note": 0},
            pad_notes=2,
        )
        labels = schema12_document_v3(
            "fixture", (row,), prediction, score, config, include_repetition=False
        )["labels"]
        self.assertEqual(len(labels), 1)
        part = labels[0]["score_part"]
        self.assertEqual(part["start_note_index"], 1)
        self.assertEqual(part["end_note_index"], 5)
        self.assertEqual(part["pad_notes"], 2)
        self.assertEqual(part["core_start_note_index"], 3)
        self.assertEqual(part["core_end_note_index"], 3)
        self.assertEqual(labels[0]["pitches"], [61, 62, 63, 64, 65])

    def test_v3_core_identity_emits_unpadded_span(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(8)
        )
        row = HeadRow(
            np.zeros(FEATURE_DIM, np.float32), "event", 0, (3, 4), 3.0, 3.5, False
        )
        prediction = HeadPrediction(
            ("wrong_note",),
            (False,),
            (0.0,),
            ("none",),
            ((0.1, 0.8, 0.05, 0.05),),
            (0.0,),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"wrong_note": 0.5},
            low_ratios={"wrong_note": 1.0},
            minimum_support={"wrong_note": 1},
            uncertainty_margins={"wrong_note": 0.0},
            merge_score_gap={"wrong_note": 0},
            pad_notes=2,
            identity_span="core",
        )
        labels = schema12_document_v3(
            "fixture", (row,), prediction, score, config, include_repetition=False
        )["labels"]
        part = labels[0]["score_part"]
        self.assertEqual(part["start_note_index"], 3)
        self.assertEqual(part["end_note_index"], 3)
        self.assertEqual(part["pad_notes"], 0)
        self.assertEqual(labels[0]["pitches"], [63])

    def test_v3_operation_gate_skips_wrong_on_match_rows(self) -> None:
        features = np.zeros(FEATURE_DIM, np.float32)
        features[FEATURE_NAMES.index("path_operation_match")] = 1.0
        row = HeadRow(features, "event", 0, (1, 2), 0.0, 0.5, False)
        prediction = HeadPrediction(
            ("wrong_note",),
            (False,),
            (0.0,),
            ("none",),
            ((0.1, 0.8, 0.05, 0.05),),
            (0.0,),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"wrong_note": 0.5},
            low_ratios={"wrong_note": 1.0},
            minimum_support={"wrong_note": 1},
            uncertainty_margins={"wrong_note": 0.0},
            merge_score_gap={"wrong_note": 0},
            require_path_operation_gate=True,
        )
        labels = schema12_document_v3(
            "fixture", (row,), prediction, _score(), config, include_repetition=False
        )["labels"]
        self.assertEqual(labels, [])

    def test_v3_operation_gate_keeps_substitute_wrong(self) -> None:
        features = np.zeros(FEATURE_DIM, np.float32)
        features[FEATURE_NAMES.index("path_operation_wrong")] = 1.0
        row = HeadRow(features, "event", 0, (1, 2), 0.0, 0.5, False)
        prediction = HeadPrediction(
            ("wrong_note",),
            (False,),
            (0.0,),
            ("none",),
            ((0.1, 0.8, 0.05, 0.05),),
            (0.0,),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"wrong_note": 0.5},
            low_ratios={"wrong_note": 1.0},
            minimum_support={"wrong_note": 1},
            uncertainty_margins={"wrong_note": 0.0},
            merge_score_gap={"wrong_note": 0},
            require_path_operation_gate=True,
        )
        labels = schema12_document_v3(
            "fixture", (row,), prediction, _score(), config, include_repetition=False
        )["labels"]
        self.assertEqual([label["type"] for label in labels], ["wrong_note"])
        self.assertEqual(labels[0]["score_part"]["pad_notes"], 1)

    def test_v3_operation_gate_skips_extra_without_extra_flag(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(4)
        )
        extra_features = np.zeros(FEATURE_DIM, np.float32)
        extra_features[FEATURE_NAMES.index("path_operation_match")] = 1.0
        rows = (
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 0, (1, 2), 0.0, 0.4, False),
            HeadRow(extra_features, "event", 1, None, 0.4, 0.6, False),
            HeadRow(np.zeros(FEATURE_DIM, np.float32), "event", 2, (2, 3), 0.6, 1.0, False),
        )
        prediction = HeadPrediction(
            ("match", "extra_note", "match"),
            (False, False, False),
            (0.0, 0.0, 0.0),
            ("none", "none", "none"),
            (
                (0.9, 0.03, 0.04, 0.03),
                (0.05, 0.05, 0.85, 0.05),
                (0.9, 0.03, 0.04, 0.03),
            ),
            (0.0, 0.0, 0.0),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"extra_note": 0.5},
            low_ratios={"extra_note": 1.0},
            minimum_support={"extra_note": 1},
            uncertainty_margins={"extra_note": 0.0},
            merge_score_gap={"extra_note": 0},
            require_path_operation_gate=True,
        )
        gated = schema12_document_v3(
            "fixture", rows, prediction, score, config, include_repetition=False
        )["labels"]
        self.assertEqual(gated, [])
        ungated = schema12_document_v3(
            "fixture",
            rows,
            prediction,
            score,
            SchemaDecodeConfig(
                high_thresholds={"extra_note": 0.5},
                low_ratios={"extra_note": 1.0},
                minimum_support={"extra_note": 1},
                uncertainty_margins={"extra_note": 0.0},
                merge_score_gap={"extra_note": 0},
            ),
            include_repetition=False,
        )["labels"]
        self.assertEqual([label["type"] for label in ungated], ["extra_note"])

    def test_v3_operation_gate_keeps_delete_miss_and_skips_match_gap(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(3)
        )
        event = np.zeros(FEATURE_DIM, np.float32)
        missed = np.zeros(FEATURE_DIM, np.float32)
        missed[FEATURE_NAMES.index("path_operation_missed")] = 1.0
        match_gap = np.zeros(FEATURE_DIM, np.float32)
        match_gap[FEATURE_NAMES.index("path_operation_match")] = 1.0
        rows = (
            HeadRow(event, "event", 0, (0, 1), 0.0, 0.5, False),
            HeadRow(event, "event", 1, (2, 3), 1.0, 1.5, False),
            HeadRow(missed, "gap", None, (1, 2), 0.5, 1.0, False),
        )
        prediction = HeadPrediction(
            ("match", "match", "missed_note"),
            (False, False, False),
            (0.0, 0.0, 0.0),
            ("none", "none", "none"),
            (
                (0.9, 0.03, 0.03, 0.04),
                (0.9, 0.03, 0.03, 0.04),
                (0.1, 0.05, 0.05, 0.8),
            ),
            (0.0, 0.0, 0.0),
        )
        config = SchemaDecodeConfig(
            high_thresholds={"missed_note": 0.5},
            low_ratios={"missed_note": 1.0},
            minimum_support={"missed_note": 1},
            uncertainty_margins={"missed_note": 0.0},
            merge_score_gap={"missed_note": 0},
            require_path_operation_gate=True,
        )
        labels = schema12_document_v3(
            "fixture", rows, prediction, score, config, include_repetition=False
        )["labels"]
        self.assertEqual([label["type"] for label in labels], ["missed_note"])
        blocked_rows = rows[:2] + (
            HeadRow(match_gap, "gap", None, (1, 2), 0.5, 1.0, False),
        )
        blocked = schema12_document_v3(
            "fixture", blocked_rows, prediction, score, config, include_repetition=False
        )["labels"]
        self.assertEqual(blocked, [])

    def test_v3_from_mapping_defaults_preserve_frozen_v3(self) -> None:
        config = SchemaDecodeConfig.from_mapping(
            {
                "high_thresholds": {"wrong_note": 0.3},
                "low_ratios": {"wrong_note": 1.0},
                "minimum_support": {"wrong_note": 1},
                "uncertainty_margins": {"wrong_note": 0.0},
                "merge_score_gap": {"wrong_note": 0},
            }
        )
        self.assertEqual(config.pad_notes, 1)
        self.assertEqual(config.identity_span, "padded")
        self.assertFalse(config.require_path_operation_gate)

    def test_direct_high_confidence_wrong_and_extra(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(3)
        )

        def features(**values: float) -> np.ndarray:
            result = np.zeros(FEATURE_DIM, dtype=np.float32)
            for name, value in values.items():
                result[FEATURE_NAMES.index(name)] = value
            return result

        rows = (
            HeadRow(features(candidate_confidence=0.9), "event", 0, (0, 1), 0.0, 0.5, False),
            HeadRow(
                features(
                    candidate_confidence=0.95,
                    upstream_path_margin=3.0,
                    path_operation_wrong=1.0,
                ),
                "event",
                1,
                (1, 2),
                0.5,
                1.0,
                False,
            ),
            HeadRow(
                features(
                    candidate_confidence=0.9,
                    upstream_path_margin=3.0,
                    upstream_extra_probability=0.95,
                    path_operation_extra=1.0,
                ),
                "event",
                2,
                None,
                1.0,
                1.2,
                False,
            ),
            HeadRow(features(candidate_confidence=0.9), "event", 3, (2, 3), 1.2, 1.7, False),
        )
        events = (
            JointEvent(60, 0.0, 0.5, (0, 1)),
            JointEvent(64, 0.5, 1.0, (1, 2), "substitute"),
            JointEvent(67, 1.0, 1.2, None, "extra"),
            JointEvent(62, 1.2, 1.7, (2, 3)),
        )
        direct, diagnostics = direct_operation_probabilities(rows, events, score)
        self.assertGreater(direct[1, LAYER2_CLASSES.index("wrong_note")], 0.5)
        self.assertGreater(direct[2, LAYER2_CLASSES.index("extra_note")], 0.5)
        self.assertEqual(diagnostics["repeat_rows_excluded"], 0)

    def test_direct_delete_requires_local_resynchronization(self) -> None:
        score = tuple(
            ScoreEvent(i, 60 + i, float(i), float(i + 1), (i,), measure=1)
            for i in range(3)
        )
        event_features = np.zeros(FEATURE_DIM, dtype=np.float32)
        event_features[FEATURE_NAMES.index("candidate_confidence")] = 0.9
        gap_features = np.zeros(FEATURE_DIM, dtype=np.float32)
        gap_features[FEATURE_NAMES.index("upstream_delete_probability")] = 1.0
        gap_features[FEATURE_NAMES.index("previous_confidence")] = 0.9
        gap_features[FEATURE_NAMES.index("next_confidence")] = 0.9
        gap_features[FEATURE_NAMES.index("path_operation_missed")] = 1.0
        rows = (
            HeadRow(event_features, "event", 0, (0, 1), 0.0, 0.5, False),
            HeadRow(event_features, "event", 1, (2, 3), 1.0, 1.5, False),
            HeadRow(gap_features, "gap", None, (1, 2), 0.5, 1.0, False),
        )
        events = (
            JointEvent(60, 0.0, 0.5, (0, 1)),
            JointEvent(62, 1.0, 1.5, (2, 3)),
        )
        direct, _diagnostics = direct_operation_probabilities(rows, events, score)
        self.assertGreater(direct[2, LAYER2_CLASSES.index("missed_note")], 0.8)
        partial_rows = (
            HeadRow(event_features, "event", 0, (2, 3), 1.0, 1.5, False),
            rows[2],
        )
        partial, _diagnostics = direct_operation_probabilities(
            partial_rows, events[1:], score
        )
        self.assertEqual(
            float(partial[1, LAYER2_CLASSES.index("missed_note")]), 0.0
        )

    def test_direct_repeat_exclusion_and_learned_blend(self) -> None:
        score = _score()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[FEATURE_NAMES.index("candidate_confidence")] = 1.0
        features[FEATURE_NAMES.index("upstream_path_margin")] = 5.0
        features[FEATURE_NAMES.index("path_operation_wrong")] = 1.0
        row = HeadRow(features, "event", 0, (0, 1), 0.0, 0.5, True)
        event = JointEvent(61, 0.0, 0.5, (0, 1), "copy", copy_pass=1)
        direct, diagnostics = direct_operation_probabilities((row,), (event,), score)
        self.assertEqual(float(np.max(direct[0, 1:])), 0.0)
        self.assertEqual(diagnostics["repeat_rows_excluded"], 1)
        learned = HeadPrediction(
            ("wrong_note",),
            (False,),
            (0.0,),
            ("none",),
            ((0.1, 0.8, 0.05, 0.05),),
            (0.0,),
        )
        hybrid = blend_direct_learned_prediction(
            learned, direct, direct_weight=0.5
        )
        self.assertEqual(hybrid.layer2, ("wrong_note",))


if __name__ == "__main__":
    unittest.main()
