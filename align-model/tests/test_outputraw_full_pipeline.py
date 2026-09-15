from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from alignmodel.joint.index import JointEvent, ScoreEvent
from alignmodel.joint.data import JointTrainingExample
from alignmodel.joint.lattice import (
    FEATURE_DIM,
    JointCandidate,
    JointEdgeScorer,
    LatticeConfig,
    SparseJointLattice,
)
from alignmodel.joint.outputraw_full import (
    FullJointPipelineModel,
    FullPipelineAugmentConfig,
    FullPipelineLossConfig,
    FullPipelineModelConfig,
    FullPipelineTargets,
    atomic_checkpoint,
    augment_difficult_timbre,
    duration_supervision_weight,
    full_pipeline_loss,
    infer_full_pipeline,
    load_checkpoint,
    verify_data_ready,
)
from alignmodel.joint.outputraw_metrics import (
    FullPipelineMetricSample,
    evaluate_full_pipeline,
)
from alignmodel.joint.outputraw_train import (
    _target_resume_events,
    collate_local_samples,
    iter_local_batches,
    prepare_local_sample,
)
from alignmodel.joint.packed_data import (
    PackedCursor,
    PackedSample,
    canonical_target,
)
from alignmodel.transcription.basic_pitch import BasicPitchFeatures


def _score() -> list[ScoreEvent]:
    return [
        ScoreEvent(index, pitch, float(index), float(index + 1), (index,))
        for index, pitch in enumerate((60, 62, 64, 65, 67))
    ]


def _candidates() -> list[JointCandidate]:
    return [
        JointCandidate(
            pitch,
            index * 0.35,
            index * 0.35 + duration,
            confidence=confidence,
            acoustic_features=(onset, 0.8, 0.5, -0.05, 0.6),
        )
        for index, (pitch, duration, confidence, onset) in enumerate(
            (
                (60, 0.070, 0.90, 0.85),
                (64, 0.110, 0.82, 0.75),
                (60, 0.160, 0.78, 0.80),
                (73, 0.090, 0.55, 0.65),
                (67, 0.250, 0.88, 0.70),
            )
        )
    ]


def _batch(model: FullJointPipelineModel):
    lattice = SparseJointLattice(
        model,
        LatticeConfig(
            max_options_per_candidate=8,
            max_states=32,
            continuation_feature_enabled=True,
        ),
    )
    spans = ((0, 1), (2, 3), (0, 1), None, (4, 5))
    keep = (True, True, True, False, True)
    rows, groups = lattice.local_warmup_edges(
        _candidates(), _score(), spans, keep
    )
    features = torch.tensor(rows, dtype=torch.float32)
    gold = [start + offset for start, _end, offset in groups]
    emission = torch.argmax(features[gold, :8], dim=-1)
    targets = FullPipelineTargets(
        groups=groups,
        keep=torch.tensor([1, 1, 1, 0, 1]),
        boundary=torch.tensor([1, 1, 0, 0, 1]),
        split=torch.tensor([0, 0, 1, 0, 0]),
        emission=emission,
        structure=torch.tensor([0, 0, 1, 1, 3]),
        layer2=torch.tensor([0, 1, 0, 2, 0]),
        rhythm=torch.tensor([0, 1, 0, 0, 1]),
        duration_target=torch.tensor([0.0, -0.2, 0.1, 0.0, 0.3]),
        duration_weight=torch.tensor(
            [duration_supervision_weight(c.end - c.start) for c in _candidates()]
        ),
        rearticulation_weight=torch.tensor([1.0, 1.0, 2.5, 1.0, 1.0]),
        clip_index=torch.zeros(5, dtype=torch.long),
        copy_count_target=torch.tensor([1]),
    )
    return features, targets


class FullPipelineArchitectureTests(unittest.TestCase):
    def test_all_repository_owned_components_receive_gradient(self) -> None:
        torch.manual_seed(101)
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=24, component_dim=12, dropout=0.0
            )
        )
        features, targets = _batch(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = full_pipeline_loss(model, features, targets)
            self.assertTrue(torch.isfinite(loss.total))
            loss.total.backward()
            optimizer.step()

        expected = {
            "legacy_path",
            "acoustic_projection",
            "score_projection",
            "structure_projection",
            "shared",
            "confidence_calibration",
            "boundary_head",
            "split_head",
            "emission_head",
            "structure_head",
            "copy_count_head",
            "layer2_head",
            "layer3_rhythm_head",
            "layer3_duration_head",
            "joint_path_head",
        }
        gradients = {name: 0.0 for name in expected}
        for name, parameter in model.named_parameters():
            component = name.split(".", 1)[0]
            self.assertIn(component, expected)
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertTrue(torch.all(torch.isfinite(parameter.grad)), name)
            gradients[component] += float(parameter.grad.abs().sum())
        self.assertTrue(all(value > 0.0 for value in gradients.values()), gradients)

    def test_stage_schedule_ends_with_every_parameter_trainable(self) -> None:
        model = FullJointPipelineModel()
        model.freeze_for_stage("acoustic")
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith("boundary_head.")
            )
        )
        self.assertFalse(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        model.freeze_for_stage("joint")
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )

    def test_short_note_weighting_covers_required_bands(self) -> None:
        self.assertEqual(duration_supervision_weight(0.079), 4.0)
        self.assertEqual(duration_supervision_weight(0.119), 3.0)
        self.assertEqual(duration_supervision_weight(0.179), 2.0)
        self.assertEqual(duration_supervision_weight(0.180), 1.0)

    def test_confidence_band_hard_negative_changes_loss(self) -> None:
        torch.manual_seed(102)
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=20,
                path_hidden_dim=20,
                path_component_dim=10,
                component_dim=10,
                dropout=0.0,
            )
        )
        features, targets = _batch(model)
        ordinary = full_pipeline_loss(
            model,
            features,
            targets,
            FullPipelineLossConfig(confidence_band_negative_weight=1.0),
        )
        weighted = full_pipeline_loss(
            model,
            features,
            targets,
            FullPipelineLossConfig(confidence_band_negative_weight=3.0),
        )
        self.assertFalse(torch.equal(ordinary.total, weighted.total))

    def test_timbre_augmentation_is_shared_across_candidate_options(self) -> None:
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=20, component_dim=10, dropout=0.0
            )
        )
        features, targets = _batch(model)
        torch.manual_seed(104)
        augmented = augment_difficult_timbre(
            features,
            targets.groups,
            FullPipelineAugmentConfig(
                difficult_timbre_probability=1.0,
                activation_noise_std=0.0,
            ),
        )
        self.assertFalse(torch.equal(features[:, -5:], augmented[:, -5:]))
        for start, end, _gold in targets.groups:
            self.assertEqual(
                torch.unique(augmented[start:end, -5]).numel(), 1
            )

    def test_deployable_inference_has_no_gold_dependency(self) -> None:
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=20, component_dim=10, dropout=0.0
            )
        )
        lattice = SparseJointLattice(
            model, LatticeConfig(max_options_per_candidate=8, max_states=32)
        )
        with mock.patch(
            "alignmodel.validated_targets.target_note_map",
            side_effect=AssertionError("gold target requested"),
        ):
            prediction = infer_full_pipeline(
                model, lattice, _candidates(), _score()
            )
        self.assertEqual(len(prediction.events), len(prediction.layer2_types))
        self.assertEqual(
            len(prediction.events), len(prediction.rhythm_probabilities)
        )
        self.assertEqual(
            len(prediction.events), len(prediction.corrected_durations_sec)
        )
        self.assertTrue(set(prediction.missed_score_events) <= set(range(5)))


class FullPipelineCheckpointTests(unittest.TestCase):
    def test_mid_epoch_resume_preserves_optimizer_determinism(self) -> None:
        torch.manual_seed(103)
        config = FullPipelineModelConfig(
            hidden_dim=20, component_dim=10, dropout=0.0
        )
        template = FullJointPipelineModel(config)
        uninterrupted = FullJointPipelineModel(config)
        interrupted = FullJointPipelineModel(config)
        uninterrupted.load_state_dict(template.state_dict())
        interrupted.load_state_dict(template.state_dict())
        features, targets = _batch(template)

        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = full_pipeline_loss(model, features, targets).total
            loss.backward()
            optimizer.step()

        full_optimizer = torch.optim.AdamW(
            uninterrupted.parameters(), lr=1e-3
        )
        update(uninterrupted, full_optimizer)
        update(uninterrupted, full_optimizer)

        partial_optimizer = torch.optim.AdamW(
            interrupted.parameters(), lr=1e-3
        )
        update(interrupted, partial_optimizer)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "last.pt"
            atomic_checkpoint(
                checkpoint,
                model=interrupted,
                optimizer=partial_optimizer,
                scheduler=None,
                scaler=None,
                progress={"stage": "joint", "epoch": 1, "cursor": 7},
                data_fingerprint="packed-fixture",
                checkpoint_metadata={
                    "config_sha256": "config-fixture",
                    "packed_cache_version": "packed-v1",
                },
            )
            resumed, payload = load_checkpoint(
                checkpoint,
                expected_data_fingerprint="packed-fixture",
                expected_checkpoint_metadata={
                    "config_sha256": "config-fixture",
                },
            )
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_checkpoint(
                    checkpoint,
                    expected_checkpoint_metadata={
                        "config_sha256": "different",
                    },
                )
            resumed_optimizer = torch.optim.AdamW(
                resumed.parameters(), lr=1e-3
            )
            resumed_optimizer.load_state_dict(
                payload["optimizer_state_dict"]
            )
            update(resumed, resumed_optimizer)

        self.assertEqual(payload["progress"]["cursor"], 7)
        for expected, actual in zip(
            uninterrupted.parameters(), resumed.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_legacy_path_checkpoint_initializes_without_aux_weights(self) -> None:
        legacy = JointEdgeScorer(
            hidden_dim=20,
            component_mode=True,
            component_dim=10,
        )
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=20,
                path_hidden_dim=20,
                path_component_dim=10,
                component_dim=10,
                dropout=0.0,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.pt"
            torch.save(
                {
                    "schema_version": "align-end-to-end-joint-v2",
                    "state_dict": legacy.state_dict(),
                },
                checkpoint,
            )
            report = model.initialize_path(checkpoint)
        self.assertGreater(report["loaded_parameter_tensors"], 0)
        for expected, actual in zip(
            legacy.network.parameters(), model.legacy_path.network.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(report["transfer_coverage"], 1.0)

    def test_path_transfer_rejects_low_exact_coverage(self) -> None:
        legacy = JointEdgeScorer(
            hidden_dim=20,
            component_mode=True,
            component_dim=10,
        )
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=20,
                path_hidden_dim=20,
                path_component_dim=20,
                component_dim=20,
                dropout=0.0,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.pt"
            torch.save({"state_dict": legacy.state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "transfer coverage"):
                model.initialize_path(
                    checkpoint,
                    transfer_mode="exact",
                    minimum_coverage=0.95,
                )

    def test_path_inflation_preserves_source_scores(self) -> None:
        torch.manual_seed(91)
        legacy = JointEdgeScorer(
            hidden_dim=20,
            dropout=0.0,
            component_mode=True,
            component_dim=10,
        ).eval()
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=20,
                path_hidden_dim=20,
                path_component_dim=20,
                component_dim=20,
                dropout=0.0,
            )
        ).eval()
        features = torch.randn(13, FEATURE_DIM)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.pt"
            torch.save({"state_dict": legacy.state_dict()}, checkpoint)
            report = model.initialize_path(
                checkpoint,
                transfer_mode="inflate",
                minimum_coverage=1.0,
            )
        self.assertEqual(report["transfer_coverage"], 1.0)
        self.assertTrue(report["inflated_parameter_tensors"])
        self.assertTrue(
            torch.allclose(
                legacy(features),
                model.legacy_path(features),
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_data_marker_verification_never_opens_test_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "DATA_READY.json"
            marker.write_text(
                json.dumps(
                    {
                        "dataset": "outputRaw_sf_10k",
                        "ready": True,
                        "fingerprint": "a" * 64,
                        "splits": {
                            "train": {"rows": 8004},
                            "val": {"rows": 999},
                            "test_locked": {"rows": 997, "sealed": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            document = verify_data_ready(marker)
        self.assertEqual(document["splits"]["val"]["rows"], 999)

    def test_release_marker_contract_accepts_metadata_only_lockbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "DATA_READY.json"
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": "align-joint-data-ready-v1",
                        "release": "joint-outputraw-full-v1",
                        "status": "ready",
                        "paths": {
                            "manifest": "audit/split.json",
                            "packed_root": "packed",
                            "packed_index": "packed/index.sqlite",
                        },
                        "counts": {"train": 8000, "val": 900},
                        "hashes": {
                            "manifest_sha256": "a" * 64,
                            "pack_id": "b" * 64,
                        },
                        "verification": {
                            "test_features_materialized": False,
                            "test_targets_materialized": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            document = verify_data_ready(marker)
        self.assertEqual(document["status"], "ready")


class PackedTrainingIntegrationTests(unittest.TestCase):
    def test_packed_sample_builds_equivalent_vectorized_batch(self) -> None:
        candidates = tuple(_candidates())
        score = tuple(_score())
        spans = ((0, 1), (2, 3), (0, 1), None, (4, 5))
        targets = (
            JointEvent(60, 0.00, 0.07, (0, 1), "match"),
            JointEvent(64, 0.35, 0.46, (2, 3), "match"),
            JointEvent(60, 0.70, 0.86, (0, 1), "copy", copy_pass=1),
            JointEvent(67, 1.40, 1.65, (4, 5), "match"),
        )
        example = JointTrainingExample(
            sample="packed-fixture",
            source="Mozart",
            candidates=candidates,
            score=score,
            gold_spans=spans,
            gold_keep_unlinked=(True, True, True, False, True),
            target_events=targets,
            target_deletions=frozenset({1, 3}),
        )
        target = canonical_target(
            example,
            valid_labels=[
                {
                    "type": "rhythm_error",
                    "start_time": 0.30,
                    "end_time": 0.50,
                }
            ],
        )
        target["sparse_lattice"] = {
            "gold_spans": [
                list(value) if value is not None else None for value in spans
            ],
            "gold_keep_unlinked": [True, True, True, False, True],
        }
        features = BasicPitchFeatures(
            note=np.zeros((2, 88), dtype=np.float32),
            onset=np.zeros((2, 88), dtype=np.float32),
            contour=np.zeros((2, 264), dtype=np.float32),
            frame_times=np.array([0.0, 0.01]),
            metadata={},
        )
        packed = PackedSample(
            ordinal=0,
            record_key="fixture",
            split="train",
            sample="packed-fixture",
            source="Mozart",
            features=features,
            candidates=candidates,
            target=target,
        )
        model = FullJointPipelineModel(
            FullPipelineModelConfig(
                hidden_dim=20, component_dim=10, dropout=0.0
            )
        )
        lattice = SparseJointLattice(
            model, LatticeConfig(max_options_per_candidate=8, max_states=32)
        )
        prepared = prepare_local_sample(packed, lattice)
        batch = collate_local_samples((prepared, prepared))
        batch.targets.validate(batch.edge_count)
        self.assertEqual(batch.samples, ("packed-fixture", "packed-fixture"))
        self.assertEqual(
            len(batch.targets.groups), 2 * len(candidates)
        )
        self.assertEqual(batch.targets.copy_count_target.tolist(), [1, 1])
        loss = full_pipeline_loss(model, batch.edge_features, batch.targets)
        self.assertTrue(torch.isfinite(loss.total))
        cursor = PackedCursor("pack", "train", 0, 17, 0)
        dataset = mock.Mock()
        dataset.iter_from_cursor.return_value = iter(
            (
                PackedCursor("pack", "train", 0, 17, position),
                packed,
            )
            for position in range(1, 4)
        )
        limited = list(
            iter_local_batches(
                dataset,
                cursor,
                lattice=lattice,
                max_edges=10**9,
                workers=0,
                prefetch=0,
                max_samples=1,
            )
        )
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0][0].position, 1)
        self.assertEqual(limited[0][1].samples, ("packed-fixture",))


class FullPipelineMetricTests(unittest.TestCase):
    def test_resume_targets_follow_lattice_not_copy_labels(self) -> None:
        events = (
            JointEvent(60, 0.0, 0.1, (0, 1), "match"),
            JointEvent(62, 0.1, 0.2, (1, 2), "match"),
            JointEvent(60, 0.2, 0.3, (0, 1), "match"),
            JointEvent(64, 0.3, 0.4, (2, 3), "match"),
        )
        lattice = SparseJointLattice(JointEdgeScorer(), LatticeConfig())
        self.assertEqual(_target_resume_events(events, lattice), (2,))

    def test_perfect_full_pipeline_reports_every_gate_metric(self) -> None:
        target = (
            JointEvent(60, 0.00, 0.07, (0, 1), "match"),
            JointEvent(63, 0.20, 0.31, (1, 2), "substitute"),
            JointEvent(
                70, 0.40, 0.56, None, "extra", rendered_index=2
            ),
            JointEvent(64, 0.70, 1.00, (2, 3), "copy", copy_pass=1),
        )
        report = evaluate_full_pipeline(
            [
                FullPipelineMetricSample(
                    predicted=target,
                    target=target,
                    predicted_layer2=(
                        "match",
                        "wrong_note",
                        "extra_note",
                        "match",
                    ),
                    predicted_rhythm=(False, False, True, False),
                    target_rhythm=(False, False, True, False),
                    predicted_deletions=frozenset({3}),
                    target_deletions=frozenset({3}),
                    predicted_resume_events=(3,),
                    target_resume_events=(3,),
                    score_event_count=4,
                )
            ],
            bootstrap_replicates=20,
        )
        self.assertEqual(report["combined"]["f1"], 1.0)
        self.assertEqual(report["repeat"]["f1"], 1.0)
        self.assertEqual(report["repeat"]["resume_accuracy"], 1.0)
        self.assertEqual(report["layer2"]["macro_f1"], 1.0)
        self.assertEqual(report["layer3"]["rhythm"]["f1"], 1.0)
        self.assertEqual(
            report["layer3"]["duration"]["within_20_percent_accuracy"],
            1.0,
        )
        self.assertEqual(
            report["mapping"]["aggregate"]["tolerances"]["20ms"]["joint"]["f1"],
            1.0,
        )
        self.assertEqual(
            report["diagnostic_timestamp_transcription_50ms"][
                "short_note_recall"
            ]["lt_80ms"]["recall"],
            1.0,
        )

    def test_wrong_auxiliary_type_reduces_honest_combined_f1(self) -> None:
        target = (JointEvent(60, 0.0, 0.2, (0, 1), "match"),)
        report = evaluate_full_pipeline(
            [
                FullPipelineMetricSample(
                    predicted=target,
                    target=target,
                    predicted_layer2=("wrong_note",),
                    score_event_count=1,
                )
            ],
            bootstrap_replicates=20,
        )
        self.assertEqual(
            report["mapping"]["aggregate"]["tolerances"]["50ms"]["joint"]["f1"],
            1.0,
        )
        self.assertEqual(report["combined"]["f1"], 0.5)

    def test_timestamp_diagnostic_cannot_change_official_score(self) -> None:
        target = (JointEvent(60, 0.0, 0.2, (0, 1), "match"),)
        predicted = (JointEvent(60, 9.0, 9.2, (0, 1), "match"),)
        report = evaluate_full_pipeline(
            [
                FullPipelineMetricSample(
                    predicted=predicted,
                    target=target,
                    predicted_layer2=("match",),
                    score_event_count=1,
                )
            ],
            bootstrap_replicates=20,
        )
        self.assertEqual(report["combined"]["f1"], 1.0)
        self.assertEqual(
            report["diagnostic_timestamp_combined_50ms"]["f1"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
