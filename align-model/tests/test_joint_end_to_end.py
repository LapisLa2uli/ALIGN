from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from music21 import note, stream

from alignmodel.joint.candidate_rescorer import (
    CandidateRescorer,
    CandidateRescorerConfig,
    _training_rows,
    candidate_features,
)
from alignmodel.joint.data import JointTrainingExample, load_inference_inputs
from alignmodel.joint.end_to_end import (
    EndToEndTrainConfig,
    _atomic_torch,
    _checkpoint_payload,
    _config_payload,
    _load_resume,
    _vectorized_group_nll,
)
from alignmodel.joint.example_cache import _deserialize, _serialize
from alignmodel.joint.index import ScoreEvent
from alignmodel.joint.lattice import (
    JointCandidate,
    JointEdgeScorer,
    LatticeConfig,
    SparseJointLattice,
)
from alignmodel.transcription.basic_pitch import (
    BASIC_PITCH_VERSION,
    CACHE_SCHEMA_VERSION,
    FRONTEND_VERSION,
    BasicPitchFeatures,
    effective_pitch_policy,
    save_basic_pitch_cache,
    selected_runtime,
)


def _score() -> list[ScoreEvent]:
    return [
        ScoreEvent(index, pitch, float(index), float(index + 1), (index,))
        for index, pitch in enumerate((60, 62, 64, 65, 67))
    ]


def _candidates() -> list[JointCandidate]:
    spans = ((0, 1), (2, 3), (0, 1), None, (4, 5))
    pitches = (60, 64, 60, 73, 67)
    return [
        JointCandidate(
            pitch,
            index * 0.45,
            index * 0.45 + 0.35,
            confidence=0.55 + index * 0.08,
            acoustic_features=(
                0.2 + 0.1 * index,
                0.3 + 0.1 * index,
                0.4 + 0.05 * index,
                -0.1 * index,
                0.5 + 0.05 * index,
            ),
        )
        for index, (pitch, _span) in enumerate(zip(pitches, spans))
    ]


class EndToEndGradientTests(unittest.TestCase):
    def test_every_repository_owned_component_receives_gradient(self) -> None:
        torch.manual_seed(11)
        model = JointEdgeScorer(
            hidden_dim=16,
            component_mode=True,
            component_dim=8,
            residual_scale=0.1,
        )
        lattice = SparseJointLattice(
            model,
            LatticeConfig(
                max_options_per_candidate=8,
                max_states=32,
                max_delete_events=8,
                continuation_feature_enabled=True,
                continuation_score_weight=0.25,
            ),
        )
        spans = [(0, 1), (2, 3), (0, 1), None, (4, 5)]
        keep = [True, True, True, False, True]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            local = lattice.local_warmup_nll(
                _candidates(), _score(), spans, keep
            )
            structured = lattice.nll(
                _candidates(), _score(), spans, keep
            ) / len(spans)
            (local + structured).backward()
            optimizer.step()

        intended = {
            "network",
            "acoustic_projection",
            "score_projection",
            "structural_projection",
            "emission_head",
            "option_head",
            "transition_head",
            "path_head",
        }
        seen = set()
        gradient_norms = {name: 0.0 for name in intended}
        for name, parameter in model.named_parameters():
            component = name.split(".", 1)[0]
            seen.add(component)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.all(torch.isfinite(parameter.grad)), name)
            gradient_norms[component] += float(torch.sum(torch.abs(parameter.grad)))
        self.assertEqual(seen, intended)
        self.assertTrue(
            all(value > 0.0 for value in gradient_norms.values()),
            gradient_norms,
        )

    def test_component_model_preserves_legacy_scores_at_initialization(self) -> None:
        torch.manual_seed(12)
        model = JointEdgeScorer(
            hidden_dim=16,
            component_mode=True,
            component_dim=8,
        )
        features = torch.randn(7, 26)
        expected = model.network(features[:, :21]).squeeze(-1)
        self.assertTrue(torch.equal(model(features), expected))

    def test_batched_local_loss_matches_per_example_loss(self) -> None:
        torch.manual_seed(13)
        model = JointEdgeScorer(
            hidden_dim=12,
            component_mode=True,
            component_dim=6,
        )
        lattice = SparseJointLattice(model, LatticeConfig(max_states=32))
        spans = [(0, 1), (2, 3), (0, 1), None, (4, 5)]
        keep = [True, True, True, False, True]
        rows, groups = lattice.local_warmup_edges(
            _candidates(), _score(), spans, keep
        )
        expected = lattice.local_warmup_nll(
            _candidates(), _score(), spans, keep
        )
        actual = _vectorized_group_nll(
            model, rows, groups, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_continuation_feature_reaches_structural_projection(self) -> None:
        score = [
            ScoreEvent(index, pitch, float(index), float(index + 1), (index,))
            for index, pitch in enumerate(
                (60, 62, 64, 65, 67, 69, 71, 72)
            )
        ]
        pitches = (
            60, 62, 64, 65, 67,
            62, 64, 65, 67,
            69, 80, 72,
        )
        candidates = [
            JointCandidate(
                pitch,
                index * 0.4,
                index * 0.4 + 0.3,
                acoustic_features=(0.4, 0.5, 0.4, -0.1, 0.5),
            )
            for index, pitch in enumerate(pitches)
        ]
        spans = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), None, (7, 8),
        ]
        model = JointEdgeScorer(
            hidden_dim=16,
            component_mode=True,
            component_dim=8,
        )
        with torch.no_grad():
            model.transition_head.weight.fill_(0.1)
            model.path_head[-1].weight.fill_(0.1)
        lattice = SparseJointLattice(
            model,
            LatticeConfig(
                max_options_per_candidate=12,
                max_states=48,
                continuation_feature_enabled=True,
                continuation_hard_negative_copies=1,
            ),
        )
        rows, _groups = lattice.local_warmup_edges(
            candidates,
            score,
            spans,
            [span is not None for span in spans],
        )
        self.assertTrue(any(abs(row[19]) > 0.25 for row in rows))
        loss = lattice.local_warmup_nll(
            candidates,
            score,
            spans,
            [span is not None for span in spans],
        )
        loss.backward()
        gradient = model.structural_projection[0].weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient[:, -1].abs().sum()), 0.0)

    def test_cached_and_in_memory_objectives_are_equivalent(self) -> None:
        spans = ((0, 1), (2, 3), (0, 1), None, (4, 5))
        keep = (True, True, True, False, True)
        original = JointTrainingExample(
            sample="fixture",
            source="fixture",
            candidates=tuple(_candidates()),
            score=tuple(_score()),
            gold_spans=spans,
            gold_keep_unlinked=keep,
            target_events=(),
            target_deletions=frozenset(),
        )
        cached = _deserialize(_serialize(original))
        torch.manual_seed(31)
        old_model = JointEdgeScorer(
            hidden_dim=16,
            component_mode=True,
            component_dim=8,
        )
        with torch.no_grad():
            old_model.path_head[-1].weight.fill_(0.01)
        new_model = JointEdgeScorer(
            hidden_dim=16,
            component_mode=True,
            component_dim=8,
        )
        new_model.load_state_dict(old_model.state_dict())
        config = LatticeConfig(max_options_per_candidate=8, max_states=32)
        old_lattice = SparseJointLattice(old_model, config)
        new_lattice = SparseJointLattice(new_model, config)
        old_rows, _ = old_lattice.local_warmup_edges(
            original.candidates,
            original.score,
            original.gold_spans,
            original.gold_keep_unlinked,
        )
        new_rows, _ = new_lattice.local_warmup_edges(
            cached.candidates,
            cached.score,
            cached.gold_spans,
            cached.gold_keep_unlinked,
        )
        self.assertEqual(old_rows, new_rows)
        old_logits = old_model(torch.tensor(old_rows))
        new_logits = new_model(torch.tensor(new_rows))
        self.assertTrue(torch.equal(old_logits, new_logits))
        old_loss = old_lattice.nll(
            original.candidates,
            original.score,
            original.gold_spans,
            original.gold_keep_unlinked,
        )
        new_loss = new_lattice.nll(
            cached.candidates,
            cached.score,
            cached.gold_spans,
            cached.gold_keep_unlinked,
        )
        self.assertTrue(torch.equal(old_loss, new_loss))
        old_loss.backward()
        new_loss.backward()
        for old_parameter, new_parameter in zip(
            old_model.parameters(), new_model.parameters()
        ):
            self.assertTrue(
                torch.equal(old_parameter.grad, new_parameter.grad)
            )
        self.assertEqual(
            old_lattice.decode(original.candidates, original.score),
            new_lattice.decode(cached.candidates, cached.score),
        )


class CandidateRescorerTests(unittest.TestCase):
    def test_short_positives_and_hard_negatives_are_weighted(self) -> None:
        candidates = (
            JointCandidate(
                60,
                0.0,
                0.06,
                0.55,
                score_hints=(1, 2),
                acoustic_features=(0.4, 0.5, 0.3, -0.1, 0.4),
            ),
            JointCandidate(
                61,
                0.1,
                0.3,
                0.64,
                acoustic_features=(0.2, 0.3, 0.2, -0.2, 0.3),
            ),
        )
        example = JointTrainingExample(
            "sample",
            "source",
            candidates,
            tuple(_score()),
            ((0, 1), None),
            (True, False),
            (),
            frozenset(),
        )
        config = CandidateRescorerConfig(
            manifest=Path("manifest.json"),
            basic_cache_root=Path("basic"),
            example_cache_path=Path("examples.sqlite"),
            output_dir=Path("output"),
        )
        features, labels, weights = _training_rows(example, config)
        self.assertEqual(features.shape, (2, 17))
        self.assertEqual(labels.tolist(), [1.0, 0.0])
        self.assertEqual(weights.tolist(), [4.0, 1.0])
        without_hints = list(candidates)
        without_hints[0] = JointCandidate(
            60,
            0.0,
            0.06,
            0.55,
            acoustic_features=candidates[0].acoustic_features,
        )
        self.assertTrue(
            np.array_equal(
                candidate_features(candidates),
                candidate_features(without_hints),
            )
        )

    def test_rescorer_starts_from_basic_pitch_confidence(self) -> None:
        model = CandidateRescorer(hidden_dim=8, dropout=0.0)
        features = torch.zeros(2, 17)
        features[:, 0] = torch.tensor([0.55, 0.70])
        self.assertTrue(
            torch.allclose(
                model(features).sigmoid(),
                features[:, 0],
                atol=1e-6,
            )
        )


class EndToEndCheckpointTests(unittest.TestCase):
    def test_checkpoint_resume_restores_model_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "split.json"
            manifest.write_text(
                json.dumps({"train": [{}], "val": [{}]}),
                encoding="utf-8",
            )
            initialize = root / "legacy.pt"
            initialize.write_bytes(b"not loaded by this test")
            config = EndToEndTrainConfig(
                manifest=manifest,
                cache_root=root / "cache",
                output_dir=root / "run",
                initialize_checkpoint=initialize,
                max_train_samples=1,
                path_train_samples=1,
                max_val_samples=1,
                local_device="cpu",
            )
            model = JointEdgeScorer(
                hidden_dim=64,
                component_mode=True,
                component_dim=32,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(4, 26)).sum().backward()
            optimizer.step()
            checkpoint = root / "last_checkpoint.pt"
            _atomic_torch(
                checkpoint,
                _checkpoint_payload(
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    optimizer_stage="vectorized_local",
                    initialization={"path": "legacy.pt"},
                    history=[{"stage": "vectorized_local", "epoch": 1}],
                    progress={"local_epoch": 1, "path_epoch": 0},
                    best_f1=0.7,
                ),
            )
            resume_config = EndToEndTrainConfig(
                **{
                    **config.__dict__,
                    "resume_checkpoint": checkpoint,
                }
            )
            restored, payload = _load_resume(
                resume_config, torch.device("cpu")
            )

        self.assertEqual(payload["progress"]["local_epoch"], 1)
        self.assertIn("optimizer_state_dict", payload)
        self.assertTrue(payload["lattice"]["continuation_feature_enabled"])
        self.assertEqual(payload["lattice"]["continuation_hard_negative_copies"], 1)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_mid_epoch_resume_is_optimizer_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "split.json"
            manifest.write_text(
                json.dumps({"train": [{}], "val": [{}]}),
                encoding="utf-8",
            )
            initialize = root / "legacy.pt"
            initialize.write_bytes(b"fixture")
            config = EndToEndTrainConfig(
                manifest=manifest,
                cache_root=root / "cache",
                output_dir=root / "run",
                initialize_checkpoint=initialize,
                max_train_samples=1,
                path_train_samples=1,
                max_val_samples=1,
                local_device="cpu",
            )
            torch.manual_seed(44)
            template = JointEdgeScorer(
                hidden_dim=64,
                component_mode=True,
                component_dim=32,
            )
            uninterrupted = JointEdgeScorer(
                hidden_dim=64,
                component_mode=True,
                component_dim=32,
            )
            interrupted = JointEdgeScorer(
                hidden_dim=64,
                component_mode=True,
                component_dim=32,
            )
            uninterrupted.load_state_dict(template.state_dict())
            interrupted.load_state_dict(template.state_dict())
            batches = [torch.randn(16, 26), torch.randn(12, 26)]

            def update(model, optimizer, batch):
                optimizer.zero_grad(set_to_none=True)
                model(batch).square().mean().backward()
                optimizer.step()

            full_optimizer = torch.optim.AdamW(
                uninterrupted.parameters(), lr=1e-3
            )
            for batch in batches:
                update(uninterrupted, full_optimizer, batch)

            partial_optimizer = torch.optim.AdamW(
                interrupted.parameters(), lr=1e-3
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                partial_optimizer, lambda _step: 1.0
            )
            update(interrupted, partial_optimizer, batches[0])
            checkpoint = root / "mid_epoch.pt"
            _atomic_torch(
                checkpoint,
                _checkpoint_payload(
                    config=config,
                    model=interrupted,
                    optimizer=partial_optimizer,
                    optimizer_stage="vectorized_local",
                    scheduler=scheduler,
                    initialization={"path": "legacy.pt"},
                    history=[],
                    progress={
                        "local_epoch": 0,
                        "local_in_epoch": 1,
                        "local_cursor": 250,
                    },
                    best_f1=0.7,
                ),
            )
            resume_config = EndToEndTrainConfig(
                **{**config.__dict__, "resume_checkpoint": checkpoint}
            )
            resumed, payload = _load_resume(
                resume_config, torch.device("cpu")
            )
            resumed_optimizer = torch.optim.AdamW(
                resumed.parameters(), lr=1e-3
            )
            resumed_optimizer.load_state_dict(
                payload["optimizer_state_dict"]
            )
            update(resumed, resumed_optimizer, batches[1])

        self.assertEqual(payload["progress"]["local_cursor"], 250)
        self.assertIsNotNone(payload["scheduler_state_dict"])
        self.assertIsNotNone(payload["rng_state"])
        for expected, actual in zip(
            uninterrupted.parameters(), resumed.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))


class FrozenFrontendAndInferenceTests(unittest.TestCase):
    def test_config_marks_frontend_frozen_and_has_no_frontend_weights(self) -> None:
        config = EndToEndTrainConfig(
            manifest=Path("split.json"),
            cache_root=Path("cache"),
            output_dir=Path("run"),
            initialize_checkpoint=Path("legacy.pt"),
        )
        self.assertTrue(_config_payload(config)["frontend"]["frozen"])
        self.assertEqual(
            _config_payload(config)["frontend"]["candidate_generation"],
            "align-joint-candidates-v2-short-rescue",
        )
        self.assertTrue(
            _config_payload(config)["lattice"][
                "continuation_feature_enabled"
            ]
        )
        model = JointEdgeScorer(component_mode=True)
        self.assertFalse(
            any("basic_pitch" in name for name, _ in model.named_parameters())
        )

    def test_inference_loader_never_requests_gold_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "performance_audio.wav"
            wav.write_bytes(b"frozen-cache-hash-input")
            score_path = root / "verified_score.musicxml"
            score = stream.Score()
            part = stream.Part()
            part.append(note.Note("C4", quarterLength=1))
            score.insert(0, part)
            score.write("musicxml", fp=str(score_path))
            metadata = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "frontend_version": FRONTEND_VERSION,
                "basic_pitch_version": BASIC_PITCH_VERSION,
                "runtime": selected_runtime(),
                "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "pitch_policy": effective_pitch_policy({}),
                "pitch_space": "written",
            }
            frames = 4
            features = BasicPitchFeatures(
                note=np.zeros((frames, 88), dtype=np.float32),
                onset=np.zeros((frames, 88), dtype=np.float32),
                contour=np.zeros((frames, 264), dtype=np.float32),
                frame_times=np.arange(frames, dtype=np.float64) * 0.01,
                metadata=metadata,
            )
            cache = save_basic_pitch_cache(root / "cache.npz", features)
            with mock.patch(
                "alignmodel.joint.data.target_note_map",
                side_effect=AssertionError("gold target requested"),
            ), mock.patch(
                "alignmodel.joint.data.basic_pitch_candidate_union",
                return_value=[],
            ):
                candidates, score_events = load_inference_inputs(
                    score_path=score_path,
                    wav_path=wav,
                    cache_path=cache,
                )

        self.assertEqual(candidates, ())
        self.assertEqual(len(score_events), 1)


if __name__ == "__main__":
    unittest.main()
