from __future__ import annotations

import copy
import unittest

import torch

from alignmodel.joint.candidates import add_score_repeat_hints
from alignmodel.joint.index import ScoreEvent
from alignmodel.joint.lattice import (
    JointCandidate,
    JointEdgeScorer,
    JointOperation,
    LatticeConfig,
    SparseJointLattice,
    StructuralState,
    continuation_compatibility,
    segment_argmax_first,
    segment_logsumexp_fp32,
)


def _score() -> list[ScoreEvent]:
    return [
        ScoreEvent(i, 60 + 2 * i, float(i), float(i + 1), (i,))
        for i in range(4)
    ]


class JointLatticeTests(unittest.TestCase):
    def test_score_repeat_hints_follow_detected_phrase(self) -> None:
        score = [
            ScoreEvent(i, pitch, float(i), float(i + 1), (i,))
            for i, pitch in enumerate((60, 62, 64, 65, 67, 69))
        ]
        phrase = [event.pitch for event in score]
        candidates = [
            JointCandidate(pitch, index * 0.5, index * 0.5 + 0.4)
            for index, pitch in enumerate((*phrase, *phrase))
        ]
        hinted = add_score_repeat_hints(candidates, score)
        self.assertEqual(
            [candidate.score_hints for candidate in hinted[6:]],
            [(0,), (1,), (2,), (3,), (4,), (5,)],
        )

    def test_repeat_replay_and_explicit_resume_transitions(self) -> None:
        lattice = SparseJointLattice(JointEdgeScorer(hidden_dim=8))
        entered = lattice._transition(StructuralState(3), (1, 2))
        self.assertIsNotNone(entered)
        replay_state, operation, _deleted = entered
        self.assertEqual(operation, JointOperation.REPEAT_ENTER)
        self.assertEqual(replay_state.resume_event, 4)

        replayed = lattice._transition(replay_state, (2, 3))
        self.assertIsNotNone(replayed)
        replay_state, operation, _deleted = replayed
        self.assertEqual(operation, JointOperation.REPLAY)

        resumed = lattice._transition(replay_state, (4, 5))
        self.assertIsNotNone(resumed)
        normal_state, operation, _deleted = resumed
        self.assertEqual(operation, JointOperation.CONTINUE)
        self.assertEqual(normal_state.mode, "normal")

    def test_continuation_accepts_one_error_and_rejects_divergence(self) -> None:
        score = [
            ScoreEvent(i, pitch, float(i), float(i + 1), (i,))
            for i, pitch in enumerate(
                (60, 62, 64, 65, 67, 69, 71, 72)
            )
        ]
        prefix = (60, 62, 64, 65, 67)
        replay = (62, 64, 65, 67)

        def candidates(continuation):
            pitches = (*prefix, *replay, *continuation)
            return [
                JointCandidate(pitch, i * 0.4, i * 0.4 + 0.3)
                for i, pitch in enumerate(pitches)
            ]

        valid = continuation_compatibility(
            candidates((69, 80, 72)),
            score,
            candidate_index=5,
            replay_span=(1, 2),
            resume_event=5,
        )
        divergent = continuation_compatibility(
            candidates((80, 81, 82)),
            score,
            candidate_index=5,
            replay_span=(1, 2),
            resume_event=5,
        )
        self.assertGreater(valid, 0.0)
        self.assertLess(divergent, 0.0)

    def test_path_crf_loss_is_finite_and_differentiable(self) -> None:
        torch.manual_seed(3)
        scorer = JointEdgeScorer(hidden_dim=8, dropout=0.0)
        lattice = SparseJointLattice(
            scorer,
            LatticeConfig(
                max_options_per_candidate=8,
                max_states=32,
                max_delete_events=4,
            ),
        )
        score = _score()
        spans = [(0, 1), (1, 2), (0, 1), (1, 2), (2, 3)]
        candidates = [
            JointCandidate(score[span[0]].pitch, i * 0.5, i * 0.5 + 0.4, 0.9)
            for i, span in enumerate(spans)
        ]
        loss = lattice.nll(candidates, score, spans)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), -1e-4)
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.all(torch.isfinite(parameter.grad))
                for parameter in scorer.parameters()
            )
        )

        scorer.zero_grad(set_to_none=True)
        warmup = lattice.local_warmup_nll(
            candidates,
            score,
            spans,
            [True] * len(spans),
        )
        self.assertTrue(torch.isfinite(warmup))
        warmup.backward()

        unlinked_candidates = [
            JointCandidate(91, -1.0, -0.7, 0.2),
            JointCandidate(89, -0.5, -0.2, 0.9),
            *candidates,
        ]
        unlinked_spans = [None, None, *spans]
        unlinked_keep = [False, True, *([True] * len(spans))]
        unlinked_loss = lattice.nll(
            unlinked_candidates,
            score,
            unlinked_spans,
            unlinked_keep,
        )
        self.assertGreaterEqual(float(unlinked_loss.detach()), -1e-4)

    def test_vectorized_fp32_dp_and_gold_gradients_match_reference(self) -> None:
        groups = [
            torch.randn(size, requires_grad=True)
            for size in (3, 7, 2, 11)
        ]
        expected = torch.stack(
            [torch.logsumexp(group.float(), dim=0) for group in groups]
        )
        actual = segment_logsumexp_fp32(groups)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        expected_winners = []
        offset = 0
        for group in groups:
            expected_winners.append(offset + int(torch.argmax(group)))
            offset += len(group)
        self.assertEqual(
            segment_argmax_first(groups).tolist(),
            expected_winners,
        )
        expected.sum().backward()
        expected_gradients = [group.grad.detach().clone() for group in groups]
        for group in groups:
            group.grad = None
        actual.sum().backward()
        for wanted, group in zip(expected_gradients, groups):
            self.assertTrue(
                torch.allclose(group.grad, wanted, atol=1e-6, rtol=1e-6)
            )

        torch.manual_seed(37)
        template = JointEdgeScorer(hidden_dim=8, dropout=0.0)
        batched_model = copy.deepcopy(template)
        scalar_model = copy.deepcopy(template)
        config = LatticeConfig(max_options_per_candidate=8, max_states=32)
        spans = [(0, 1), (1, 2), (0, 1), (1, 2), (2, 3)]
        candidates = [
            JointCandidate(
                _score()[span[0]].pitch,
                index * 0.5,
                index * 0.5 + 0.4,
                0.9,
            )
            for index, span in enumerate(spans)
        ]
        batched = SparseJointLattice(batched_model, config).gold_path_score(
            candidates,
            _score(),
            spans,
            [True] * len(spans),
            batch_edge_scoring=True,
        )
        scalar = SparseJointLattice(scalar_model, config).gold_path_score(
            candidates,
            _score(),
            spans,
            [True] * len(spans),
            batch_edge_scoring=False,
        )
        self.assertTrue(torch.allclose(batched, scalar, atol=1e-6, rtol=1e-6))
        batched.backward()
        scalar.backward()
        for left, right in zip(
            batched_model.parameters(), scalar_model.parameters()
        ):
            self.assertTrue(
                torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-6)
            )

    def test_decode_emits_one_decision_per_candidate(self) -> None:
        torch.manual_seed(5)
        scorer = JointEdgeScorer(hidden_dim=8, dropout=0.0)
        lattice = SparseJointLattice(
            scorer,
            LatticeConfig(max_options_per_candidate=6, max_states=24),
        )
        candidates = [
            JointCandidate(60, 0.0, 0.4),
            JointCandidate(62, 0.5, 0.9),
        ]
        path = lattice.decode(candidates, _score())
        self.assertEqual(len(path.steps), len(candidates))
        self.assertLessEqual(len(path.joint_events(candidates)), len(candidates))


if __name__ == "__main__":
    unittest.main()
