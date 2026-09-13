from __future__ import annotations

import unittest

import torch

from alignmodel.cached_alignment_train import monotonic_sequence_nll


class CachedSequenceLossTests(unittest.TestCase):
    def test_monotonic_sequence_nll_is_finite_and_has_gradients(self) -> None:
        # Two played notes, two score notes, final column is "extra".
        logits = torch.tensor(
            [[3.0, -1.0, -2.0], [-1.0, 3.0, -2.0]],
            requires_grad=True,
        )
        loss = monotonic_sequence_nll(logits, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_extra_note_is_a_valid_sequence_edge(self) -> None:
        logits = torch.tensor(
            [[3.0, -1.0, -2.0], [-2.0, -2.0, 3.0]],
            requires_grad=True,
        )
        loss = monotonic_sequence_nll(logits, torch.tensor([0, 2]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
