from __future__ import annotations

import math

import torch

from alignmodel.joint.drop_emit_lattice_v1 import (
    DropEmitScorer,
    build_drop_emit_lattice,
    decode_drop_emit,
    drop_emit_nll,
    exact_identity_round_trip,
    prove_gold_path_coverage,
    teacher_decode,
)
from alignmodel.joint.identity_crf_v1 import IdentityTarget
from alignmodel.joint.index import JointEvent


def _candidate(pitch: int, start: float, *, confidence: float = 0.8) -> dict:
    return {
        "pitch": pitch,
        "start": start,
        "end": start + 0.2,
        "confidence": confidence,
        "note_peak": confidence,
        "note_mean": confidence * 0.9,
        "onset_peak": confidence * 0.7,
        "contour_peak": confidence * 0.6,
        "onset_contrast": 0.1,
        "lower_harmonic": 0.05,
        "upper_harmonic": 0.04,
        "duration_frames": 8,
        "source_kind": "standard_decode",
        "alternatives": [pitch, pitch + 1, pitch - 1, pitch + 2],
        "alternative_confidences": [0.5, 0.2, 0.15, 0.1],
    }


def test_synthetic_coverage_and_round_trip_with_drop_and_insert() -> None:
    # Three acoustic groups; gold emits first and third, inserts an extra between.
    pool = [
        _candidate(60, 0.0),
        _candidate(60, 0.0, confidence=0.4),  # same onset group
        _candidate(62, 0.5),  # will be DROPped
        _candidate(64, 1.0),
    ]
    targets = (
        JointEvent(60, 0.0, 0.2, (0, 1), "match", 0, rendered_index=0),
        JointEvent(61, 0.25, 0.3, None, "extra", 0, rendered_index=1),
        JointEvent(64, 1.0, 1.2, (1, 2), "match", 0, rendered_index=2),
    )
    assignments = [
        {
            "target_rendered_index": 0,
            "selected_interval_candidate": 0,
            "candidate_group": 0,
        },
        {
            "target_rendered_index": 2,
            "selected_interval_candidate": 3,
            "candidate_group": 2,
        },
    ]
    lattice = build_drop_emit_lattice(
        sample="synthetic",
        split="train",
        pool_candidates=pool,
        targets=targets,
        assignments=assignments,
    )
    coverage = prove_gold_path_coverage(lattice)
    assert coverage["passed"]
    assert coverage["acoustic_emits"] == 2
    assert coverage["template_inserts"] == 1
    assert coverage["dropped_groups"] == 1
    round_trip = exact_identity_round_trip(lattice)
    assert round_trip["passed"]
    teacher = teacher_decode(lattice)
    assert [step.action for step in teacher] == ["emit", "insert_extra", "emit"]


def test_nll_has_gradients_and_gold_mass() -> None:
    pool = [_candidate(60, 0.0), _candidate(61, 0.4)]
    targets = (
        IdentityTarget("match", (0, 1), 0, 0, 60),
        IdentityTarget("extra", None, 0, 1, 61),
    )
    assignments = [
        {
            "target_rendered_index": 0,
            "selected_interval_candidate": 0,
            "candidate_group": 0,
        },
        {
            "target_rendered_index": 1,
            "selected_interval_candidate": 1,
            "candidate_group": 1,
        },
    ]
    lattice = build_drop_emit_lattice(
        sample="tiny",
        split="train",
        pool_candidates=pool,
        targets=targets,
        assignments=assignments,
    )
    model = DropEmitScorer(hidden=16)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    loss, parts = drop_emit_nll(model, lattice, normalize=False)
    assert torch.isfinite(loss)
    assert parts["log_partition"] >= parts["gold_log_partition"] - 1e-5
    loss.backward()
    assert model.emit_bias.grad is not None
    assert torch.isfinite(model.emit_bias.grad)


def test_decode_respects_expected_emission_count() -> None:
    pool = [
        _candidate(60, 0.0, confidence=0.9),
        _candidate(62, 0.3, confidence=0.1),
        _candidate(64, 0.6, confidence=0.8),
    ]
    targets = (
        IdentityTarget("match", (0, 1), 0, 0, 60),
        IdentityTarget("match", (1, 2), 0, 1, 64),
    )
    assignments = [
        {
            "target_rendered_index": 0,
            "selected_interval_candidate": 0,
            "candidate_group": 0,
        },
        {
            "target_rendered_index": 1,
            "selected_interval_candidate": 2,
            "candidate_group": 2,
        },
    ]
    lattice = build_drop_emit_lattice(
        sample="decode",
        split="calibration",
        pool_candidates=pool,
        targets=targets,
        assignments=assignments,
    )
    model = DropEmitScorer(hidden=16)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    # Encourage EMIT over DROP/INSERT mildly via bias.
    with torch.no_grad():
        model.emit_bias.fill_(2.0)
        model.insert_bias.fill_(-4.0)
    emits, diagnostics = decode_drop_emit(
        model, lattice, expected_emissions=2, max_inserts_ahead=1
    )
    assert diagnostics["fallback"] is False
    assert len(emits) == 2
    assert math.isfinite(diagnostics["score"])
