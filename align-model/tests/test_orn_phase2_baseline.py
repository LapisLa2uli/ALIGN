from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "eval_orn_phase2_baseline_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "eval_orn_phase2_baseline_v1", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def _event(
    index: int,
    *,
    relationship: str = "match",
    score_span: tuple[int, int] | None = None,
) -> baseline.JointEvent:
    return baseline.JointEvent(
        pitch=60 + index,
        start=float(index),
        end=float(index) + 0.5,
        score_span=score_span,
        relationship=relationship,
        rendered_index=index,
    )


def test_full_report_uses_exact_exclusive_note_identity() -> None:
    events = (
        _event(0, score_span=(0, 1)),
        _event(1, relationship="extra"),
    )
    sample = baseline.JointMetricSample(
        predicted=events,
        target=events,
        source="source-a",
        score_event_count=1,
    )
    report = baseline._full_report([sample], seed=7, replicates=10)
    assert report["f1"] == 1.0
    assert report["per_type"]["match"]["gold"] == 1
    assert report["per_type"]["extra"]["gold"] == 1
    assert report["source_macro"]["macro_f1"] == 1.0


def test_transcription_projection_preserves_rendered_extra_identity() -> None:
    event = _event(3, relationship="extra")
    projected = baseline._transcription_prediction(event, 3)
    assert projected.score_span is None
    assert projected.relationship == "extra"
    assert projected.rendered_index == 3


def test_lcs_and_bootstrap_counts_are_deterministic() -> None:
    assert baseline._lcs([60, 61, 62], [60, 62]) == 2
    counts = [(2.0, 3, 2), (1.0, 1, 2)]
    first = baseline._bootstrap_counts(counts, seed=11, replicates=20)
    second = baseline._bootstrap_counts(counts, seed=11, replicates=20)
    assert first == second
