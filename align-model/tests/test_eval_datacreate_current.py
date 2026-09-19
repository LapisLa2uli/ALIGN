from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignmodel.melody import match_note_wise_labels_detail


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "eval_datacreate_current.py"
)
SPEC = importlib.util.spec_from_file_location("eval_datacreate_current", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_sample_id_selects_only_requested_expected_directory(
    tmp_path: Path,
) -> None:
    assert [path.name for path in evaluator._selected_samples(tmp_path, "011")] == [
        "011"
    ]
    assert len(evaluator._selected_samples(tmp_path)) == 94
    with pytest.raises(ValueError, match="Unknown DataCreate sample id"):
        evaluator._selected_samples(tmp_path, "../011")


def test_core_projection_excludes_declared_context_padding() -> None:
    notes = [SimpleNamespace(pitch=value) for value in (60, 62, 64, 65)]
    label = {
        "id": "agent_0000",
        "type": "wrong_note",
        "score_part": {
            "start_note_index": 0,
            "end_note_index": 2,
            "pad_notes": 1,
            "core_start_note_index": 1,
            "core_end_note_index": 1,
        },
        "pitches": [60, 62, 64],
        "note_ids": ["note_0000", "note_0001", "note_0002"],
        "core_note_ids": ["note_0001"],
    }
    projected, audit = evaluator._canonical_projection_audit(label, notes)
    assert audit["accepted"] is True
    assert audit["projection_basis"] == "score_part_explicit_core"
    assert projected["score_event_indices"] == [1]


def test_explicit_core_range_is_authoritative_over_auxiliary_core_ids() -> None:
    notes = [SimpleNamespace(pitch=value) for value in (60, 62, 64)]
    label = {
        "type": "wrong_note",
        "score_part": {
            "start_note_index": 0,
            "end_note_index": 2,
            "pad_notes": 1,
            "core_start_note_index": 1,
            "core_end_note_index": 1,
        },
        "pitches": [60, 62, 64],
        "core_note_ids": ["note_0002"],
    }
    projected, audit = evaluator._canonical_projection_audit(label, notes)
    assert audit["accepted"] is True
    assert audit["warnings"] == [
        "core_note_ids_disagree_with_core_score_part"
    ]
    assert projected["score_event_indices"] == [1]


def test_official_core_identity_is_exclusive_and_type_fractional() -> None:
    notes = [SimpleNamespace(pitch=value) for value in (60, 62, 64)]
    base = {
        "score_part": {
            "start_note_index": 0,
            "end_note_index": 2,
            "pad_notes": 1,
            "core_start_note_index": 1,
            "core_end_note_index": 1,
        },
        "pitches": [60, 62, 64],
        "core_note_ids": ["note_0001"],
    }
    gold, gold_audit = evaluator._canonical_projection_audit(
        {**base, "type": "wrong_note"}, notes
    )
    wrong_type, pred_audit = evaluator._canonical_projection_audit(
        {**base, "type": "rhythm_error"}, notes
    )
    assert gold_audit["accepted"] and pred_audit["accepted"]
    detail = match_note_wise_labels_detail(
        [gold], [wrong_type, dict(wrong_type)], score_event_count=len(notes)
    )
    assert detail["credit"] == 0.5
    assert detail["precision"] == 0.25
    assert detail["recall"] == 0.5


def test_strict_guard_allows_declared_reads_and_rejects_other_access(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "011"
    sample.mkdir()
    guard = evaluator._SampleReadGuard(
        [sample], evaluator._FREEZE_SAMPLE_INPUTS, "freeze"
    )
    guard("open", (str(sample / "performance_audio.wav"), "r", 0))
    with pytest.raises(PermissionError, match="may not open"):
        guard("open", (str(sample / "labels.json"), "r", 0))
    with pytest.raises(PermissionError, match="may not write"):
        guard("open", (str(sample / "metadata.json"), "w", 0))
