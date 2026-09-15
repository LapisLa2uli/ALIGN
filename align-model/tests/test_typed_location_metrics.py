from __future__ import annotations

import pytest

from alignmodel.joint.metrics import evaluate_typed_location_labels


def _label(kind: str, start: float, end: float) -> dict[str, object]:
    return {"type": kind, "start_time": start, "end_time": end}


def test_exact_type_and_location_receives_full_credit() -> None:
    result = evaluate_typed_location_labels(
        [_label("wrong_note", 1.0, 2.0)],
        [_label("wrong_note", 1.0, 2.0)],
    )
    assert result["credit"] == 1.0
    assert result["f1"] == 1.0
    assert result["pair_counts"] == {
        "full_credit": 1,
        "half_credit": 0,
        "zero_credit": 0,
    }


def test_wrong_type_same_location_receives_half_credit() -> None:
    result = evaluate_typed_location_labels(
        [_label("extra_note", 1.0, 2.0)],
        [_label("wrong_note", 1.0, 2.0)],
        type_mismatch_credit=0.5,
    )
    assert result["credit"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["pair_counts"]["half_credit"] == 1


def test_right_type_wrong_location_receives_zero_credit() -> None:
    result = evaluate_typed_location_labels(
        [_label("wrong_note", 4.0, 5.0)],
        [_label("wrong_note", 1.0, 2.0)],
    )
    assert result["credit"] == 0.0
    assert result["f1"] == 0.0
    assert result["pair_counts"]["zero_credit"] == 1


def test_duplicate_predictions_compete_one_to_one() -> None:
    result = evaluate_typed_location_labels(
        [
            _label("wrong_note", 1.0, 2.0),
            _label("wrong_note", 1.0, 2.0),
        ],
        [_label("wrong_note", 1.0, 2.0)],
    )
    assert result["credit"] == 1.0
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["unmatched_predictions"] == 1


def test_global_assignment_beats_greedy_first_choice() -> None:
    # P0 can take either target; P1 can only take G0. Greedily taking the
    # P0/G0 full-credit edge scores 1.0, while the global optimum scores 1.5.
    result = evaluate_typed_location_labels(
        [
            _label("a", 0.04, 0.5),
            _label("a", -0.05, 0.4),
        ],
        [
            _label("a", 0.0, 0.5),
            _label("b", 0.09, 0.6),
        ],
        criterion="onset_100ms",
    )
    assert result["credit"] == 1.5
    assert result["pair_counts"]["full_credit"] == 1
    assert result["pair_counts"]["half_credit"] == 1


@pytest.mark.parametrize(
    ("predicted", "gold", "expected_f1"),
    [
        ([], [], 1.0),
        ([_label("a", 0.0, 1.0)], [], 0.0),
        ([], [_label("a", 0.0, 1.0)], 0.0),
    ],
)
def test_empty_cases(
    predicted: list[dict[str, object]],
    gold: list[dict[str, object]],
    expected_f1: float,
) -> None:
    result = evaluate_typed_location_labels(predicted, gold)
    assert result["f1"] == expected_f1


def test_fractional_counts_are_deterministic() -> None:
    predicted = [
        _label("wrong_note", 0.0, 1.0),
        _label("extra_note", 2.0, 3.0),
        _label("rhythm_error", 8.0, 9.0),
    ]
    gold = [
        _label("wrong_note", 0.0, 1.0),
        _label("missed_note", 2.0, 3.0),
        _label("rhythm_error", 5.0, 6.0),
    ]
    first = evaluate_typed_location_labels(predicted, gold)
    second = evaluate_typed_location_labels(list(reversed(predicted)), gold)
    assert first["credit"] == second["credit"] == 1.5
    assert first["pair_counts"] == second["pair_counts"] == {
        "full_credit": 1,
        "half_credit": 1,
        "zero_credit": 1,
    }
    assert first["f1"] == second["f1"] == 0.5
