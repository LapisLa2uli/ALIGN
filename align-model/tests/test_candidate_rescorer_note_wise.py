import torch
from torch import nn

from alignmodel.joint.candidate_rescorer import (
    _selected_joint_events,
    evaluate_candidate_rescorer,
)
from alignmodel.joint.data import JointTrainingExample
from alignmodel.joint.index import JointEvent, ScoreEvent
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import evaluate_joint_events


def _example(*, gold_keep_unlinked: tuple[bool, ...] = (True, False)) -> JointTrainingExample:
    target = JointEvent(
        pitch=60,
        start=0.0,
        end=0.4,
        score_span=(0, 1),
        relationship="match",
        rendered_index=0,
    )
    extra = JointCandidate(pitch=72, start=1.0, end=1.3, confidence=0.9)
    keep = JointCandidate(pitch=60, start=0.0, end=0.4, confidence=0.9)
    score = ScoreEvent(
        index=0,
        pitch=60,
        ql_start=0.0,
        ql_end=1.0,
        source_indices=(0,),
    )
    return JointTrainingExample(
        sample="clip",
        source="test",
        candidates=(keep, extra),
        score=(score,),
        gold_spans=((0, 1), None),
        gold_keep_unlinked=gold_keep_unlinked,
        target_events=(target,),
        target_deletions=frozenset(),
    )


def test_selected_candidates_use_official_note_wise_not_index_set():
    example = _example()
    keep_only = evaluate_joint_events(
        _selected_joint_events(example, {0}),
        example.target_events,
        target_deletions=example.target_deletions,
        score_event_count=1,
    )["official_note_wise"]
    keep_and_false_extra = evaluate_joint_events(
        _selected_joint_events(example, {0, 1}),
        example.target_events,
        target_deletions=example.target_deletions,
        score_event_count=1,
    )["official_note_wise"]
    assert keep_only["f1"] == 1.0
    assert keep_and_false_extra["f1"] < keep_only["f1"]
    assert keep_and_false_extra["predicted"] == 2
    assert keep_only["predicted"] == 1


class _FixedLogits(nn.Module):
    def __init__(self, logits: tuple[float, ...]) -> None:
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits[: features.size(0)].to(features.device)


def test_threshold_selection_prefers_note_wise_over_index_set():
    example = _example(gold_keep_unlinked=(True, True))
    threshold, detail = evaluate_candidate_rescorer(
        _FixedLogits((3.0, 0.0)),
        [example],
        device=torch.device("cpu"),
        thresholds=(0.9, 0.1),
    )
    assert threshold == 0.9
    assert detail["metric"] == "official_note_wise"
    assert detail["f1"] == 1.0
    assert detail["diagnostic_index_set_f1"] < detail["f1"]

