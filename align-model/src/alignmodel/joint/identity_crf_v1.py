"""Globally normalized ornament/replay identity alignment CRF."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from math import inf
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .grammar_mapper_v2 import GrammarHypothesis, grammar_hypotheses
from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate
from .ornament_mapper_v1 import (
    OrnamentTemplateUnit,
    expand_ornament_hypothesis,
    score_ornament_patterns,
)


SCHEMA_VERSION = "align-ornament-identity-crf-v1"
LINK = 0
ORNAMENT_EXTRA = 1
CANDIDATE_EXTRA = 2
CANDIDATE_EXTRA_COPY = 3
SCORE_DELETE = 4
SKIP_ORNAMENT = 5
ACTION_COUNT = 6
ACTION_NAMES = (
    "link",
    "ornament_extra",
    "candidate_extra",
    "candidate_extra_copy",
    "score_delete",
    "skip_ornament",
)
NEGATIVE_INFINITY = -1e9


@dataclass(frozen=True)
class IdentityCandidate:
    pitch: int
    start: float
    end: float
    confidence: float = 1.0
    alternatives: tuple[int, ...] = ()
    alternative_confidences: tuple[float, ...] = ()

    @classmethod
    def from_joint(cls, value: JointCandidate) -> "IdentityCandidate":
        return cls(value.pitch, value.start, value.end, value.confidence)

    def to_joint(self) -> JointCandidate:
        return JointCandidate(
            self.pitch, self.start, self.end, self.confidence
        )


@dataclass(frozen=True)
class IdentityTarget:
    relationship: str
    score_span: tuple[int, int] | None
    copy_pass: int
    rendered_index: int
    pitch: int

    @classmethod
    def from_event(cls, value: JointEvent) -> "IdentityTarget":
        if value.rendered_index is None:
            raise ValueError("Gold rendered event lacks exclusive identity")
        return cls(
            relationship=(
                "copy" if value.is_copy else value.relationship
            ),
            score_span=value.score_span,
            copy_pass=value.copy_pass,
            rendered_index=int(value.rendered_index),
            pitch=value.pitch,
        )


@dataclass(frozen=True)
class LatticeHypothesis:
    grammar: GrammarHypothesis
    template: tuple[OrnamentTemplateUnit, ...]
    is_gold_compatible: bool
    hard_negative_rank: float


@dataclass(frozen=True)
class IdentityLattice:
    candidates: tuple[IdentityCandidate, ...]
    score: tuple[ScoreEvent, ...]
    hypotheses: tuple[LatticeHypothesis, ...]
    targets: tuple[IdentityTarget, ...] | None = None
    target_deletions: frozenset[int] = frozenset()


def _linked_identity(unit: OrnamentTemplateUnit) -> tuple[int, int] | None:
    if unit.kind != "linked" or unit.score_index is None:
        return None
    return int(unit.score_index), int(unit.copy_pass)


def _target_linked_identity(
    target: IdentityTarget,
) -> tuple[int, int, int] | None:
    if target.score_span is None:
        return None
    start, end = target.score_span
    return start, end, int(target.copy_pass)


def _subsequence(left: Sequence[Any], right: Sequence[Any]) -> bool:
    cursor = 0
    for value in right:
        if cursor < len(left) and value == left[cursor]:
            cursor += 1
    return cursor == len(left)


def _gold_compatible(
    template: Sequence[OrnamentTemplateUnit],
    targets: Sequence[IdentityTarget],
) -> bool:
    if max((unit.copy_pass for unit in template), default=0) != max(
        (target.copy_pass for target in targets), default=0
    ):
        return False
    linked_targets = [
        (score_index, copy_pass)
        for target in targets
        if (identity := _target_linked_identity(target)) is not None
        for score_index in range(identity[0], identity[1])
        for copy_pass in (identity[2],)
    ]
    linked_template = [
        identity
        for unit in template
        if (identity := _linked_identity(unit)) is not None
    ]
    return _subsequence(linked_targets, linked_template)


def _hypothesis_rank(
    hypothesis: GrammarHypothesis,
    template: Sequence[OrnamentTemplateUnit],
    candidates: Sequence[IdentityCandidate],
    score: Sequence[ScoreEvent],
) -> float:
    template_pitch = [unit.pitch for unit in template]
    candidate_pitch = [event.pitch for event in candidates]
    similarity = SequenceMatcher(
        None, candidate_pitch, template_pitch, autojunk=False
    ).ratio()
    return (
        abs(len(template) - len(candidates))
        + 4.0 * (1.0 - similarity)
        + 0.25 * hypothesis.copies
        + 0.01 * len(score)
    )


def build_identity_lattice(
    candidates: Sequence[IdentityCandidate | JointCandidate],
    score: Sequence[ScoreEvent],
    score_path: Path | str,
    *,
    targets: Sequence[JointEvent] | None = None,
    target_deletions: Sequence[int] = (),
    max_negative_hypotheses: int = 8,
    max_inference_hypotheses: int = 16,
) -> IdentityLattice:
    identity_candidates = tuple(
        value
        if isinstance(value, IdentityCandidate)
        else IdentityCandidate.from_joint(value)
        for value in candidates
    )
    score_tuple = tuple(score)
    patterns = score_ornament_patterns(score_path, score_tuple)
    grammar = grammar_hypotheses(
        score_tuple,
        len(identity_candidates),
        edit_slack=100000,
    )
    identity_targets = (
        tuple(IdentityTarget.from_event(value) for value in targets)
        if targets is not None
        else None
    )
    rows = []
    for hypothesis in grammar:
        template = expand_ornament_hypothesis(
            score_tuple, patterns, hypothesis
        )
        compatible = (
            _gold_compatible(template, identity_targets)
            if identity_targets is not None
            else False
        )
        rows.append(
            LatticeHypothesis(
                grammar=hypothesis,
                template=template,
                is_gold_compatible=compatible,
                hard_negative_rank=_hypothesis_rank(
                    hypothesis,
                    template,
                    identity_candidates,
                    score_tuple,
                ),
            )
        )
    if identity_targets is not None:
        gold = [row for row in rows if row.is_gold_compatible]
        if not gold:
            raise ValueError("Identity lattice has no gold-compatible replay path")
        negatives = sorted(
            (row for row in rows if not row.is_gold_compatible),
            key=lambda row: (
                row.hard_negative_rank,
                row.grammar.copies,
                row.grammar.source_span or (-1, -1),
            ),
        )[:max_negative_hypotheses]
        selected = [*gold, *negatives]
    else:
        selected = sorted(
            rows,
            key=lambda row: (
                row.hard_negative_rank,
                row.grammar.copies,
                row.grammar.source_span or (-1, -1),
            ),
        )[:max_inference_hypotheses]
    selected.sort(
        key=lambda row: (
            not row.is_gold_compatible,
            row.hard_negative_rank,
            row.grammar.copies,
            row.grammar.source_span or (-1, -1),
        )
    )
    return IdentityLattice(
        candidates=identity_candidates,
        score=score_tuple,
        hypotheses=tuple(selected),
        targets=identity_targets,
        target_deletions=frozenset(int(value) for value in target_deletions),
    )


def _event_features(
    candidates: Sequence[IdentityCandidate],
) -> list[list[float]]:
    start = min((value.start for value in candidates), default=0.0)
    end = max((value.end for value in candidates), default=start + 1.0)
    extent = max(end - start, 1e-6)
    output = []
    for index, event in enumerate(candidates):
        previous = candidates[index - 1] if index else None
        following = (
            candidates[index + 1] if index + 1 < len(candidates) else None
        )
        output.append(
            [
                event.pitch / 128.0,
                min(event.end - event.start, 2.0),
                event.confidence,
                (event.start - start) / extent,
                (event.start - previous.end) if previous else 0.0,
                (following.start - event.end) if following else 0.0,
                (event.pitch - previous.pitch) / 12.0 if previous else 0.0,
                (following.pitch - event.pitch) / 12.0 if following else 0.0,
                float(previous is not None and previous.pitch == event.pitch),
                float(following is not None and following.pitch == event.pitch),
                len(event.alternatives) / 4.0,
                max(event.alternative_confidences or (0.0,)),
            ]
        )
    return output


def _ornament_kind(value: str | None) -> list[float]:
    names = ("grace", "mordent", "inverted_mordent", "turn", "trill")
    text = str(value or "")
    return [float(text == name) for name in names]


def _template_features(
    template: Sequence[OrnamentTemplateUnit],
    score: Sequence[ScoreEvent],
) -> list[list[float]]:
    start = min((value.time for value in template), default=0.0)
    end = max((value.time for value in template), default=start + 1.0)
    extent = max(end - start, 1e-6)
    output = []
    for index, unit in enumerate(template):
        score_event = (
            score[unit.score_index]
            if unit.score_index is not None
            else None
        )
        output.append(
            [
                unit.pitch / 128.0,
                (unit.time - start) / extent,
                float(unit.kind == "linked"),
                float(unit.kind == "ornament_extra"),
                float(unit.copy_pass == 0),
                float(unit.copy_pass == 1),
                float(unit.copy_pass >= 2),
                (
                    float(score_event.ql_end - score_event.ql_start)
                    if score_event is not None
                    else 0.0
                ),
                index / max(len(template) - 1, 1),
                *_ornament_kind(unit.ornament_kind),
            ]
        )
    return output


def _hypothesis_features(
    hypothesis: LatticeHypothesis,
    candidates: Sequence[IdentityCandidate],
    score: Sequence[ScoreEvent],
) -> list[float]:
    span = hypothesis.grammar.source_span or (0, 0)
    return [
        float(hypothesis.grammar.copies == 0),
        float(hypothesis.grammar.copies == 1),
        float(hypothesis.grammar.copies >= 2),
        span[0] / max(len(score), 1),
        span[1] / max(len(score), 1),
        (span[1] - span[0]) / max(len(score), 1),
        len(hypothesis.template) / max(len(score), 1),
        len(candidates) / max(len(score), 1),
        (len(candidates) - len(hypothesis.template))
        / max(len(score), 1),
    ]


class OrnamentIdentityCRF(nn.Module):
    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.event_encoder = nn.Sequential(
            nn.Linear(12, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.template_encoder = nn.Sequential(
            nn.Linear(14, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.pair_score = nn.Sequential(
            nn.Linear(hidden * 2 + 6, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.extra_score = nn.Linear(hidden, 2)
        self.delete_score = nn.Linear(hidden, 2)
        self.hypothesis_score = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.initial = nn.Parameter(torch.zeros(ACTION_COUNT))
        self.transitions = nn.Parameter(
            torch.zeros(ACTION_COUNT, ACTION_COUNT)
        )

    def scores(
        self,
        lattice: IdentityLattice,
        hypothesis: LatticeHypothesis,
    ) -> dict[str, Tensor]:
        device = self.initial.device
        event_raw = torch.tensor(
            _event_features(lattice.candidates),
            dtype=torch.float32,
            device=device,
        )
        template_raw = torch.tensor(
            _template_features(hypothesis.template, lattice.score),
            dtype=torch.float32,
            device=device,
        )
        events = self.event_encoder(event_raw)
        template = self.template_encoder(template_raw)
        rows, columns = len(lattice.candidates), len(hypothesis.template)
        left = events[:, None, :].expand(rows, columns, self.hidden)
        right = template[None, :, :].expand(rows, columns, self.hidden)
        interactions = []
        for event in lattice.candidates:
            row = []
            for unit in hypothesis.template:
                alternative_match = max(
                    (
                        confidence
                        for pitch, confidence in zip(
                            event.alternatives,
                            event.alternative_confidences,
                        )
                        if pitch == unit.pitch
                    ),
                    default=0.0,
                )
                row.append(
                    [
                        (event.pitch - unit.pitch) / 12.0,
                        float(event.pitch == unit.pitch),
                        float(unit.pitch in event.alternatives),
                        alternative_match,
                        min(event.end - event.start, 2.0),
                        event.confidence,
                    ]
                )
            interactions.append(row)
        interaction_tensor = torch.tensor(
            interactions, dtype=torch.float32, device=device
        )
        pair = self.pair_score(
            torch.cat((left, right, interaction_tensor), dim=-1)
        )
        hypothesis_raw = torch.tensor(
            _hypothesis_features(
                hypothesis, lattice.candidates, lattice.score
            ),
            dtype=torch.float32,
            device=device,
        )
        return {
            "link": pair[:, :, 0],
            "ornament": pair[:, :, 1],
            "extra": self.extra_score(events),
            "delete": self.delete_score(template),
            "hypothesis": self.hypothesis_score(hypothesis_raw).squeeze(-1),
        }


def _emitted_relationship(
    candidate: IdentityCandidate,
    unit: OrnamentTemplateUnit,
) -> str:
    if unit.copy_pass:
        return "copy"
    return "match" if candidate.pitch == unit.pitch else "substitute"


def _action_allowed(
    action: int,
    candidate: IdentityCandidate | None,
    unit: OrnamentTemplateUnit | None,
    target: IdentityTarget | None,
    lattice: IdentityLattice,
    *,
    gold_only: bool,
) -> bool:
    if action == LINK:
        if candidate is None or unit is None or unit.kind != "linked":
            return False
        if not gold_only:
            return True
        assert target is not None
        target_identity = _target_linked_identity(target)
        return (
            target_identity is not None
            and target_identity[1] - 1 == int(unit.score_index)
            and target_identity[2] == unit.copy_pass
            and target.relationship == _emitted_relationship(candidate, unit)
        )
    if action == ORNAMENT_EXTRA:
        if (
            candidate is None
            or unit is None
            or unit.kind != "ornament_extra"
        ):
            return False
        if not gold_only:
            return True
        assert target is not None
        return (
            target.relationship == "extra"
            and candidate.pitch == unit.pitch
        )
    if action == CANDIDATE_EXTRA:
        if candidate is None:
            return False
        return not gold_only or (
            target is not None and target.relationship == "extra"
        )
    if action == CANDIDATE_EXTRA_COPY:
        if candidate is None:
            return False
        return not gold_only or (
            target is not None
            and target.relationship == "copy"
            and target.score_span is None
        )
    if action == SCORE_DELETE:
        if unit is None or unit.kind != "linked":
            return False
        if not gold_only:
            return True
        identity = _linked_identity(unit)
        assert identity is not None
        linked = {
            (score_index, copy_pass)
            for value in (
                _target_linked_identity(item)
                for item in (lattice.targets or ())
            )
            if value is not None
            for score_index in range(value[0], value[1])
            for copy_pass in (value[2],)
        }
        score_index, copy_pass = identity
        if identity in linked:
            return False
        return copy_pass > 0 or score_index in lattice.target_deletions
    if action == SKIP_ORNAMENT:
        return unit is not None and unit.kind == "ornament_extra"
    return False


def _predecessor(
    action: int, row: int, column: int
) -> tuple[int, int] | None:
    if action in {LINK, ORNAMENT_EXTRA}:
        return (row - 1, column - 1) if row and column else None
    if action in {CANDIDATE_EXTRA, CANDIDATE_EXTRA_COPY}:
        return (row - 1, column) if row else None
    if action in {SCORE_DELETE, SKIP_ORNAMENT}:
        return (row, column - 1) if column else None
    return None


def _link_span_options(
    template: Sequence[OrnamentTemplateUnit],
    column: int,
    *,
    max_score_events: int = 8,
) -> list[tuple[int, tuple[int, int], int]]:
    """Return predecessor columns and canonical spans ending at ``column``."""

    if column <= 0:
        return []
    end_unit = template[column - 1]
    if end_unit.kind != "linked" or end_unit.score_index is None:
        return []
    copy_pass = int(end_unit.copy_pass)
    expected = int(end_unit.score_index)
    output = [(column - 1, (expected, expected + 1), copy_pass)]
    linked_count = 1
    for index in range(column - 2, -1, -1):
        unit = template[index]
        if unit.kind != "linked":
            continue
        if (
            unit.copy_pass != copy_pass
            or unit.score_index is None
            or int(unit.score_index) != expected - 1
        ):
            break
        expected -= 1
        linked_count += 1
        output.append(
            (index, (expected, int(end_unit.score_index) + 1), copy_pass)
        )
        if linked_count >= max_score_events:
            break
    return output


def _emission(
    scores: Mapping[str, Tensor],
    action: int,
    row: int,
    column: int,
) -> Tensor:
    if action == LINK:
        return scores["link"][row - 1, column - 1]
    if action == ORNAMENT_EXTRA:
        return scores["ornament"][row - 1, column - 1]
    if action == CANDIDATE_EXTRA:
        return scores["extra"][row - 1, 0]
    if action == CANDIDATE_EXTRA_COPY:
        return scores["extra"][row - 1, 1]
    if action == SCORE_DELETE:
        return scores["delete"][column - 1, 0]
    if action == SKIP_ORNAMENT:
        return scores["delete"][column - 1, 1]
    raise ValueError(action)


def hypothesis_log_partition(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
    hypothesis: LatticeHypothesis,
    *,
    gold_only: bool = False,
) -> Tensor:
    if gold_only and lattice.targets is None:
        raise ValueError("Gold partition requires exact identity targets")
    scores = model.scores(lattice, hypothesis)
    rows, columns = len(lattice.candidates), len(hypothesis.template)
    grid: list[list[Tensor | None]] = [
        [None] * (columns + 1) for _ in range(rows + 1)
    ]
    for diagonal in range(1, rows + columns + 1):
        first_row = max(0, diagonal - columns)
        last_row = min(rows, diagonal)
        for row in range(first_row, last_row + 1):
            column = diagonal - row
            if not 0 <= column <= columns:
                continue
            values = []
            candidate = lattice.candidates[row - 1] if row else None
            target = (
                lattice.targets[row - 1]
                if gold_only and row and lattice.targets is not None
                else None
            )
            unit = hypothesis.template[column - 1] if column else None
            for action in range(ACTION_COUNT):
                if action == LINK and candidate is not None:
                    link_values = []
                    for (
                        previous_column,
                        score_span,
                        copy_pass,
                    ) in _link_span_options(hypothesis.template, column):
                        if gold_only:
                            assert target is not None
                            if (
                                target.score_span != score_span
                                or target.copy_pass != copy_pass
                                or target.relationship
                                != _emitted_relationship(candidate, unit)
                            ):
                                continue
                        previous_row = row - 1
                        if previous_row == 0 and previous_column == 0:
                            prefix = model.initial[action]
                        else:
                            previous = grid[previous_row][previous_column]
                            if previous is None:
                                continue
                            prefix = torch.logsumexp(
                                previous + model.transitions[:, action],
                                dim=0,
                            )
                        skipped = model.initial.new_zeros(())
                        for template_index in range(
                            previous_column, column - 1
                        ):
                            skipped_unit = hypothesis.template[template_index]
                            skipped = skipped + scores["delete"][
                                template_index,
                                1
                                if skipped_unit.kind == "ornament_extra"
                                else 0,
                            ]
                        link_values.append(
                            prefix
                            + _emission(scores, action, row, column)
                            + skipped
                        )
                    values.append(
                        torch.logsumexp(torch.stack(link_values), dim=0)
                        if link_values
                        else model.initial.new_tensor(NEGATIVE_INFINITY)
                    )
                    continue
                predecessor = _predecessor(action, row, column)
                if predecessor is None or not _action_allowed(
                    action,
                    candidate,
                    unit,
                    target,
                    lattice,
                    gold_only=gold_only,
                ):
                    values.append(model.initial.new_tensor(NEGATIVE_INFINITY))
                    continue
                previous_row, previous_column = predecessor
                if previous_row == 0 and previous_column == 0:
                    prefix = model.initial[action]
                else:
                    previous = grid[previous_row][previous_column]
                    if previous is None:
                        prefix = model.initial.new_tensor(NEGATIVE_INFINITY)
                    else:
                        prefix = torch.logsumexp(
                            previous + model.transitions[:, action], dim=0
                        )
                values.append(
                    prefix + _emission(scores, action, row, column)
                )
            grid[row][column] = torch.stack(values)
    final = grid[rows][columns]
    if final is None:
        return model.initial.new_tensor(NEGATIVE_INFINITY)
    return torch.logsumexp(final, dim=0) + scores["hypothesis"]


def identity_crf_nll(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
    *,
    normalize: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    if lattice.targets is None:
        raise ValueError("CRF training requires exact identity targets")
    all_values = [
        hypothesis_log_partition(model, lattice, hypothesis)
        for hypothesis in lattice.hypotheses
    ]
    gold_values = [
        hypothesis_log_partition(
            model, lattice, hypothesis, gold_only=True
        )
        for hypothesis in lattice.hypotheses
        if hypothesis.is_gold_compatible
    ]
    if not gold_values:
        raise ValueError("Selected lattice contains no gold hypothesis")
    log_partition = torch.logsumexp(torch.stack(all_values), dim=0)
    gold_partition = torch.logsumexp(torch.stack(gold_values), dim=0)
    loss = log_partition - gold_partition
    if normalize:
        loss = loss / max(
            len(lattice.candidates) + len(lattice.score), 1
        )
    return loss, {
        "loss": loss.detach(),
        "log_partition": log_partition.detach(),
        "gold_log_partition": gold_partition.detach(),
    }


def _viterbi_hypothesis(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
    hypothesis: LatticeHypothesis,
) -> tuple[float, list[tuple[int, int, int]]]:
    with torch.no_grad():
        scores = model.scores(lattice, hypothesis)
    rows, columns = len(lattice.candidates), len(hypothesis.template)
    values = [
        [[-inf] * ACTION_COUNT for _ in range(columns + 1)]
        for _ in range(rows + 1)
    ]
    back: dict[tuple[int, int, int], tuple[int, int, int | None]] = {}
    transition = model.transitions.detach().cpu().numpy()
    initial = model.initial.detach().cpu().numpy()
    for diagonal in range(1, rows + columns + 1):
        for row in range(max(0, diagonal - columns), min(rows, diagonal) + 1):
            column = diagonal - row
            if not 0 <= column <= columns:
                continue
            candidate = lattice.candidates[row - 1] if row else None
            unit = hypothesis.template[column - 1] if column else None
            for action in range(ACTION_COUNT):
                if action == LINK and candidate is not None:
                    ranked_spans = []
                    for (
                        previous_column,
                        _score_span,
                        _copy_pass,
                    ) in _link_span_options(hypothesis.template, column):
                        previous_row = row - 1
                        if previous_row == 0 and previous_column == 0:
                            prefix = float(initial[action])
                            previous_action = None
                        else:
                            ranked_previous = [
                                (
                                    values[previous_row][previous_column][old]
                                    + float(transition[old, action]),
                                    old,
                                )
                                for old in range(ACTION_COUNT)
                            ]
                            prefix, previous_action = max(
                                ranked_previous,
                                key=lambda value: (value[0], -value[1]),
                            )
                        skipped = 0.0
                        for template_index in range(
                            previous_column, column - 1
                        ):
                            skipped_unit = hypothesis.template[template_index]
                            skipped += float(
                                scores["delete"][
                                    template_index,
                                    1
                                    if skipped_unit.kind == "ornament_extra"
                                    else 0,
                                ]
                                .detach()
                                .cpu()
                            )
                        ranked_spans.append(
                            (
                                prefix
                                + float(
                                    _emission(
                                        scores, action, row, column
                                    )
                                    .detach()
                                    .cpu()
                                )
                                + skipped,
                                previous_row,
                                previous_column,
                                previous_action,
                            )
                        )
                    if ranked_spans:
                        (
                            value,
                            previous_row,
                            previous_column,
                            previous_action,
                        ) = max(
                            ranked_spans,
                            key=lambda item: (
                                item[0],
                                item[2],
                                -1
                                if item[3] is None
                                else -int(item[3]),
                            ),
                        )
                        values[row][column][action] = value
                        back[(row, column, action)] = (
                            previous_row,
                            previous_column,
                            previous_action,
                        )
                    continue
                predecessor = _predecessor(action, row, column)
                if predecessor is None or not _action_allowed(
                    action,
                    candidate,
                    unit,
                    None,
                    lattice,
                    gold_only=False,
                ):
                    continue
                previous_row, previous_column = predecessor
                if previous_row == 0 and previous_column == 0:
                    prefix = float(initial[action])
                    previous_action = None
                else:
                    ranked = [
                        (
                            values[previous_row][previous_column][old]
                            + float(transition[old, action]),
                            old,
                        )
                        for old in range(ACTION_COUNT)
                    ]
                    prefix, previous_action = max(
                        ranked, key=lambda value: (value[0], -value[1])
                    )
                emission = float(
                    _emission(scores, action, row, column).detach().cpu()
                )
                values[row][column][action] = prefix + emission
                back[(row, column, action)] = (
                    previous_row,
                    previous_column,
                    previous_action,
                )
    final_score, action = max(
        (
            (
                values[rows][columns][action]
                + float(scores["hypothesis"].detach().cpu()),
                action,
            )
            for action in range(ACTION_COUNT)
        ),
        key=lambda value: (value[0], -value[1]),
    )
    path = []
    row, column = rows, columns
    while row or column:
        previous_row, previous_column, previous_action = back[
            (row, column, action)
        ]
        path.append(
            (
                action,
                row,
                column,
                previous_row,
                previous_column,
            )
        )
        row, column = previous_row, previous_column
        if previous_action is None:
            break
        action = previous_action
    path.reverse()
    return final_score, path


def decode_identity_crf(
    model: OrnamentIdentityCRF,
    lattice: IdentityLattice,
) -> tuple[tuple[JointEvent, ...], frozenset[int], dict[str, Any]]:
    ranked = [
        (*_viterbi_hypothesis(model, lattice, hypothesis), hypothesis)
        for hypothesis in lattice.hypotheses
    ]
    score_value, path, hypothesis = max(
        ranked,
        key=lambda value: (
            value[0],
            -value[2].grammar.copies,
            value[2].grammar.source_span or (-1, -1),
        ),
    )
    events = []
    deletions = set()
    for action, row, column, _previous_row, previous_column in path:
        candidate = lattice.candidates[row - 1] if row else None
        unit = hypothesis.template[column - 1] if column else None
        if action == LINK:
            assert candidate is not None and unit is not None
            relationship = _emitted_relationship(candidate, unit)
            linked_units = [
                value
                for value in hypothesis.template[previous_column:column]
                if value.kind == "linked"
                and value.score_index is not None
                and value.copy_pass == unit.copy_pass
            ]
            span_start = min(
                int(value.score_index) for value in linked_units
            )
            span_end = max(
                int(value.score_index) for value in linked_units
            ) + 1
            events.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=(span_start, span_end),
                    relationship=relationship,
                    copy_pass=unit.copy_pass,
                    origin_relationship=(
                        "match"
                        if candidate.pitch == unit.pitch
                        else "substitute"
                    ),
                    rendered_index=row - 1,
                    confidence=candidate.confidence,
                )
            )
        elif action in {
            ORNAMENT_EXTRA,
            CANDIDATE_EXTRA,
            CANDIDATE_EXTRA_COPY,
        }:
            assert candidate is not None
            events.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=None,
                    relationship=(
                        "copy"
                        if action == CANDIDATE_EXTRA_COPY
                        else "extra"
                    ),
                    copy_pass=1 if action == CANDIDATE_EXTRA_COPY else 0,
                    rendered_index=row - 1,
                    confidence=candidate.confidence,
                )
            )
        elif (
            action == SCORE_DELETE
            and unit is not None
            and unit.copy_pass == 0
            and unit.score_index is not None
        ):
            deletions.add(int(unit.score_index))
    return tuple(events), frozenset(deletions), {
        "score": score_value,
        "copies": hypothesis.grammar.copies,
        "source_span": hypothesis.grammar.source_span,
        "hypotheses": len(lattice.hypotheses),
        "actions": {
            name: sum(
                action == index
                for action, _row, _column, _previous_row, _previous_column in path
            )
            for index, name in enumerate(ACTION_NAMES)
        },
    }


def exact_identity_round_trip(
    lattice: IdentityLattice,
) -> dict[str, Any]:
    """Prove that at least one deterministic gold path reproduces every identity."""

    if lattice.targets is None:
        raise ValueError("Round-trip proof requires targets")
    compatible = [
        value for value in lattice.hypotheses if value.is_gold_compatible
    ]
    if not compatible:
        raise ValueError("No gold-compatible hypothesis")
    hypothesis = compatible[0]
    template = hypothesis.template
    column = 0
    emitted = []
    deleted = set()
    for row, (candidate, target) in enumerate(
        zip(lattice.candidates, lattice.targets)
    ):
        identity = _target_linked_identity(target)
        if identity is None:
            emitted.append(target)
            continue
        found: tuple[int, int] | None = None
        for end_column in range(column + 1, len(template) + 1):
            for previous_column, score_span, copy_pass in _link_span_options(
                template, end_column
            ):
                if (
                    previous_column >= column
                    and score_span == identity[:2]
                    and copy_pass == identity[2]
                ):
                    found = (previous_column, end_column)
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError(f"Gold identity is absent from template: {identity}")
        previous_column, end_column = found
        for unit in template[column:previous_column]:
            linked = _linked_identity(unit)
            if linked is not None and linked[1] == 0:
                deleted.add(linked[0])
        end_unit = template[end_column - 1]
        emitted.append(
            IdentityTarget(
                relationship=_emitted_relationship(
                    candidate, end_unit
                ),
                score_span=identity[:2],
                copy_pass=identity[2],
                rendered_index=row,
                pitch=candidate.pitch,
            )
        )
        column = end_column
    for unit in template[column:]:
        linked = _linked_identity(unit)
        if linked is not None and linked[1] == 0:
            deleted.add(linked[0])
    target_values = list(lattice.targets)
    passed = emitted == target_values and deleted == set(
        lattice.target_deletions
    )
    return {
        "passed": passed,
        "events": len(emitted),
        "deletions": len(deleted),
        "gold_hypotheses": len(compatible),
        "target_events_equal": emitted == target_values,
        "target_deletions_equal": deleted
        == set(lattice.target_deletions),
    }
