"""Jointly normalized acoustic DROP/EMIT lattice with emitted-event count.

This lattice sits outside the frozen identity CRF. Candidate groups may be
DROPped without emitting, EMITted as the next exclusive rendered identity, or
bridged by INSERT_EXTRA when template/score expectations account for events
missing from the acoustic pool. Globally normalized training marginalizes
gold-consistent emit/drop/insert paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .identity_crf_v1 import IdentityCandidate, IdentityTarget
from .index import JointEvent
from ..transcription.activation_candidate_scorer_v1 import candidate_features


SCHEMA_VERSION = "align-drop-emit-lattice-v1"
DROP = 0
EMIT = 1
INSERT_EXTRA = 2
ACTION_NAMES = ("drop", "emit", "insert_extra")
ACTION_COUNT = 3
NEGATIVE_INFINITY = -1e9
GROUP_FEATURE_DIM = 20
EMIT_FEATURE_DIM = 28
INSERT_FEATURE_DIM = 16


@dataclass(frozen=True)
class AcousticGroup:
    pitch: int
    onset_frame: int
    candidate_indices: tuple[int, ...]
    candidates: tuple[Mapping[str, Any], ...]

    @property
    def best_candidate(self) -> Mapping[str, Any]:
        return max(
            self.candidates,
            key=lambda value: (
                float(value["confidence"]),
                float(value["onset_peak"]),
                float(value["note_peak"]),
            ),
        )


@dataclass(frozen=True)
class GoldEmitStep:
    """One exclusive gold emission in rendered-index order."""

    rendered_index: int
    target: IdentityTarget
    group_index: int | None
    candidate_index: int | None
    source: str


@dataclass(frozen=True)
class DropEmitLattice:
    sample: str
    split: str
    groups: tuple[AcousticGroup, ...]
    pool_candidates: tuple[Mapping[str, Any], ...]
    targets: tuple[IdentityTarget, ...]
    gold_steps: tuple[GoldEmitStep, ...]
    target_deletions: frozenset[int] = frozenset()

    @property
    def gold_path_covered(self) -> bool:
        return len(self.gold_steps) == len(self.targets) and all(
            step.rendered_index == index for index, step in enumerate(self.gold_steps)
        )


def _identity_from_event(event: JointEvent) -> IdentityTarget:
    return IdentityTarget.from_event(event)


def build_acoustic_groups(
    pool_candidates: Sequence[Mapping[str, Any]],
    *,
    hop: float = 256.0 / 22050.0,
) -> tuple[AcousticGroup, ...]:
    buckets: dict[tuple[int, int], list[tuple[int, Mapping[str, Any]]]] = {}
    for index, candidate in enumerate(pool_candidates):
        onset_frame = int(round(float(candidate["start"]) / hop))
        key = (int(candidate["pitch"]), onset_frame)
        buckets.setdefault(key, []).append((index, candidate))
    groups = []
    for (pitch, onset_frame), rows in sorted(
        buckets.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        indices = tuple(index for index, _candidate in rows)
        candidates = tuple(candidate for _index, candidate in rows)
        groups.append(
            AcousticGroup(
                pitch=pitch,
                onset_frame=onset_frame,
                candidate_indices=indices,
                candidates=candidates,
            )
        )
    return tuple(groups)


def build_gold_emit_steps(
    groups: Sequence[AcousticGroup],
    targets: Sequence[IdentityTarget | JointEvent],
    assignments: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[GoldEmitStep, ...]:
    identity_targets = tuple(
        value if isinstance(value, IdentityTarget) else _identity_from_event(value)
        for value in targets
    )
    assignment_by_target: dict[int, Mapping[str, Any]] = {}
    for row in assignments or ():
        assignment_by_target[int(row["target_rendered_index"])] = row
    group_by_key = {
        (group.pitch, group.onset_frame): index for index, group in enumerate(groups)
    }
    # Also index by any candidate belonging to the group for robust lookup.
    group_by_candidate: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for candidate_index in group.candidate_indices:
            group_by_candidate[candidate_index] = group_index

    steps: list[GoldEmitStep] = []
    for target in identity_targets:
        assignment = assignment_by_target.get(int(target.rendered_index))
        if assignment is not None:
            candidate_index = int(assignment["selected_interval_candidate"])
            group_index = group_by_candidate.get(candidate_index)
            if group_index is None:
                # Fall back to pitch/onset group if present in assignment.
                group_index = int(assignment.get("candidate_group", -1))
                if group_index < 0 or group_index >= len(groups):
                    group_index = None
            steps.append(
                GoldEmitStep(
                    rendered_index=int(target.rendered_index),
                    target=target,
                    group_index=group_index,
                    candidate_index=candidate_index if group_index is not None else None,
                    source="acoustic_emit",
                )
            )
            continue
        steps.append(
            GoldEmitStep(
                rendered_index=int(target.rendered_index),
                target=target,
                group_index=None,
                candidate_index=None,
                source="template_insert",
            )
        )
    return tuple(steps)


def build_drop_emit_lattice(
    *,
    sample: str,
    split: str,
    pool_candidates: Sequence[Mapping[str, Any]],
    targets: Sequence[IdentityTarget | JointEvent],
    assignments: Sequence[Mapping[str, Any]] | None = None,
    target_deletions: Sequence[int] = (),
    hop: float = 256.0 / 22050.0,
) -> DropEmitLattice:
    groups = build_acoustic_groups(pool_candidates, hop=hop)
    identity_targets = tuple(
        value if isinstance(value, IdentityTarget) else _identity_from_event(value)
        for value in targets
    )
    gold_steps = build_gold_emit_steps(groups, identity_targets, assignments)
    return DropEmitLattice(
        sample=sample,
        split=split,
        groups=groups,
        pool_candidates=tuple(pool_candidates),
        targets=identity_targets,
        gold_steps=gold_steps,
        target_deletions=frozenset(int(value) for value in target_deletions),
    )


def prove_gold_path_coverage(lattice: DropEmitLattice) -> dict[str, Any]:
    """Fail-closed coverage: every gold identity is either EMITted or INSERTed."""

    if len(lattice.gold_steps) != len(lattice.targets):
        return {
            "passed": False,
            "reason": "gold_step_count_mismatch",
            "gold_steps": len(lattice.gold_steps),
            "targets": len(lattice.targets),
        }
    used_groups: set[int] = set()
    previous_group = -1
    for index, step in enumerate(lattice.gold_steps):
        if step.rendered_index != index:
            return {
                "passed": False,
                "reason": "rendered_index_gap",
                "index": index,
                "rendered_index": step.rendered_index,
            }
        if step.target != lattice.targets[index]:
            return {
                "passed": False,
                "reason": "target_identity_mismatch",
                "index": index,
            }
        if step.group_index is not None:
            if step.group_index in used_groups:
                return {
                    "passed": False,
                    "reason": "group_reused",
                    "group_index": step.group_index,
                }
            if step.group_index < previous_group:
                return {
                    "passed": False,
                    "reason": "non_monotonic_group_cursor",
                    "group_index": step.group_index,
                    "previous_group": previous_group,
                }
            used_groups.add(step.group_index)
            previous_group = step.group_index
            if step.candidate_index not in lattice.groups[step.group_index].candidate_indices:
                return {
                    "passed": False,
                    "reason": "candidate_outside_group",
                    "candidate_index": step.candidate_index,
                    "group_index": step.group_index,
                }
        elif step.source != "template_insert":
            return {
                "passed": False,
                "reason": "missing_insert_source",
                "index": index,
            }
    return {
        "passed": True,
        "targets": len(lattice.targets),
        "acoustic_emits": sum(
            step.group_index is not None for step in lattice.gold_steps
        ),
        "template_inserts": sum(
            step.group_index is None for step in lattice.gold_steps
        ),
        "groups": len(lattice.groups),
        "dropped_groups": len(lattice.groups) - len(used_groups),
    }


def exact_identity_round_trip(lattice: DropEmitLattice) -> dict[str, Any]:
    """Rebuild exclusive identities from the deterministic gold DROP/EMIT path."""

    coverage = prove_gold_path_coverage(lattice)
    if not coverage["passed"]:
        return {"passed": False, "coverage": coverage, "events": ()}
    emitted: list[IdentityTarget] = []
    for step in lattice.gold_steps:
        if step.group_index is None:
            emitted.append(
                IdentityTarget(
                    relationship=step.target.relationship,
                    score_span=step.target.score_span,
                    copy_pass=step.target.copy_pass,
                    rendered_index=len(emitted),
                    pitch=step.target.pitch,
                )
            )
            continue
        group = lattice.groups[step.group_index]
        assert step.candidate_index is not None
        candidate = lattice.pool_candidates[step.candidate_index]
        emitted.append(
            IdentityTarget(
                relationship=step.target.relationship,
                score_span=step.target.score_span,
                copy_pass=step.target.copy_pass,
                rendered_index=len(emitted),
                pitch=int(candidate["pitch"]),
            )
        )
    target_values = [
        IdentityTarget(
            relationship=target.relationship,
            score_span=target.score_span,
            copy_pass=target.copy_pass,
            rendered_index=index,
            pitch=target.pitch,
        )
        for index, target in enumerate(lattice.targets)
    ]
    # Pitch may differ only for substitute relationships when acoustic pitch wins;
    # official identity still requires the gold relationship/span/copy_pass and the
    # exclusive rendered index. Round-trip checks identity fields plus acoustic pitch
    # equality when the gold relationship is not substitute.
    matched = True
    for predicted, gold in zip(emitted, target_values):
        if (
            predicted.relationship != gold.relationship
            or predicted.score_span != gold.score_span
            or predicted.copy_pass != gold.copy_pass
            or predicted.rendered_index != gold.rendered_index
        ):
            matched = False
            break
        if gold.relationship != "substitute" and predicted.pitch != gold.pitch:
            matched = False
            break
    return {
        "passed": matched and len(emitted) == len(target_values),
        "coverage": coverage,
        "events": len(emitted),
        "target_events_equal": matched,
    }


def gold_emit_candidates(lattice: DropEmitLattice) -> tuple[IdentityCandidate, ...]:
    """Materialize the acoustic+insert candidate sequence for the gold path."""

    output: list[IdentityCandidate] = []
    for step in lattice.gold_steps:
        if step.group_index is None or step.candidate_index is None:
            # Synthetic insert: zero-duration placeholder at a monotonic time.
            start = float(len(output)) * 0.01
            output.append(
                IdentityCandidate(
                    pitch=step.target.pitch,
                    start=start,
                    end=start + 0.05,
                    confidence=0.05,
                )
            )
            continue
        candidate = lattice.pool_candidates[step.candidate_index]
        alternatives = tuple(int(value) for value in candidate.get("alternatives") or ())
        alt_conf = tuple(
            float(value) for value in candidate.get("alternative_confidences") or ()
        )
        output.append(
            IdentityCandidate(
                pitch=int(candidate["pitch"]),
                start=float(candidate["start"]),
                end=float(candidate["end"]),
                confidence=float(candidate["confidence"]),
                alternatives=alternatives,
                alternative_confidences=alt_conf,
            )
        )
    return tuple(output)


def _group_features(group: AcousticGroup) -> list[float]:
    best = group.best_candidate
    confidences = [float(value["confidence"]) for value in group.candidates]
    return [
        float(group.pitch) / 127.0,
        np.log1p(float(group.onset_frame)) / 12.0,
        np.log1p(float(len(group.candidates))) / 3.0,
        float(np.max(confidences)),
        float(np.mean(confidences)),
        float(np.min(confidences)),
        float(best["note_peak"]),
        float(best["note_mean"]),
        float(best["onset_peak"]),
        float(best["contour_peak"]),
        float(best["onset_contrast"]),
        float(best["lower_harmonic"]),
        float(best["upper_harmonic"]),
        np.log1p(float(best["duration_frames"])) / 5.0,
        float(str(best.get("source_kind") or "") == "standard_decode"),
        float(str(best.get("source_kind") or "").startswith("activation_run_")),
        float(str(best.get("source_kind") or "").startswith("fixed_")),
        float(len(group.candidates) == 1),
        float(max(confidences) - min(confidences)),
        float(np.std(confidences) if len(confidences) > 1 else 0.0),
    ]


def _emit_features(
    group: AcousticGroup,
    candidate: Mapping[str, Any],
    *,
    emitted_so_far: int,
    remaining_groups: int,
    target_hint: IdentityTarget | None = None,
) -> list[float]:
    base = candidate_features(candidate)  # 16 dims
    values = base + [
        float(emitted_so_far) / 64.0,
        float(remaining_groups) / 64.0,
        np.log1p(float(len(group.candidates))) / 3.0,
        float(group.pitch) / 127.0,
        np.log1p(float(group.onset_frame)) / 12.0,
        float(candidate["confidence"]),
        float(target_hint is not None),
        (
            float(target_hint.relationship == "extra")
            if target_hint is not None
            else 0.0
        ),
        (
            float(target_hint.relationship in {"match", "substitute"})
            if target_hint is not None
            else 0.0
        ),
        (
            float(target_hint.relationship == "copy")
            if target_hint is not None
            else 0.0
        ),
        (
            float(target_hint.pitch == int(candidate["pitch"]))
            if target_hint is not None
            else 0.0
        ),
        float(str(candidate.get("source_kind") or "") == "standard_decode"),
    ]
    if len(values) != EMIT_FEATURE_DIM:
        raise ValueError(
            f"emit feature dim {len(values)} != {EMIT_FEATURE_DIM}"
        )
    return values


def _insert_features(
    target: IdentityTarget,
    *,
    emitted_so_far: int,
    remaining_groups: int,
) -> list[float]:
    span = target.score_span or (-1, -1)
    return [
        float(target.pitch) / 127.0,
        float(target.relationship == "extra"),
        float(target.relationship == "copy"),
        float(target.relationship == "match"),
        float(target.relationship == "substitute"),
        float(target.copy_pass > 0),
        float(span[0] >= 0),
        float(span[1] - span[0]) / 8.0 if span[0] >= 0 else 0.0,
        float(emitted_so_far) / 64.0,
        float(remaining_groups) / 64.0,
        float(target.rendered_index) / 64.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


class DropEmitScorer(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.drop_net = nn.Sequential(
            nn.Linear(GROUP_FEATURE_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.emit_net = nn.Sequential(
            nn.Linear(EMIT_FEATURE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.insert_net = nn.Sequential(
            nn.Linear(INSERT_FEATURE_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.transitions = nn.Parameter(torch.zeros(ACTION_COUNT, ACTION_COUNT))
        self.initial = nn.Parameter(torch.zeros(ACTION_COUNT))
        self.emit_bias = nn.Parameter(torch.zeros(()))
        self.insert_bias = nn.Parameter(torch.tensor(-1.0))

    def drop_score(self, features: Tensor) -> Tensor:
        return self.drop_net(features.float()).squeeze(-1)

    def emit_score(self, features: Tensor) -> Tensor:
        return self.emit_net(features.float()).squeeze(-1) + self.emit_bias

    def insert_score(self, features: Tensor) -> Tensor:
        return self.insert_net(features.float()).squeeze(-1) + self.insert_bias


def _logsumexp(values: Sequence[Tensor], *, like: Tensor) -> Tensor:
    if not values:
        return like.new_tensor(NEGATIVE_INFINITY)
    if len(values) == 1:
        return values[0]
    return torch.logsumexp(torch.stack(list(values)), dim=0)


def _crf_path_bonus(
    emitted: Sequence[IdentityCandidate],
    targets: Sequence[IdentityTarget],
    *,
    frozen_crf_score: float | None,
) -> float:
    """Optional frozen-CRF coupling term (path-level, detached)."""

    if frozen_crf_score is not None:
        return float(frozen_crf_score)
    # Cheap surrogate when no CRF score is supplied: reward exact count match.
    return -abs(len(emitted) - len(targets)) * 0.25


def gold_path_log_prob(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    frozen_crf_score: float | None = None,
) -> Tensor:
    """Deterministic gold DROP/EMIT/INSERT path score (not marginalized)."""

    from .drop_emit_dp_fast_v1 import fast_gold_path_log_prob

    return fast_gold_path_log_prob(
        model, lattice, frozen_crf_score=frozen_crf_score
    )
    coverage = prove_gold_path_coverage(lattice)
    if not coverage["passed"]:
        raise ValueError(f"Incomplete gold path coverage: {coverage}")
    device = model.initial.device
    dtype = model.initial.dtype
    score = model.initial.new_zeros(())
    previous_action: int | None = None
    group_cursor = 0
    for step in lattice.gold_steps:
        # DROP every group strictly before the next acoustic emit.
        if step.group_index is not None:
            while group_cursor < step.group_index:
                group = lattice.groups[group_cursor]
                features = torch.tensor(
                    _group_features(group), dtype=dtype, device=device
                )
                action_score = model.drop_score(features)
                if previous_action is None:
                    action_score = action_score + model.initial[DROP]
                else:
                    action_score = (
                        action_score + model.transitions[previous_action, DROP]
                    )
                score = score + action_score
                previous_action = DROP
                group_cursor += 1
            group = lattice.groups[step.group_index]
            assert step.candidate_index is not None
            candidate = lattice.pool_candidates[step.candidate_index]
            features = torch.tensor(
                _emit_features(
                    group,
                    candidate,
                    emitted_so_far=step.rendered_index,
                    remaining_groups=len(lattice.groups) - step.group_index - 1,
                    target_hint=step.target,
                ),
                dtype=dtype,
                device=device,
            )
            action_score = model.emit_score(features)
            if previous_action is None:
                action_score = action_score + model.initial[EMIT]
            else:
                action_score = action_score + model.transitions[previous_action, EMIT]
            score = score + action_score
            previous_action = EMIT
            group_cursor = step.group_index + 1
            continue
        features = torch.tensor(
            _insert_features(
                step.target,
                emitted_so_far=step.rendered_index,
                remaining_groups=len(lattice.groups) - group_cursor,
            ),
            dtype=dtype,
            device=device,
        )
        action_score = model.insert_score(features)
        if previous_action is None:
            action_score = action_score + model.initial[INSERT_EXTRA]
        else:
            action_score = (
                action_score + model.transitions[previous_action, INSERT_EXTRA]
            )
        score = score + action_score
        previous_action = INSERT_EXTRA
    while group_cursor < len(lattice.groups):
        group = lattice.groups[group_cursor]
        features = torch.tensor(_group_features(group), dtype=dtype, device=device)
        action_score = model.drop_score(features)
        if previous_action is None:
            action_score = action_score + model.initial[DROP]
        else:
            action_score = action_score + model.transitions[previous_action, DROP]
        score = score + action_score
        previous_action = DROP
        group_cursor += 1
    bonus = _crf_path_bonus(
        gold_emit_candidates(lattice),
        lattice.targets,
        frozen_crf_score=frozen_crf_score,
    )
    return score + score.new_tensor(bonus)


def _transition(
    model: DropEmitScorer,
    previous_action: int | None,
    action: int,
    emission: Tensor,
) -> Tensor:
    if previous_action is None:
        return emission + model.initial[action]
    return emission + model.transitions[previous_action, action]


def drop_emit_log_partition(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    gold_only: bool = False,
    max_inserts_ahead: int = 6,
    frozen_crf_score: float | None = None,
) -> Tensor:
    """Forward algorithm over (group_cursor, emitted_count, previous_action)."""

    from .drop_emit_dp_fast_v1 import fast_drop_emit_log_partition

    return fast_drop_emit_log_partition(
        model,
        lattice,
        gold_only=gold_only,
        max_inserts_ahead=max_inserts_ahead,
        frozen_crf_score=frozen_crf_score,
    )
    if gold_only:
        return gold_path_log_prob(
            model, lattice, frozen_crf_score=frozen_crf_score
        )

    device = model.initial.device
    dtype = model.initial.dtype
    groups = lattice.groups
    targets = lattice.targets
    n_groups = len(groups)
    n_targets = len(targets)
    start = model.initial.new_zeros(())
    # mass[(g, e, prev_action_or_-1)]
    mass: dict[tuple[int, int, int], Tensor] = {(0, 0, -1): start}

    def absorb(
        table: dict[tuple[int, int, int], Tensor],
        key: tuple[int, int, int],
        value: Tensor,
    ) -> None:
        previous = table.get(key)
        table[key] = (
            value if previous is None else _logsumexp([previous, value], like=value)
        )

    # Layered DP: at each group boundary apply up to max_inserts_ahead inserts,
    # then consume the group via DROP or EMIT.
    for group_index in range(n_groups + 1):
        # Expand inserts at this boundary.
        for _ in range(max_inserts_ahead):
            expanded: dict[tuple[int, int, int], Tensor] = {}
            for (g_index, emitted, previous_action), value in mass.items():
                if g_index != group_index:
                    absorb(expanded, (g_index, emitted, previous_action), value)
                    continue
                absorb(expanded, (g_index, emitted, previous_action), value)
                if emitted >= n_targets:
                    continue
                remaining_needed = n_targets - emitted
                remaining_groups = n_groups - group_index
                if remaining_needed - remaining_groups > max_inserts_ahead:
                    continue
                target = targets[emitted]
                features = torch.tensor(
                    _insert_features(
                        target,
                        emitted_so_far=emitted,
                        remaining_groups=remaining_groups,
                    ),
                    dtype=dtype,
                    device=device,
                )
                action_score = _transition(
                    model,
                    None if previous_action < 0 else previous_action,
                    INSERT_EXTRA,
                    model.insert_score(features),
                )
                absorb(
                    expanded,
                    (group_index, emitted + 1, INSERT_EXTRA),
                    value + action_score,
                )
            mass = expanded
        if group_index == n_groups:
            break
        nxt: dict[tuple[int, int, int], Tensor] = {}
        group = groups[group_index]
        drop_features = torch.tensor(
            _group_features(group), dtype=dtype, device=device
        )
        drop_emission = model.drop_score(drop_features)
        for (g_index, emitted, previous_action), value in mass.items():
            if g_index != group_index:
                continue
            prev = None if previous_action < 0 else previous_action
            absorb(
                nxt,
                (group_index + 1, emitted, DROP),
                value + _transition(model, prev, DROP, drop_emission),
            )
            if emitted < n_targets:
                target_hint = targets[emitted]
                emit_values = []
                for candidate in group.candidates:
                    features = torch.tensor(
                        _emit_features(
                            group,
                            candidate,
                            emitted_so_far=emitted,
                            remaining_groups=n_groups - group_index - 1,
                            target_hint=target_hint,
                        ),
                        dtype=dtype,
                        device=device,
                    )
                    emit_values.append(model.emit_score(features))
                emit_emission = _logsumexp(emit_values, like=value)
                absorb(
                    nxt,
                    (group_index + 1, emitted + 1, EMIT),
                    value + _transition(model, prev, EMIT, emit_emission),
                )
        mass = nxt

    finals = [
        value
        for (group_index, emitted, _action), value in mass.items()
        if group_index == n_groups and emitted == n_targets
    ]
    partition = _logsumexp(finals, like=start)
    bonus = _crf_path_bonus(
        gold_emit_candidates(lattice),
        lattice.targets,
        frozen_crf_score=frozen_crf_score,
    )
    return partition + start.new_tensor(bonus)


def drop_emit_nll(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    normalize: bool = True,
    frozen_crf_score: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if not prove_gold_path_coverage(lattice)["passed"]:
        raise ValueError("Refusing to train on incomplete gold-path coverage")
    log_partition = drop_emit_log_partition(
        model, lattice, gold_only=False, frozen_crf_score=frozen_crf_score
    )
    gold_partition = drop_emit_log_partition(
        model, lattice, gold_only=True, frozen_crf_score=frozen_crf_score
    )
    loss = log_partition - gold_partition
    if normalize:
        loss = loss / max(len(lattice.groups) + len(lattice.targets), 1)
    return loss, {
        "loss": loss.detach(),
        "log_partition": log_partition.detach(),
        "gold_log_partition": gold_partition.detach(),
    }


@dataclass(frozen=True)
class DecodedEmit:
    action: str
    group_index: int | None
    candidate_index: int | None
    target_hint: IdentityTarget | None
    candidate: IdentityCandidate


def _candidate_identity(candidate: Mapping[str, Any]) -> IdentityCandidate:
    return IdentityCandidate(
        pitch=int(candidate["pitch"]),
        start=float(candidate["start"]),
        end=float(candidate["end"]),
        confidence=float(candidate["confidence"]),
        alternatives=tuple(
            int(value) for value in candidate.get("alternatives") or ()
        ),
        alternative_confidences=tuple(
            float(value)
            for value in candidate.get("alternative_confidences") or ()
        ),
    )


def decode_drop_emit(
    model: DropEmitScorer,
    lattice: DropEmitLattice,
    *,
    expected_emissions: int | None = None,
    max_inserts_ahead: int = 6,
    max_emissions: int | None = None,
) -> tuple[tuple[DecodedEmit, ...], dict[str, Any]]:
    """Viterbi DROP/EMIT/INSERT decoding with emitted-count state.

    ``expected_emissions`` may come from a score/template prior. When omitted,
    the decoder maximizes path score over reachable emission counts after every
    acoustic group is consumed. Gold targets are never required at decode time.
    """

    from .drop_emit_dp_fast_v1 import fast_decode_drop_emit

    return fast_decode_drop_emit(
        model,
        lattice,
        expected_emissions=expected_emissions,
        max_inserts_ahead=max_inserts_ahead,
        max_emissions=max_emissions,
    )
    model.eval()
    device = model.initial.device
    dtype = model.initial.dtype
    groups = lattice.groups
    n_groups = len(groups)
    emit_cap = int(
        max_emissions
        if max_emissions is not None
        else max(
            expected_emissions or 0,
            n_groups + max_inserts_ahead,
            len(lattice.targets) if lattice.targets else 0,
        )
    )
    emit_cap = max(emit_cap, 1)
    beam: dict[tuple[int, int, int], tuple[float, list[DecodedEmit]]] = {
        (0, 0, -1): (0.0, [])
    }

    def push(
        table: dict[tuple[int, int, int], tuple[float, list[DecodedEmit]]],
        key: tuple[int, int, int],
        score: float,
        path: list[DecodedEmit],
    ) -> None:
        previous = table.get(key)
        if previous is None or score > previous[0]:
            table[key] = (score, path)

    for group_index in range(n_groups + 1):
        for _ in range(max_inserts_ahead):
            expanded: dict[tuple[int, int, int], tuple[float, list[DecodedEmit]]] = {}
            for (g_index, emitted, previous_action), (mass, path) in beam.items():
                if g_index != group_index:
                    push(expanded, (g_index, emitted, previous_action), mass, path)
                    continue
                push(expanded, (g_index, emitted, previous_action), mass, path)
                if emitted >= emit_cap:
                    continue
                hint = (
                    lattice.targets[emitted]
                    if emitted < len(lattice.targets)
                    else IdentityTarget("extra", None, 0, emitted, 60)
                )
                # At decode time avoid gold pitch leakage when targets absent or
                # when expected length differs; use a neutral insert prior.
                insert_target = IdentityTarget(
                    relationship="extra",
                    score_span=None,
                    copy_pass=0,
                    rendered_index=emitted,
                    pitch=hint.pitch if expected_emissions is not None else 60,
                )
                features = torch.tensor(
                    _insert_features(
                        insert_target,
                        emitted_so_far=emitted,
                        remaining_groups=n_groups - group_index,
                    ),
                    dtype=dtype,
                    device=device,
                )
                action_score = float(model.insert_score(features).detach().cpu())
                if previous_action < 0:
                    action_score += float(model.initial[INSERT_EXTRA].detach().cpu())
                else:
                    action_score += float(
                        model.transitions[previous_action, INSERT_EXTRA]
                        .detach()
                        .cpu()
                    )
                start_t = float(emitted) * 0.01
                push(
                    expanded,
                    (group_index, emitted + 1, INSERT_EXTRA),
                    mass + action_score,
                    path
                    + [
                        DecodedEmit(
                            "insert_extra",
                            None,
                            None,
                            insert_target,
                            IdentityCandidate(
                                pitch=insert_target.pitch,
                                start=start_t,
                                end=start_t + 0.05,
                                confidence=0.05,
                            ),
                        )
                    ],
                )
            beam = expanded
        if group_index == n_groups:
            break
        nxt: dict[tuple[int, int, int], tuple[float, list[DecodedEmit]]] = {}
        group = groups[group_index]
        drop_features = torch.tensor(
            _group_features(group), dtype=dtype, device=device
        )
        drop_emission = float(model.drop_score(drop_features).detach().cpu())
        for (g_index, emitted, previous_action), (mass, path) in beam.items():
            if g_index != group_index:
                continue
            action_score = drop_emission
            if previous_action < 0:
                action_score += float(model.initial[DROP].detach().cpu())
            else:
                action_score += float(
                    model.transitions[previous_action, DROP].detach().cpu()
                )
            push(
                nxt,
                (group_index + 1, emitted, DROP),
                mass + action_score,
                path
                + [
                    DecodedEmit(
                        "drop",
                        group_index,
                        None,
                        None,
                        IdentityCandidate(0, 0.0, 0.0, 0.0),
                    )
                ],
            )
            if emitted >= emit_cap:
                continue
            best_score = -inf
            best_candidate_index = group.candidate_indices[0]
            best_candidate = group.candidates[0]
            for local_index, candidate in enumerate(group.candidates):
                features = torch.tensor(
                    _emit_features(
                        group,
                        candidate,
                        emitted_so_far=emitted,
                        remaining_groups=n_groups - group_index - 1,
                        target_hint=None,
                    ),
                    dtype=dtype,
                    device=device,
                )
                score = float(model.emit_score(features).detach().cpu())
                if score > best_score:
                    best_score = score
                    best_candidate_index = group.candidate_indices[local_index]
                    best_candidate = candidate
            action_score = best_score
            if previous_action < 0:
                action_score += float(model.initial[EMIT].detach().cpu())
            else:
                action_score += float(
                    model.transitions[previous_action, EMIT].detach().cpu()
                )
            push(
                nxt,
                (group_index + 1, emitted + 1, EMIT),
                mass + action_score,
                path
                + [
                    DecodedEmit(
                        "emit",
                        group_index,
                        best_candidate_index,
                        None,
                        _candidate_identity(best_candidate),
                    )
                ],
            )
        beam = nxt

    if expected_emissions is not None:
        ranked = [
            (score, path)
            for (group_index, emitted, _action), (score, path) in beam.items()
            if group_index == n_groups and emitted == int(expected_emissions)
        ]
    else:
        ranked = [
            (score, path)
            for (group_index, emitted, _action), (score, path) in beam.items()
            if group_index == n_groups
        ]
    if not ranked:
        return (), {"score": -inf, "fallback": True, "actions": {}}
    score, path = max(ranked, key=lambda value: value[0])
    emits = tuple(step for step in path if step.action != "drop")
    return emits, {
        "score": score,
        "fallback": False,
        "actions": {
            name: sum(step.action == name for step in path)
            for name in ACTION_NAMES
        },
        "emitted": len(emits),
        "expected_emissions": expected_emissions,
    }


def emits_to_identity_candidates(
    emits: Sequence[DecodedEmit],
) -> tuple[IdentityCandidate, ...]:
    return tuple(step.candidate for step in emits)


def emits_to_joint_events_from_targets(
    emits: Sequence[DecodedEmit],
) -> tuple[JointEvent, ...]:
    """Assign exclusive rendered identities from decode target hints.

    Used when the outer lattice has already committed to emitted-count-aligned
    identities (gold-teacher or insert hints). Frozen CRF can replace this.
    """

    events = []
    for index, step in enumerate(emits):
        hint = step.target_hint
        candidate = step.candidate
        if hint is None:
            events.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=None,
                    relationship="extra",
                    copy_pass=0,
                    origin_relationship="extra",
                    rendered_index=index,
                    confidence=candidate.confidence,
                )
            )
            continue
        events.append(
            JointEvent(
                pitch=candidate.pitch,
                start=candidate.start,
                end=candidate.end,
                score_span=hint.score_span,
                relationship=hint.relationship,
                copy_pass=hint.copy_pass,
                origin_relationship=hint.relationship,
                rendered_index=index,
                confidence=candidate.confidence,
            )
        )
    return tuple(events)


def teacher_decode(lattice: DropEmitLattice) -> tuple[DecodedEmit, ...]:
    """Oracle DROP/EMIT path used for coverage proofs and upper-bound checks."""

    coverage = prove_gold_path_coverage(lattice)
    if not coverage["passed"]:
        raise ValueError(f"Teacher decode requires coverage: {coverage}")
    output: list[DecodedEmit] = []
    for step in lattice.gold_steps:
        if step.group_index is None or step.candidate_index is None:
            start = float(len(output)) * 0.01
            output.append(
                DecodedEmit(
                    "insert_extra",
                    None,
                    None,
                    step.target,
                    IdentityCandidate(
                        pitch=step.target.pitch,
                        start=start,
                        end=start + 0.05,
                        confidence=0.05,
                    ),
                )
            )
            continue
        candidate = lattice.pool_candidates[step.candidate_index]
        output.append(
            DecodedEmit(
                "emit",
                step.group_index,
                step.candidate_index,
                step.target,
                IdentityCandidate(
                    pitch=int(candidate["pitch"]),
                    start=float(candidate["start"]),
                    end=float(candidate["end"]),
                    confidence=float(candidate["confidence"]),
                    alternatives=tuple(
                        int(value) for value in candidate.get("alternatives") or ()
                    ),
                    alternative_confidences=tuple(
                        float(value)
                        for value in candidate.get("alternative_confidences") or ()
                    ),
                ),
            )
        )
    return tuple(output)
