from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate


FEATURE_DIM = 48


@dataclass(frozen=True)
class MapperState:
    cursor: int = -1
    mode: str = "normal"
    resume: int = -1


class ContextualRepeatMapper(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def transition(state: MapperState, option: int) -> tuple[MapperState, int]:
    """Return destination and operation class.

    Classes: MATCH, SUBSTITUTE, EXTRA, DELETE, REPEAT_ENTER, REPLAY,
    CONTINUE. DELETE is represented by a forward skip and supervised through
    the transition features rather than as a candidate-consuming option.
    """

    if option < 0:
        return state, 2
    if state.mode == "normal" and state.cursor >= 0 and option < state.cursor:
        return MapperState(option, "replay", state.cursor + 1), 4
    if state.mode == "replay":
        if option >= state.resume:
            return MapperState(option), 6
        return MapperState(option, "replay", state.resume), 5
    operation = 3 if state.cursor >= 0 and option > state.cursor + 1 else 0
    return MapperState(option), operation


def _pitch(events: Sequence[JointCandidate | JointEvent], index: int) -> int | None:
    return events[index].pitch if 0 <= index < len(events) else None


def edge_features(
    events: Sequence[JointCandidate | JointEvent],
    score: Sequence[ScoreEvent],
    event_index: int,
    option: int,
    state: MapperState,
) -> np.ndarray:
    event = events[event_index]
    destination, operation = transition(state, option)
    if option < 0:
        score_pitch = event.pitch
        delta = 0
        score_position = -1.0
    else:
        score_pitch = score[option].pitch
        delta = option - state.cursor if state.cursor >= 0 else option + 1
        score_position = option / max(len(score) - 1, 1)
        if operation == 0 and event.pitch != score_pitch:
            operation = 1
    previous_interval = (
        event.pitch - events[event_index - 1].pitch if event_index else 0
    )
    score_interval = (
        score_pitch - score[state.cursor].pitch
        if option >= 0 and 0 <= state.cursor < len(score)
        else 0
    )
    context = []
    for offset in (*range(-12, 0), *range(1, 13)):
        observed = _pitch(events, event_index + offset)
        expected_index = option + offset
        expected = (
            score[expected_index].pitch
            if option >= 0 and 0 <= expected_index < len(score)
            else None
        )
        context.append(float(observed is not None and observed == expected))
    next_interval = (
        events[event_index + 1].pitch - event.pitch
        if event_index + 1 < len(events)
        else 0
    )
    score_next_interval = (
        score[option + 1].pitch - score_pitch
        if option >= 0 and option + 1 < len(score)
        else 0
    )
    confidence = float(getattr(event, "confidence", 1.0))
    duration = max(float(event.end - event.start), 1e-3)
    ioi = (
        float(event.start - events[event_index - 1].start)
        if event_index
        else duration
    )
    continuation = []
    for length in (8, 16):
        compared = matched = 0
        for offset in range(length):
            observed = _pitch(events, event_index + offset)
            expected_index = option + offset
            if (
                observed is None
                or option < 0
                or not 0 <= expected_index < len(score)
            ):
                continue
            compared += 1
            matched += int(observed == score[expected_index].pitch)
        continuation.append(matched / max(compared, 1))
    one_hot = [float(index == operation) for index in range(7)]
    values = [
        *one_hot,
        max(-2.0, min(2.0, (event.pitch - score_pitch) / 12.0)),
        float(event.pitch == score_pitch),
        max(-2.0, min(2.0, delta / 16.0)),
        float(delta == 1),
        float(delta < 0),
        event_index / max(len(events) - 1, 1),
        score_position,
        max(-2.0, min(2.0, (previous_interval - score_interval) / 12.0)),
        max(-2.0, min(2.0, (next_interval - score_next_interval) / 12.0)),
        *context,
        confidence,
        min(duration, 2.0),
        min(max(ioi, 0.0), 2.0),
        float(state.mode == "replay"),
        (
            max(-2.0, min(2.0, (option - state.resume) / 16.0))
            if state.mode == "replay" and option >= 0
            else 0.0
        ),
        float(state.mode == "replay" and option == state.resume),
        *continuation,
    ]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Mapper feature shape mismatch: {result.shape}")
    return result


def option_indices(
    events: Sequence[JointCandidate | JointEvent],
    score: Sequence[ScoreEvent],
    event_index: int,
    state: MapperState,
    *,
    gold: int | None = None,
    max_pitch_options: int = 36,
) -> tuple[int, ...]:
    event = events[event_index]
    exact = [index for index, value in enumerate(score) if value.pitch == event.pitch]
    ranked = sorted(
        exact,
        key=lambda option: (
            -sum(
                edge_features(events, score, event_index, option, state)[16:40]
            ),
            abs(option - (state.cursor + 1)),
        ),
    )[:max_pitch_options]
    local = range(
        max(0, state.cursor),
        min(len(score), max(state.cursor + 6, 6)),
    )
    options = {-1, *ranked, *local}
    if state.mode == "replay":
        options.update(
            range(max(0, state.resume - 2), min(len(score), state.resume + 4))
        )
    if gold is not None:
        options.add(gold)
    return tuple(sorted(options))


def _exact_run(
    events: Sequence[JointCandidate | JointEvent],
    score: Sequence[ScoreEvent],
    event_index: int,
    score_index: int,
    *,
    limit: int = 64,
) -> int:
    matched = 0
    while (
        matched < limit
        and event_index + matched < len(events)
        and score_index + matched < len(score)
        and events[event_index + matched].pitch == score[score_index + matched].pitch
    ):
        matched += 1
    return matched


def decode_anchor_mapper(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
) -> tuple[JointEvent, ...]:
    """Long-context deterministic mapper used as the segmental anchor path."""

    cursor = -1
    replay_resume = -1
    replay = False
    output = []
    for event_index, event in enumerate(events):
        expected = cursor + 1
        exact = [
            index for index, value in enumerate(score) if value.pitch == event.pitch
        ]
        runs = {
            option: _exact_run(events, score, event_index, option)
            for option in exact
        }
        best = max(
            exact,
            key=lambda option: (
                runs[option],
                option == expected,
                -abs(option - expected),
            ),
            default=-1,
        )
        expected_run = runs.get(expected, 0)
        best_run = runs.get(best, 0)
        option = -1
        relationship = "extra"
        if 0 <= expected < len(score):
            next_pitch = (
                events[event_index + 1].pitch
                if event_index + 1 < len(events)
                else None
            )
            if event.pitch == score[expected].pitch:
                option = (
                    best
                    if best_run >= max(3, expected_run + 2)
                    else expected
                )
            elif next_pitch == score[expected].pitch:
                option = -1
            elif (
                expected + 1 < len(score)
                and next_pitch == score[expected + 1].pitch
            ):
                option = expected
                relationship = "substitute"
            elif best_run >= 2:
                option = best
            else:
                local = [
                    index
                    for index in range(expected, min(len(score), expected + 5))
                    if score[index].pitch == event.pitch
                ]
                if local:
                    option = max(
                        local,
                        key=lambda value: _exact_run(
                            events, score, event_index, value
                        ),
                    )
                else:
                    option = expected
                    relationship = "substitute"
        elif best >= 0:
            option = best
        if option >= 0:
            if relationship != "substitute":
                if option < cursor:
                    replay_resume = cursor + 1
                    replay = True
                    relationship = "copy"
                elif replay and option < replay_resume:
                    relationship = "copy"
                else:
                    replay = False
                    relationship = (
                        "match"
                        if event.pitch == score[option].pitch
                        else "substitute"
                    )
            cursor = option
        output.append(
            JointEvent(
                pitch=event.pitch,
                start=event.start,
                end=event.end,
                score_span=(option, option + 1) if option >= 0 else None,
                relationship=relationship,
                copy_pass=1 if relationship == "copy" else 0,
                rendered_index=event_index if relationship == "extra" else None,
                confidence=event.confidence,
            )
        )
    return tuple(output)


def decode_segmental_mapper(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    *,
    beam_width: int = 128,
    continuity_reward: float = 5.0,
    repeat_penalty: float = 4.0,
    extra_penalty: float = 5.0,
) -> tuple[JointEvent, ...]:
    """Beam pair-HMM with explicit repeat and continuation state."""

    beams = [(0.0, MapperState(), (), ())]
    for event_index, event in enumerate(events):
        expanded = []
        for total, state, path, operations in beams:
            options = option_indices(
                events,
                score,
                event_index,
                state,
                max_pitch_options=48,
            )
            for option in options:
                destination, operation = transition(state, option)
                if option < 0:
                    edge = -extra_penalty
                else:
                    pitch_match = event.pitch == score[option].pitch
                    edge = 4.0 if pitch_match else -4.0
                    context = edge_features(
                        events, score, event_index, option, state
                    )[16:40]
                    edge += 0.65 * float(np.sum(context))
                    delta = option - state.cursor if state.cursor >= 0 else option + 1
                    if delta == 1:
                        edge += continuity_reward
                    elif delta > 1:
                        edge -= min(8.0, 0.8 * (delta - 1))
                    else:
                        edge -= repeat_penalty
                        edge += 0.35 * _exact_run(
                            events, score, event_index, option
                        )
                expanded.append(
                    (
                        total + edge,
                        destination,
                        path + (option,),
                        operations + (operation,),
                    )
                )
        best_by_state = {}
        for row in sorted(expanded, key=lambda value: value[0], reverse=True):
            key = (row[1].cursor, row[1].mode, row[1].resume)
            if key not in best_by_state:
                best_by_state[key] = row
            if len(best_by_state) >= beam_width:
                break
        beams = list(best_by_state.values())
    _total, _state, path, operations = max(beams, key=lambda value: value[0])
    output = []
    for event_index, (event, option, operation) in enumerate(
        zip(events, path, operations)
    ):
        relationship = (
            "extra"
            if option < 0
            else "copy"
            if operation in {4, 5}
            else "substitute"
            if event.pitch != score[option].pitch
            else "match"
        )
        output.append(
            JointEvent(
                pitch=event.pitch,
                start=event.start,
                end=event.end,
                score_span=(option, option + 1) if option >= 0 else None,
                relationship=relationship,
                copy_pass=1 if relationship == "copy" else 0,
                rendered_index=event_index if relationship == "extra" else None,
                confidence=event.confidence,
            )
        )
    return tuple(output)


@torch.inference_mode()
def decode_mapper(
    model: ContextualRepeatMapper,
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    *,
    beam_width: int = 96,
) -> tuple[JointEvent, ...]:
    if not events:
        return ()
    device = next(model.parameters()).device
    beams: list[tuple[float, MapperState, tuple[int, ...], tuple[int, ...]]] = [
        (0.0, MapperState(), (), ())
    ]
    for event_index in range(len(events)):
        features = []
        metadata = []
        for total, state, path, operations in beams:
            for option in option_indices(events, score, event_index, state):
                features.append(
                    edge_features(events, score, event_index, option, state)
                )
                destination, operation = transition(state, option)
                if (
                    option >= 0
                    and operation == 0
                    and events[event_index].pitch != score[option].pitch
                ):
                    operation = 1
                metadata.append(
                    (total, destination, path + (option,), operations + (operation,))
                )
        scores = model(torch.from_numpy(np.stack(features)).to(device)).cpu().tolist()
        candidates = [
            (total + float(score_value), state, path, operations)
            for score_value, (total, state, path, operations) in zip(scores, metadata)
        ]
        best_by_state = {}
        for row in sorted(candidates, key=lambda value: value[0], reverse=True):
            key = (row[1].cursor, row[1].mode, row[1].resume)
            if key not in best_by_state:
                best_by_state[key] = row
            if len(best_by_state) >= beam_width:
                break
        beams = list(best_by_state.values())
    _total, _state, path, operations = max(beams, key=lambda value: value[0])
    output = []
    for event_index, (event, option, operation) in enumerate(
        zip(events, path, operations)
    ):
        relationship = (
            "extra"
            if option < 0
            else "copy"
            if operation in {4, 5}
            else "substitute"
            if event.pitch != score[option].pitch
            else "match"
        )
        output.append(
            JointEvent(
                pitch=event.pitch,
                start=event.start,
                end=event.end,
                score_span=(option, option + 1) if option >= 0 else None,
                relationship=relationship,
                copy_pass=1 if relationship == "copy" else 0,
                rendered_index=event_index if relationship == "extra" else None,
                confidence=event.confidence,
            )
        )
    return tuple(output)
