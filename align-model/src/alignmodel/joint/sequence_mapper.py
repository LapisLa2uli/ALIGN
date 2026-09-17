from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import numpy as np

from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate
from .mel_mapper import ContextualRepeatMapper, MapperState, edge_features


TYPE_NAMES = ("match", "copy", "substitute", "extra")
MAX_SPAN_LENGTH = 128


class FixedScoreSequenceMapper(nn.Module):
    """Offline event/score sequence mapper with dynamic score identities."""

    def __init__(self, score_events: int | None = None, hidden_dim: int = 32) -> None:
        super().__init__()
        self.score_events = int(score_events) if score_events is not None else None
        self.hidden_dim = int(hidden_dim)
        self.pitch = nn.Embedding(128, 16)
        self.score_position = nn.Linear(1, 16)
        self.input = nn.Linear(20, hidden_dim)
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.score_encoder = nn.GRU(
            32,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads=4,
            dropout=0.10,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim * 2)
        self.extra = nn.Linear(hidden_dim * 2, 1)
        self.operation = nn.Linear(hidden_dim * 2, len(TYPE_NAMES))
        self.span_length = nn.Linear(hidden_dim * 2, MAX_SPAN_LENGTH + 1)

    def forward(
        self,
        pitch: torch.Tensor,
        continuous: torch.Tensor,
        score_pitch: torch.Tensor,
        score_position: torch.Tensor,
        score_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.pitch(pitch.clamp(0, 127))
        encoded, _state = self.encoder(
            torch.nn.functional.gelu(
                self.input(torch.cat((embedded, continuous), dim=-1))
            )
        )
        score_embedded = torch.cat(
            (
                self.pitch(score_pitch.clamp(0, 127)),
                torch.nn.functional.gelu(
                    self.score_position(score_position[..., None])
                ),
            ),
            dim=-1,
        )
        query = self.query(encoded)
        key, _score_state = self.score_encoder(score_embedded)
        attended, _weights = self.cross_attention(
            query,
            key,
            key,
            key_padding_mask=(~score_mask if score_mask is not None else None),
            need_weights=False,
        )
        query = self.cross_norm(query + attended)
        logits = torch.einsum("bth,bsh->bts", query, key) / (
            query.shape[-1] ** 0.5
        )
        if score_mask is not None:
            logits = logits.masked_fill(~score_mask[:, None, :], -1e4)
        location = torch.cat((logits, self.extra(encoded)), dim=-1)
        return location, self.operation(encoded), self.span_length(encoded)


def sequence_tensors(
    events: Sequence[JointCandidate | JointEvent],
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    pitch = torch.tensor(
        [event.pitch for event in events], dtype=torch.long, device=device
    )
    rows = []
    total_duration = max((event.end for event in events), default=1.0)
    for index, event in enumerate(events):
        previous = events[index - 1] if index else None
        following = events[index + 1] if index + 1 < len(events) else None
        rows.append(
            (
                event.start / max(total_duration, 1e-3),
                min(event.end - event.start, 2.0),
                min(event.start - previous.start, 2.0) if previous else 0.0,
                ((event.pitch - previous.pitch) / 12.0) if previous else 0.0,
            )
        )
    continuous = torch.tensor(rows, dtype=torch.float32, device=device)
    return pitch, continuous


@torch.inference_mode()
def decode_sequence_mapper(
    model: FixedScoreSequenceMapper,
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    *,
    pitch_match_bias: float = 3.0,
    edge_model: ContextualRepeatMapper | None = None,
    edge_weight: float = 0.8,
) -> tuple[JointEvent, ...]:
    if model.score_events is not None and len(score) != model.score_events:
        raise ValueError("Sequence mapper score-event count mismatch")
    if not events:
        return ()
    device = next(model.parameters()).device
    pitch, continuous = sequence_tensors(events, device=device)
    score_pitch_input = torch.tensor(
        [[event.pitch for event in score]], dtype=torch.long, device=device
    )
    score_end = max((event.ql_end for event in score), default=1.0)
    score_position = torch.tensor(
        [[event.ql_start / max(score_end, 1e-6) for event in score]],
        dtype=torch.float32,
        device=device,
    )
    location, operation, span_length = model(
        pitch[None],
        continuous[None],
        score_pitch_input,
        score_position,
    )
    score_pitch = torch.tensor(
        [event.pitch for event in score], device=device
    )
    matching = pitch[:, None] == score_pitch[None, :]
    location[0, :, : len(score)] += matching * pitch_match_bias
    location_values = location[0].cpu()
    operation_values = operation[0].cpu()
    lengths = span_length[0].argmax(-1).cpu().tolist()
    beams = [(0.0, -1, False, -1, (), ())]
    for event_index, event in enumerate(events):
        top = torch.topk(
            location_values[event_index],
            min(36, location_values.shape[-1]),
        ).indices.tolist()
        expanded = []
        edge_rows = []
        for total, cursor, replay, resume, path, types in beams:
            options = set(top)
            options.update(
                range(max(0, cursor), min(len(score), max(cursor + 6, 6)))
            )
            options.add(len(score))
            for option_index in options:
                if option_index == len(score):
                    type_index = TYPE_NAMES.index("extra")
                    destination = (cursor, replay, resume)
                    transition_score = -2.0
                else:
                    if replay and option_index < resume:
                        type_index = TYPE_NAMES.index("copy")
                        destination = (option_index, True, resume)
                    elif cursor >= 0 and option_index < cursor:
                        type_index = TYPE_NAMES.index("copy")
                        destination = (option_index, True, cursor + 1)
                    else:
                        type_index = TYPE_NAMES.index(
                            "match"
                            if event.pitch == score[option_index].pitch
                            else "substitute"
                        )
                        destination = (option_index, False, -1)
                    delta = option_index - cursor if cursor >= 0 else option_index + 1
                    transition_score = (
                        3.0
                        if delta == 1
                        else -0.35 * (delta - 1)
                        if delta > 1
                        else -2.5
                    )
                edge = (
                    float(location_values[event_index, option_index])
                    + 0.8 * float(operation_values[event_index, type_index])
                    + transition_score
                )
                expanded.append(
                    (
                        total + edge,
                        *destination,
                        path + (option_index,),
                        types + (type_index,),
                    )
                )
                if edge_model is not None:
                    edge_rows.append(
                        edge_features(
                            events,
                            score,
                            event_index,
                            option_index if option_index < len(score) else -1,
                            MapperState(
                                cursor,
                                "replay" if replay else "normal",
                                resume,
                            ),
                        )
                    )
        if edge_model is not None and edge_rows:
            edge_device = next(edge_model.parameters()).device
            edge_scores = edge_model(
                torch.from_numpy(np.stack(edge_rows)).to(edge_device)
            ).cpu().tolist()
            expanded = [
                (
                    row[0] + edge_weight * float(edge_score),
                    *row[1:],
                )
                for row, edge_score in zip(expanded, edge_scores)
            ]
        best_by_state = {}
        for row in sorted(expanded, key=lambda value: value[0], reverse=True):
            key = row[1:4]
            if key not in best_by_state:
                best_by_state[key] = row
            if len(best_by_state) >= 96:
                break
        beams = list(best_by_state.values())
    _total, _cursor, _replay, _resume, locations, operations = max(
        beams, key=lambda value: value[0]
    )
    output = []
    cursor = -1
    active_copy_pass = 0
    passes_by_source: dict[int, int] = {}
    for event_index, (event, location_index, operation_index, length) in enumerate(
        zip(events, locations, operations, lengths)
    ):
        if location_index == len(score):
            relationship = "extra"
            span = None
        else:
            relationship = TYPE_NAMES[operation_index]
            if relationship == "extra":
                relationship = (
                    "match"
                    if event.pitch == score[location_index].pitch
                    else "substitute"
                )
            span = (
                location_index,
                min(len(score), location_index + max(1, int(length))),
            )
        if relationship == "copy":
            if location_index < cursor:
                active_copy_pass = passes_by_source.get(location_index, 0) + 1
                passes_by_source[location_index] = active_copy_pass
        else:
            active_copy_pass = 0
        if location_index < len(score):
            cursor = location_index
        output.append(
            JointEvent(
                pitch=event.pitch,
                start=event.start,
                end=event.end,
                score_span=span,
                relationship=relationship,
                copy_pass=active_copy_pass if relationship == "copy" else 0,
                rendered_index=event_index if relationship == "extra" else None,
                confidence=event.confidence,
            )
        )
    return tuple(output)


@torch.inference_mode()
def predict_operation_types(
    model: FixedScoreSequenceMapper,
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
) -> tuple[str, ...]:
    if not events:
        return ()
    device = next(model.parameters()).device
    pitch, continuous = sequence_tensors(events, device=device)
    score_pitch = torch.tensor(
        [[event.pitch for event in score]], dtype=torch.long, device=device
    )
    score_end = max((event.ql_end for event in score), default=1.0)
    score_position = torch.tensor(
        [[event.ql_start / max(score_end, 1e-6) for event in score]],
        dtype=torch.float32,
        device=device,
    )
    _location, operation, _span = model(
        pitch[None],
        continuous[None],
        score_pitch,
        score_position,
    )
    return tuple(TYPE_NAMES[index] for index in operation[0].argmax(-1).tolist())


@torch.inference_mode()
def predict_mapper_scores(
    model: FixedScoreSequenceMapper,
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    pitch, continuous = sequence_tensors(events, device=device)
    score_pitch = torch.tensor(
        [[event.pitch for event in score]], dtype=torch.long, device=device
    )
    score_end = max((event.ql_end for event in score), default=1.0)
    score_position = torch.tensor(
        [[event.ql_start / max(score_end, 1e-6) for event in score]],
        dtype=torch.float32,
        device=device,
    )
    location, operation, _span = model(
        pitch[None],
        continuous[None],
        score_pitch,
        score_position,
    )
    matching = pitch[:, None] == score_pitch[0][None, :]
    location[0, :, : len(score)] += matching * 3.0
    return (
        location[0].log_softmax(-1).cpu().numpy(),
        operation[0].softmax(-1).cpu().numpy(),
    )
