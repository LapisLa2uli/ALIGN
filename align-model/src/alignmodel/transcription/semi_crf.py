"""Sparse monophonic interval Semi-CRF utilities.

The graph contains only proposed note intervals.  Silence is represented by a
one-frame skip edge, so the dynamic programs never allocate a ``T x T`` tensor.
Adjacent intervals (including intervals with the same pitch) remain distinct
edges and are therefore valid repeated notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class IntervalCandidates:
    """A sparse set of half-open ``[start, end)`` note intervals."""

    starts: Tensor
    ends: Tensor
    pitches: Tensor
    scores: Tensor
    num_frames: int

    def __post_init__(self) -> None:
        sizes = {int(self.starts.numel()), int(self.ends.numel()),
                 int(self.pitches.numel()), int(self.scores.numel())}
        if len(sizes) != 1:
            raise ValueError("candidate tensors must have the same length")
        if self.starts.ndim != 1 or self.ends.ndim != 1 or self.scores.ndim != 1:
            raise ValueError("candidate tensors must be one-dimensional")
        if self.num_frames < 0:
            raise ValueError("num_frames must be non-negative")
        if self.starts.numel():
            if bool((self.starts < 0).any()) or bool((self.ends > self.num_frames).any()):
                raise ValueError("candidate boundary outside clip")
            if bool((self.ends <= self.starts).any()):
                raise ValueError("candidate intervals must have positive duration")

    def __len__(self) -> int:
        return int(self.scores.numel())

    @classmethod
    def empty(
        cls,
        num_frames: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "IntervalCandidates":
        index = torch.empty(0, dtype=torch.long, device=device)
        return cls(index, index.clone(), index.clone(), torch.empty(
            0, dtype=dtype, device=device
        ), num_frames)


def _ending_at(candidates: IntervalCandidates) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(candidates.num_frames + 1)]
    for index, end in enumerate(candidates.ends.detach().cpu().tolist()):
        groups[int(end)].append(index)
    return groups


def semi_crf_log_partition(candidates: IntervalCandidates) -> Tensor:
    """Log-partition over all non-overlapping subsets of sparse intervals."""

    groups = _ending_at(candidates)
    zero = candidates.scores.sum() * 0.0
    alpha = [zero]
    for end in range(1, candidates.num_frames + 1):
        choices = [alpha[end - 1]]
        for index in groups[end]:
            start = int(candidates.starts[index].item())
            choices.append(alpha[start] + candidates.scores[index])
        alpha.append(torch.logsumexp(torch.stack(choices), dim=0))
    return alpha[-1]


def _validate_gold(
    candidates: IntervalCandidates, gold_indices: Tensor | Sequence[int]
) -> Tensor:
    indices = torch.as_tensor(
        gold_indices, dtype=torch.long, device=candidates.scores.device
    ).reshape(-1)
    if indices.numel() == 0:
        return indices
    if bool((indices < 0).any()) or bool((indices >= len(candidates)).any()):
        raise ValueError("gold candidate index out of range")
    order = torch.argsort(candidates.starts[indices])
    ordered = indices[order]
    if ordered.numel() > 1:
        if bool(
            (
                candidates.ends[ordered[:-1]]
                > candidates.starts[ordered[1:]]
            ).any()
        ):
            raise ValueError("gold intervals overlap")
    return ordered


def find_gold_indices(
    candidates: IntervalCandidates,
    intervals: Iterable[tuple[int, int, int]],
) -> Tensor:
    """Resolve exact ``(start, end, pitch-index)`` gold edges in the graph."""

    lookup = {
        (
            int(candidates.starts[i]),
            int(candidates.ends[i]),
            int(candidates.pitches[i]),
        ): i
        for i in range(len(candidates))
    }
    result = []
    for interval in intervals:
        key = tuple(int(value) for value in interval)
        if key not in lookup:
            raise ValueError(f"gold interval {key} is absent from candidate graph")
        result.append(lookup[key])
    return torch.tensor(result, dtype=torch.long, device=candidates.scores.device)


def semi_crf_nll(
    candidates: IntervalCandidates,
    gold_indices: Tensor | Sequence[int] | None = None,
    *,
    gold_intervals: Iterable[tuple[int, int, int]] | None = None,
    normalize: bool = False,
) -> Tensor:
    """Negative log likelihood of one non-overlapping gold interval set."""

    if (gold_indices is None) == (gold_intervals is None):
        raise ValueError("provide exactly one of gold_indices or gold_intervals")
    if gold_intervals is not None:
        gold_indices = find_gold_indices(candidates, gold_intervals)
    assert gold_indices is not None
    indices = _validate_gold(candidates, gold_indices)
    gold_score = (
        candidates.scores[indices].sum()
        if indices.numel()
        else candidates.scores.new_zeros(())
    )
    loss = semi_crf_log_partition(candidates) - gold_score
    return loss / max(candidates.num_frames, 1) if normalize else loss


def weighted_interval_decode(candidates: IntervalCandidates) -> list[int]:
    """Maximum-weight non-overlapping interval set, returned in time order."""

    groups = _ending_at(candidates)
    zero = candidates.scores.new_zeros(())
    best = [zero]
    back: list[tuple[int, int | None]] = [(0, None)]
    for end in range(1, candidates.num_frames + 1):
        value = best[end - 1]
        pointer = (end - 1, None)
        for index in groups[end]:
            start = int(candidates.starts[index].item())
            proposed = best[start] + candidates.scores[index]
            if bool(proposed > value):
                value = proposed
                pointer = (start, index)
        best.append(value)
        back.append(pointer)

    chosen: list[int] = []
    cursor = candidates.num_frames
    while cursor:
        previous, index = back[cursor]
        if index is not None:
            chosen.append(index)
        cursor = previous
    chosen.reverse()
    return chosen


viterbi_decode = weighted_interval_decode
log_partition = semi_crf_log_partition


def _local_peak_indices(probability: Tensor, threshold: float) -> list[int]:
    if probability.numel() == 0:
        return []
    left = torch.cat([probability.new_full((1,), -1.0), probability[:-1]])
    right = torch.cat([probability[1:], probability.new_full((1,), -1.0)])
    mask = (probability >= threshold) & (probability >= left) & (probability >= right)
    return torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()


def _cap_boundaries(indices: list[int], values: Tensor, limit: int) -> list[int]:
    unique = sorted(set(int(index) for index in indices))
    if len(unique) <= limit:
        return unique
    ranked = sorted(unique, key=lambda i: float(values[i].detach()), reverse=True)
    return sorted(ranked[:limit])


def candidate_pruned_intervals(
    voiced_logits: Tensor,
    onset_logits: Tensor,
    offset_logits: Tensor,
    pitch_logits: Tensor,
    *,
    min_duration: int = 3,
    max_duration: int = 400,
    pitches_per_interval: int = 3,
    boundary_threshold: float = 0.25,
    max_boundaries: int = 192,
    max_ends_per_start: int = 12,
    required_intervals: Iterable[tuple[int, int, int]] = (),
) -> IntervalCandidates:
    """Build a bounded sparse graph directly from one clip's frame heads.

    Boundary proposals are local peaks plus voiced/pitch transitions.  Only a
    capped number of offsets are paired with each onset, giving memory
    ``O(T + candidates)`` rather than ``O(T^2)``.
    """

    required_intervals = tuple(
        (int(start), int(end), int(pitch))
        for start, end, pitch in required_intervals
    )
    if pitch_logits.ndim != 2:
        raise ValueError("pitch_logits must have shape [T, P]")
    frames, n_pitches = pitch_logits.shape
    if any(value.shape != (frames,) for value in (
        voiced_logits, onset_logits, offset_logits
    )):
        raise ValueError("frame heads must all have shape [T]")
    if min_duration < 1 or max_duration < min_duration:
        raise ValueError("invalid duration bounds")
    if frames == 0:
        return IntervalCandidates.empty(
            0, device=pitch_logits.device, dtype=pitch_logits.dtype
        )
    if any(
        not (0 <= start < end <= frames and 0 <= pitch < n_pitches)
        for start, end, pitch in required_intervals
    ):
        raise ValueError("required interval outside frame or pitch range")

    voiced_probability = voiced_logits.sigmoid()
    onset_probability = onset_logits.sigmoid()
    offset_probability = offset_logits.sigmoid()
    frame_pitch = pitch_logits.argmax(dim=-1)
    voiced_start = torch.nonzero(
        (voiced_probability >= boundary_threshold)
        & torch.cat([
            torch.ones(1, dtype=torch.bool, device=voiced_logits.device),
            voiced_probability[:-1] < boundary_threshold,
        ]),
        as_tuple=False,
    ).flatten().detach().cpu().tolist()
    transitions = (
        torch.nonzero(
            frame_pitch[1:] != frame_pitch[:-1], as_tuple=False
        ).flatten().add(1).detach().cpu().tolist()
    )
    starts = _local_peak_indices(onset_probability, boundary_threshold)
    starts.extend(voiced_start)
    starts.extend(transitions)
    required_starts = {start for start, _, _ in required_intervals}
    starts.extend(required_starts)
    starts = sorted(set(
        _cap_boundaries(starts, onset_probability, max_boundaries)
    ) | required_starts)

    ends = _local_peak_indices(
        offset_probability, boundary_threshold
    )
    ends.extend(transitions)
    ends.append(frames)
    required_ends = {end for _, end, _ in required_intervals}
    ends.extend(required_ends)
    # Rank an exclusive end by the final frame's offset evidence.
    end_values = torch.cat([offset_probability, offset_probability[-1:]])
    ends = [end for end in ends if min_duration <= end <= frames]
    ends = sorted(set(
        _cap_boundaries(ends, end_values, max_boundaries)
    ) | required_ends)

    required = {
        (int(start), int(end), int(pitch))
        for start, end, pitch in required_intervals
        if 0 <= start < end <= frames and 0 <= pitch < n_pitches
    }
    records: dict[tuple[int, int, int], Tensor] = {}
    pitch_probability = pitch_logits.softmax(dim=-1)
    for start in starts:
        valid_ends = [
            end for end in ends
            if min_duration <= end - start <= max_duration
        ]
        if len(valid_ends) > max_ends_per_start:
            nearest = min(valid_ends)
            ranked = sorted(
                valid_ends,
                key=lambda end: float(end_values[end].detach()),
                reverse=True,
            )
            required_for_start = {
                end for req_start, end, _ in required if req_start == start
            }
            valid_ends = sorted(
                set([nearest, *ranked[:max_ends_per_start]])
                | required_for_start
            )
        for end in valid_ends:
            pitch_evidence = pitch_probability[start:end].mean(dim=0)
            top = torch.topk(
                pitch_evidence, k=min(pitches_per_interval, n_pitches)
            ).indices.detach().cpu().tolist()
            top.extend(
                pitch for req_start, req_end, pitch in required
                if req_start == start and req_end == end
            )
            boundary_score = 2.0 * onset_probability[start]
            boundary_score = boundary_score + (
                offset_probability[end]
                if end < frames
                else offset_probability[end - 1]
            )
            voice_score = voiced_probability[start:end].mean()
            for pitch in sorted(set(int(value) for value in top)):
                # Centered probability score keeps the frozen Basic Pitch
                # residual initialization decodable while remaining fully
                # differentiable for Semi-CRF likelihood training.
                score = (
                    boundary_score
                    + 1.5 * voice_score
                    + 1.5 * pitch_evidence[pitch]
                    - 1.5
                )
                records[(start, end, pitch)] = score

    if not records:
        return IntervalCandidates.empty(
            frames, device=pitch_logits.device, dtype=pitch_logits.dtype
        )
    keys = sorted(records)
    device = pitch_logits.device
    return IntervalCandidates(
        starts=torch.tensor([key[0] for key in keys], dtype=torch.long, device=device),
        ends=torch.tensor([key[1] for key in keys], dtype=torch.long, device=device),
        pitches=torch.tensor([key[2] for key in keys], dtype=torch.long, device=device),
        scores=torch.stack([records[key] for key in keys]),
        num_frames=frames,
    )


class SemiCRF:
    """Small façade for sparse partition, NLL, and Viterbi operations."""

    @staticmethod
    def log_partition(candidates: IntervalCandidates) -> Tensor:
        return semi_crf_log_partition(candidates)

    @staticmethod
    def nll(
        candidates: IntervalCandidates,
        gold_indices: Tensor | Sequence[int] | None = None,
        *,
        gold_intervals: Iterable[tuple[int, int, int]] | None = None,
        normalize: bool = False,
    ) -> Tensor:
        return semi_crf_nll(
            candidates,
            gold_indices,
            gold_intervals=gold_intervals,
            normalize=normalize,
        )

    @staticmethod
    def decode(candidates: IntervalCandidates) -> list[int]:
        return weighted_interval_decode(candidates)
