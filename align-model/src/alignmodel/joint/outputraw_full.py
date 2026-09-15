"""OutputRaw-only full joint pipeline architecture.

The frozen upstream AMT frontend proposes candidates.  Every module in this
file is repository-owned and trainable.  The scalar ``forward`` method is
compatible with :class:`SparseJointLattice`; auxiliary heads share the same
representations so staged unfreezing can end in one jointly fine-tuned model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .index import JointEvent, ScoreEvent
from .lattice import (
    FEATURE_DIM,
    JointCandidate,
    JointEdgeScorer,
    JointOperation,
    LatticePath,
    SparseJointLattice,
    StructuralState,
)


SCHEMA_VERSION = "align-outputraw-full-joint-v1"
MODEL_FAMILY = "joint-outputraw-full-v1"
LAYER2_CLASSES = ("match", "wrong_note", "extra_note", "missed_note")
STRUCTURE_CLASSES = ("ordinary", "repeat_enter", "replay", "resume")
INTONATION_MASKED = True

# Existing edge features are stable and serialized by the packed-data worker.
_OPERATION_COUNT = len(tuple(JointOperation))
_REPEAT_ENTER_INDEX = list(JointOperation).index(JointOperation.REPEAT_ENTER)
_REPLAY_INDEX = list(JointOperation).index(JointOperation.REPLAY)
_CONTINUE_INDEX = list(JointOperation).index(JointOperation.CONTINUE)
_CONFIDENCE_INDEX = _OPERATION_COUNT + 3
_DURATION_RATIO_INDEX = _OPERATION_COUNT + 4
_REPLAY_MODE_INDEX = _OPERATION_COUNT + 10
_CONTINUATION_INDEX = _OPERATION_COUNT + 11
_ACOUSTIC_START = FEATURE_DIM - 5


class StructureClass(IntEnum):
    ORDINARY = 0
    REPEAT_ENTER = 1
    REPLAY = 2
    RESUME = 3


@dataclass(frozen=True)
class FullPipelineModelConfig:
    edge_feature_dim: int = FEATURE_DIM
    hidden_dim: int = 128
    path_hidden_dim: int = 64
    path_component_dim: int = 32
    path_residual_scale: float = 0.10
    component_dim: int = 64
    dropout: float = 0.08
    residual_scale: float = 0.20
    structure_path_weight: float = 0.0
    layer2_classes: int = len(LAYER2_CLASSES)
    structure_classes: int = len(STRUCTURE_CLASSES)
    copy_count_classes: int = 3


@dataclass(frozen=True)
class FullPipelineLossConfig:
    path: float = 1.0
    confidence: float = 0.35
    boundary: float = 0.45
    split: float = 0.50
    emission: float = 0.45
    structure: float = 0.65
    repeat_margin: float = 0.40
    copy_count: float = 0.35
    layer2: float = 0.60
    layer3: float = 0.55
    duration: float = 0.20
    confidence_band_negative_weight: float = 2.0
    strong_rearticulation_weight: float = 2.5


@dataclass(frozen=True)
class FullPipelineAugmentConfig:
    difficult_timbre_probability: float = 0.35
    activation_scale_min: float = 0.55
    activation_scale_max: float = 0.90
    activation_noise_std: float = 0.02
    pitch_margin_drop_max: float = 0.12


@dataclass(frozen=True)
class FullPipelineTargets:
    """One target row per local candidate group.

    ``groups`` stores ``(edge_start, edge_end, gold_offset)``.  All remaining
    vectors have one value per group except ``clip_index`` and
    ``copy_count_target``, which identify clip-level pooling and targets.
    """

    groups: Sequence[tuple[int, int, int]]
    keep: torch.Tensor
    boundary: torch.Tensor
    split: torch.Tensor
    emission: torch.Tensor
    structure: torch.Tensor
    layer2: torch.Tensor
    rhythm: torch.Tensor
    duration_target: torch.Tensor
    duration_weight: torch.Tensor
    rearticulation_weight: torch.Tensor
    clip_index: torch.Tensor
    copy_count_target: torch.Tensor

    def validate(self, edge_count: int) -> None:
        count = len(self.groups)
        fields = {
            "keep": self.keep,
            "boundary": self.boundary,
            "split": self.split,
            "emission": self.emission,
            "structure": self.structure,
            "layer2": self.layer2,
            "rhythm": self.rhythm,
            "duration_target": self.duration_target,
            "duration_weight": self.duration_weight,
            "rearticulation_weight": self.rearticulation_weight,
            "clip_index": self.clip_index,
        }
        for name, value in fields.items():
            if value.ndim != 1 or len(value) != count:
                raise ValueError(
                    f"{name} must have one value per group ({count})"
                )
        previous_end = 0
        for start, end, gold_offset in self.groups:
            if start != previous_end or not start < end <= edge_count:
                raise ValueError("Local edge groups must be contiguous")
            if not 0 <= gold_offset < end - start:
                raise ValueError("Gold edge offset is outside its group")
            previous_end = end
        if previous_end != edge_count:
            raise ValueError("Local edge groups do not cover every edge")
        if self.clip_index.numel():
            clip_count = int(torch.max(self.clip_index).item()) + 1
            if len(self.copy_count_target) != clip_count:
                raise ValueError("copy_count_target must have one value per clip")


@dataclass(frozen=True)
class FullPipelineOutput:
    path_score: torch.Tensor
    keep_logit: torch.Tensor
    boundary_logits: torch.Tensor
    split_logit: torch.Tensor
    emission_logits: torch.Tensor
    structure_logits: torch.Tensor
    layer2_logits: torch.Tensor
    rhythm_logit: torch.Tensor
    duration_residual: torch.Tensor
    representation: torch.Tensor


@dataclass(frozen=True)
class FullPipelineLoss:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class FullPipelinePrediction:
    path: LatticePath
    events: tuple[JointEvent, ...]
    confidence_probabilities: tuple[float, ...]
    boundary_probabilities: tuple[float, ...]
    split_probabilities: tuple[float, ...]
    layer2_types: tuple[str, ...]
    rhythm_probabilities: tuple[float, ...]
    duration_residuals: tuple[float, ...]
    corrected_durations_sec: tuple[float, ...]
    structure_types: tuple[str, ...]
    copy_count: int
    missed_score_events: tuple[int, ...]


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _inflate_tensor(
    key: str,
    source: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor | None:
    """Repeat adjacent channels while preserving linear-layer outputs."""

    if source.ndim != len(target_shape):
        return None
    ratios = []
    for found, expected in zip(source.shape, target_shape):
        if expected < found or expected % found:
            return None
        ratios.append(expected // found)
    if not any(ratio > 1 for ratio in ratios):
        return source
    result = source
    for dimension, ratio in enumerate(ratios):
        if ratio > 1:
            result = result.repeat_interleave(ratio, dim=dimension)
    # Duplicated input channels must share the original incoming weight.
    if source.ndim == 2 and key.endswith(".weight") and ratios[1] > 1:
        result = result / ratios[1]
    return result


class FullJointPipelineModel(nn.Module):
    """Shared path scorer and all repository-owned inference heads."""

    def __init__(
        self,
        config: FullPipelineModelConfig = FullPipelineModelConfig(),
    ) -> None:
        super().__init__()
        if config.edge_feature_dim != FEATURE_DIM:
            raise ValueError(
                f"Expected stable edge feature dimension {FEATURE_DIM}"
            )
        self.config = config
        dim = config.component_dim
        self.legacy_path = JointEdgeScorer(
            hidden_dim=config.path_hidden_dim,
            dropout=config.dropout,
            component_mode=True,
            component_dim=config.path_component_dim,
            residual_scale=config.path_residual_scale,
        )
        self.acoustic_projection = _mlp(
            5 + 3, config.hidden_dim, dim, config.dropout
        )
        self.score_projection = _mlp(
            _OPERATION_COUNT + 7, config.hidden_dim, dim, config.dropout
        )
        self.structure_projection = _mlp(
            _OPERATION_COUNT + 4, config.hidden_dim, dim, config.dropout
        )
        self.shared = _mlp(dim * 3, config.hidden_dim, dim, config.dropout)

        self.confidence_calibration = nn.Linear(dim, 1)
        self.boundary_head = nn.Linear(dim, 2)
        self.split_head = nn.Linear(dim, 1)
        self.emission_head = nn.Linear(dim, len(tuple(JointOperation)))
        self.structure_head = nn.Linear(dim, config.structure_classes)
        self.copy_count_head = _mlp(
            dim, config.hidden_dim, config.copy_count_classes, config.dropout
        )
        self.layer2_head = nn.Linear(dim, config.layer2_classes)
        self.layer3_rhythm_head = nn.Linear(dim, 1)
        self.layer3_duration_head = nn.Linear(dim, 1)
        self.joint_path_head = nn.Linear(dim, 1)

        # Preserve an imported path decoder exactly before the first update.
        nn.init.zeros_(self.joint_path_head.weight)
        nn.init.zeros_(self.joint_path_head.bias)

    @staticmethod
    def _inputs(
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        operation = edge_features[..., :_OPERATION_COUNT]
        acoustic = torch.cat(
            (
                edge_features[..., _ACOUSTIC_START:],
                edge_features[
                    ...,
                    [
                        _CONFIDENCE_INDEX,
                        _DURATION_RATIO_INDEX,
                        _OPERATION_COUNT + 12,
                    ],
                ],
            ),
            dim=-1,
        )
        score = torch.cat(
            (
                operation,
                edge_features[
                    ...,
                    [
                        _OPERATION_COUNT,
                        _OPERATION_COUNT + 1,
                        _OPERATION_COUNT + 2,
                        _OPERATION_COUNT + 5,
                        _OPERATION_COUNT + 6,
                        _OPERATION_COUNT + 7,
                        _OPERATION_COUNT + 9,
                    ],
                ],
            ),
            dim=-1,
        )
        structure = torch.cat(
            (
                operation,
                edge_features[
                    ...,
                    [
                        _OPERATION_COUNT + 8,
                        _REPLAY_MODE_INDEX,
                        _CONTINUATION_INDEX,
                        _OPERATION_COUNT + 12,
                    ],
                ],
            ),
            dim=-1,
        )
        return acoustic, score, structure

    def heads(self, edge_features: torch.Tensor) -> FullPipelineOutput:
        if (
            edge_features.ndim != 2
            or edge_features.shape[-1] != self.config.edge_feature_dim
        ):
            raise ValueError(
                f"Expected edge features [N, {self.config.edge_feature_dim}]"
            )
        acoustic, score, structure = self._inputs(edge_features)
        acoustic_embedding = self.acoustic_projection(acoustic)
        score_embedding = self.score_projection(score)
        structure_embedding = self.structure_projection(structure)
        representation = self.shared(
            torch.cat(
                (acoustic_embedding, score_embedding, structure_embedding),
                dim=-1,
            )
        )
        keep_logit = self.confidence_calibration(acoustic_embedding).squeeze(-1)
        boundary_logits = self.boundary_head(acoustic_embedding)
        split_logit = self.split_head(acoustic_embedding).squeeze(-1)
        emission_logits = self.emission_head(
            acoustic_embedding + score_embedding
        )
        structure_logits = self.structure_head(
            structure_embedding + representation
        )
        layer2_logits = self.layer2_head(representation)
        rhythm_logit = self.layer3_rhythm_head(representation).squeeze(-1)
        duration_residual = self.layer3_duration_head(
            representation
        ).squeeze(-1)

        # Candidate keep/noise calibration and score-conditioned emissions are
        # part of the actual lattice score, not detached auxiliary classifiers.
        noise_axis = list(JointOperation).index(JointOperation.NOISE)
        noise = edge_features[:, noise_axis]
        keep_sign = 1.0 - 2.0 * noise
        selected_emission = torch.sum(
            emission_logits * edge_features[:, :_OPERATION_COUNT], dim=-1
        )
        operation_to_layer2 = torch.tensor(
            (0, 1, 2, 3, 0, 0, 0, 0),
            dtype=torch.long,
            device=edge_features.device,
        )
        selected_layer2 = torch.sum(
            layer2_logits[:, None, :]
            * F.one_hot(
                operation_to_layer2, num_classes=self.config.layer2_classes
            ).to(layer2_logits.dtype)
            * edge_features[:, :_OPERATION_COUNT, None],
            dim=(-2, -1),
        )
        operation_to_structure = torch.tensor(
            (0, 0, 0, 0, 1, 2, 3, 0),
            dtype=torch.long,
            device=edge_features.device,
        )
        selected_structure = torch.sum(
            structure_logits[:, None, :]
            * F.one_hot(
                operation_to_structure,
                num_classes=self.config.structure_classes,
            ).to(structure_logits.dtype)
            * edge_features[:, :_OPERATION_COUNT, None],
            dim=(-2, -1),
        )
        path_score = (
            self.legacy_path(edge_features)
            + self.config.residual_scale
            * (
                self.joint_path_head(representation).squeeze(-1)
                + 0.25 * keep_sign * keep_logit
                + 0.10
                * keep_sign
                * (
                    boundary_logits[:, 1] - boundary_logits[:, 0]
                    + split_logit
                )
                + 0.25 * selected_emission
                + 0.10 * selected_layer2
                + self.config.structure_path_weight * selected_structure
            )
        )
        return FullPipelineOutput(
            path_score=path_score,
            keep_logit=keep_logit,
            boundary_logits=boundary_logits,
            split_logit=split_logit,
            emission_logits=emission_logits,
            structure_logits=structure_logits,
            layer2_logits=layer2_logits,
            rhythm_logit=rhythm_logit,
            duration_residual=duration_residual,
            representation=representation,
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Return scalar edge scores for ``SparseJointLattice``."""

        return self.heads(edge_features).path_score

    def initialize_path(
        self,
        checkpoint: Path | str,
        *,
        transfer_mode: str = "exact",
        minimum_coverage: float = 1.0,
    ) -> dict[str, Any]:
        """Import path weights with an explicit, coverage-gated policy."""

        path = Path(checkpoint)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source_state = payload.get("state_dict")
        if not isinstance(source_state, Mapping):
            raise ValueError("Initialization checkpoint has no state_dict")
        source_model = payload.get("model") or {}
        source_residual_scale = source_model.get("residual_scale")
        if source_residual_scale is not None and not math.isclose(
            float(source_residual_scale),
            self.config.path_residual_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Path residual scale mismatch: "
                f"source={source_residual_scale}, "
                f"target={self.config.path_residual_scale}"
            )
        if transfer_mode not in {"exact", "inflate"}:
            raise ValueError(f"Unsupported path transfer mode: {transfer_mode}")
        if not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be between zero and one")
        legacy_state = self.legacy_path.state_dict()
        prefixed = {}
        shape_mismatches = {}
        inflated = []
        source_parameter_count = 0
        transferred_parameter_count = 0
        for key, value in source_state.items():
            candidate_key = key.removeprefix("legacy_path.")
            if candidate_key in legacy_state:
                source_parameter_count += int(value.numel())
                if legacy_state[candidate_key].shape == value.shape:
                    prefixed[candidate_key] = value
                    transferred_parameter_count += int(value.numel())
                elif transfer_mode == "inflate":
                    expanded = _inflate_tensor(
                        candidate_key,
                        value,
                        legacy_state[candidate_key].shape,
                    )
                    if expanded is not None:
                        prefixed[candidate_key] = expanded
                        inflated.append(candidate_key)
                        transferred_parameter_count += int(value.numel())
                        continue
                    shape_mismatches[candidate_key] = {
                        "expected": list(legacy_state[candidate_key].shape),
                        "found": list(value.shape),
                    }
                else:
                    shape_mismatches[candidate_key] = {
                        "expected": list(legacy_state[candidate_key].shape),
                        "found": list(value.shape),
                    }
        coverage = transferred_parameter_count / max(source_parameter_count, 1)
        if coverage < minimum_coverage:
            raise ValueError(
                "Path initialization transfer coverage "
                f"{coverage:.3f} is below required {minimum_coverage:.3f}; "
                f"shape mismatches={sorted(shape_mismatches)}"
            )
        incompatible = self.legacy_path.load_state_dict(prefixed, strict=False)
        loaded = sorted(set(prefixed) - set(incompatible.unexpected_keys))
        if not loaded:
            raise ValueError("No compatible path parameters were imported")
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "source_schema": payload.get("schema_version"),
            "source_model_config": source_model,
            "loaded_parameter_tensors": len(loaded),
            "inflated_parameter_tensors": sorted(inflated),
            "transfer_mode": transfer_mode,
            "transfer_coverage": coverage,
            "transferred_source_parameters": transferred_parameter_count,
            "source_parameters": source_parameter_count,
            "minimum_transfer_coverage": minimum_coverage,
            "missing_parameter_tensors": list(incompatible.missing_keys),
            "unexpected_parameter_tensors": list(incompatible.unexpected_keys),
            "shape_mismatches": shape_mismatches,
        }

    def freeze_for_stage(self, stage: str) -> None:
        """Apply the explicit staged-unfreeze schedule."""

        stages = {
            "acoustic": {
                "acoustic_projection",
                "confidence_calibration",
                "boundary_head",
                "split_head",
                "emission_head",
                "shared",
                "joint_path_head",
            },
            "structure": {
                "score_projection",
                "structure_projection",
                "structure_head",
                "copy_count_head",
                "emission_head",
                "shared",
                "joint_path_head",
            },
            "errors": {
                "layer2_head",
                "layer3_rhythm_head",
                "layer3_duration_head",
                "shared",
            },
            "joint": {name.split(".", 1)[0] for name, _ in self.named_parameters()},
        }
        if stage not in stages:
            raise ValueError(f"Unknown training stage {stage!r}")
        selected = stages[stage]
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.split(".", 1)[0] in selected


def _segment_logsumexp(
    scores: torch.Tensor,
    groups: Sequence[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.tensor(
        [value[0] for value in groups],
        dtype=torch.long,
        device=scores.device,
    )
    lengths = torch.tensor(
        [value[1] - value[0] for value in groups],
        dtype=torch.long,
        device=scores.device,
    )
    gold = starts + torch.tensor(
        [value[2] for value in groups],
        dtype=torch.long,
        device=scores.device,
    )
    group_ids = torch.repeat_interleave(
        torch.arange(len(groups), device=scores.device), lengths
    )
    maxima = torch.full(
        (len(groups),),
        -torch.inf,
        dtype=torch.float32,
        device=scores.device,
    )
    float_scores = scores.float()
    maxima.scatter_reduce_(
        0, group_ids, float_scores, reduce="amax", include_self=True
    )
    sums = torch.zeros_like(maxima)
    sums.scatter_add_(
        0, group_ids, torch.exp(float_scores - maxima[group_ids])
    )
    return maxima + torch.log(sums.clamp_min(1e-20)), gold


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(dtype=value.dtype, device=value.device)
    return torch.sum(value * weight) / torch.sum(weight).clamp_min(1e-12)


def _balanced_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    classes = logits.shape[-1]
    counts = torch.bincount(target, minlength=classes).to(logits)
    present = counts > 0
    weights = torch.ones_like(counts)
    if int(present.sum()) > 1:
        mean = counts[present].mean()
        weights[present] = torch.sqrt(mean / counts[present].clamp_min(1.0))
        weights = torch.clamp(weights, 0.5, 5.0)
    return F.cross_entropy(logits, target, weight=weights)


def _balanced_binary(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    target = target.to(logits).float()
    positives = torch.sum(target)
    negatives = target.numel() - positives
    pos_weight = (
        torch.clamp(
            negatives / positives.clamp_min(1.0),
            min=1.0,
            max=8.0,
        )
        if positives > 0
        else torch.ones((), dtype=logits.dtype, device=logits.device)
    )
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction=reduction,
    )


def full_pipeline_loss(
    model: FullJointPipelineModel,
    edge_features: torch.Tensor,
    targets: FullPipelineTargets,
    config: FullPipelineLossConfig = FullPipelineLossConfig(),
) -> FullPipelineLoss:
    """Compute FP32 path math plus all jointly optimized inference losses."""

    targets.validate(len(edge_features))
    output = model.heads(edge_features)
    log_partition, gold = _segment_logsumexp(output.path_score, targets.groups)
    gold_score = output.path_score.float()[gold]

    group_weight = torch.ones_like(gold_score)
    confidence = edge_features[gold, _CONFIDENCE_INDEX]
    band_negative = (
        (targets.keep.to(confidence.device) == 0)
        & (confidence >= 0.45)
        & (confidence <= 0.65)
    )
    group_weight = torch.where(
        band_negative,
        group_weight * config.confidence_band_negative_weight,
        group_weight,
    )
    path = _weighted_mean(log_partition - gold_score, group_weight)
    lengths = torch.tensor(
        [end - start for start, end, _offset in targets.groups],
        dtype=torch.long,
        device=gold.device,
    )
    group_ids = torch.repeat_interleave(
        torch.arange(len(targets.groups), device=gold.device),
        lengths,
    )
    operation = torch.argmax(edge_features[:, :_OPERATION_COUNT], dim=-1)
    structural_edge = (
        (operation == _REPEAT_ENTER_INDEX)
        | (operation == _REPLAY_INDEX)
        | (operation == _CONTINUE_INDEX)
    )
    ordinary_scores = torch.where(
        structural_edge,
        torch.full_like(output.path_score.float(), -torch.inf),
        output.path_score.float(),
    )
    strongest_ordinary = torch.full(
        (len(targets.groups),),
        -torch.inf,
        device=gold.device,
    )
    strongest_ordinary.scatter_reduce_(
        0,
        group_ids,
        ordinary_scores,
        reduce="amax",
        include_self=True,
    )
    gold_structural = structural_edge[gold]
    repeat_margin_loss = (
        F.relu(
            1.0
            + strongest_ordinary[gold_structural]
            - gold_score[gold_structural]
        ).mean()
        if bool(gold_structural.any())
        else gold_score.sum() * 0.0
    )

    keep = F.binary_cross_entropy_with_logits(
        output.keep_logit[gold],
        targets.keep.to(output.keep_logit).float(),
        reduction="none",
    )
    confidence_loss = _weighted_mean(keep, group_weight)
    boundary = F.cross_entropy(
        output.boundary_logits[gold],
        targets.boundary.to(gold.device).long(),
        reduction="none",
    )
    rearticulation = targets.rearticulation_weight.to(boundary)
    rearticulation = torch.clamp(
        rearticulation, min=1.0, max=config.strong_rearticulation_weight
    )
    boundary_loss = _weighted_mean(boundary, rearticulation)
    split = _balanced_binary(
        output.split_logit[gold],
        targets.split.to(output.split_logit).float(),
        reduction="none",
    )
    split_loss = _weighted_mean(split, rearticulation)
    emission_loss = F.cross_entropy(
        output.emission_logits[gold],
        targets.emission.to(gold.device).long(),
    )
    structure_loss = _balanced_cross_entropy(
        output.structure_logits[gold],
        targets.structure.to(gold.device).long(),
    )
    layer2_loss = _balanced_cross_entropy(
        output.layer2_logits[gold],
        targets.layer2.to(gold.device).long(),
    )
    rhythm_loss = _balanced_binary(
        output.rhythm_logit[gold],
        targets.rhythm.to(output.rhythm_logit).float(),
        reduction="none",
    )
    duration_weight = targets.duration_weight.to(rhythm_loss)
    rhythm_loss = _weighted_mean(rhythm_loss, duration_weight)
    duration_loss = F.smooth_l1_loss(
        output.duration_residual[gold],
        targets.duration_target.to(output.duration_residual),
        reduction="none",
    )
    duration_loss = _weighted_mean(duration_loss, duration_weight)

    clip_index = targets.clip_index.to(gold.device).long()
    clip_count = len(targets.copy_count_target)
    pooled = torch.zeros(
        (clip_count, output.representation.shape[-1]),
        dtype=output.representation.dtype,
        device=output.representation.device,
    )
    pooled.index_add_(0, clip_index, output.representation[gold])
    counts = torch.bincount(clip_index, minlength=clip_count).clamp_min(1)
    pooled = pooled / counts[:, None]
    copy_count_loss = _balanced_cross_entropy(
        model.copy_count_head(pooled),
        targets.copy_count_target.to(gold.device).long(),
    )

    components = {
        "path": path,
        "confidence": confidence_loss,
        "boundary": boundary_loss,
        "split": split_loss,
        "emission": emission_loss,
        "structure": structure_loss,
        "repeat_margin": repeat_margin_loss,
        "copy_count": copy_count_loss,
        "layer2": layer2_loss,
        "layer3": rhythm_loss,
        "duration": duration_loss,
    }
    weights = asdict(config)
    total = sum(
        components[name] * float(weights[name]) for name in components
    )
    return FullPipelineLoss(total=total, components=components)


def duration_supervision_weight(duration_sec: float) -> float:
    """Nested duration weighting for <80/<120/<180 ms supervision."""

    duration = float(duration_sec)
    return (
        4.0
        if duration < 0.080
        else 3.0
        if duration < 0.120
        else 2.0
        if duration < 0.180
        else 1.0
    )


def augment_difficult_timbre(
    edge_features: torch.Tensor,
    groups: Sequence[tuple[int, int, int]],
    config: FullPipelineAugmentConfig = FullPipelineAugmentConfig(),
) -> torch.Tensor:
    """Apply label-preserving filtered/breathy activation augmentation.

    One random transform is shared by every option for a candidate group, so
    augmentation cannot leak the selected edge.
    """

    if config.difficult_timbre_probability <= 0.0:
        return edge_features
    device = edge_features.device
    lengths = torch.tensor(
        [end - start for start, end, _gold in groups],
        dtype=torch.long,
        device=device,
    )
    group_ids = torch.repeat_interleave(
        torch.arange(len(groups), device=device), lengths
    )
    selected = (
        torch.rand(len(groups), device=device)
        < config.difficult_timbre_probability
    )
    scale = torch.empty(len(groups), device=device).uniform_(
        config.activation_scale_min, config.activation_scale_max
    )
    scale = torch.where(selected, scale, torch.ones_like(scale))
    output = edge_features.clone()
    positive_columns = torch.tensor(
        (0, 1, 2, 4), dtype=torch.long, device=device
    ) + _ACOUSTIC_START
    values = output[:, positive_columns] * scale[group_ids, None]
    if config.activation_noise_std > 0.0:
        noise = (
            torch.randn_like(values)
            * config.activation_noise_std
            * selected[group_ids, None]
        )
        values = values + noise
    output[:, positive_columns] = torch.clamp(values, 0.0, 1.0)
    margin_drop = torch.rand(len(groups), device=device) * float(
        config.pitch_margin_drop_max
    )
    margin_drop = margin_drop * selected
    output[:, _ACOUSTIC_START + 3] = torch.clamp(
        output[:, _ACOUSTIC_START + 3] - margin_drop[group_ids],
        -1.0,
        0.0,
    )
    return output


def _selected_edge_rows(
    model: FullJointPipelineModel,
    lattice: SparseJointLattice,
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    path: LatticePath,
) -> torch.Tensor:
    """Reconstruct only decoded edge inputs; this never accesses gold."""

    state = StructuralState()
    tempo = lattice._tempo_scale(candidates, score)
    rows: list[list[float]] = []
    for step in path.steps:
        candidate = candidates[step.candidate_index]
        previous = (
            candidates[step.candidate_index - 1]
            if step.candidate_index
            else None
        )
        if step.score_span is None:
            destination = state
            operation = step.operation
        else:
            transition = lattice._transition(
                state, step.score_span, allow_long_delete=True
            )
            if transition is None:
                raise RuntimeError("Decoded path contains an invalid transition")
            destination, structural, deleted = transition
            base = (
                JointOperation.MATCH
                if candidate.pitch == score[step.score_span[0]].pitch
                else JointOperation.SUBSTITUTE
            )
            operation = structural or (
                JointOperation.DELETE if deleted else base
            )
        rows.append(
            lattice._edge_features(
                operation,
                candidate,
                step.score_span,
                state,
                destination,
                len(step.deleted_events),
                score,
                tempo,
                previous,
                candidates,
                step.candidate_index,
            )
        )
        state = destination
    device = next(model.parameters()).device
    return torch.tensor(rows, dtype=torch.float32, device=device)


@torch.inference_mode()
def infer_full_pipeline(
    model: FullJointPipelineModel,
    lattice: SparseJointLattice,
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
) -> FullPipelinePrediction:
    """Run every inference layer from candidates and verified score only."""

    model.eval()
    path = lattice.decode(candidates, score)
    events = tuple(path.joint_events(candidates))
    rows = _selected_edge_rows(model, lattice, candidates, score, path)
    output = model.heads(rows)
    kept = torch.tensor(
        [
            step.operation != JointOperation.NOISE
            for step in path.steps
        ],
        dtype=torch.bool,
        device=rows.device,
    )
    confidence = tuple(
        float(value)
        for value in torch.sigmoid(output.keep_logit[kept]).cpu().tolist()
    )
    boundary = tuple(
        float(value)
        for value in torch.softmax(
            output.boundary_logits[kept], dim=-1
        )[:, 1].cpu().tolist()
    )
    split = tuple(
        float(value)
        for value in torch.sigmoid(output.split_logit[kept]).cpu().tolist()
    )
    layer2 = tuple(
        LAYER2_CLASSES[int(value)]
        for value in torch.argmax(
            output.layer2_logits[kept, :3], dim=-1
        ).cpu().tolist()
    )
    rhythm = tuple(
        float(value)
        for value in torch.sigmoid(output.rhythm_logit[kept]).cpu().tolist()
    )
    duration_residuals = tuple(
        float(value)
        for value in output.duration_residual[kept].cpu().tolist()
    )
    corrected_durations = tuple(
        max(
            0.001,
            float(event.end - event.start)
            * math.exp(max(-2.0, min(2.0, residual))),
        )
        for event, residual in zip(events, duration_residuals)
    )
    structure = tuple(
        STRUCTURE_CLASSES[int(value)]
        for value in torch.argmax(
            output.structure_logits[kept], dim=-1
        ).cpu().tolist()
    )
    pooled = output.representation.mean(dim=0, keepdim=True)
    copy_count = int(torch.argmax(model.copy_count_head(pooled), dim=-1).item())
    missed = set(path.trailing_deletions)
    for step in path.steps:
        missed.update(step.deleted_events)
    return FullPipelinePrediction(
        path=path,
        events=events,
        confidence_probabilities=confidence,
        boundary_probabilities=boundary,
        split_probabilities=split,
        layer2_types=layer2,
        rhythm_probabilities=rhythm,
        duration_residuals=duration_residuals,
        corrected_durations_sec=corrected_durations,
        structure_types=structure,
        copy_count=copy_count,
        missed_score_events=tuple(sorted(missed)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )


def atomic_checkpoint(
    path: Path,
    *,
    model: FullJointPipelineModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    progress: Mapping[str, Any],
    data_fingerprint: str,
    history: Sequence[Mapping[str, Any]] = (),
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically persist model, cursor, optimizer, scaler, and RNG state."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": (
            scaler.state_dict() if scaler is not None else None
        ),
        "progress": dict(progress),
        "data_fingerprint": str(data_fingerprint),
        "rng_state": rng_state(),
        "history": list(history),
        "checkpoint_metadata": dict(checkpoint_metadata or {}),
        "intonation_masked": INTONATION_MASKED,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    expected_data_fingerprint: str | None = None,
    expected_checkpoint_metadata: Mapping[str, Any] | None = None,
) -> tuple[FullJointPipelineModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported outputRaw full-pipeline checkpoint")
    if (
        expected_data_fingerprint is not None
        and payload.get("data_fingerprint") != expected_data_fingerprint
    ):
        raise ValueError("Checkpoint data fingerprint mismatch")
    saved_metadata = payload.get("checkpoint_metadata") or {}
    for key, expected in (expected_checkpoint_metadata or {}).items():
        if saved_metadata.get(key) != expected:
            raise ValueError(f"Checkpoint metadata mismatch for {key!r}")
    model_config = dict(payload["model_config"])
    if "path_component_dim" not in model_config:
        projection = payload["state_dict"].get(
            "legacy_path.acoustic_projection.0.weight"
        )
        if projection is None:
            raise ValueError(
                "Legacy checkpoint cannot infer path_component_dim"
            )
        model_config["path_component_dim"] = int(projection.shape[0])
    if "path_residual_scale" not in model_config:
        model_config["path_residual_scale"] = float(
            model_config["residual_scale"]
        )
    model = FullJointPipelineModel(
        FullPipelineModelConfig(**model_config)
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, payload


def verify_data_ready(
    marker: Path,
    *,
    expected_dataset: str = "outputRaw_sf_10k",
) -> dict[str, Any]:
    """Validate the release marker without opening any locked-test feature."""

    document = json.loads(marker.read_text(encoding="utf-8"))
    if document.get("schema_version") == "align-joint-data-ready-v1":
        if document.get("release") != MODEL_FAMILY:
            raise ValueError("Packed release marker names the wrong release")
        if document.get("status") != "ready":
            raise ValueError("Packed release is not marked ready")
        paths = document.get("paths") or {}
        counts = document.get("counts") or {}
        hashes = document.get("hashes") or {}
        verification = document.get("verification") or {}
        for required in ("manifest", "packed_root", "packed_index"):
            if not paths.get(required):
                raise ValueError(f"Packed release has no {required!r} path")
        for required in ("train", "val"):
            if int(counts.get(required, 0)) <= 0:
                raise ValueError(f"Packed release has no {required!r} rows")
        if not hashes.get("manifest_sha256") or not hashes.get("pack_id"):
            raise ValueError("Packed release has no stable data fingerprint")
        if (
            verification.get("test_features_materialized") is not False
            or verification.get("test_targets_materialized") is not False
        ):
            raise ValueError("Locked-test features or targets were materialized")
        return document

    dataset = document.get("dataset") or document.get("dataset_name")
    if dataset != expected_dataset:
        raise ValueError(
            f"Expected dataset {expected_dataset!r}, got {dataset!r}"
        )
    if not bool(document.get("ready", True)):
        raise ValueError("Packed release is not marked ready")
    splits = document.get("splits") or {}
    for required in ("train", "val"):
        if required not in splits:
            raise ValueError(f"Packed release has no {required!r} split")
    if any(key in document for key in ("test_features", "test_materialized")):
        raise ValueError("Release marker exposes locked-test feature materialization")
    fingerprint = document.get("fingerprint") or document.get("sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) < 32:
        raise ValueError("Packed release marker has no stable fingerprint")
    return document
