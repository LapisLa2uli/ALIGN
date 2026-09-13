"""Compact clarinet note refiner operating on precomputed AMT maps.

This module intentionally has no dependency on Basic Pitch or PESTO packages:
callers provide aligned tensors produced by those frontends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decode import TransNote
from .semi_crf import (
    IntervalCandidates,
    candidate_pruned_intervals,
    semi_crf_nll,
    weighted_interval_decode,
)


BASIC_PITCH_MIDI_MIN = 21
BASIC_PITCH_BINS = 88
BASIC_PITCH_CONTOUR_BINS = 264
BASIC_PITCH_HOP_SEC = 256.0 / 22050.0


@dataclass
class NoteRefinerConfig:
    """Architecture, pitch convention, and sparse decoder configuration."""

    midi_min: int = 36
    midi_max: int = 108
    basic_midi_min: int = BASIC_PITCH_MIDI_MIN
    basic_pitch_bins: int = BASIC_PITCH_BINS
    contour_bins_per_semitone: int = 3
    pesto_pitch_unit: str = "hz"
    pesto_written_shift: float = 2.0
    channels: int = 96
    temporal_blocks: int = 6
    kernel_size: int = 5
    dropout: float = 0.10
    max_boundary_correction_frames: float = 2.0
    top_pitch_candidates: int = 3
    min_note_frames: int = 5
    max_note_frames: int = 600
    boundary_threshold: float = 0.25
    max_boundaries: int = 192
    max_ends_per_start: int = 12
    interval_pitches: int = 3
    hop_sec: float = BASIC_PITCH_HOP_SEC

    def __post_init__(self) -> None:
        if self.midi_min > self.midi_max:
            raise ValueError("midi_min must not exceed midi_max")
        basic_max = self.basic_midi_min + self.basic_pitch_bins - 1
        if self.midi_min < self.basic_midi_min or self.midi_max > basic_max:
            raise ValueError(
                f"written range must lie inside Basic Pitch "
                f"[{self.basic_midi_min}, {basic_max}]"
            )
        if self.channels < 8 or self.temporal_blocks < 1:
            raise ValueError("refiner requires channels >= 8 and at least one block")
        if self.kernel_size < 3 or not self.kernel_size % 2:
            raise ValueError("kernel_size must be odd and at least 3")
        if self.pesto_pitch_unit not in {"hz", "midi", "auto"}:
            raise ValueError("pesto_pitch_unit must be 'hz', 'midi', or 'auto'")
        if self.top_pitch_candidates < 1:
            raise ValueError("top_pitch_candidates must be positive")
        if self.min_note_frames < 1 or self.max_note_frames < self.min_note_frames:
            raise ValueError("invalid note duration bounds")

    @property
    def n_pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    @property
    def axis_start(self) -> int:
        return self.midi_min - self.basic_midi_min

    @property
    def axis_end(self) -> int:
        return self.axis_start + self.n_pitches

    @property
    def input_channels(self) -> int:
        return self.n_pitches * (2 + self.contour_bins_per_semitone) + 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "NoteRefinerConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{
            key: value for key, value in dict(payload or {}).items() if key in allowed
        })


class _TemporalResidual(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        groups = 8
        while channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(value)


def _probability_logit(value: Tensor, eps: float = 1e-4) -> Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


class NoteRefiner(nn.Module):
    """Residual temporal model for Basic Pitch maps and aligned PESTO F0."""

    def __init__(self, config: NoteRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or NoteRefinerConfig()
        cfg = self.config
        self.input_projection = nn.Conv1d(cfg.input_channels, cfg.channels, 1)
        self.input_norm = nn.GroupNorm(
            8 if cfg.channels % 8 == 0 else 1, cfg.channels
        )
        dilations = [2 ** (index % 5) for index in range(cfg.temporal_blocks)]
        self.temporal = nn.Sequential(*[
            _TemporalResidual(
                cfg.channels, cfg.kernel_size, dilation, cfg.dropout
            )
            for dilation in dilations
        ])
        self.voiced_residual_head = nn.Conv1d(cfg.channels, 1, 1)
        self.onset_residual_head = nn.Conv1d(cfg.channels, 1, 1)
        self.offset_residual_head = nn.Conv1d(cfg.channels, 1, 1)
        self.pitch_residual_head = nn.Conv1d(cfg.channels, cfg.n_pitches, 1)
        self.boundary_head = nn.Conv1d(cfg.channels, 2, 1)
        self.confidence_residual_head = nn.Conv1d(cfg.channels, 1, 1)
        self.cents_residual_head = nn.Conv1d(cfg.channels, 1, 1)
        self._zero_residual_heads()

    def _zero_residual_heads(self) -> None:
        for layer in (
            self.voiced_residual_head,
            self.onset_residual_head,
            self.offset_residual_head,
            self.pitch_residual_head,
            self.boundary_head,
            self.confidence_residual_head,
            self.cents_residual_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _validate_inputs(
        self, note: Tensor, onset: Tensor, contour: Tensor, pesto: Tensor
    ) -> None:
        if note.ndim != 3 or note.shape[-1] != self.config.basic_pitch_bins:
            raise ValueError(
                f"note must have shape [B,T,{self.config.basic_pitch_bins}]"
            )
        if onset.shape != note.shape:
            raise ValueError("onset must have the same shape as note")
        expected_contour = (
            self.config.basic_pitch_bins
            * self.config.contour_bins_per_semitone
        )
        if contour.shape != (*note.shape[:2], expected_contour):
            raise ValueError(
                f"contour must have shape [B,T,{expected_contour}]"
            )
        if pesto.shape != (*note.shape[:2], 2):
            raise ValueError("pesto must have shape [B,T,2]")

    def _pesto_midi(self, pesto_pitch: Tensor) -> Tensor:
        unit = self.config.pesto_pitch_unit
        hz_midi = 69.0 + 12.0 * torch.log2(pesto_pitch.clamp_min(1e-4) / 440.0)
        if unit == "hz":
            midi = hz_midi
        elif unit == "midi":
            midi = pesto_pitch
        else:
            midi = torch.where(pesto_pitch > 127.0, hz_midi, pesto_pitch)
        return midi + self.config.pesto_written_shift

    def _base_predictions(
        self, note: Tensor, onset: Tensor, contour: Tensor, pesto: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        note = note[:, :, cfg.axis_start:cfg.axis_end].clamp(0.0, 1.0)
        onset = onset[:, :, cfg.axis_start:cfg.axis_end].clamp(0.0, 1.0)
        contour_all = contour.reshape(
            *contour.shape[:2],
            cfg.basic_pitch_bins,
            cfg.contour_bins_per_semitone,
        )
        contour = contour_all[:, :, cfg.axis_start:cfg.axis_end].clamp(0.0, 1.0)
        pesto_confidence = pesto[..., 1].clamp(0.0, 1.0)
        pesto_midi = self._pesto_midi(pesto[..., 0])

        voiced_probability = note.amax(dim=-1)
        onset_probability = onset.amax(dim=-1)
        previous = F.pad(voiced_probability[:, :-1], (1, 0))
        offset_probability = (previous - voiced_probability).clamp(0.0, 1.0)

        contour_pitch = contour.amax(dim=-1)
        midi_axis = torch.arange(
            cfg.midi_min, cfg.midi_max + 1, device=note.device, dtype=note.dtype
        )
        pesto_distance = pesto_midi.unsqueeze(-1) - midi_axis
        pesto_prior = torch.exp(-0.5 * (pesto_distance / 0.45).square())
        pesto_prior = pesto_prior * pesto_confidence.unsqueeze(-1)
        pitch_evidence = (
            0.60 * note + 0.25 * contour_pitch + 0.15 * pesto_prior
        ).clamp_min(1e-5)
        pitch_base = torch.log(pitch_evidence)

        subbin = torch.linspace(
            -100.0 / 3.0,
            100.0 / 3.0,
            cfg.contour_bins_per_semitone,
            device=note.device,
            dtype=note.dtype,
        )
        contour_mass = contour.sum(dim=(-1, -2)).clamp_min(1e-5)
        contour_cents = (
            contour * subbin.view(1, 1, 1, -1)
        ).sum(dim=(-1, -2)) / contour_mass
        nearest = pesto_midi.round()
        pesto_cents = 100.0 * (pesto_midi - nearest)
        cents_base = (
            (1.0 - pesto_confidence) * contour_cents
            + pesto_confidence * pesto_cents
        ).clamp(-100.0, 100.0)

        features = torch.cat(
            [
                note,
                onset,
                contour.flatten(start_dim=2),
                torch.stack(
                    [
                        pesto_midi / 128.0,
                        pesto_confidence,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        return (
            features,
            _probability_logit(voiced_probability),
            _probability_logit(onset_probability),
            _probability_logit(offset_probability),
            pitch_base,
            cents_base,
            pesto_confidence,
        )

    def forward(
        self,
        note: Tensor,
        onset: Tensor,
        contour: Tensor,
        pesto: Tensor,
    ) -> dict[str, Tensor]:
        self._validate_inputs(note, onset, contour, pesto)
        base_note_probability = note[
            :, :, self.config.axis_start : self.config.axis_end
        ].clamp(0.0, 1.0)
        base_onset_probability = onset[
            :, :, self.config.axis_start : self.config.axis_end
        ].clamp(0.0, 1.0)
        (
            features,
            voiced_base,
            onset_base,
            offset_base,
            pitch_base,
            cents_base,
            pesto_confidence,
        ) = self._base_predictions(note, onset, contour, pesto)
        hidden = self.input_projection(features.transpose(1, 2))
        hidden = self.temporal(F.silu(self.input_norm(hidden)))

        voiced_residual = self.voiced_residual_head(hidden).squeeze(1)
        onset_residual = self.onset_residual_head(hidden).squeeze(1)
        offset_residual = self.offset_residual_head(hidden).squeeze(1)
        pitch_residual = self.pitch_residual_head(hidden).transpose(1, 2)
        confidence_residual = self.confidence_residual_head(hidden).squeeze(1)
        cents_residual = 100.0 * torch.tanh(
            self.cents_residual_head(hidden).squeeze(1)
        )

        voiced_logits = voiced_base + voiced_residual
        onset_logits = onset_base + onset_residual
        offset_logits = offset_base + offset_residual
        pitch_logits = pitch_base + pitch_residual
        confidence_logits = (
            0.65 * voiced_base
            + 0.35 * _probability_logit(pesto_confidence)
            + confidence_residual
        )
        cents = (cents_base + cents_residual).clamp(-100.0, 100.0)
        correction = self.config.max_boundary_correction_frames * torch.tanh(
            self.boundary_head(hidden).transpose(1, 2)
        )
        candidate_scores, candidate_classes = torch.topk(
            pitch_logits.softmax(dim=-1),
            k=min(self.config.top_pitch_candidates, self.config.n_pitches),
            dim=-1,
        )
        candidate_midis = candidate_classes + self.config.midi_min
        return {
            "voiced_logits": voiced_logits,
            "onset_logits": onset_logits,
            "offset_logits": offset_logits,
            "pitch_logits": pitch_logits,
            "boundary_correction": correction,
            "confidence_logits": confidence_logits,
            "confidence": confidence_logits.sigmoid(),
            "cents": cents,
            "pitch_candidates": candidate_midis,
            "pitch_candidate_scores": candidate_scores,
            "voiced_residual": voiced_residual,
            "onset_residual": onset_residual,
            "offset_residual": offset_residual,
            "pitch_residual": pitch_residual,
            "confidence_residual": confidence_residual,
            "cents_residual": cents_residual,
            "basic_note_probability": base_note_probability,
            "basic_onset_probability": base_onset_probability,
        }


BasicPitchRefiner = NoteRefiner
ClarinetIntervalRefiner = NoteRefiner
RefinerConfig = NoteRefinerConfig


def _masked_bce(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    value = F.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none"
    )
    return (value * valid).sum() / valid.sum().clamp_min(1)


def note_refiner_loss(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor | Sequence[Sequence[tuple[int, int, int]]]],
    config: NoteRefinerConfig | None = None,
    *,
    interval_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Frame losses plus optional sparse Semi-CRF interval NLL."""

    cfg = config or NoteRefinerConfig()
    valid_value = targets.get("frame_mask")
    valid = (
        valid_value.bool()  # type: ignore[union-attr]
        if torch.is_tensor(valid_value)
        else torch.ones_like(outputs["voiced_logits"], dtype=torch.bool)
    )
    voiced_target = targets["voiced"]
    onset_target = targets["onset"]
    offset_target = targets["offset"]
    if not all(torch.is_tensor(item) for item in (
        voiced_target, onset_target, offset_target
    )):
        raise TypeError("frame targets must be tensors")
    voiced = _masked_bce(outputs["voiced_logits"], voiced_target, valid)
    onset = _masked_bce(outputs["onset_logits"], onset_target, valid)
    offset = _masked_bce(outputs["offset_logits"], offset_target, valid)

    pitch_target = targets["pitch"]
    if not torch.is_tensor(pitch_target):
        raise TypeError("pitch target must be a tensor")
    local_pitch = pitch_target.long()
    nonnegative = local_pitch >= 0
    if (
        bool(nonnegative.any())
        and int(local_pitch[nonnegative].min()) >= cfg.midi_min
        and int(local_pitch[nonnegative].max()) <= cfg.midi_max
    ):
        local_pitch = local_pitch - cfg.midi_min
    pitch_mask = valid & nonnegative & (voiced_target > 0)  # type: ignore[operator]
    local_pitch = local_pitch.masked_fill(~pitch_mask, -1)
    pitch = F.cross_entropy(
        outputs["pitch_logits"].transpose(1, 2),
        local_pitch,
        ignore_index=-1,
        reduction="sum",
    ) / pitch_mask.sum().clamp_min(1)

    zero = outputs["voiced_logits"].sum() * 0.0
    cents = zero
    cents_target = targets.get("cents")
    if torch.is_tensor(cents_target):
        cents = (
            F.smooth_l1_loss(
                outputs["cents"] / 100.0,
                cents_target.float() / 100.0,
                reduction="none",
            )
            * pitch_mask
        ).sum() / pitch_mask.sum().clamp_min(1)
    boundary = zero
    boundary_target = targets.get("boundary_correction")
    if torch.is_tensor(boundary_target):
        boundary_valid = valid.unsqueeze(-1).expand_as(
            outputs["boundary_correction"]
        )
        boundary = (
            F.smooth_l1_loss(
                outputs["boundary_correction"],
                boundary_target.float(),
                reduction="none",
            )
            * boundary_valid
        ).sum() / boundary_valid.sum().clamp_min(1)
    confidence_target = targets.get("confidence")
    confidence = _masked_bce(
        outputs["confidence_logits"],
        confidence_target if torch.is_tensor(confidence_target) else voiced_target,
        valid,
    )

    interval = zero
    target_intervals = targets.get("intervals")
    if target_intervals is not None and interval_weight > 0:
        losses = []
        for batch_index, raw_intervals in enumerate(target_intervals):
            intervals = [
                (
                    int(start),
                    int(end),
                    (
                        int(pitch) - cfg.midi_min
                        if cfg.midi_min <= int(pitch) <= cfg.midi_max
                        else int(pitch)
                    ),
                )
                for start, end, pitch in raw_intervals
                if cfg.min_note_frames
                <= int(end) - int(start)
                <= cfg.max_note_frames
            ]
            graph = build_refiner_candidates(
                outputs, cfg, batch_index=batch_index,
                required_intervals=intervals,
            )
            losses.append(
                semi_crf_nll(
                    graph, gold_intervals=intervals, normalize=True
                )
            )
        if losses:
            interval = torch.stack(losses).mean()

    total = (
        voiced
        + 2.0 * onset
        + 1.5 * offset
        + pitch
        + 0.4 * cents
        + 0.25 * boundary
        + 0.25 * confidence
        + interval_weight * interval
    )
    return total, {
        "loss": total.detach(),
        "voiced": voiced.detach(),
        "onset": onset.detach(),
        "offset": offset.detach(),
        "pitch": pitch.detach(),
        "cents": cents.detach(),
        "boundary": boundary.detach(),
        "confidence": confidence.detach(),
        "interval": interval.detach(),
    }


def _one_clip(value: Tensor, batch_index: int) -> Tensor:
    if value.ndim >= 2 and value.shape[0] > batch_index:
        return value[batch_index]
    if batch_index:
        raise IndexError("batch_index outside output batch")
    return value


def build_refiner_candidates(
    outputs: dict[str, Tensor],
    config: NoteRefinerConfig | None = None,
    *,
    batch_index: int = 0,
    required_intervals: Sequence[tuple[int, int, int]] = (),
) -> IntervalCandidates:
    cfg = config or NoteRefinerConfig()
    base_intervals = _basic_pitch_intervals(
        _one_clip(outputs.get("basic_note_probability"), batch_index)
        if outputs.get("basic_note_probability") is not None
        else None,
        _one_clip(outputs.get("basic_onset_probability"), batch_index)
        if outputs.get("basic_onset_probability") is not None
        else None,
        cfg,
    )
    all_required = tuple(required_intervals) + tuple(base_intervals)
    return candidate_pruned_intervals(
        _one_clip(outputs["voiced_logits"], batch_index),
        _one_clip(outputs["onset_logits"], batch_index),
        _one_clip(outputs["offset_logits"], batch_index),
        _one_clip(outputs["pitch_logits"], batch_index),
        min_duration=cfg.min_note_frames,
        max_duration=cfg.max_note_frames,
        pitches_per_interval=cfg.interval_pitches,
        boundary_threshold=cfg.boundary_threshold,
        max_boundaries=cfg.max_boundaries,
        max_ends_per_start=cfg.max_ends_per_start,
        required_intervals=all_required,
    )


def _basic_pitch_intervals(
    note_probability: Tensor | None,
    onset_probability: Tensor | None,
    config: NoteRefinerConfig,
) -> list[tuple[int, int, int]]:
    """Propose per-pitch runs so the residual decoder preserves its teacher."""

    if note_probability is None or onset_probability is None:
        return []
    note = note_probability.detach().cpu()
    onset = onset_probability.detach().cpu()
    frames, pitches = note.shape
    intervals: list[tuple[int, int, int]] = []
    for pitch in range(pitches):
        active = note[:, pitch] >= 0.40
        index = 0
        while index < frames:
            if not bool(active[index]):
                index += 1
                continue
            start = index
            index += 1
            while index < frames and bool(active[index]):
                if (
                    index - start >= config.min_note_frames
                    and float(onset[index, pitch]) >= 0.50
                ):
                    intervals.append((start, index, pitch))
                    start = index
                index += 1
            if index - start >= config.min_note_frames:
                intervals.append((start, index, pitch))
    return intervals


@torch.no_grad()
def decode_refined_notes(
    outputs: dict[str, Tensor],
    config: NoteRefinerConfig | None = None,
    *,
    batch_index: int = 0,
) -> list[TransNote]:
    """Decode one batch item into existing :class:`TransNote` values."""

    cfg = config or NoteRefinerConfig()
    graph = build_refiner_candidates(outputs, cfg, batch_index=batch_index)
    selected = weighted_interval_decode(graph)
    if not selected:
        return []
    pitch_probability = _one_clip(
        outputs["pitch_logits"], batch_index
    ).softmax(dim=-1)
    cents = _one_clip(outputs["cents"], batch_index)
    confidence = _one_clip(outputs["confidence"], batch_index)
    correction = _one_clip(outputs["boundary_correction"], batch_index)
    notes: list[TransNote] = []
    previous_end_frame = 0.0
    for index in selected:
        nominal_start = int(graph.starts[index])
        nominal_end = int(graph.ends[index])
        start = float(nominal_start + correction[nominal_start, 0].item())
        end_frame_index = min(nominal_end - 1, correction.shape[0] - 1)
        end = float(nominal_end + correction[end_frame_index, 1].item())
        start = max(previous_end_frame, start, 0.0)
        end = min(float(graph.num_frames), end)
        if end - start < cfg.min_note_frames:
            end = min(float(graph.num_frames), start + cfg.min_note_frames)
        if end - start < cfg.min_note_frames:
            continue

        pitch_index = int(graph.pitches[index])
        segment_pitch = pitch_probability[nominal_start:nominal_end].mean(dim=0)
        top = torch.topk(
            segment_pitch,
            k=min(cfg.top_pitch_candidates, cfg.n_pitches),
        ).indices
        candidates = tuple(
            int(value) + cfg.midi_min for value in top.detach().cpu().tolist()
        )
        note_confidence = float(
            confidence[nominal_start:nominal_end].mean().clamp(0.0, 1.0)
        )
        note_cents = float(cents[nominal_start:nominal_end].median())
        pitch = pitch_index + cfg.midi_min
        if pitch not in candidates:
            candidates = (pitch, *candidates[:-1])
        notes.append(
            TransNote(
                pitch=pitch,
                start=round(start * cfg.hop_sec, 6),
                end=round(end * cfg.hop_sec, 6),
                confidence=round(note_confidence, 6),
                cents=round(note_cents, 2),
                pitch_candidates=candidates,
            )
        )
        previous_end_frame = end
    return notes


decode_notes = decode_refined_notes
refiner_loss = note_refiner_loss


def save_note_refiner(
    path: Path | str,
    model: NoteRefiner,
    *,
    extra: dict | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "extra": dict(extra or {}),
        },
        destination,
    )
    return destination


def load_note_refiner(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[NoteRefiner, dict]:
    payload = torch.load(
        Path(path), map_location=torch.device(device), weights_only=False
    )
    model = NoteRefiner(NoteRefinerConfig.from_dict(payload.get("model_config")))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, dict(payload.get("extra") or {})


save_refiner_checkpoint = save_note_refiner
load_refiner_checkpoint = load_note_refiner
