from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from datacreate.melody import padded_melody

from alignmodel.config import (
    ALIGN_TO_MELODY,
    MELODY_ALIGN_TYPES,
    MELODY_NOTE_CLASSES,
    ModelConfig,
)
from alignmodel.melody import ScoreSoundingNote
from alignmodel.model import AudioEncoder, HierarchicalFusion, ScoreEncoder

MATCH_DECODE_BIAS = 0.0
DECODE_PAD_NOTES = 2
MAX_RUN_FRAC = 0.45
MAX_RUN_NOTES = 16


def class_index(name: str) -> int:
    return MELODY_NOTE_CLASSES.index(name)


class MelodyFirst(nn.Module):
    """Score-note type heads plus clip-level extra_copies. Primary output is schema 1.2."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.audio = AudioEncoder(self.cfg)
        self.score = ScoreEncoder(self.cfg)
        self.fusion = HierarchicalFusion(self.cfg)
        self.type_head = nn.Linear(self.cfg.d_model, len(MELODY_NOTE_CLASSES))
        self.copies_head = nn.Linear(self.cfg.d_model, 3)
        with torch.no_grad():
            self.type_head.bias.zero_()
            self.type_head.bias[class_index("match")] = 0.2
            self.type_head.bias[class_index("repetition")] = -0.2

    def forward(
        self,
        mel: Tensor,
        mel_mask: Tensor,
        pitch: Tensor,
        onset: Tensor,
        duration: Tensor,
        note_mask: Tensor,
        hop_sec: float,
    ) -> dict[str, Tensor]:
        audio_h = self.audio(mel, mel_mask, hop_sec)
        t_audio = audio_h.size(1)
        stride = self.cfg.audio_stride
        if mel_mask.size(-1) >= t_audio * stride:
            audio_mask = (
                mel_mask[:, : t_audio * stride]
                .view(mel_mask.size(0), t_audio, stride)
                .any(dim=-1)
            )
        else:
            audio_mask = torch.ones(
                audio_h.size(0), t_audio, dtype=torch.bool, device=mel.device
            )
        score_h = self.score(pitch, onset, duration, note_mask)
        fused_score, _ = self.fusion(score_h, audio_h, note_mask, audio_mask)
        type_logits = self.type_head(fused_score)
        pooled = (fused_score * note_mask.unsqueeze(-1)).sum(1) / note_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        copies_logits = self.copies_head(pooled)
        return {
            "type_logits": type_logits,
            "copies_logits": copies_logits,
            "audio_mask": audio_mask,
        }


def types_from_logits(type_logits: Tensor, match_bias: float = MATCH_DECODE_BIAS) -> list[str]:
    """Argmax types with a match-class bias so repetition cannot swallow the clip."""
    logits = type_logits
    if match_bias:
        logits = logits.clone()
        logits[..., class_index("match")] = logits[..., class_index("match")] + float(match_bias)
    ids = logits.argmax(-1).tolist()
    if isinstance(ids, int):
        ids = [ids]
    return [MELODY_NOTE_CLASSES[int(i)] for i in ids]


def decode_note_runs(
    types: list[str],
    notes: list[ScoreSoundingNote],
    extra_copies: int,
    pad_notes: int = DECODE_PAD_NOTES,
    max_run_frac: float = MAX_RUN_FRAC,
    max_run_notes: int = MAX_RUN_NOTES,
) -> list[dict[str, Any]]:
    """Contiguous same-type non-match cores → schema 1.2 labels.

    Cores are padded by ``pad_notes`` so pred pitch lists line up with gold.
    A run that covers most of the score is split into 5-note chunks so one
    clip-wide repetition cannot be the only prediction.
    """
    n = min(len(types), len(notes))
    labels: list[dict[str, Any]] = []
    i = 0
    while i < n:
        kind = types[i]
        if kind == "match" or kind not in MELODY_ALIGN_TYPES:
            i += 1
            continue
        j = i + 1
        while j < n and types[j] == kind:
            j += 1
        core_len = j - i
        too_long = core_len > max_run_notes or (n > 0 and core_len / n > max_run_frac)
        starts = list(range(i, j, 5)) if too_long else [i]
        ends = [min(s + 5, j) for s in starts] if too_long else [j]
        for lo, hi in zip(starts, ends):
            if hi <= lo:
                continue
            span = padded_melody(notes, lo, hi, pad_notes)
            span_notes = notes[span.start_note_index : span.end_note_index + 1]
            item: dict[str, Any] = {
                "id": f"mel_{len(labels):03d}",
                "source": "melody",
                "type": MELODY_ALIGN_TYPES[kind],
                "start_time": round(span_notes[0].start, 4),
                "end_time": round(span_notes[-1].end, 4),
                "score_part": {
                    "start_note_index": span.start_note_index,
                    "end_note_index": span.end_note_index,
                    "pad_notes": span.pad_notes,
                    "start_measure": span.start_measure,
                    "end_measure": span.end_measure,
                },
                "pitches": list(span.pitches),
                "note_ids": list(span.note_ids),
            }
            if kind == "repetition":
                item["extra_copies"] = 1 if extra_copies <= 0 else min(2, extra_copies)
                item["repeats_label_range"] = {
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                }
            labels.append(item)
        i = j
    return labels


def align_type_to_class(align_type: str) -> int | None:
    name = ALIGN_TO_MELODY.get(align_type)
    if name is None:
        return None
    return class_index(name)
