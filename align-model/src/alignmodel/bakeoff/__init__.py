"""Odd-team bakeoff variants (V3/V5/V7). V1 stays on MelodyFirst."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from alignmodel.config import ModelConfig
from alignmodel.melody import ScoreSoundingNote
from alignmodel.melody_model import MelodyFirst

OUR_VARIANTS = {"v3", "v5", "v7", "bio", "dice", "combo"}


def normalize_variant(name: str | None) -> str:
    v = (name or "v1").lower().strip()
    aliases = {
        "control": "v1",
        "bio": "v3",
        "dice": "v5",
        "combo": "v7",
        "": "v1",
    }
    return aliases.get(v, v)


def build_model(variant: str, cfg: ModelConfig | None = None) -> nn.Module:
    v = normalize_variant(variant)
    model_cfg = cfg or ModelConfig()
    if v in {"v1", "control"}:
        return MelodyFirst(model_cfg)
    if v == "v3":
        from alignmodel.bakeoff.v3 import MelodyBIO

        return MelodyBIO(model_cfg)
    if v == "v5":
        from alignmodel.bakeoff.v5 import MelodyDice

        return MelodyDice(model_cfg)
    if v == "v7":
        return MelodyFirst(model_cfg)
    raise ValueError(f"Unknown bakeoff variant {variant!r} (odd team owns v1/v3/v5/v7)")


def compute_loss(variant: str, outputs: dict[str, Tensor], batch: dict, cfg):
    v = normalize_variant(variant)
    if v == "v3":
        from alignmodel.bakeoff.v3 import compute_loss as fn

        return fn(outputs, batch, cfg)
    if v == "v5":
        from alignmodel.bakeoff.v5 import compute_loss as fn

        return fn(outputs, batch, cfg)
    if v == "v7":
        from alignmodel.bakeoff.v7 import compute_loss as fn

        return fn(outputs, batch, cfg)
    from alignmodel.melody_train import compute_loss as v1_loss

    return v1_loss(outputs, batch, cfg)


def decode_sample(
    variant: str,
    outputs: dict[str, Tensor],
    index: int,
    n: int,
    notes: list[ScoreSoundingNote],
) -> list[dict[str, Any]]:
    v = normalize_variant(variant)
    if v == "v3":
        from alignmodel.bakeoff.v3 import decode_sample as fn

        return fn(outputs, index, n, notes)
    if v == "v5":
        from alignmodel.bakeoff.v5 import decode_sample as fn

        return fn(outputs, index, n, notes)
    if v == "v7":
        from alignmodel.bakeoff.v7 import decode_sample as fn

        return fn(outputs, index, n, notes)
    from alignmodel.melody_model import decode_note_runs, types_from_logits

    types = types_from_logits(outputs["type_logits"][index, :n])
    copies = int(outputs["copies_logits"][index].argmax(-1).detach().cpu())
    return decode_note_runs(types, notes, copies)
