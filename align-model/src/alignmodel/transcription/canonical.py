"""Versioned canonical note-decoder loader and inference dispatch."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .basic_pitch import (
    basic_pitch_cache_path as default_basic_cache_path,
    decode_frozen_basic_pitch,
    extract_basic_pitch_features,
    load_audio_metadata,
)
from .decode import DecodeConfig, infer_sample_notes, load_note_transcriber
from .fine_pitch import (
    apply_pesto_cents,
    extract_pesto_features,
    pesto_cache_path as default_pesto_cache_path,
)
from .refiner import NoteRefiner, decode_refined_notes, load_note_refiner


@dataclass
class CanonicalNoteDecoder:
    kind: str
    model: Any
    decode_config: Any
    device: torch.device
    extra: dict


def load_note_decoder(
    checkpoint: Path | str,
    device: torch.device | str = "cpu",
) -> CanonicalNoteDecoder:
    path = Path(checkpoint)
    torch_device = torch.device(device)
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("kind") != "basic-pitch":
            raise ValueError(f"Unknown note decoder manifest {path}")
        return CanonicalNoteDecoder(
            kind="basic-pitch",
            model=None,
            decode_config=document.get("decode") or {},
            device=torch_device,
            extra=document,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = dict(payload.get("extra") or {})
    if int(extra.get("format_version") or 0) >= 3:
        model, extra = load_note_refiner(path, torch_device)
        return CanonicalNoteDecoder(
            kind="basic-pitch-refiner",
            model=model,
            decode_config=model.config,
            device=torch_device,
            extra=extra,
        )
    model, decode = load_note_transcriber(path, torch_device)
    return CanonicalNoteDecoder(
        kind="note-frame-net",
        model=model,
        decode_config=decode,
        device=torch_device,
        extra={},
    )


@torch.no_grad()
def infer_note_decoder(
    decoder: CanonicalNoteDecoder,
    sample_dir: Path | str,
    *,
    basic_cache_path: Path | str | None = None,
    pesto_cache_path: Path | str | None = None,
) -> list:
    sample = Path(sample_dir)
    if decoder.kind == "note-frame-net":
        return infer_sample_notes(
            decoder.model,
            sample,
            decoder.device,
            decode_config=(
                decoder.decode_config
                if isinstance(decoder.decode_config, DecodeConfig)
                else None
            ),
        )
    if decoder.kind == "basic-pitch":
        metadata = load_audio_metadata(sample)
        fallback_transpose = decoder.extra.get("effective_audio_transpose")
        if (
            metadata.get("effective_audio_transpose") is None
            and fallback_transpose is not None
        ):
            metadata["effective_audio_transpose"] = int(fallback_transpose)
            metadata["audio_pitch_space"] = "transposed"
        if basic_cache_path is None and decoder.extra.get("basic_cache_root"):
            basic_cache_path = default_basic_cache_path(
                Path(str(decoder.extra["basic_cache_root"])),
                sample,
                str(decoder.extra.get("corpus") or "procedural12k"),
            )
        basic = extract_basic_pitch_features(
            sample / "performance_audio.wav",
            source_metadata=metadata,
            cache_path=basic_cache_path,
        )
        notes = decode_frozen_basic_pitch(basic)
        if decoder.extra.get("fine_pitch") == "pesto-2.0.1":
            if pesto_cache_path is None and decoder.extra.get("pesto_cache_root"):
                pesto_cache_path = default_pesto_cache_path(
                    Path(str(decoder.extra["pesto_cache_root"])),
                    sample,
                    str(decoder.extra.get("corpus") or "procedural12k"),
                )
            pesto = extract_pesto_features(
                sample / "performance_audio.wav",
                basic,
                source_metadata=metadata,
                cache_path=pesto_cache_path,
                device=str(decoder.device),
            )
            notes = apply_pesto_cents(notes, basic, pesto)
        return notes
    if decoder.kind != "basic-pitch-refiner" or not isinstance(
        decoder.model, NoteRefiner
    ):
        raise ValueError(f"Unknown note decoder kind {decoder.kind!r}")
    metadata = load_audio_metadata(sample)
    fallback_transpose = decoder.extra.get("effective_audio_transpose")
    if (
        metadata.get("effective_audio_transpose") is None
        and fallback_transpose is not None
    ):
        metadata["effective_audio_transpose"] = int(fallback_transpose)
        metadata["audio_pitch_space"] = "transposed"
    train_config = dict(decoder.extra.get("train_config") or {})
    corpus = str(decoder.extra.get("corpus") or "procedural12k")
    if basic_cache_path is None and train_config.get("basic_cache_root"):
        basic_cache_path = default_basic_cache_path(
            Path(str(train_config["basic_cache_root"])), sample, corpus
        )
    if pesto_cache_path is None and train_config.get("pesto_cache_root"):
        pesto_cache_path = default_pesto_cache_path(
            Path(str(train_config["pesto_cache_root"])), sample, corpus
        )
    basic = extract_basic_pitch_features(
        sample / "performance_audio.wav",
        source_metadata=metadata,
        cache_path=basic_cache_path,
    )
    pesto = extract_pesto_features(
        sample / "performance_audio.wav",
        basic,
        source_metadata=metadata,
        cache_path=pesto_cache_path,
        device=str(decoder.device),
    )
    outputs = decoder.model(
        torch.from_numpy(basic.note)[None].to(decoder.device),
        torch.from_numpy(basic.onset)[None].to(decoder.device),
        torch.from_numpy(basic.contour)[None].to(decoder.device),
        torch.from_numpy(pesto)[None].to(decoder.device),
    )
    return decode_refined_notes(
        {key: value.float().cpu() for key, value in outputs.items()},
        decoder.model.config,
    )
