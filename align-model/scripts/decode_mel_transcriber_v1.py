"""Gold-isolated Track B inference from cached high-resolution mels."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    infer_mel_probabilities,
    load_mel_checkpoint,
)
from alignmodel.transcription.mel_v1_data import MelPackedCache
from alignmodel.training_resources import resource_lease


FORBIDDEN_INFERENCE_NAMES = frozenset({
    "labels.json",
    "note_map.json",
    "performance_audio.mid",
    "performance_score.musicxml",
    "verified_score.musicxml",
    "canonical_dev_targets.sqlite",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=2048)
    parser.add_argument("--overlap-frames", type=int, default=512)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--resource-status", type=Path)
    parser.add_argument("--allow-authorized-lockbox-cache", action="store_true")
    args = parser.parse_args()
    lease = None
    if args.resource_status is not None:
        lease = resource_lease(
            args.resource_status,
            "gpu",
            track=f"mel-transcriber-v1-decode-{args.split}",
            command=[__file__, *map(str, vars(args).values())],
            metadata={"split": args.split, "locked_test_materialized": False},
        )
        lease.__enter__()
        atexit.register(lease.__exit__, None, None, None)
    device = torch.device(args.device)
    model, frontend, decode, payload = load_mel_checkpoint(
        args.checkpoint, device
    )
    if args.min_confidence is not None:
        if not 0.0 <= args.min_confidence <= 1.0:
            raise ValueError("--min-confidence must be within [0, 1]")
        decode = replace(decode, min_confidence=args.min_confidence)
    cache = MelPackedCache(args.cache, deep=False)
    if cache.pack_id != payload["data"]["cache_pack_id"]:
        authorized_test = (
            args.split == "test"
            and args.allow_authorized_lockbox_cache
            and cache.metadata.get("source", {}).get("locked_test_materialized")
            is True
            and cache.metadata.get("source", {}).get("pack_id")
            == payload["data"]["release_pack_id"]
        )
        if not authorized_test:
            raise ValueError("Checkpoint/cache fingerprint mismatch")
    if cache.frontend != frontend:
        raise ValueError("Checkpoint/cache frontend mismatch")
    # This SQL projection intentionally omits the target BLOB.
    records = cache.records(args.split, include_targets=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent,
        suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        started = time.perf_counter()
        for position, record in enumerate(records, 1):
            mel = np.asarray(cache.mel(record), np.float32)
            probabilities = infer_mel_probabilities(
                model,
                mel,
                device,
                window_frames=args.window_frames,
                overlap_frames=args.overlap_frames,
                batch_size=args.batch_size,
            )
            notes = decode_mel_notes(
                probabilities,
                midi_min=model.config.midi_min,
                hop_sec=frontend.hop_sec,
                config=decode,
            )
            stream.write(json.dumps({
                "sample": record.sample,
                "source": record.source,
                "split": record.split,
                "effective_audio_transpose": record.effective_audio_transpose,
                "notes": [note.to_dict() for note in notes],
            }, sort_keys=True))
            stream.write("\n")
            if position == 1 or position % 25 == 0 or position == len(records):
                elapsed = time.perf_counter() - started
                rate = position / max(elapsed, 1e-9)
                print(
                    f"decode={position}/{len(records)} rows_per_sec={rate:.2f} "
                    f"eta_sec={(len(records)-position)/max(rate,1e-9):.1f}",
                    flush=True,
                )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    cache.close()
    manifest = {
        "schema_version": "align-mel-transcriber-freeze-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "cache_pack_id": payload["data"]["cache_pack_id"],
        "release_pack_id": payload["data"]["release_pack_id"],
        "predictions": str(args.output.resolve()),
        "predictions_sha256": _sha256(args.output),
        "rows": len(records),
        "split": args.split,
        "inference_inputs": ["packed log-mel"],
        "pitch_output_space": "written",
        "pitch_policy": (
            "written pitch learned from sounding-audio supervision; "
            "transpose metadata audited but not consumed by the model"
        ),
        "effective_audio_transpose_consumed": False,
        "effective_audio_transpose_values": sorted({
            int(record.effective_audio_transpose) for record in records
        }),
        "decode_config": decode.to_dict(),
        "decode_override": {
            "min_confidence": args.min_confidence,
        },
        "score_input": False,
        "target_column_read": False,
        "forbidden_files_opened": [],
        "forbidden_basenames": sorted(FORBIDDEN_INFERENCE_NAMES),
        "basic_pitch_dependency": False,
        "locked_test_materialized": False,
    }
    manifest_path = args.output.with_name("freeze_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if lease is not None:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


if __name__ == "__main__":
    main()
