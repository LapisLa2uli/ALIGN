"""Build a train/val mel cache for dataset 9.2 with forced-aligned targets.

Targets keep the gold written-pitch order from note_map rendered_notes but take
their frame positions from forced Viterbi alignment on a frozen mel
transcriber's posteriors, because note_map timestamps do not match the Muse
Sounds audio. Degraded (9.2) audio is always cached; the clean Muse Sounds
render of each train clip can be added with the same targets, since the
degradation is time-aligned. Test clips are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.forced_align_v1 import ForcedAlignConfig, forced_align
from alignmodel.transcription.mel_v1 import (
    CACHE_SCHEMA_VERSION,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from alignmodel.transcription.mel_v1_data import (
    _atomic_json,
    _canonical_json,
    _new_index,
    _sha256_bytes,
)


SCHEMA_VERSION = "align-realistic92-aligned-cache-v1"


def _align_job(payload: tuple) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probabilities, pitches, midi_min, hop_sec, config = payload
    aligned, stats = forced_align(
        probabilities, pitches, midi_min=midi_min, config=config
    )
    targets = [
        {
            "pitch": note.pitch,
            "start_sec": round(note.start_frame * hop_sec, 6),
            "end_sec": round(note.end_frame * hop_sec, 6),
            "gold_index": note.index,
        }
        for note in aligned
    ]
    return targets, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--aligner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--max-skip-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--onset-weight", type=float, default=0.6)
    parser.add_argument("--use-boundary-heads", action="store_true")
    parser.add_argument("--aligner-kind", choices=("mel_v1", "ctc_base"), default="mel_v1")
    parser.add_argument("--dual-mel", action="store_true",
                        help="store stacked long+short log-mels (v3 input) instead of the long mel")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.split) != args.expected_split_sha256:
        raise ValueError("Frozen split mismatch")
    split = json.loads(args.split.read_text(encoding="utf-8"))["splits"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.aligner_kind == "ctc_base":
        from alignmodel.transcription.mel_ctc_v1 import load_ctc_checkpoint
        ctc_model, frontend, _payload = load_ctc_checkpoint(args.aligner_checkpoint, device)
        model = ctc_model.base
    else:
        model, frontend, _decode, _payload = load_mel_checkpoint(
            args.aligner_checkpoint, device
        )
    if args.dual_mel:
        from alignmodel.transcription.mel_ctc_v3 import dual_frontend_metadata, extract_dual_mel
    config = ForcedAlignConfig(
        onset_weight=args.onset_weight,
        use_boundary_heads=args.use_boundary_heads,
    )
    rows = [(name, "train") for name in split["train"]]
    rows += [(name, "val") for name in split["val"]]
    if args.limit:
        rows = rows[:args.limit]

    staging = args.output.with_name(f".{args.output.name}.building")
    staging.mkdir(parents=True, exist_ok=False)
    index_path = staging / "index.sqlite"
    connection = _new_index(index_path)
    stats_path = staging / "alignment_stats.jsonl"
    stats_stream = stats_path.open("w", encoding="utf-8")
    pool = ProcessPoolExecutor(max_workers=args.workers)

    pending: list[tuple[str, str, Any, np.ndarray, dict, Any]] = []
    shard_arrays: list[np.ndarray] = []
    shard_records: list[tuple] = []
    shard = 0
    frame_offset = 0
    written = excluded = 0
    started = time.perf_counter()

    def mel_for(path: Path) -> tuple[np.ndarray, dict]:
        audio = load_audio_mono(path, frontend.sample_rate)
        mel, normalization = extract_log_mel(audio, frontend, device=device)
        return np.asarray(mel, np.float32), normalization

    def stored_mel(path: Path, long_mel: np.ndarray, long_norm: dict) -> tuple[np.ndarray, dict]:
        if not args.dual_mel:
            return long_mel, long_norm
        audio = load_audio_mono(path, frontend.sample_rate)
        stacked, normalization = extract_dual_mel(audio, device)
        return np.asarray(stacked, np.float32), normalization

    def add_record(sample: str, split_name: str, source: str, render: str,
                   audio_sha: str, mel: np.ndarray, normalization: dict,
                   targets: list[dict[str, Any]]) -> None:
        nonlocal frame_offset
        contiguous = np.ascontiguousarray(mel.T, dtype="<f2")
        shard_arrays.append(contiguous)
        shard_records.append((
            sample, split_name, source, shard, frame_offset,
            int(contiguous.shape[0]), float(mel.shape[1] * frontend.hop_sec),
            render, 2, audio_sha, _sha256_bytes(contiguous.tobytes(order="C")),
            json.dumps(normalization, sort_keys=True),
            sqlite3.Binary(zlib.compress(_canonical_json(targets), level=1)),
        ))
        frame_offset += int(contiguous.shape[0])

    def flush_shard() -> None:
        nonlocal shard, frame_offset
        if not shard_records:
            return
        path = staging / f"shard-{shard:05d}.mel.float16.bin"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            for array in shard_arrays:
                stream.write(array.tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        connection.executemany(
            "INSERT INTO records(sample,split,source,shard,frame_offset,"
            "frame_count,duration_sec,audio_render,effective_audio_transpose,"
            "audio_sha256,mel_sha256,normalization,target) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            shard_records,
        )
        connection.commit()
        shard_arrays.clear()
        shard_records.clear()
        shard += 1
        frame_offset = 0

    def drain(block_all: bool) -> None:
        nonlocal written, excluded
        while pending and (block_all or len(pending) > args.workers * 2):
            name, split_name, future, mel, normalization, extra = pending.pop(0)
            targets, stats = future.result()
            stats.update({"sample": name, "split": split_name})
            skip_fraction = stats["skipped"] / max(stats["notes"], 1)
            keep = skip_fraction <= args.max_skip_fraction and targets
            stats["kept"] = bool(keep)
            stats_stream.write(json.dumps(stats) + "\n")
            if not keep:
                excluded += 1
                continue
            source = "procedural" if name.startswith("synth_gen_") else "rawdata"
            add_record(name, split_name, source, "musesounds_v1+realistic_v1",
                       extra["audio_sha"], mel, normalization, targets)
            written += 1
            if extra.get("clean") is not None:
                clean_mel, clean_norm, clean_sha = extra["clean"]
                add_record(f"{name}#clean", split_name, source, "musesounds_v1",
                           clean_sha, clean_mel, clean_norm, targets)
                written += 1
            if len(shard_records) >= args.shard_rows:
                flush_shard()

    try:
        for position, (name, split_name) in enumerate(rows, 1):
            sample = args.root / name
            gold = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))[
                "rendered_notes"
            ]
            pitches = [int(row["pitch_midi_written"]) for row in gold]
            audio_path = sample / "performance_audio.wav"
            mel, normalization = mel_for(audio_path)
            probabilities = infer_mel_probabilities(
                model, mel, device, window_frames=2048, overlap_frames=512,
                batch_size=4,
            )
            extra: dict[str, Any] = {"audio_sha": sha256_file(audio_path)}
            mel, normalization = stored_mel(audio_path, mel, normalization)
            clean_path = sample / "performance_audio_clean.wav"
            if args.include_clean and split_name == "train" and clean_path.is_file():
                if args.dual_mel:
                    clean_mel, clean_norm = stored_mel(clean_path, None, None)
                else:
                    clean_mel, clean_norm = mel_for(clean_path)
                extra["clean"] = (clean_mel, clean_norm, sha256_file(clean_path))
            future = pool.submit(_align_job, (
                {key: probabilities[key] for key in
                 ("voiced", "pitch", "onset", "boundary", "rearticulation")},
                pitches, model.config.midi_min, frontend.hop_sec, config,
            ))
            pending.append((name, split_name, future, mel, normalization, extra))
            drain(block_all=False)
            if position == 1 or position % 100 == 0 or position == len(rows):
                rate = position / max(time.perf_counter() - started, 1e-9)
                print(
                    f"rows={position}/{len(rows)} written={written} "
                    f"excluded={excluded} rows_per_sec={rate:.2f} "
                    f"eta_min={(len(rows) - position) / max(rate, 1e-9) / 60:.1f}",
                    flush=True,
                )
        drain(block_all=True)
        flush_shard()
    finally:
        pool.shutdown(wait=True)
        stats_stream.close()
    connection.execute("PRAGMA optimize")
    connection.commit()
    counts = dict(connection.execute(
        "SELECT split, COUNT(*) FROM records GROUP BY split"
    ).fetchall())
    connection.close()

    shards = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(staging.glob("shard-*.bin"))
    ]
    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "frontend_config": dual_frontend_metadata() if args.dual_mel else frontend.to_dict(),
        "dtype": "float16",
        "layout": "time_major_contiguous",
        "shard_rows": args.shard_rows,
        "split_counts": {"train": int(counts.get("train", 0)), "val": int(counts.get("val", 0))},
        "record_count": int(sum(counts.values())),
        "source": {
            "release": "realistic92-transcriber-v1",
            "cache_builder": SCHEMA_VERSION,
            "split_sha256": args.expected_split_sha256,
            "pack_id": hashlib.sha256(
                f"{args.expected_split_sha256}:{sha256_file(args.aligner_checkpoint)}".encode()
            ).hexdigest(),
            "aligner_checkpoint_sha256": sha256_file(args.aligner_checkpoint),
            "forced_align_config": config.__dict__,
            "aligner_kind": args.aligner_kind,
            "dual_mel": bool(args.dual_mel),
            "include_clean": bool(args.include_clean),
            "excluded_clips": excluded,
            "locked_test_materialized": False,
            "targets_included": True,
        },
        "index": {
            "name": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
        },
        "shards": shards,
    }
    metadata["pack_id"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
    _atomic_json(staging / "metadata.json", metadata)
    os.replace(staging, args.output)
    print(json.dumps({
        "output": str(args.output),
        "split_counts": metadata["split_counts"],
        "excluded_clips": excluded,
        "seconds": time.perf_counter() - started,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
