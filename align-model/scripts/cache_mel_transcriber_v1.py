"""Build the checksum-verified train/val-only Track B mel cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import MelFrontendConfig
from alignmodel.transcription.mel_v1_data import MelPackedCache, build_mel_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--hop-length", type=int, choices=(128, 256), default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--split", choices=("train_val", "test"), default="train_val")
    parser.add_argument("--allow-locked-test", action="store_true")
    args = parser.parse_args()
    frontend = MelFrontendConfig(hop_length=args.hop_length)
    started = time.perf_counter()
    last = [started, 0]

    def progress(done: int, total: int, sample: str) -> None:
        now = time.perf_counter()
        if done == 1 or done == total or done % 25 == 0:
            rate = done / max(now - started, 1e-9)
            print(
                f"cache={done}/{total} rows_per_sec={rate:.2f} "
                f"eta_sec={(total-done)/max(rate,1e-9):.1f} sample={sample}",
                flush=True,
            )
        last[:] = [now, done]

    with resource_lease(
        args.resource_status,
        "gpu",
        track=f"mel-transcriber-v1-cache-hop{args.hop_length}",
        command=["cache_mel_transcriber_v1.py", *map(str, vars(args).values())],
        metadata={
            "ready_marker": str(args.ready_marker.resolve()),
            "hop_length": args.hop_length,
            "lockbox_materialization": args.split == "test",
            "split": args.split,
        },
    ):
        output = build_mel_cache(
            args.ready_marker,
            args.output,
            frontend,
            device=args.device,
            shard_rows=args.shard_rows,
            progress_callback=progress,
            splits=(("test",) if args.split == "test" else ("train", "val")),
            include_targets=args.split != "test",
            allow_locked_test=args.allow_locked_test,
        )
    elapsed = time.perf_counter() - started
    cache = MelPackedCache(output, deep=False)
    report = {
        "schema_version": "align-mel-cache-build-report-v1",
        "output": str(output),
        "pack_id": cache.pack_id,
        "frontend": frontend.to_dict(),
        "records": cache.metadata["record_count"],
        "split_counts": cache.metadata["split_counts"],
        "seconds": elapsed,
        "rows_per_second": cache.metadata["record_count"] / max(elapsed, 1e-9),
        "bytes": sum(int(row["bytes"]) for row in cache.metadata["shards"]),
        "locked_test_materialized": args.split == "test",
    }
    cache.close()
    report_path = args.output.parent / f"cache-hop{args.hop_length}-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
