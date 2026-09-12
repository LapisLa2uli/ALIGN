"""Cache PESTO written-pitch features for a strict procedural manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alignmodel.transcription.basic_pitch import (
    basic_pitch_cache_path,
    load_audio_metadata,
    load_basic_pitch_cache,
)
from alignmodel.transcription.fine_pitch import (
    extract_pesto_features,
    pesto_cache_path,
)


def _is_raw(row: dict[str, Any]) -> bool:
    return str(row.get("corpus") or row.get("root") or "").lower() == "raw2k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--basic-cache-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--procedural-only", action="store_true")
    args = parser.parse_args()

    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = args.split or ["train", "val", "test_id"]
    summary: dict[str, Any] = {"splits": {}, "failures": []}
    for split in splits:
        source_rows = list(document.get(split) or [])
        if args.procedural_only and any(_is_raw(dict(row)) for row in source_rows):
            raise ValueError(
                f"--procedural-only rejected raw2k row in {split}"
            )
        if args.max_samples:
            source_rows = source_rows[: args.max_samples]
        completed = 0
        for index, raw in enumerate(source_rows, 1):
            row = dict(raw)
            sample = Path(row["sample_dir"])
            corpus = str(row.get("corpus") or row.get("root") or "unknown")
            wav = sample / "performance_audio.wav"
            metadata = load_audio_metadata(sample)
            metadata.update(
                {
                    key: row[key]
                    for key in (
                        "audio_pitch_space",
                        "effective_audio_transpose",
                    )
                    if key in row
                }
            )
            basic_path = basic_pitch_cache_path(
                args.basic_cache_root, sample, corpus
            )
            basic = load_basic_pitch_cache(basic_path, wav, metadata)
            if basic is None:
                summary["failures"].append(
                    {
                        "split": split,
                        "sample": sample.name,
                        "error": "missing_or_stale_basic_pitch_cache",
                    }
                )
                continue
            try:
                extract_pesto_features(
                    wav,
                    basic,
                    source_metadata=metadata,
                    cache_path=pesto_cache_path(
                        args.cache_root, sample, corpus
                    ),
                    device=args.device,
                )
                completed += 1
            except Exception as exc:  # noqa: BLE001
                summary["failures"].append(
                    {
                        "split": split,
                        "sample": sample.name,
                        "error": str(exc),
                    }
                )
            if index == 1 or index % 50 == 0 or index == len(source_rows):
                print(
                    f"{split} {index}/{len(source_rows)} cached={completed}",
                    flush=True,
                )
        summary["splits"][split] = {
            "selected": len(source_rows),
            "cached": completed,
        }
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failures"]:
        raise RuntimeError(
            f"PESTO caching failed for {len(summary['failures'])} rows"
        )


if __name__ == "__main__":
    main()
