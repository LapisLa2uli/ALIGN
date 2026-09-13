"""Compare procedural intonation labels with cached acoustic PESTO pitch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignmodel.transcription.refiner_data import (
    load_cached_refiner_features,
    load_refiner_examples,
    load_rendered_target_notes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--basic-cache-root", type=Path, required=True)
    parser.add_argument("--pesto-cache-root", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for split in args.split or ["val", "test_id"]:
        examples = load_refiner_examples(
            args.manifest, split, procedural_only=True
        )
        if args.max_samples:
            examples = examples[: args.max_samples]
        for example in examples:
            basic, pesto = load_cached_refiner_features(
                example, args.basic_cache_root, args.pesto_cache_root
            )
            for pitch, start, end, cents in load_rendered_target_notes(example):
                if abs(cents) < 1e-6:
                    continue
                mask = (
                    (basic.frame_times >= start)
                    & (basic.frame_times < end)
                    & (pesto[:, 1] >= 0.8)
                    & (pesto[:, 0] > 0)
                )
                if not np.any(mask):
                    continue
                measured = 100.0 * float(
                    np.median(pesto[mask, 0] - int(pitch))
                )
                rows.append(
                    {
                        "split": split,
                        "sample": example.sample_id,
                        "pitch": pitch,
                        "start": start,
                        "end": end,
                        "label_cents": cents,
                        "measured_cents": measured,
                        "absolute_error": abs(measured - cents),
                    }
                )
    labels = np.asarray([row["label_cents"] for row in rows], np.float64)
    measured = np.asarray([row["measured_cents"] for row in rows], np.float64)
    correlation = (
        float(np.corrcoef(labels, measured)[0, 1])
        if len(rows) > 1 and np.std(labels) and np.std(measured)
        else None
    )
    summary = {
        "procedural_only": True,
        "n_measured": len(rows),
        "label_cents_median_abs": (
            float(np.median(np.abs(labels))) if len(rows) else None
        ),
        "measured_cents_median_abs": (
            float(np.median(np.abs(measured))) if len(rows) else None
        ),
        "cents_mae": (
            float(np.mean(np.abs(measured - labels))) if len(rows) else None
        ),
        "correlation": correlation,
        "measured_within_20_cents_of_zero_fraction": (
            float(np.mean(np.abs(measured) <= 20.0)) if len(rows) else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
