from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.joint import OracleHarnessConfig, run_oracle_harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CPU-only joint oracle controls on an explicit non-test split."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        help=(
            "Optional decoded-note or validated raw Basic Pitch cache root. "
            "No candidate cache is discovered implicitly."
        ),
    )
    parser.add_argument(
        "--note-map-root",
        type=Path,
        help="Optional exact-lineage root; defaults to each manifest sample_dir.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-pairing-ms",
        type=float,
        default=50.0,
        help="Oracle acoustic-ceiling pairing tolerance (default: 50).",
    )
    parser.add_argument(
        "--minimum-candidate-confidence",
        type=float,
        default=0.0,
    )
    parser.add_argument("--repetition-min-notes", type=int, default=6)
    parser.add_argument("--repetition-max-notes", type=int, default=64)
    parser.add_argument("--repetition-min-confidence", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_oracle_harness(
        OracleHarnessConfig(
            manifest=args.manifest,
            split=args.split,
            output=args.output,
            candidate_root=args.candidate_root,
            note_map_root=args.note_map_root,
            limit=args.limit,
            candidate_pairing_tolerance_sec=args.candidate_pairing_ms / 1000.0,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
            repetition_min_notes=args.repetition_min_notes,
            repetition_max_notes=args.repetition_max_notes,
            repetition_min_confidence=args.repetition_min_confidence,
        )
    )
    summary = {
        "output": str(args.output.resolve()),
        "split": result["split"],
        "n_samples": result["n_samples"],
        "stages": sorted(result["stages"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
