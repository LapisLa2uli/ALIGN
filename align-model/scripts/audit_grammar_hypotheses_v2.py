from __future__ import annotations

import argparse
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path

from alignmodel.joint.grammar_mapper_v2 import grammar_hypotheses
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    manifest = json.loads(
        Path(str(ready["paths"]["manifest"])).read_text(encoding="utf-8")
    )
    source_root = Path(str(manifest["source_root"]))
    rows = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        ordinals = sorted(
            dataset.ordinals("train"),
            key=lambda value: hashlib.sha256(
                f"{args.seed}:{value}".encode()
            ).digest(),
        )[: args.limit]
        for ordinal in ordinals:
            packed = dataset[ordinal]
            example = packed.training_example()
            score = ScoreEventIndex.from_musicxml(
                source_root / packed.sample / "verified_score.musicxml"
            ).events
            target = tuple(
                (event.score_span[-1] - 1, event.copy_pass)
                for event in example.target_events
                if event.score_span is not None
            )
            hypotheses = grammar_hypotheses(score, len(example.target_events))
            coverage, best = max(
                (
                    (
                        SequenceMatcher(
                            None, target, hypothesis.units
                        ).ratio(),
                        hypothesis,
                    )
                    for hypothesis in hypotheses
                ),
                key=lambda row: row[0],
            )
            rows.append(
                {
                    "sample": packed.sample,
                    "target_copy_count": max(
                        (event.copy_pass for event in example.target_events),
                        default=0,
                    ),
                    "best_copy_count": best.copies,
                    "best_source_span": best.source_span,
                    "unit_sequence_ratio": coverage,
                    "hypotheses": len(hypotheses),
                }
            )
    report = {
        "schema_version": "align-grammar-hypothesis-audit-v1",
        "rows": len(rows),
        "mean_unit_sequence_ratio": sum(
            row["unit_sequence_ratio"] for row in rows
        )
        / max(len(rows), 1),
        "copy_count_accuracy": sum(
            row["target_copy_count"] == row["best_copy_count"] for row in rows
        )
        / max(len(rows), 1),
        "exact_unit_sequences": sum(
            row["unit_sequence_ratio"] == 1.0 for row in rows
        ),
        "samples": rows,
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
