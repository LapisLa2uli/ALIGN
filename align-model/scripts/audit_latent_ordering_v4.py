from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from alignmodel.joint.grammar_mapper_v2 import grammar_hypotheses
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from train_replay_plan_ranker_v3 import _gold_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    with gzip.open(args.supervision, "rt", encoding="utf-8") as stream:
        supervision = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }
    reasons = Counter()
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
            copies, source = _gold_plan(
                supervision[packed.sample], example.score
            )
            plans = [
                plan
                for plan in grammar_hypotheses(
                    example.score, len(example.target_events)
                )
                if plan.copies == copies and plan.source_span == source
            ]
            row_reasons = Counter()
            if not plans:
                row_reasons["gold_plan_absent"] += 1
            else:
                plan = plans[0]
                positions = {}
                for position, identity in enumerate(plan.units):
                    positions.setdefault(identity, []).append(position)
                cursor = -1
                for event in example.target_events:
                    if event.score_span is None:
                        continue
                    width = event.score_span[1] - event.score_span[0]
                    if width > 8:
                        row_reasons["span_width_gt_8"] += 1
                    identity = (event.score_span[-1] - 1, event.copy_pass)
                    candidates = positions.get(identity, [])
                    following = [value for value in candidates if value >= cursor]
                    if following:
                        cursor = following[0]
                    elif candidates:
                        row_reasons["within_pass_nonmonotonic"] += 1
                        cursor = candidates[0]
                    else:
                        row_reasons["identity_absent_from_plan"] += 1
            reasons.update(row_reasons)
            rows.append(
                {
                    "sample": packed.sample,
                    "reasons": dict(row_reasons),
                }
            )
    report = {
        "schema_version": "align-latent-ordering-audit-v4",
        "rows": len(rows),
        "rows_without_structural_violation": sum(
            not row["reasons"] for row in rows
        ),
        "reason_counts": dict(reasons),
        "samples": rows,
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
