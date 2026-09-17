from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from alignmodel.joint.grammar_mapper_v2 import grammar_hypotheses
from alignmodel.joint.latent_crf_v4 import LatentSpanCRF, latent_crf_loss
from alignmodel.joint.lattice import JointCandidate
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
    model = LatentSpanCRF()
    admitted = Counter()
    excluded = []
    edge_counts = []
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
            if not plans:
                excluded.append(
                    {"sample": packed.sample, "reason": "gold plan absent"}
                )
                continue
            candidates = tuple(
                JointCandidate(
                    event.pitch, event.start, event.end, 1.0
                )
                for event in example.target_events
            )
            try:
                _loss, report = latent_crf_loss(
                    model,
                    candidates,
                    example.score,
                    plans[0],
                    example.target_events,
                    example.target_deletions,
                )
                admitted[str(copies)] += 1
                edge_counts.append(int(report["edges"]))
            except ValueError as error:
                excluded.append(
                    {"sample": packed.sample, "reason": str(error)}
                )
    report = {
        "schema_version": "align-latent-crf-coverage-audit-v4",
        "requested": args.limit,
        "admitted": sum(admitted.values()),
        "admitted_by_copy_count": dict(admitted),
        "excluded": excluded,
        "gold_path_coverage": sum(admitted.values()) / max(args.limit, 1),
        "mean_edges": sum(edge_counts) / max(len(edge_counts), 1),
        "extra_identity_representation": "rendered event transition at fixed event index",
        "timestamp_metrics_used": False,
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
