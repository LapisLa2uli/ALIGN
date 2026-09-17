from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from alignmodel.joint.grammar_mapper_v2 import decode_grammar_mapper
from alignmodel.joint.index import JointEvent
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset


def _event(row: dict) -> JointEvent:
    return JointEvent(
        pitch=int(row["pitch_written"]),
        start=float(row["start_sec"]),
        end=float(row["end_sec"]),
        score_span=(
            tuple(int(value) for value in row["score_span"])
            if row["score_span"] is not None
            else None
        ),
        relationship=str(row["relationship"]),
        copy_pass=int(row["copy_pass"]),
        origin_relationship=row.get("origin_relationship"),
        rendered_index=int(row["rendered_event_id"]),
        source_indices=tuple(int(value) for value in row["source_indices"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    with gzip.open(args.supervision, "rt", encoding="utf-8") as stream:
        targets = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }
    with args.predictions.open("r", encoding="utf-8") as stream:
        predictions = {
            row["sample"]: row["notes"]
            for row in (json.loads(line) for line in stream if line.strip())
        }
    samples = []
    per_clip = []
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
            candidates = tuple(
                JointCandidate(
                    int(row["pitch"]),
                    float(row["start"]),
                    float(row["end"]),
                    float(row["confidence"]),
                )
                for row in predictions[packed.sample]
            )
            target = tuple(_event(row) for row in targets[packed.sample]["events"])
            mapped, _grammar = decode_grammar_mapper(
                candidates, packed.training_example().score
            )
            sample = JointMetricSample(
                predicted=mapped,
                target=target,
                source=packed.source,
                score_event_count=len(packed.training_example().score),
            )
            samples.append(sample)
            metric = evaluate_joint_dataset([sample])["aggregate"][
                "official_note_wise"
            ]
            per_clip.append(
                (
                    float(metric["credit"]),
                    int(metric["predicted"]),
                    int(metric["gold"]),
                )
            )
    metric = evaluate_joint_dataset(samples)
    generator = np.random.default_rng(args.seed)
    bootstrap = []
    for _ in range(args.bootstrap_replicates):
        selected = generator.integers(0, len(per_clip), len(per_clip))
        credit = sum(per_clip[index][0] for index in selected)
        predicted = sum(per_clip[index][1] for index in selected)
        gold = sum(per_clip[index][2] for index in selected)
        precision = credit / max(predicted, 1)
        recall = credit / max(gold, 1)
        bootstrap.append(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    report = {
        "schema_version": "align-repaired-target-eval-v3",
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "split": "train",
        "rows": len(samples),
        "metrics": metric,
        "bootstrap_95": {
            "lower": float(np.quantile(bootstrap, 0.025)),
            "median": float(np.quantile(bootstrap, 0.5)),
            "upper": float(np.quantile(bootstrap, 0.975)),
            "replicates": args.bootstrap_replicates,
        },
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
