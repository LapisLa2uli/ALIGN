from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from alignmodel.joint.grammar_mapper_v2 import (
    grammar_hypotheses,
    plan_features,
)
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.training_resources import resource_lease


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gold_plan(
    document: dict, score: tuple
) -> tuple[int, tuple[int, int] | None]:
    plans = [
        plan for plan in document["replay_plans"] if plan["source_span"] is not None
    ]
    if not plans:
        return 0, None
    raw_start = min(int(plan["source_span"][0]) for plan in plans)
    raw_end = max(int(plan["source_span"][1]) for plan in plans)
    measures = {
        score[index].measure for index in range(raw_start, raw_end)
    }
    measure_indices = [
        index for index, event in enumerate(score) if event.measure in measures
    ]
    return (
        max(int(plan["copy_pass"]) for plan in plans),
        (min(measure_indices), max(measure_indices) + 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    lease = resource_lease(
        args.resource_status,
        "heavy_cpu",
        track="mel-mapper-v3-replay-plan-ranker",
        command=[str(value) for value in __import__("sys").argv],
        metadata={"rows": args.rows, "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    ready = verify_data_ready(args.ready_marker)
    with gzip.open(args.supervision, "rt", encoding="utf-8") as stream:
        supervision = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }
    with args.predictions.open("r", encoding="utf-8") as stream:
        predictions = {
            row["sample"]: row["notes"]
            for row in (json.loads(line) for line in stream if line.strip())
        }
    train_x, train_y, heldout_groups = [], [], []
    missing_exact = []
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
        )[: args.rows]
        for position, ordinal in enumerate(ordinals, 1):
            packed = dataset[ordinal]
            example = packed.training_example()
            candidates = tuple(
                JointCandidate(
                    int(row["pitch"]),
                    float(row["start"]),
                    float(row["end"]),
                    float(row["confidence"]),
                )
                for row in predictions[packed.sample]
            )
            hypotheses = grammar_hypotheses(example.score, len(candidates))
            gold_copies, gold_source = _gold_plan(
                supervision[packed.sample], example.score
            )
            features = np.stack(
                [
                    plan_features(candidates, example.score, hypothesis)
                    for hypothesis in hypotheses
                ]
            )
            labels = np.asarray(
                [
                    hypothesis.copies == gold_copies
                    and hypothesis.source_span == gold_source
                    for hypothesis in hypotheses
                ],
                dtype=np.int64,
            )
            if not labels.any():
                missing_exact.append(
                    {
                        "sample": packed.sample,
                        "copies": gold_copies,
                        "source_span": gold_source,
                    }
                )
                continue
            if position % 5 == 0:
                heldout_groups.append(
                    (features, labels, gold_copies, gold_source, hypotheses)
                )
            else:
                train_x.extend(features)
                train_y.extend(labels)
            if position == 1 or position % 25 == 0:
                print(f"plan_rows={position}/{len(ordinals)}", flush=True)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(np.stack(train_x), np.asarray(train_y))
    plan_correct = copy_correct = source_correct = resume_correct = 0
    for features, _labels, gold_copies, gold_source, hypotheses in heldout_groups:
        probability = model.predict_proba(features)[:, 1]
        selected = hypotheses[int(np.argmax(probability))]
        copy_correct += int(selected.copies == gold_copies)
        source_correct += int(selected.source_span == gold_source)
        plan_correct += int(
            selected.copies == gold_copies
            and selected.source_span == gold_source
        )
        resume_correct += int(
            (selected.source_span[1] if selected.source_span else -1)
            == (gold_source[1] if gold_source else -1)
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    joblib.dump(model, temporary)
    os.replace(temporary, output / "plan_ranker.joblib")
    count = len(heldout_groups)
    report = {
        "schema_version": "align-replay-plan-ranker-v3",
        "model": "plan_ranker.joblib",
        "model_sha256": _sha256(output / "plan_ranker.joblib"),
        "data_fingerprint": ready["hashes"]["pack_id"],
        "supervision_sha256": _sha256(args.supervision),
        "predictions_sha256": _sha256(args.predictions),
        "requested_rows": args.rows,
        "admitted_rows": args.rows - len(missing_exact),
        "missing_exact_hypothesis": missing_exact,
        "heldout_rows": count,
        "plan_accuracy": plan_correct / max(count, 1),
        "copy_count_accuracy": copy_correct / max(count, 1),
        "source_span_accuracy": source_correct / max(count, 1),
        "resume_accuracy": resume_correct / max(count, 1),
        "timestamp_metrics_used": False,
        "locked_test_touched": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    atexit.unregister(lease.__exit__)
    lease.__exit__(None, None, None)


if __name__ == "__main__":
    main()
