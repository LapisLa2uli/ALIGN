from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from alignmodel.joint.grammar_mapper_v2 import (
    GrammarCosts,
    decode_grammar_mapper,
    grammar_hypotheses,
    plan_features,
)
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.training_resources import resource_lease


class JointPlanPathRanker(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _path_features(
    plan: np.ndarray, mapped: tuple, score: tuple, costs: GrammarCosts
) -> np.ndarray:
    count = max(len(mapped), 1)
    relationships = [
        "copy" if event.is_copy else event.relationship for event in mapped
    ]
    linked = [event for event in mapped if event.score_span is not None]
    spans = [
        event.score_span[1] - event.score_span[0] for event in linked
    ]
    starts = [event.score_span[0] for event in linked]
    backward = sum(
        current < previous for previous, current in zip(starts, starts[1:])
    )
    return np.asarray(
        [
            *plan,
            *(relationships.count(name) / count for name in ("match", "copy", "substitute", "extra")),
            float(np.mean(spans)) if spans else 0.0,
            float(np.max(spans)) / max(len(score), 1) if spans else 0.0,
            backward / count,
            sum(event.copy_pass == 2 for event in mapped) / count,
            costs.substitution,
            costs.extra,
            costs.deletion,
            costs.repeat,
            costs.timing,
            costs.duration,
        ],
        dtype=np.float32,
    )


def _official(predicted: tuple, target: tuple, source: str, score_count: int):
    sample = JointMetricSample(
        predicted=predicted,
        target=target,
        source=source,
        score_event_count=score_count,
    )
    metric = evaluate_joint_dataset([sample])["aggregate"]["official_note_wise"]
    return sample, float(metric["f1"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    lease = resource_lease(
        args.resource_status,
        "heavy_cpu",
        track="mel-mapper-v3-joint-plan-path-ranker",
        command=[str(value) for value in __import__("sys").argv],
        metadata={"rows": args.rows, "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    ready = verify_data_ready(args.ready_marker)
    with args.predictions.open("r", encoding="utf-8") as stream:
        predictions = {
            row["sample"]: row["notes"]
            for row in (json.loads(line) for line in stream if line.strip())
        }
    groups = []
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
            plan_rows = [
                plan_features(candidates, example.score, hypothesis)
                for hypothesis in hypotheses
            ]
            selected_indices = sorted(
                range(len(hypotheses)),
                key=lambda index: plan_rows[index][10],
                reverse=True,
            )[: args.max_candidates]
            if 0 not in selected_indices:
                selected_indices[-1] = 0
            cost_variants = (
                GrammarCosts(),
                GrammarCosts(
                    substitution=1.5, extra=0.3, deletion=0.3, repeat=0.2
                ),
                GrammarCosts(substitution=2.1, extra=0.7, deletion=1.0),
                GrammarCosts(timing=0.15, duration=0.2),
            )
            feature_rows, utilities, samples = [], [], []
            baseline_candidate = None
            for index in sorted(set(selected_indices)):
                for variant_index, costs in enumerate(cost_variants):
                    mapped, _grammar = decode_grammar_mapper(
                        candidates,
                        example.score,
                        costs=costs,
                        forced_hypothesis=hypotheses[index],
                    )
                    sample, utility = _official(
                        mapped,
                        example.target_events,
                        packed.source,
                        len(example.score),
                    )
                    feature_rows.append(
                        _path_features(
                            plan_rows[index], mapped, example.score, costs
                        )
                    )
                    utilities.append(utility)
                    samples.append(sample)
                    if variant_index == 0 and (
                        baseline_candidate is None
                        or plan_rows[index][9]
                        < plan_rows[baseline_candidate[0]][9]
                    ):
                        baseline_candidate = (
                            index,
                            len(samples) - 1,
                        )
            groups.append(
                {
                    "features": np.stack(feature_rows),
                    "utilities": np.asarray(utilities, dtype=np.float32),
                    "samples": samples,
                    "heldout": position % 5 == 0,
                    "baseline": baseline_candidate[1],
                }
            )
            if position == 1 or position % 25 == 0:
                print(f"joint_candidates={position}/{len(ordinals)}", flush=True)
    train_groups = [group for group in groups if not group["heldout"]]
    heldout_groups = [group for group in groups if group["heldout"]]
    all_train = np.concatenate([group["features"] for group in train_groups])
    mean = all_train.mean(0)
    scale = all_train.std(0).clip(1e-4)
    torch.manual_seed(args.seed)
    model = JointPlanPathRanker(all_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        order = torch.randperm(len(train_groups)).tolist()
        for group_index in order:
            group = train_groups[group_index]
            features = torch.from_numpy(
                (group["features"] - mean) / scale
            ).float()
            utilities = torch.from_numpy(group["utilities"]).float()
            scores = model(features)
            target = int(torch.argmax(utilities))
            probabilities = scores.softmax(0)
            regret = torch.max(utilities) - utilities
            loss = -scores.log_softmax(0)[target] + 0.5 * torch.sum(
                probabilities * regret
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss)
        if epoch == 1 or epoch % 20 == 0:
            history.append({"epoch": epoch, "loss": total / len(train_groups)})
            print(json.dumps(history[-1]), flush=True)
    selected_samples = []
    oracle_samples = []
    baseline_samples = []
    plan_correct = 0
    for group in heldout_groups:
        features = torch.from_numpy(
            (group["features"] - mean) / scale
        ).float()
        with torch.inference_mode():
            selected = int(torch.argmax(model(features)))
        oracle = int(np.argmax(group["utilities"]))
        baseline = int(group["baseline"])
        plan_correct += int(selected == oracle)
        selected_samples.append(group["samples"][selected])
        oracle_samples.append(group["samples"][oracle])
        baseline_samples.append(group["samples"][baseline])
    selected_metric = evaluate_joint_dataset(selected_samples)["aggregate"][
        "official_note_wise"
    ]
    oracle_metric = evaluate_joint_dataset(oracle_samples)["aggregate"][
        "official_note_wise"
    ]
    baseline_metric = evaluate_joint_dataset(baseline_samples)["aggregate"][
        "official_note_wise"
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last_checkpoint.pt"
    _atomic(
        checkpoint,
        {
            "schema_version": "align-joint-plan-path-ranker-v3",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "feature_dim": int(all_train.shape[1]),
                "hidden_dim": 64,
            },
            "normalization": {"mean": mean, "scale": scale},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "rng_state": {"torch": torch.get_rng_state()},
            "cursor": {"epoch": args.epochs + 1, "position": 0},
            "data_fingerprint": ready["hashes"]["pack_id"],
            "history": history,
        },
    )
    report = {
        "schema_version": "align-joint-plan-path-ranker-report-v3",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "data_fingerprint": ready["hashes"]["pack_id"],
        "train_rows": len(train_groups),
        "heldout_rows": len(heldout_groups),
        "max_candidates": args.max_candidates,
        "candidate_selection": "top sequence-similarity legal plans plus no-repeat",
        "history": history,
        "heldout_candidate_accuracy": plan_correct / max(len(heldout_groups), 1),
        "heldout_canonical": selected_metric,
        "heldout_candidate_oracle": oracle_metric,
        "heldout_fixed_cost_baseline": baseline_metric,
        "heldout_absolute_gain": (
            float(selected_metric["f1"]) - float(baseline_metric["f1"])
        ),
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
