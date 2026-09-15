from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from alignmodel.joint.data import build_inference_example
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.lattice import SparseJointLattice
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.train import load_joint_model
from alignmodel.validated_targets import target_note_map


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _bootstrap(
    samples: list[JointMetricSample],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int]:
    counts = []
    for sample in samples:
        row = evaluate_joint_dataset([sample])["aggregate"]["tolerances"][
            "50ms"
        ]["counts"]
        counts.append(
            (
                int(row["joint_correct"]),
                int(row["predicted"]),
                int(row["target"]),
            )
        )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.integers(0, len(counts), len(counts))
        correct = sum(counts[index][0] for index in selected)
        predicted = sum(counts[index][1] for index in selected)
        target = sum(counts[index][2] for index in selected)
        precision = correct / max(predicted, 1)
        recall = correct / max(target, 1)
        values.append(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen sparse joint decoder without gold inference inputs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--noise-inference-bias", type=float)
    parser.add_argument("--minimum-candidate-confidence", type=float)
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help="Required for the single final test_id evaluation.",
    )
    args = parser.parse_args()

    protected = "test" in args.split.lower() or "sealed" in args.split.lower()
    if protected and not args.allow_locked_test:
        raise ValueError("Protected split requires --allow-locked-test")
    if args.allow_locked_test and not protected:
        raise ValueError("--allow-locked-test is valid only for a protected split")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest.get(args.split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No non-empty manifest split {args.split!r}")
    if args.limit is not None:
        if protected:
            raise ValueError("A locked test cannot be partially inspected")
        rows = rows[: args.limit]

    model, lattice_config, checkpoint_payload = load_joint_model(
        args.checkpoint, device=args.device
    )
    if args.noise_inference_bias is not None:
        lattice_config = replace(
            lattice_config,
            noise_inference_bias=args.noise_inference_bias,
        )
    manifest_hash = _sha256(args.manifest)
    if checkpoint_payload.get("manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint was trained against a different manifest")
    lattice = SparseJointLattice(model, lattice_config)
    minimum_candidate_confidence = float(
        args.minimum_candidate_confidence
        if args.minimum_candidate_confidence is not None
        else checkpoint_payload.get("training", {}).get(
            "minimum_candidate_confidence", 0.0
        )
    )
    samples: list[JointMetricSample] = []
    for position, row in enumerate(rows, 1):
        # Complete inference before loading any evaluation target.
        inference = build_inference_example(
            row,
            args.cache_root,
            minimum_candidate_confidence=minimum_candidate_confidence,
        )
        path = lattice.decode(inference.candidates, inference.score)
        predicted = path.joint_events(inference.candidates)
        predicted_deletions = set(path.trailing_deletions)
        for step in path.steps:
            predicted_deletions.update(step.deleted_events)

        lineage = target_note_map(row)
        target_index = ScoreEventIndex.from_musicxml(
            Path(str(row["sample_dir"])) / "verified_score.musicxml",
            lineage,
        )
        if target_index.events != inference.score:
            raise ValueError(f"Gold altered inference score index for {inference.sample}")
        samples.append(
            JointMetricSample(
                predicted=predicted,
                target=target_index.rendered_events,
                source=inference.source,
                predicted_deletions=frozenset(predicted_deletions),
                target_deletions=target_index.deleted_event_indices,
                score_event_count=len(inference.score),
            )
        )
        if position == 1 or position % 50 == 0 or position == len(rows):
            print(f"evaluated {position}/{len(rows)}", flush=True)

    report = {
        "schema_version": "align-joint-evaluation-v1",
        "split": args.split,
        "locked_evaluation": protected,
        "n_samples": len(samples),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "cache_root": str(args.cache_root.resolve()),
        "minimum_candidate_confidence": minimum_candidate_confidence,
        "noise_inference_bias": lattice_config.noise_inference_bias,
        "inference_inputs": [
            "performance_audio.wav-derived Basic Pitch activation cache",
            "verified_score.musicxml",
        ],
        "inference_forbidden_inputs": [
            "labels.json",
            "note_map.json",
            "performance_audio.mid",
            "performance_score.musicxml",
            "validated target sidecars",
        ],
        "metrics": evaluate_joint_dataset(samples),
        "bootstrap_joint_f1_50ms": _bootstrap(
            samples,
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        ),
    }
    _write_json(args.output, report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
