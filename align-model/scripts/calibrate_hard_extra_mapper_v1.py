"""Calibrate a hard learned-EXTRA gate before global grammar mapping."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

import calibrate_ornament_mapper_v1 as calibrator
import eval_orn_phase2_baseline_v1 as baseline
import train_ornament_mapper_prior_v1 as prior
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.ornament_mapper_v1 import (
    OrnamentMapperCosts,
    decode_hard_extra_mapper,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


SCHEMA_VERSION = "align-hard-extra-mapper-calibration-v1"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def _evaluate(
    contexts: Sequence[Mapping[str, Any]],
    classifier: Any,
    costs: OrnamentMapperCosts,
    *,
    threshold: float,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    combined_samples = []
    oracle_samples = []
    for position, context in enumerate(contexts, 1):
        candidates, probability = prior._probabilities(
            classifier, context, costs, oracle=False
        )
        predicted, deletions, _diagnostics = decode_hard_extra_mapper(
            candidates,
            context["index"].events,
            probability >= threshold,
        )
        oracle_candidates, oracle_probability = prior._probabilities(
            classifier, context, costs, oracle=True
        )
        oracle, oracle_deletions, _oracle_diagnostics = (
            decode_hard_extra_mapper(
                oracle_candidates,
                context["index"].events,
                oracle_probability >= threshold,
            )
        )
        combined_samples.append(
            JointMetricSample(
                predicted=predicted,
                target=context["target"],
                source=context["source"],
                predicted_deletions=deletions,
                target_deletions=context["index"].deleted_event_indices,
                score_event_count=len(context["index"].events),
            )
        )
        oracle_samples.append(
            JointMetricSample(
                predicted=oracle,
                target=context["target"],
                source=context["source"],
                predicted_deletions=oracle_deletions,
                target_deletions=context["index"].deleted_event_indices,
                score_event_count=len(context["index"].events),
            )
        )
        if position == 1 or position % 10 == 0 or position == len(contexts):
            print(
                f"threshold={threshold:.2f} rows={position}/{len(contexts)}",
                flush=True,
            )
    return {
        "threshold": threshold,
        "combined": baseline._full_report(
            combined_samples, seed=seed, replicates=replicates
        ),
        "oracle_note_mapper": baseline._full_report(
            oracle_samples, seed=seed + 1, replicates=replicates
        ),
    }


def run_calibrate(args: argparse.Namespace) -> None:
    release = baseline._verify_release(args.release_manifest)
    baseline._assert_lockbox_sealed(release)
    prior_candidate = baseline._load_json(args.prior_candidate)
    model_path = Path(prior_candidate["extra_prior"])
    if sha256_file(model_path) != prior_candidate["extra_prior_sha256"]:
        raise ValueError("EXTRA-prior artifact mismatch")
    classifier = joblib.load(model_path)
    costs = OrnamentMapperCosts(**prior_candidate["costs"])
    predictions = (
        args.prior_candidate.parents[1]
        / "ornament-mapper-v2"
        / "calibration-trackb18-predictions.jsonl"
    )
    contexts = calibrator._contexts(release, "calibration", predictions)
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="hard-extra-mapper-v1-calibration",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(contexts), "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        variants = [
            _evaluate(
                contexts,
                classifier,
                costs,
                threshold=threshold,
                seed=args.seed + index * 100,
                replicates=args.bootstrap_replicates,
            )
            for index, threshold in enumerate(THRESHOLDS)
        ]
        selected = max(
            variants,
            key=lambda row: (
                float(row["combined"]["f1"]),
                float(row["oracle_note_mapper"]["f1"]),
                -abs(float(row["threshold"]) - 0.5),
            ),
        )
        report_path = args.output_dir / "calibration_report.json"
        baseline._atomic_json(
            report_path,
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "calibration_only",
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "prior_candidate_sha256": sha256_file(
                    args.prior_candidate
                ),
                "predeclared_thresholds": list(THRESHOLDS),
                "selected": selected,
                "variants": variants,
                "open_validation_read": False,
                "lockbox_targets_read": False,
            },
        )
        candidate_path = args.output_dir / "HARD_EXTRA_CANDIDATE.json"
        baseline._atomic_json(
            candidate_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-candidate",
                "frozen": True,
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "prior_candidate": str(args.prior_candidate.resolve()),
                "prior_candidate_sha256": sha256_file(
                    args.prior_candidate
                ),
                "threshold": selected["threshold"],
                "calibration_combined": selected["combined"],
                "calibration_oracle_note_mapper": selected[
                    "oracle_note_mapper"
                ],
                "calibration_report": str(report_path.resolve()),
                "calibration_report_sha256": sha256_file(report_path),
                "open_validation_read": False,
                "lockbox_targets_read": False,
            },
        )
        baseline._atomic_json(
            args.output_dir / "STATUS.json",
            {
                "schema_version": f"{SCHEMA_VERSION}-status",
                "status": "candidate_frozen_for_open_validation",
                "candidate": str(candidate_path.resolve()),
                "candidate_sha256": sha256_file(candidate_path),
                "calibration_combined_f1": selected["combined"]["f1"],
                "calibration_oracle_mapper_f1": selected[
                    "oracle_note_mapper"
                ]["f1"],
                "lockbox_opened": False,
            },
        )
    finally:
        baseline._assert_lockbox_sealed(release)
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=baseline.DEFAULT_RELEASE,
    )
    parser.add_argument(
        "--prior-candidate",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-generalization-v1/"
            "development/ornament-mapper-extra-prior-v2/"
            "EXTRA_PRIOR_CANDIDATE.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-generalization-v1/"
            "development/hard-extra-mapper-v1"
        ),
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=baseline.DEFAULT_RESOURCE_STATUS,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=baseline.DEFAULT_SEED)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    for name in (
        "release_manifest",
        "prior_candidate",
        "output_dir",
        "resource_status",
    ):
        value = getattr(args, name)
        setattr(args, name, value if value.is_absolute() else repo / value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_calibrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
