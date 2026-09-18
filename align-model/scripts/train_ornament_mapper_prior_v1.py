"""Train a target-free mapper EXTRA prior on ORN train and calibrate on 53 rows."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import calibrate_ornament_mapper_v1 as calibrator
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.ornament_mapper_v1 import (
    OrnamentMapperCosts,
    decode_ornament_mapper,
    ornament_mapper_features,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


SCHEMA_VERSION = "align-ornament-mapper-extra-prior-v1"
EXTRA_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def _train_contexts(
    release: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest_rows = {
        row["sample"]: row
        for row in release["splits"]["development"]["train"]
    }
    targets = baseline._load_targets(release, "train")
    if set(manifest_rows) != set(targets):
        raise ValueError("ORN train target population mismatch")
    output = []
    for sample in sorted(manifest_rows):
        row = manifest_rows[sample]
        target_row = targets[sample]
        lineage = target_row["lineage"]
        if (
            hashlib.sha256(baseline._canonical_bytes(lineage)).hexdigest()
            != row["target_lineage_sha256"]
        ):
            raise ValueError(f"ORN train target hash mismatch: {sample}")
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        if (
            sha256_file(score_path)
            != row["source_hashes"]["verified_score.musicxml"]
        ):
            raise ValueError(f"ORN train score changed: {sample}")
        index = ScoreEventIndex.from_musicxml(score_path, lineage)
        output.append(
            {
                "sample": sample,
                "source": row["leakage_group"],
                "score_path": score_path,
                "index": index,
                "target": index.rendered_events,
                "candidate": baseline._oracle_candidates(
                    index.rendered_events
                ),
                "provenance": row["provenance"],
            }
        )
    return output


def _extra_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = probabilities >= 0.5
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "rows": int(len(labels)),
        "positive_support": int(labels.sum()),
        "predicted_positive": int(predicted.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(set(labels.tolist())) > 1
            else None
        ),
        "support": int(support) if support is not None else None,
    }


def _fit_prior(
    contexts: Sequence[Mapping[str, Any]],
    costs: OrnamentMapperCosts,
    *,
    seed: int,
) -> tuple[ExtraTreesClassifier, dict[str, Any]]:
    features = []
    labels = []
    provenance = []
    for position, context in enumerate(contexts, 1):
        mapped, _deletions, _diagnostics = decode_ornament_mapper(
            context["candidate"],
            context["index"].events,
            context["score_path"],
            costs=costs,
        )
        features.extend(
            ornament_mapper_features(
                context["candidate"],
                mapped,
                context["index"].events,
                context["score_path"],
            )
        )
        labels.extend(event.is_extra for event in context["target"])
        provenance.extend(
            [context["provenance"]] * len(context["target"])
        )
        if position == 1 or position % 25 == 0 or position == len(contexts):
            print(f"prior-features={position}/{len(contexts)}", flush=True)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    classifier = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    classifier.fit(x, y)
    probability = classifier.predict_proba(x)[:, 1]
    by_provenance = {}
    provenance_values = np.asarray(provenance)
    for name in sorted(set(provenance)):
        chosen = provenance_values == name
        by_provenance[name] = _extra_metrics(
            y[chosen], probability[chosen]
        )
    return classifier, {
        "examples": int(len(y)),
        "features": int(x.shape[1]),
        "metrics": _extra_metrics(y, probability),
        "by_provenance": by_provenance,
    }


def _probabilities(
    classifier: ExtraTreesClassifier,
    context: Mapping[str, Any],
    costs: OrnamentMapperCosts,
    *,
    oracle: bool,
) -> tuple[Sequence[Any], np.ndarray]:
    candidates = (
        baseline._oracle_candidates(context["target"])
        if oracle
        else context["candidate"]
    )
    initial, _deletions, _diagnostics = decode_ornament_mapper(
        candidates,
        context["index"].events,
        context["score_path"],
        costs=costs,
    )
    features = ornament_mapper_features(
        candidates,
        initial,
        context["index"].events,
        context["score_path"],
    )
    probability = classifier.predict_proba(
        np.asarray(features, dtype=np.float32)
    )[:, 1]
    return candidates, probability


def _evaluate_weight(
    contexts: Sequence[Mapping[str, Any]],
    classifier: ExtraTreesClassifier,
    costs: OrnamentMapperCosts,
    *,
    extra_weight: float,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    combined_samples = []
    oracle_samples = []
    combined_labels = []
    combined_probabilities = []
    oracle_labels = []
    oracle_probabilities = []
    for position, context in enumerate(contexts, 1):
        candidates, probability = _probabilities(
            classifier, context, costs, oracle=False
        )
        predicted, predicted_deletions, _diagnostics = (
            decode_ornament_mapper(
                candidates,
                context["index"].events,
                context["score_path"],
                costs=costs,
                extra_probabilities=probability,
                extra_weight=extra_weight,
            )
        )
        oracle_candidates, oracle_probability = _probabilities(
            classifier, context, costs, oracle=True
        )
        oracle, oracle_deletions, _oracle_diagnostics = (
            decode_ornament_mapper(
                oracle_candidates,
                context["index"].events,
                context["score_path"],
                costs=costs,
                extra_probabilities=oracle_probability,
                extra_weight=extra_weight,
            )
        )
        combined_samples.append(
            JointMetricSample(
                predicted=predicted,
                target=context["target"],
                source=context["source"],
                predicted_deletions=predicted_deletions,
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
        target_by_rendered = {
            int(event.rendered_index): event
            for event in context["target"]
            if event.rendered_index is not None
        }
        pairs = baseline.pair_exact_pitch_onset(
            baseline._candidate_events(candidates),
            context["target"],
            tolerance_sec=0.050,
        )
        pair_by_candidate = dict(pairs)
        combined_labels.extend(
            (
                target_by_rendered[
                    int(context["target"][pair_by_candidate[index]].rendered_index)
                ].is_extra
                if index in pair_by_candidate
                else True
            )
            for index in range(len(candidates))
        )
        combined_probabilities.extend(probability.tolist())
        oracle_labels.extend(event.is_extra for event in context["target"])
        oracle_probabilities.extend(oracle_probability.tolist())
        if position == 1 or position % 10 == 0 or position == len(contexts):
            print(
                f"prior-weight={extra_weight:g} rows={position}/{len(contexts)}",
                flush=True,
            )
    return {
        "extra_weight": float(extra_weight),
        "combined": baseline._full_report(
            combined_samples, seed=seed, replicates=replicates
        ),
        "oracle_note_mapper": baseline._full_report(
            oracle_samples, seed=seed + 1, replicates=replicates
        ),
        "extra_classifier_on_trackb_candidates": _extra_metrics(
            np.asarray(combined_labels, dtype=np.int64),
            np.asarray(combined_probabilities, dtype=np.float64),
        ),
        "extra_classifier_on_oracle_candidates": _extra_metrics(
            np.asarray(oracle_labels, dtype=np.int64),
            np.asarray(oracle_probabilities, dtype=np.float64),
        ),
    }


def run_train(args: argparse.Namespace) -> None:
    release = baseline._verify_release(args.release_manifest)
    baseline._assert_lockbox_sealed(release)
    structural_candidate = baseline._load_json(args.structural_candidate)
    if not structural_candidate.get("frozen"):
        raise ValueError("Structural ornament mapper is not frozen")
    costs = OrnamentMapperCosts(**structural_candidate["costs"])
    train_contexts = _train_contexts(release)
    calibration_predictions = (
        args.structural_candidate.parent
        / "calibration-trackb18-predictions.jsonl"
    )
    calibration_contexts = calibrator._contexts(
        release, "calibration", calibration_predictions
    )
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="ornament-mapper-extra-prior-v1",
        command=[sys.executable, *sys.argv],
        metadata={
            "train_rows": len(train_contexts),
            "calibration_rows": len(calibration_contexts),
            "locked_test": False,
        },
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        classifier, train_report = _fit_prior(
            train_contexts, costs, seed=args.seed
        )
        model_path = args.output_dir / "extra-prior.joblib"
        joblib.dump(classifier, model_path, compress=3)
        variants = []
        for index, weight in enumerate(EXTRA_WEIGHTS):
            variants.append(
                _evaluate_weight(
                    calibration_contexts,
                    classifier,
                    costs,
                    extra_weight=weight,
                    seed=args.seed + index * 100,
                    replicates=args.bootstrap_replicates,
                )
            )
        selected = max(
            variants,
            key=lambda row: (
                float(row["combined"]["f1"]),
                float(row["oracle_note_mapper"]["f1"]),
                -abs(float(row["extra_weight"]) - 1.0),
            ),
        )
        report_path = args.output_dir / "calibration_report.json"
        baseline._atomic_json(
            report_path,
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "train_plus_calibration",
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "structural_candidate_sha256": sha256_file(
                    args.structural_candidate
                ),
                "train": {
                    "rows": len(train_contexts),
                    "exact_exposed_orn500_rows": 0,
                    "report": train_report,
                },
                "calibration": {
                    "rows": len(calibration_contexts),
                    "predeclared_extra_weights": list(EXTRA_WEIGHTS),
                    "selected_extra_weight": selected["extra_weight"],
                    "selected_combined_f1": selected["combined"]["f1"],
                    "selected_oracle_mapper_f1": selected[
                        "oracle_note_mapper"
                    ]["f1"],
                    "variants": variants,
                },
                "open_validation_read": False,
                "lockbox_targets_read": False,
            },
        )
        candidate_path = args.output_dir / "EXTRA_PRIOR_CANDIDATE.json"
        baseline._atomic_json(
            candidate_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-candidate",
                "frozen": True,
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "structural_candidate": str(
                    args.structural_candidate.resolve()
                ),
                "structural_candidate_sha256": sha256_file(
                    args.structural_candidate
                ),
                "costs": asdict(costs),
                "extra_prior": str(model_path.resolve()),
                "extra_prior_sha256": sha256_file(model_path),
                "extra_weight": selected["extra_weight"],
                "calibration_report": str(report_path.resolve()),
                "calibration_report_sha256": sha256_file(report_path),
                "calibration_combined": selected["combined"],
                "calibration_oracle_note_mapper": selected[
                    "oracle_note_mapper"
                ],
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


def run_validate(args: argparse.Namespace) -> None:
    release = baseline._verify_release(args.release_manifest)
    baseline._assert_lockbox_sealed(release)
    candidate_path = args.output_dir / "EXTRA_PRIOR_CANDIDATE.json"
    candidate = baseline._load_json(candidate_path)
    if not candidate.get("frozen") or candidate.get("open_validation_read"):
        raise ValueError("EXTRA-prior candidate was not frozen before validation")
    model_path = Path(candidate["extra_prior"])
    if sha256_file(model_path) != candidate["extra_prior_sha256"]:
        raise ValueError("EXTRA-prior artifact hash mismatch")
    classifier = joblib.load(model_path)
    costs = OrnamentMapperCosts(**candidate["costs"])
    baseline_freeze = baseline._load_json(
        args.baseline_dir / "prediction_freeze.json"
    )
    predictions_path = Path(baseline_freeze["predictions"])
    if (
        sha256_file(predictions_path)
        != baseline_freeze["predictions_sha256"]
    ):
        raise ValueError("Frozen open-validation predictions changed")
    contexts = calibrator._contexts(
        release, "open_validation", predictions_path
    )
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="ornament-mapper-extra-prior-open-validation",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(contexts), "locked_test": False},
    ):
        result = _evaluate_weight(
            contexts,
            classifier,
            costs,
            extra_weight=float(candidate["extra_weight"]),
            seed=args.seed + 1000,
            replicates=args.bootstrap_replicates,
        )
    report_path = args.output_dir / "open_validation_report.json"
    baseline._atomic_json(
        report_path,
        {
            "schema_version": f"{SCHEMA_VERSION}-open-validation",
            "phase": "single_prespecified_open_validation_check",
            "release_manifest_sha256": sha256_file(args.release_manifest),
            "candidate_sha256": sha256_file(candidate_path),
            **result,
            "timestamps_used_for_selection": False,
            "lockbox_targets_read": False,
        },
    )
    combined_f1 = float(result["combined"]["f1"])
    oracle_f1 = float(result["oracle_note_mapper"]["f1"])
    baseline._atomic_json(
        args.output_dir / "VALIDATION_STATUS.json",
        {
            "schema_version": f"{SCHEMA_VERSION}-validation-status",
            "status": (
                "extra_prior_validated"
                if oracle_f1 >= 0.85
                else "mapper_oracle_below_required_support"
            ),
            "combined_f1_with_frozen_track_b": combined_f1,
            "oracle_note_mapper_f1": oracle_f1,
            "combined_gate_threshold": 0.85,
            "combined_gate_passed": combined_f1 >= 0.85,
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "candidate_sha256": sha256_file(candidate_path),
            "lockbox_action": (
                "eligible_for_single_open"
                if combined_f1 >= 0.85
                else "remain_sealed"
            ),
            "lockbox_opened": False,
        },
    )
    baseline._assert_lockbox_sealed(release)
    print(
        json.dumps(
            {
                "combined_f1": combined_f1,
                "oracle_note_mapper_f1": oracle_f1,
            },
            indent=2,
        ),
        flush=True,
    )


def _resolve(repo: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo / value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "validate"))
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--release-manifest", type=Path, default=baseline.DEFAULT_RELEASE
    )
    parser.add_argument(
        "--structural-candidate",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-generalization-v1/"
            "development/ornament-mapper-v2/ORNAMENT_MAPPER_CANDIDATE.json"
        ),
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=baseline.DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-generalization-v1/"
            "development/ornament-mapper-extra-prior-v1"
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
    args.repo = args.repo.resolve()
    for name in (
        "release_manifest",
        "structural_candidate",
        "baseline_dir",
        "output_dir",
        "resource_status",
    ):
        setattr(args, name, _resolve(args.repo, getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        run_train(args)
    else:
        run_validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
