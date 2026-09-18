"""Official calibration scoring for the selected score-free ORN frontend."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

import eval_orn_phase2_baseline_v1 as baseline
import train_ornament_mapper_prior_v1 as prior
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.ornament_mapper_v1 import OrnamentMapperCosts
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


SCHEMA_VERSION = "align-orn-frontend-official-calibration-v1"


def _contexts(
    release: Mapping[str, Any],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = {
        row["sample"]: row
        for row in release["splits"]["development"]["calibration"]
    }
    targets = baseline._load_targets(release, "calibration")
    if set(rows) != set(targets) or set(targets) != set(predictions):
        raise ValueError("Frontend official calibration population mismatch")
    output = []
    for sample in sorted(rows):
        row = rows[sample]
        lineage = targets[sample]["lineage"]
        if (
            hashlib.sha256(baseline._canonical_bytes(lineage)).hexdigest()
            != row["target_lineage_sha256"]
        ):
            raise ValueError(f"Target hash mismatch: {sample}")
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        index = ScoreEventIndex.from_musicxml(score_path, lineage)
        output.append(
            {
                "sample": sample,
                "source": row["leakage_group"],
                "score_path": score_path,
                "index": index,
                "target": index.rendered_events,
                "candidate": baseline._candidates(predictions[sample]),
            }
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--frontend-calibration", type=Path, required=True)
    parser.add_argument("--mapper-candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args(argv)
    release = baseline._verify_release(args.release_manifest)
    baseline._assert_lockbox_sealed(release)
    frontend = baseline._load_json(args.frontend_calibration)
    if frontend.get("open_validation_read") or frontend.get(
        "lockbox_targets_read"
    ):
        raise ValueError("Frontend calibration isolation failed")
    predictions = frontend["selected_predictions"]
    contexts = _contexts(release, predictions)
    mapper = baseline._load_json(args.mapper_candidate)
    model_path = Path(mapper["extra_prior"])
    if sha256_file(model_path) != mapper["extra_prior_sha256"]:
        raise ValueError("Mapper prior hash mismatch")
    classifier = joblib.load(model_path)
    costs = OrnamentMapperCosts(**mapper["costs"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-frontend-official-calibration-v1",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(contexts), "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        result = prior._evaluate_weight(
            contexts,
            classifier,
            costs,
            extra_weight=float(mapper["extra_weight"]),
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        )
        report_path = args.output_dir / "report.json"
        baseline._atomic_json(
            report_path,
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "calibration_only",
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "frontend_calibration_sha256": sha256_file(
                    args.frontend_calibration
                ),
                "mapper_candidate_sha256": sha256_file(
                    args.mapper_candidate
                ),
                "frontend_selection": frontend["selection"],
                **result,
                "official_metric": (
                    "exclusive one-to-one canonical score-event/rendered EXTRA "
                    "identity; exact type=1, wrong type at exact location=0.5"
                ),
                "open_validation_read": False,
                "lockbox_targets_read": False,
            },
        )
        candidate_path = args.output_dir / "FRONTEND_CANDIDATE.json"
        baseline._atomic_json(
            candidate_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-candidate",
                "frozen": True,
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "frontend_calibration": str(
                    args.frontend_calibration.resolve()
                ),
                "frontend_calibration_sha256": sha256_file(
                    args.frontend_calibration
                ),
                "selected_frontend": frontend["selection"]["selected"],
                "mapper_candidate": str(args.mapper_candidate.resolve()),
                "mapper_candidate_sha256": sha256_file(
                    args.mapper_candidate
                ),
                "calibration_report": str(report_path.resolve()),
                "calibration_report_sha256": sha256_file(report_path),
                "calibration_combined": result["combined"],
                "calibration_oracle_note_mapper": result[
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
                "calibration_combined_f1": result["combined"]["f1"],
                "calibration_oracle_mapper_f1": result[
                    "oracle_note_mapper"
                ]["f1"],
                "lockbox_opened": False,
            },
        )
        print(
            json.dumps(
                {
                    "combined_f1": result["combined"]["f1"],
                    "oracle_note_mapper_f1": result[
                        "oracle_note_mapper"
                    ]["f1"],
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        baseline._assert_lockbox_sealed(release)
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
