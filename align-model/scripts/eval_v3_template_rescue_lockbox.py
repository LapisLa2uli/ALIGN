"""One-shot ORN v3 evaluation of the frozen local-template-rescue candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import calibrate_v2_acoustic_crf as acoustic
from alignmodel.joint.packed_data import sha256_file
from eval_v2_template_rescue_lockbox import (
    _decrypt_targets,
    _freeze_predictions,
    _score,
)


SCHEMA_VERSION = "align-orn-v3-template-rescue-lockbox-v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    expected = (
        (args.lockbox_manifest, args.expected_manifest_sha256),
        (args.encrypted_targets, args.expected_targets_sha256),
        (args.freeze, args.expected_freeze_sha256),
        (args.candidate, args.expected_candidate_sha256),
        (args.base_candidate, args.expected_base_candidate_sha256),
        (args.checkpoint, args.expected_checkpoint_sha256),
        (args.protocol, args.expected_protocol_sha256),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact mismatch: {path}")
    manifest = _load_json(args.lockbox_manifest)
    freeze = _load_json(args.freeze)
    candidate = _load_json(args.candidate)
    base_candidate = _load_json(args.base_candidate)
    protocol = _load_json(args.protocol)
    if freeze.get("lockbox_opened"):
        raise ValueError("ORN v3 FREEZE already marks the lockbox opened")
    if int(manifest["rows"]) != 114 or len(manifest["inputs"]) != 114:
        raise ValueError("Unexpected ORN v3 lockbox population")
    if (
        freeze["lockbox_manifest"]["sha256"]
        != args.expected_manifest_sha256
        or freeze["lockbox_targets_encrypted"]["sha256"]
        != args.expected_targets_sha256
    ):
        raise ValueError("Freeze/lockbox mismatch")
    if (
        not candidate.get("frozen")
        or candidate["holdout_freeze"]["sha256"]
        != args.expected_freeze_sha256
        or candidate["holdout_freeze"]["targets_read"]
        or candidate["holdout_freeze"]["key_read"]
    ):
        raise ValueError("Candidate was not frozen against sealed ORN v3")
    method = candidate["method"]
    if (
        method["mode"] != "direct_template_local_warp"
        or method["base_candidate_sha256"]
        != args.expected_base_candidate_sha256
        or method["identity_crf_checkpoint_sha256"]
        != args.expected_checkpoint_sha256
    ):
        raise ValueError("Unexpected candidate dependencies")
    if protocol["evaluator_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Evaluator changed after protocol freeze")
    if (
        protocol["candidate_sha256"] != args.expected_candidate_sha256
        or protocol["freeze_sha256"] != args.expected_freeze_sha256
        or protocol["lockbox_manifest_sha256"]
        != args.expected_manifest_sha256
        or protocol["encrypted_targets_sha256"]
        != args.expected_targets_sha256
    ):
        raise ValueError("Protocol/frozen artifact mismatch")
    if float(protocol["success_threshold_f1"]) != 0.80:
        raise ValueError("Unexpected success threshold")
    model, payload = acoustic._load_model(args.checkpoint)
    if (
        payload["data"]["release_manifest_sha256"]
        != base_candidate["release_manifest_sha256"]
    ):
        raise ValueError("Checkpoint/base-candidate release mismatch")
    return manifest, candidate, base_candidate, model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockbox-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--encrypted-targets", type=Path, required=True)
    parser.add_argument("--expected-targets-sha256", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--expected-base-candidate-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--public-smoke", action="store_true")
    args = parser.parse_args(argv)

    manifest, candidate, base_candidate, model = _validate(args)
    if args.public_smoke:
        from eval_v2_template_rescue_lockbox import _predict_row

        result = _predict_row(
            row=manifest["inputs"][0],
            candidate=candidate,
            base_candidate=base_candidate,
            model=model,
            cache_root=args.feature_cache,
        )
        print(
            json.dumps(
                {
                    "row_id": result["row_id"],
                    "predicted_count": len(result["predicted"]),
                    "score_event_count": result["score_event_count"],
                    "target_or_key_read": False,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "FROZEN_PREDICTIONS.json"
    sentinel_path = args.output_dir / "LOCKBOX_OPENED.json"
    repository_sentinel = (
        args.lockbox_manifest.parent / "LOCKBOX_OPENED.json"
    )
    report_path = args.output_dir / "LOCKBOX_REPORT.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")

    if sentinel_path.exists() or repository_sentinel.exists():
        if not sentinel_path.exists() or not repository_sentinel.exists():
            raise ValueError("Inconsistent ORN v3 opening sentinels")
        sentinel = _load_json(sentinel_path)
        if (
            not prediction_path.exists()
            or sentinel["frozen_predictions_sha256"]
            != sha256_file(prediction_path)
        ):
            raise ValueError("Frozen predictions are missing or changed")
        predictions = _load_json(prediction_path)["rows"]
    else:
        predictions = _freeze_predictions(
            manifest=manifest,
            candidate=candidate,
            base_candidate=base_candidate,
            model=model,
            cache_root=args.feature_cache,
            destination=prediction_path,
        )
        opening = {
            "schema_version": f"{SCHEMA_VERSION}-opened",
            "opened_utc": _utc(),
            "opening_number": 1,
            "protocol_sha256": args.expected_protocol_sha256,
            "freeze_sha256": args.expected_freeze_sha256,
            "candidate_sha256": args.expected_candidate_sha256,
            "lockbox_manifest_sha256": args.expected_manifest_sha256,
            "encrypted_targets_sha256": args.expected_targets_sha256,
            "frozen_predictions_sha256": sha256_file(prediction_path),
            "rows": len(predictions),
            "predictions_frozen_before_target_or_key_read": True,
        }
        acoustic._atomic_json(sentinel_path, opening)
        acoustic._atomic_json(repository_sentinel, opening)

    targets = _decrypt_targets(
        key_path=args.key,
        encrypted_path=args.encrypted_targets,
        manifest=manifest,
    )
    result = _score(
        predictions=predictions,
        targets=targets,
        manifest=manifest,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    protocol = _load_json(args.protocol)
    threshold = float(protocol["success_threshold_f1"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "protocol_sha256": args.expected_protocol_sha256,
        "freeze_sha256": args.expected_freeze_sha256,
        "candidate_sha256": args.expected_candidate_sha256,
        "lockbox_manifest_sha256": args.expected_manifest_sha256,
        "encrypted_targets_sha256": args.expected_targets_sha256,
        "frozen_predictions_sha256": sha256_file(prediction_path),
        "population": {"split": "orn_v3_lockbox", "rows": len(targets)},
        "official_note_wise": result,
        "success_threshold_f1": threshold,
        "success": float(result["f1"]) >= threshold,
        "post_lockbox_tuning_allowed": False,
    }
    acoustic._atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "bootstrap_95": result.get("bootstrap_95"),
                "success_threshold_f1": threshold,
                "success": report["success"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(report_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
