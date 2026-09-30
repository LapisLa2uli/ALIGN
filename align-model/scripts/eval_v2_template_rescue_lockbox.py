"""One-shot encrypted-lockbox evaluation of a frozen local-rescue candidate.

All predictions are written and hashed from public audio/score inputs before
the opening sentinel is created and before the AES key or targets are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import calibrate_v2_acoustic_crf as acoustic
import calibrate_v2_template_rescue as rescue
import eval_orn_phase2_baseline_v1 as baseline
import freeze_orn_generalization_v2 as freeze
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    decode_basic_pitch_features,
    extract_basic_pitch_features,
)


SCHEMA_VERSION = "align-orn-v2-template-rescue-local-lockbox-v1"
PITCH_METADATA = {
    "sounding_transpose": -2,
    "midi_pitch_space": "sounding",
    "audio_pitch_space": "sounding",
    "effective_audio_transpose": 2,
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_from_json(raw: Mapping[str, Any]) -> JointEvent:
    span = raw.get("score_span")
    return JointEvent(
        pitch=int(raw["pitch"]),
        start=float(raw["start"]),
        end=float(raw["end"]),
        score_span=tuple(map(int, span)) if span is not None else None,
        relationship=str(raw.get("relationship") or "match"),
        copy_pass=int(raw.get("copy_pass") or 0),
        origin_relationship=raw.get("origin_relationship"),
        rendered_index=(
            int(raw["rendered_index"])
            if raw.get("rendered_index") is not None
            else None
        ),
        source_indices=tuple(map(int, raw.get("source_indices") or ())),
        confidence=float(raw.get("confidence") or 0.0),
    )


def _validate_artifact(raw: Mapping[str, Any]) -> Path:
    path = Path(raw["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(raw["bytes"]) != path.stat().st_size:
        raise ValueError(f"Artifact size changed: {path}")
    if str(raw["sha256"]) != sha256_file(path):
        raise ValueError(f"Artifact hash changed: {path}")
    return path


def _load_and_validate(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    expected = (
        (args.lockbox_manifest, args.expected_manifest_sha256),
        (args.encrypted_targets, args.expected_targets_sha256),
        (args.candidate, args.expected_candidate_sha256),
        (args.base_candidate, args.expected_base_candidate_sha256),
        (args.checkpoint, args.expected_checkpoint_sha256),
        (args.protocol, args.expected_protocol_sha256),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact mismatch: {path}")
    manifest = json.loads(args.lockbox_manifest.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    base_candidate = json.loads(
        args.base_candidate.read_text(encoding="utf-8")
    )
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["evaluator_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Evaluator changed after protocol freeze")
    if protocol["candidate_sha256"] != args.expected_candidate_sha256:
        raise ValueError("Protocol/candidate mismatch")
    if (
        protocol["lockbox"]["manifest_sha256"]
        != args.expected_manifest_sha256
    ):
        raise ValueError("Protocol/manifest mismatch")
    if (
        protocol["lockbox"]["encrypted_targets_sha256"]
        != args.expected_targets_sha256
    ):
        raise ValueError("Protocol/target envelope mismatch")
    method = candidate["method"]
    if (
        not candidate.get("frozen")
        or method["mode"] != "direct_template_local_warp"
        or method["base_candidate_sha256"]
        != args.expected_base_candidate_sha256
        or candidate["frozen_inputs"]["identity_crf_checkpoint_sha256"]
        != args.expected_checkpoint_sha256
        or base_candidate.get("mode") != "direct_template"
    ):
        raise ValueError("Unexpected frozen local-rescue candidate")
    if method["implementation_sha256"] != sha256_file(
        Path(__file__).with_name("calibrate_v2_template_rescue.py")
    ):
        raise ValueError("Local-rescue implementation changed after freeze")
    if manifest["target_envelope"]["sha256"] != args.expected_targets_sha256:
        raise ValueError("Manifest/target envelope mismatch")
    if int(manifest["rows"]) != 57 or len(manifest["inputs"]) != 57:
        raise ValueError("Unexpected lockbox population")
    model, payload = acoustic._load_model(args.checkpoint)
    if (
        payload["data"]["release_manifest_sha256"]
        != candidate["frozen_inputs"]["release_manifest_sha256"]
    ):
        raise ValueError("Checkpoint/release mismatch")
    return manifest, candidate, base_candidate, model


def _predict_row(
    *,
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    base_candidate: Mapping[str, Any],
    model: Any,
    cache_root: Path,
) -> dict[str, Any]:
    audio_path = _validate_artifact(row["audio"])
    score_path = _validate_artifact(row["score"])
    row_id = str(row["row_id"])
    features = extract_basic_pitch_features(
        audio_path,
        source_metadata=PITCH_METADATA,
        cache_path=cache_root / f"{row_id}.npz",
    )
    base = decode_basic_pitch_features(
        features, BasicPitchDecodeConfig(**base_candidate["base_decode"])
    )
    # Public-input inference only: no target lineage is available here.
    index = ScoreEventIndex.from_musicxml(score_path)
    base_candidates = acoustic._identity_candidates(base, features)
    baseline_sample, base_diagnostics = rescue._map(
        model, base_candidates, index, score_path
    )
    final_candidates = rescue._direct_rescue(
        base,
        index.events,
        score_path,
        base_diagnostics,
        features,
        evidence_threshold=float(candidate["method"]["evidence_threshold"]),
        window_sec=float(candidate["method"]["window_sec"]),
        anchor_events=baseline_sample.predicted,
    )
    final_sample, final_diagnostics = rescue._map(
        model, final_candidates, index, score_path
    )
    return {
        "ordinal": int(row["ordinal"]),
        "row_id": row_id,
        "leakage_group": str(row["leakage_group"]),
        "audio_sha256": row["audio"]["sha256"],
        "score_sha256": row["score"]["sha256"],
        "score_event_count": len(index.events),
        "base_candidate_count": len(base),
        "rescued_candidate_count": len(final_candidates) - len(base),
        "predicted": [asdict(event) for event in final_sample.predicted],
        "predicted_deletions": sorted(final_sample.predicted_deletions),
        "decoder": {
            "copies": int(final_diagnostics.get("copies") or 0),
            "source_span": final_diagnostics.get("source_span"),
        },
    }


def _freeze_predictions(
    *,
    manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    base_candidate: Mapping[str, Any],
    model: Any,
    cache_root: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    rows = []
    cache_root.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(manifest["inputs"], 1):
        rows.append(
            _predict_row(
                row=row,
                candidate=candidate,
                base_candidate=base_candidate,
                model=model,
                cache_root=cache_root,
            )
        )
        if position == 1 or position % 5 == 0 or position == len(
            manifest["inputs"]
        ):
            print(f"freeze={position}/{len(manifest['inputs'])}", flush=True)
    acoustic._atomic_json(
        destination,
        {
            "schema_version": f"{SCHEMA_VERSION}-predictions",
            "created_utc": _utc(),
            "target_envelope_read": False,
            "key_read": False,
            "rows": rows,
        },
    )
    return rows


def _decrypt_targets(
    *,
    key_path: Path,
    encrypted_path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    key = key_path.read_bytes()
    expected_key_sha = manifest["target_envelope"]["key_sha256"]
    if hashlib.sha256(key).hexdigest() != expected_key_sha:
        raise ValueError("Lockbox key mismatch")
    targets = freeze._decrypt(encrypted_path.read_bytes(), key)
    public_ids = [
        (int(row["ordinal"]), str(row["row_id"]))
        for row in manifest["inputs"]
    ]
    private_ids = [
        (int(row["ordinal"]), str(row["row_id"])) for row in targets
    ]
    if private_ids != public_ids:
        raise ValueError("Decrypted target population/order mismatch")
    return targets


def _score(
    *,
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    prediction_by_id = {str(row["row_id"]): row for row in predictions}
    public_by_id = {
        str(row["row_id"]): row for row in manifest["inputs"]
    }
    samples = []
    for position, target in enumerate(targets, 1):
        row_id = str(target["row_id"])
        predicted = prediction_by_id[row_id]
        public = public_by_id[row_id]
        index = ScoreEventIndex.from_musicxml(
            Path(public["score"]["path"]), target["lineage"]
        )
        if len(index.events) != int(predicted["score_event_count"]):
            raise ValueError(f"{row_id}: score changed after prediction freeze")
        samples.append(
            JointMetricSample(
                predicted=tuple(
                    _event_from_json(event)
                    for event in predicted["predicted"]
                ),
                target=index.rendered_events,
                source=str(public["leakage_group"]),
                predicted_deletions=frozenset(
                    map(int, predicted["predicted_deletions"])
                ),
                target_deletions=index.deleted_event_indices,
                score_event_count=len(index.events),
            )
        )
        if position == 1 or position % 10 == 0 or position == len(targets):
            print(f"score={position}/{len(targets)}", flush=True)
    return baseline._full_report(
        samples, seed=seed, replicates=replicates
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockbox-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--encrypted-targets", type=Path, required=True)
    parser.add_argument("--expected-targets-sha256", required=True)
    parser.add_argument("--key", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--public-smoke", action="store_true")
    args = parser.parse_args(argv)

    manifest, candidate, base_candidate, model = _load_and_validate(args)
    if args.public_smoke:
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
                    "score_event_count": result["score_event_count"],
                    "predicted_count": len(result["predicted"]),
                    "rescued_candidate_count": result[
                        "rescued_candidate_count"
                    ],
                    "target_or_key_read": False,
                },
                indent=2,
            )
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
            raise ValueError("Inconsistent lockbox opening sentinels")
        if not prediction_path.exists():
            raise ValueError("Lockbox opened without frozen predictions")
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        if sentinel["frozen_predictions_sha256"] != sha256_file(
            prediction_path
        ):
            raise ValueError("Predictions changed after lockbox opening")
        predictions = json.loads(
            prediction_path.read_text(encoding="utf-8")
        )["rows"]
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
            "protocol_sha256": args.expected_protocol_sha256,
            "candidate_sha256": args.expected_candidate_sha256,
            "lockbox_manifest_sha256": args.expected_manifest_sha256,
            "encrypted_targets_sha256": args.expected_targets_sha256,
            "frozen_predictions_sha256": sha256_file(prediction_path),
            "rows": len(predictions),
            "predictions_frozen_before_target_or_key_read": True,
            "opening_number": 1,
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
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    threshold = float(protocol["success_threshold_f1"])
    acoustic._atomic_json(
        report_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "protocol_sha256": args.expected_protocol_sha256,
            "candidate_sha256": args.expected_candidate_sha256,
            "lockbox_manifest_sha256": args.expected_manifest_sha256,
            "encrypted_targets_sha256": args.expected_targets_sha256,
            "frozen_predictions_sha256": sha256_file(prediction_path),
            "population": {"split": "encrypted_lockbox", "rows": len(targets)},
            "official_note_wise": result,
            "success_threshold_f1": threshold,
            "success": float(result["f1"]) >= threshold,
            "original_v2_gate_f1": 0.85,
            "original_v2_gate_changed": False,
            "original_v2_promotion": False,
            "post_lockbox_tuning_allowed": False,
        },
    )
    print(
        json.dumps(
            {
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "bootstrap_95": result.get("bootstrap_95"),
                "success_threshold_f1": threshold,
                "success": float(result["f1"]) >= threshold,
                "original_v2_promotion": False,
            },
            indent=2,
        ),
        flush=True,
    )
    print(report_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
