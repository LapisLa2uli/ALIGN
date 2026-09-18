"""Calibrate ornament-aware mapper on 53 rows, then validate once on 66 rows."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.ornament_mapper_v1 import (
    OrnamentMapperCosts,
    decode_ornament_mapper,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


SCHEMA_VERSION = "align-ornament-mapper-calibration-v1"
COST_GRID = (
    OrnamentMapperCosts(),
    OrnamentMapperCosts(
        substitution=1.40,
        candidate_extra=0.60,
        score_deletion=1.00,
        ornament_miss=0.15,
        timing=0.10,
    ),
    OrnamentMapperCosts(
        substitution=1.25,
        candidate_extra=0.75,
        score_deletion=0.65,
        ornament_miss=0.10,
        timing=0.20,
    ),
    OrnamentMapperCosts(
        substitution=1.60,
        candidate_extra=1.00,
        score_deletion=0.80,
        ornament_miss=0.30,
        timing=0.05,
    ),
    OrnamentMapperCosts(
        substitution=1.35,
        candidate_extra=0.60,
        score_deletion=0.85,
        ornament_miss=0.10,
        timing=0.50,
    ),
    OrnamentMapperCosts(
        substitution=1.35,
        candidate_extra=0.60,
        score_deletion=0.85,
        ornament_miss=0.10,
        timing=1.00,
    ),
    OrnamentMapperCosts(
        substitution=1.35,
        candidate_extra=0.60,
        score_deletion=0.85,
        ornament_miss=0.10,
        timing=2.00,
    ),
    OrnamentMapperCosts(
        substitution=1.25,
        candidate_extra=0.40,
        score_deletion=0.90,
        ornament_miss=0.10,
        timing=3.00,
    ),
)


def _calibration_predictions(
    args: argparse.Namespace,
    release: Mapping[str, Any],
) -> Path:
    output = args.output_dir / "calibration-trackb18-predictions.jsonl"
    freeze_path = args.output_dir / "calibration-prediction-freeze.json"
    if freeze_path.is_file():
        freeze = baseline._load_json(freeze_path)
        if (
            freeze["release_manifest_sha256"]
            != baseline.EXPECTED_RELEASE_SHA256
            or sha256_file(output) != freeze["predictions_sha256"]
        ):
            raise ValueError("Calibration prediction cache integrity failure")
        return output
    rows = release["splits"]["development"]["calibration"]
    device = torch.device(args.device)
    with resource_lease(
        args.resource_status,
        "gpu" if device.type == "cuda" else "cpu_validation",
        track="ornament-mapper-v1-calibration-infer",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(rows), "locked_test": False},
    ):
        model, frontend, decode, _payload = load_mel_checkpoint(
            args.checkpoint, device
        )
        decode = replace(decode, min_confidence=0.8)
        predictions = []
        for position, row in enumerate(rows, 1):
            audio_path = Path(row["sample_dir"]) / "performance_audio.wav"
            if (
                sha256_file(audio_path)
                != row["source_hashes"]["performance_audio.wav"]
            ):
                raise ValueError(f"Calibration audio changed: {row['sample']}")
            audio = load_audio_mono(audio_path, frontend.sample_rate)
            mel, _normalization = extract_log_mel(
                audio, frontend, device=device
            )
            probabilities = infer_mel_probabilities(
                model,
                mel,
                device,
                window_frames=2048,
                overlap_frames=512,
                batch_size=args.batch_size,
            )
            notes = [
                value.to_dict()
                for value in decode_mel_notes(
                    probabilities,
                    midi_min=model.config.midi_min,
                    hop_sec=frontend.hop_sec,
                    config=decode,
                )
            ]
            predictions.append(
                {
                    "sample": row["sample"],
                    "audio_sha256": row["source_hashes"][
                        "performance_audio.wav"
                    ],
                    "notes": notes,
                }
            )
            if position == 1 or position % 10 == 0 or position == len(rows):
                print(f"calibration-infer={position}/{len(rows)}", flush=True)
        baseline._atomic_jsonl(output, predictions)
        baseline._atomic_json(
            freeze_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-prediction-freeze",
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "predictions": str(output.resolve()),
                "predictions_sha256": sha256_file(output),
                "rows": len(predictions),
                "split": "calibration",
                "score_input_to_acoustic_model": False,
                "lockbox_targets_read": False,
            },
        )
    return output


def _contexts(
    release: Mapping[str, Any],
    split: str,
    predictions_path: Path,
) -> list[dict[str, Any]]:
    manifest_rows = {
        row["sample"]: row
        for row in release["splits"]["development"][split]
    }
    targets = baseline._load_targets(release, split)
    predictions = {
        row["sample"]: row for row in baseline._read_jsonl(predictions_path)
    }
    if set(manifest_rows) != set(targets) or set(targets) != set(predictions):
        raise ValueError(f"{split} mapper population mismatch")
    output = []
    for sample in sorted(manifest_rows):
        row = manifest_rows[sample]
        target_row = targets[sample]
        lineage = target_row["lineage"]
        if (
            hashlib.sha256(baseline._canonical_bytes(lineage)).hexdigest()
            != row["target_lineage_sha256"]
        ):
            raise ValueError(f"Target hash mismatch: {sample}")
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        if (
            sha256_file(score_path)
            != row["source_hashes"]["verified_score.musicxml"]
        ):
            raise ValueError(f"Score hash mismatch: {sample}")
        index = ScoreEventIndex.from_musicxml(score_path, lineage)
        output.append(
            {
                "sample": sample,
                "source": row["leakage_group"],
                "score_path": score_path,
                "index": index,
                "target": index.rendered_events,
                "candidate": baseline._candidates(
                    predictions[sample]["notes"]
                ),
            }
        )
    return output


def _evaluate_costs(
    contexts: Sequence[Mapping[str, Any]],
    costs: OrnamentMapperCosts,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    combined_samples = []
    oracle_samples = []
    per_row = []
    diagnostics = []
    for position, context in enumerate(contexts, 1):
        index = context["index"]
        predicted, predicted_deletions, mapper_diagnostic = (
            decode_ornament_mapper(
                context["candidate"],
                index.events,
                context["score_path"],
                costs=costs,
            )
        )
        oracle, oracle_deletions, oracle_diagnostic = (
            decode_ornament_mapper(
                baseline._oracle_candidates(context["target"]),
                index.events,
                context["score_path"],
                costs=costs,
            )
        )
        combined_sample = JointMetricSample(
            predicted=predicted,
            target=context["target"],
            source=context["source"],
            predicted_deletions=predicted_deletions,
            target_deletions=index.deleted_event_indices,
            score_event_count=len(index.events),
        )
        oracle_sample = JointMetricSample(
            predicted=oracle,
            target=context["target"],
            source=context["source"],
            predicted_deletions=oracle_deletions,
            target_deletions=index.deleted_event_indices,
            score_event_count=len(index.events),
        )
        combined_samples.append(combined_sample)
        oracle_samples.append(oracle_sample)
        per_row.append(
            {
                "sample": context["sample"],
                "combined": baseline._fractional_prf(
                    *baseline._counts(combined_sample)
                ),
                "oracle": baseline._fractional_prf(
                    *baseline._counts(oracle_sample)
                ),
            }
        )
        diagnostics.append(
            {
                "sample": context["sample"],
                "predicted": mapper_diagnostic,
                "oracle": oracle_diagnostic,
            }
        )
        if position == 1 or position % 10 == 0 or position == len(contexts):
            print(f"mapper={position}/{len(contexts)}", flush=True)
    return {
        "costs": asdict(costs),
        "combined": baseline._full_report(
            combined_samples, seed=seed, replicates=replicates
        ),
        "oracle_note_mapper": baseline._full_report(
            oracle_samples, seed=seed + 1, replicates=replicates
        ),
        "per_row": per_row,
        "diagnostics": diagnostics,
    }


def run_calibrate(args: argparse.Namespace) -> None:
    release = baseline._verify_release(args.release_manifest)
    baseline._assert_lockbox_sealed(release)
    if sha256_file(args.checkpoint) != baseline.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Track B checkpoint mismatch")
    predictions_path = _calibration_predictions(args, release)
    contexts = _contexts(release, "calibration", predictions_path)
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="ornament-mapper-v1-calibration",
        command=[sys.executable, *sys.argv],
        metadata={
            "rows": len(contexts),
            "variants": len(COST_GRID),
            "locked_test": False,
        },
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        variants = []
        for index, costs in enumerate(COST_GRID):
            print(f"variant={index + 1}/{len(COST_GRID)}", flush=True)
            result = _evaluate_costs(
                contexts,
                costs,
                seed=args.seed + 100 * index,
                replicates=args.bootstrap_replicates,
            )
            result["variant"] = index
            variants.append(result)
        selected = max(
            variants,
            key=lambda row: (
                float(row["combined"]["f1"]),
                float(row["oracle_note_mapper"]["f1"]),
                -int(row["variant"]),
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
                "population": {
                    "split": "calibration",
                    "rows": len(contexts),
                    "open_validation_read": False,
                    "lockbox_targets_read": False,
                },
                "selection": {
                    "metric": "combined canonical note-wise F1",
                    "predeclared_variants": len(COST_GRID),
                    "selected_variant": selected["variant"],
                    "selected_costs": selected["costs"],
                    "selected_combined_f1": selected["combined"]["f1"],
                    "selected_oracle_mapper_f1": selected[
                        "oracle_note_mapper"
                    ]["f1"],
                },
                "variants": variants,
            },
        )
        candidate_path = args.output_dir / "ORNAMENT_MAPPER_CANDIDATE.json"
        baseline._atomic_json(
            candidate_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-candidate",
                "frozen": True,
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "calibration_rows": len(contexts),
                "calibration_report": str(report_path.resolve()),
                "calibration_report_sha256": sha256_file(report_path),
                "selected_variant": selected["variant"],
                "costs": selected["costs"],
                "calibration_combined": selected["combined"],
                "calibration_oracle_note_mapper": selected[
                    "oracle_note_mapper"
                ],
                "open_validation_read": False,
                "lockbox_targets_read": False,
            },
        )
        baseline._atomic_json(
            args.output_dir / "CALIBRATION_STATUS.json",
            {
                "schema_version": f"{SCHEMA_VERSION}-status",
                "status": "candidate_frozen_for_open_validation",
                "candidate": str(candidate_path.resolve()),
                "candidate_sha256": sha256_file(candidate_path),
                "calibration_combined_f1": selected["combined"]["f1"],
                "calibration_oracle_mapper_f1": selected[
                    "oracle_note_mapper"
                ]["f1"],
                "open_validation_read": False,
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
    candidate_path = args.output_dir / "ORNAMENT_MAPPER_CANDIDATE.json"
    candidate = baseline._load_json(candidate_path)
    if not candidate.get("frozen") or candidate.get("open_validation_read"):
        raise ValueError("Mapper candidate was not frozen before validation")
    baseline_dir = args.baseline_dir.resolve()
    baseline_freeze = baseline._load_json(
        baseline_dir / "prediction_freeze.json"
    )
    predictions_path = Path(baseline_freeze["predictions"])
    if (
        sha256_file(predictions_path)
        != baseline_freeze["predictions_sha256"]
    ):
        raise ValueError("Baseline open-validation predictions changed")
    contexts = _contexts(release, "open_validation", predictions_path)
    costs = OrnamentMapperCosts(**candidate["costs"])
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="ornament-mapper-v1-open-validation",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(contexts), "locked_test": False},
    ):
        result = _evaluate_costs(
            contexts,
            costs,
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
            "baseline_prediction_freeze_sha256": sha256_file(
                baseline_dir / "prediction_freeze.json"
            ),
            "rows": len(contexts),
            **result,
            "metric": (
                "exclusive one-to-one canonical score-event/rendered EXTRA "
                "identity; exact type=1, wrong type at exact location=0.5"
            ),
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
                "mapper_candidate_validated"
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
    parser.add_argument("phase", choices=("calibrate", "validate"))
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--release-manifest", type=Path, default=baseline.DEFAULT_RELEASE
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=baseline.DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=baseline.DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-generalization-v1/"
            "development/ornament-mapper-v1"
        ),
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=baseline.DEFAULT_RESOURCE_STATUS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=baseline.DEFAULT_SEED)
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    for name in (
        "release_manifest",
        "checkpoint",
        "baseline_dir",
        "output_dir",
        "resource_status",
    ):
        setattr(args, name, _resolve(args.repo, getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "calibrate":
        run_calibrate(args)
    else:
        run_validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
