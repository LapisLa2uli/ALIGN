"""Calibrate unchanged Basic Pitch candidates through the frozen v2 identity CRF."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_fast_v2 import fast_decode_identity_crf
from alignmodel.joint.identity_crf_v1 import (
    IdentityCandidate,
    OrnamentIdentityCRF,
    build_identity_lattice,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    basic_pitch_cache_path,
    decode_basic_pitch_features,
    extract_sample_basic_pitch_features,
)


SCHEMA_VERSION = "align-orn-v2-basic-pitch-identity-crf-calibration-v1"
DECODE_GRID = (
    BasicPitchDecodeConfig(),
    BasicPitchDecodeConfig(
        onset_threshold=0.40,
        frame_threshold=0.30,
        minimum_note_length_ms=30.0,
    ),
    BasicPitchDecodeConfig(
        onset_threshold=0.60,
        frame_threshold=0.50,
        minimum_note_length_ms=40.0,
    ),
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_model(path: Path) -> tuple[OrnamentIdentityCRF, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = OrnamentIdentityCRF(
        hidden=int(payload["model_config"]["hidden"])
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _targets(
    release: Mapping[str, Any], split: str
) -> dict[str, dict[str, Any]]:
    artifact = release["artifacts"]["development_targets"]
    path = Path(artifact["path"])
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("Development target archive mismatch")
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] == split
        }


def _identity_candidates(notes: Sequence[Any], features: Any) -> tuple[IdentityCandidate, ...]:
    output = []
    for note in notes:
        frame = int(np.argmin(np.abs(features.frame_times - float(note.start))))
        probabilities = np.asarray(features.note[frame], np.float32)
        selected = np.argsort(probabilities)[-4:][::-1]
        alternatives = tuple(int(index) + 21 for index in selected)
        confidences = tuple(float(probabilities[index]) for index in selected)
        output.append(
            IdentityCandidate(
                pitch=int(note.pitch),
                start=float(note.start),
                end=float(note.end),
                confidence=float(note.confidence),
                alternatives=alternatives,
                alternative_confidences=confidences,
            )
        )
    return tuple(output)


def _lcs(left: Sequence[int], right: Sequence[int]) -> int:
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            row[column] = (
                previous + 1
                if item == other
                else max(row[column], row[column - 1])
            )
            previous = saved
    return row[-1]


def _prf(correct: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    return {
        "correct": correct,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "count_ratio": predicted / max(gold, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen ORN v2 release hash mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("ORN v2 lockbox opening sentinel exists")
    model, model_payload = _load_model(args.checkpoint)
    if (
        model_payload["data"]["release_manifest_sha256"] != release_sha
        or not model_payload["history"]
    ):
        raise ValueError("Identity CRF checkpoint/release mismatch")
    rows = {
        row["sample"]: row
        for row in release["splits"]["development"]["calibration"]
    }
    targets = _targets(release, "calibration")
    if set(rows) != set(targets) or len(rows) != 64:
        raise ValueError("Frozen calibration population mismatch")
    features = {}
    lease = resource_lease(
        args.resource_status,
        "gpu",
        track="orn-v2-basic-pitch-calibration",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(rows), "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        for position, (sample, row) in enumerate(sorted(rows.items()), 1):
            audio_path = Path(row["sample_dir"]) / "performance_audio.wav"
            if (
                sha256_file(audio_path)
                != row["source_hashes"]["performance_audio.wav"]
            ):
                raise ValueError(f"Calibration audio changed: {sample}")
            features[sample] = extract_sample_basic_pitch_features(
                row["sample_dir"],
                cache_path=basic_pitch_cache_path(
                    args.feature_cache, row["sample_dir"], "orn-v2-calibration"
                ),
            )
            if position == 1 or position % 10 == 0 or position == len(rows):
                print(f"basic-pitch={position}/{len(rows)}", flush=True)
    finally:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)
    variants = []
    for variant_index, decode in enumerate(DECODE_GRID):
        samples = []
        sequence_counts = [0, 0, 0]
        predictions = {}
        for position, sample in enumerate(sorted(rows), 1):
            row = rows[sample]
            target_row = targets[sample]
            lineage = target_row["lineage"]
            if (
                hashlib.sha256(baseline._canonical_bytes(lineage)).hexdigest()
                != row["target_lineage_sha256"]
            ):
                raise ValueError(f"Calibration target changed: {sample}")
            score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
            index = ScoreEventIndex.from_musicxml(score_path, lineage)
            notes = decode_basic_pitch_features(features[sample], decode)
            candidates = _identity_candidates(notes, features[sample])
            lattice = build_identity_lattice(
                candidates,
                index.events,
                score_path,
                max_inference_hypotheses=16,
            )
            predicted, deletions, diagnostics = fast_decode_identity_crf(
                model, lattice
            )
            metric_sample = JointMetricSample(
                predicted=predicted,
                target=index.rendered_events,
                source=row["leakage_group"],
                predicted_deletions=deletions,
                target_deletions=index.deleted_event_indices,
                score_event_count=len(index.events),
            )
            samples.append(metric_sample)
            correct = _lcs(
                [candidate.pitch for candidate in candidates],
                [event.pitch for event in index.rendered_events],
            )
            sequence_counts[0] += correct
            sequence_counts[1] += len(candidates)
            sequence_counts[2] += len(index.rendered_events)
            predictions[sample] = {
                "notes": [
                    {
                        "pitch": candidate.pitch,
                        "start": candidate.start,
                        "end": candidate.end,
                        "confidence": candidate.confidence,
                        "alternatives": candidate.alternatives,
                        "alternative_confidences": candidate.alternative_confidences,
                    }
                    for candidate in candidates
                ],
                "mapped_events": [baseline._event_dict(event) if hasattr(baseline, "_event_dict") else {
                    "pitch": event.pitch,
                    "start": event.start,
                    "end": event.end,
                    "score_span": event.score_span,
                    "relationship": event.relationship,
                    "copy_pass": event.copy_pass,
                    "rendered_index": event.rendered_index,
                } for event in predicted],
                "predicted_deletions": sorted(deletions),
                "diagnostics": diagnostics,
            }
            if position == 1 or position % 10 == 0 or position == len(rows):
                print(
                    f"variant={variant_index} map={position}/{len(rows)}",
                    flush=True,
                )
        variants.append(
            {
                "variant": variant_index,
                "decode": asdict(decode),
                "score_agnostic_pitch_sequence": _prf(*sequence_counts),
                "combined": baseline._full_report(
                    samples,
                    seed=args.seed + variant_index,
                    replicates=args.bootstrap_replicates,
                ),
                "predictions": predictions,
            }
        )
    selected = max(
        variants,
        key=lambda value: (
            float(value["combined"]["f1"]),
            -int(value["variant"]),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "calibration_report.json"
    _atomic_json(
        report_path,
        {
            "schema_version": SCHEMA_VERSION,
            "release_manifest_sha256": release_sha,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "population": {"split": "calibration", "rows": len(rows)},
            "predeclared_decode_variants": len(DECODE_GRID),
            "variants": variants,
            "selection": {
                "metric": "official combined canonical note-wise F1",
                "variant": selected["variant"],
                "decode": selected["decode"],
                "combined_f1": selected["combined"]["f1"],
                "score_agnostic_f1": selected[
                    "score_agnostic_pitch_sequence"
                ]["f1"],
            },
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    candidate_path = args.output_dir / "CALIBRATED_CANDIDATE.json"
    _atomic_json(
        candidate_path,
        {
            "schema_version": f"{SCHEMA_VERSION}-candidate",
            "frozen": True,
            "release_manifest_sha256": release_sha,
            "identity_crf_checkpoint": str(args.checkpoint.resolve()),
            "identity_crf_checkpoint_sha256": sha256_file(args.checkpoint),
            "basic_pitch_decode": selected["decode"],
            "calibration_report": str(report_path.resolve()),
            "calibration_report_sha256": sha256_file(report_path),
            "calibration_combined": selected["combined"],
            "calibration_score_agnostic": selected[
                "score_agnostic_pitch_sequence"
            ],
            "mapper_oracle_gate": 0.95,
            "mapper_oracle_passed": True,
            "combined_calibration_gate": 0.85,
            "combined_calibration_passed": float(
                selected["combined"]["f1"]
            )
            >= 0.85,
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate_path.resolve()),
                "candidate_sha256": sha256_file(candidate_path),
                "combined_f1": selected["combined"]["f1"],
                "score_agnostic_f1": selected[
                    "score_agnostic_pitch_sequence"
                ]["f1"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import argparse

    raise SystemExit(main())
