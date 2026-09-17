"""Train-fold-only diagnosis and duration-conditioned Track A calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from alignmodel.joint.candidate_rescorer import (
    CANONICAL_FEATURE_DIM,
    candidate_features,
    load_candidate_rescorer,
)
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset


CALIBRATIONS = {
    "global-0.62": ((9.0, 0.62),),
    "short-a": ((0.080, 0.45), (0.120, 0.52), (0.180, 0.58), (9.0, 0.62)),
    "short-b": ((0.080, 0.50), (0.120, 0.55), (0.180, 0.60), (9.0, 0.62)),
    "short-c": ((0.080, 0.55), (0.120, 0.58), (0.180, 0.60), (9.0, 0.62)),
}
OBJECTIVE = {
    "primary": "maximize mean recall(<80,<120,<180ms)",
    "constraints": {
        "candidate_identity_f1_drop_max": 0.005,
        "false_split_rate_multiplier_max": 1.05,
        "count_ratio_must_not_exceed_global_train_fold_baseline": True,
    },
    "tie_break": "higher candidate identity F1",
}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _heldout_groups(manifest: dict, seed: int) -> tuple[set[str], dict[str, dict]]:
    rows = {str(row["sample"]): row for row in manifest["train"]}
    groups = sorted({str(row["leakage_group"]) for row in rows.values()})
    heldout = {
        group
        for group in groups
        if hashlib.sha256(f"{seed}:{group}".encode()).digest()[0] < 26
    }
    return heldout, rows


def _threshold(duration: float, schedule) -> float:
    for maximum, threshold in schedule:
        if duration < maximum:
            return threshold
    raise AssertionError("Calibration schedule has no terminal bucket")


def _prf(correct: int, predicted: int, target: int) -> dict[str, float]:
    precision = correct / max(predicted, 1)
    recall = correct / max(target, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    ready = verify_data_ready(args.ready_marker)
    manifest_path = Path(str(ready["paths"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_groups, manifest_rows = _heldout_groups(manifest, args.seed)
    model, checkpoint_threshold, checkpoint = load_candidate_rescorer(
        args.checkpoint, "cpu"
    )
    if model.feature_dim != CANONICAL_FEATURE_DIM:
        raise ValueError("Track A requires the canonical feature rescorer")

    examples = []
    diagnosis = defaultdict(lambda: [0, 0.0, 0.0])
    ablations = defaultdict(list)
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        selected_ordinals = [
            ordinal
            for ordinal in dataset.ordinals("train")
            if str(
                manifest_rows[dataset[ordinal].sample]["leakage_group"]
            ) in heldout_groups
        ]
        for ordinal in selected_ordinals:
            packed = dataset[ordinal]
            example = packed.training_example()
            features = candidate_features(
                example.candidates,
                example.score,
                feature_dim=CANONICAL_FEATURE_DIM,
            )
            with torch.inference_mode():
                probability = (
                    model(torch.from_numpy(features)).sigmoid().numpy()
                )
                no_duration = features.copy()
                no_duration[:, 1:3] = 0.0
                probability_no_duration = (
                    model(torch.from_numpy(no_duration)).sigmoid().numpy()
                )
            labels = np.asarray(example.gold_keep_unlinked, dtype=bool)
            durations = np.asarray(
                [candidate.end - candidate.start for candidate in example.candidates]
            )
            for index, (label, duration) in enumerate(zip(labels, durations)):
                bucket = (
                    "lt80" if duration < 0.080 else
                    "lt120" if duration < 0.120 else
                    "lt180" if duration < 0.180 else "long"
                )
                key = f"{bucket}:{'positive' if label else 'negative'}"
                diagnosis[key][0] += 1
                diagnosis[key][1] += float(probability[index])
                diagnosis[key][2] += float(
                    probability[index] - probability_no_duration[index]
                )
                ablations["confidence"].append(float(features[index, 0]))
            examples.append(
                (packed, example, probability, labels, durations, features)
            )

    results = {}
    for name, schedule in CALIBRATIONS.items():
        correct = predicted = target = 0
        short = {80: [0, 0], 120: [0, 0], 180: [0, 0]}
        false_splits = true_rearticulations = 0
        false_types = Counter()
        render_counts = defaultdict(lambda: [0, 0, 0])
        for packed, example, probability, labels, durations, features in examples:
            selected = np.asarray(
                [
                    value >= _threshold(float(duration), schedule)
                    for value, duration in zip(probability, durations)
                ]
            )
            correct += int(np.sum(selected & labels))
            predicted += int(np.sum(selected))
            target += int(np.sum(labels))
            row = manifest_rows[packed.sample]
            render = str(row.get("audio_render") or "unknown")
            render_counts[render][0] += int(np.sum(selected & labels))
            render_counts[render][1] += int(np.sum(selected))
            render_counts[render][2] += int(np.sum(labels))
            for maximum in short:
                bucket = durations < maximum / 1000.0
                short[maximum][0] += int(np.sum(selected & labels & bucket))
                short[maximum][1] += int(np.sum(labels & bucket))
            kept = np.flatnonzero(selected)
            for left, right in zip(kept, kept[1:]):
                first, second = example.candidates[left], example.candidates[right]
                if (
                    first.pitch == second.pitch
                    and second.start - first.end <= 0.100
                ):
                    if labels[left] and labels[right]:
                        true_rearticulations += 1
                    else:
                        false_splits += 1
            for index in np.flatnonzero(selected & ~labels):
                confidence = float(features[index, 0])
                onset, frame_peak, frame_mean, margin, contour = features[index, 3:8]
                if durations[index] < 0.180:
                    false_types["short"] += 1
                if 0.45 <= confidence <= 0.65:
                    false_types["confidence_45_65"] += 1
                if onset >= 0.5 and frame_mean < 0.2 and contour < 0.2:
                    false_types["onset_only"] += 1
                if margin < -0.25:
                    false_types["weak_pitch_margin_or_harmonic"] += 1
        metrics = {
            **_prf(correct, predicted, target),
            "correct": correct,
            "predicted": predicted,
            "target": target,
            "count_ratio": predicted / max(target, 1),
            "duration_recall": {
                f"lt_{maximum}ms": {
                    "matched": values[0],
                    "target": values[1],
                    "recall": values[0] / max(values[1], 1),
                }
                for maximum, values in short.items()
            },
            "short_recall_objective": float(
                np.mean([values[0] / max(values[1], 1) for values in short.values()])
            ),
            "false_split_count": false_splits,
            "false_split_rate": false_splits / max(predicted, 1),
            "preserved_true_rearticulations": true_rearticulations,
            "false_candidate_types": dict(false_types),
            "per_render": {
                key: {
                    **_prf(*values),
                    "correct": values[0],
                    "predicted": values[1],
                    "target": values[2],
                }
                for key, values in render_counts.items()
            },
            "schedule": [list(value) for value in schedule],
        }
        results[name] = metrics

    baseline = results["global-0.62"]
    eligible = []
    for name, result in results.items():
        constraints = {
            "f1": result["f1"] >= baseline["f1"] - 0.005,
            "split": result["false_split_rate"]
            <= baseline["false_split_rate"] * 1.05,
            "count": result["count_ratio"] <= baseline["count_ratio"],
        }
        result["objective_constraints"] = constraints
        if all(constraints.values()):
            eligible.append(name)
    winner = max(
        eligible,
        key=lambda name: (
            results[name]["short_recall_objective"],
            results[name]["f1"],
        ),
    )
    diagnostic_report = {
        key: {
            "count": values[0],
            "mean_probability": values[1] / max(values[0], 1),
            "mean_duration_feature_effect": values[2] / max(values[0], 1),
        }
        for key, values in diagnosis.items()
    }
    _atomic_json(
        args.output,
        {
            "schema_version": "align-track-a-train-fold-calibration-v1",
            "data_fingerprint": ready["hashes"]["pack_id"],
            "split": "train_group_disjoint_holdout",
            "heldout_leakage_groups": len(heldout_groups),
            "heldout_rows": len(examples),
            "validation_rows_opened": 0,
            "lockbox_touched": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_selected_threshold": checkpoint_threshold,
            "checkpoint_training_validation": checkpoint.get("validation"),
            "predeclared_objective": OBJECTIVE,
            "duration_feature_diagnosis": diagnostic_report,
            "results": results,
            "eligible": eligible,
            "winner": winner,
            "winner_schedule": results[winner]["schedule"],
        },
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
