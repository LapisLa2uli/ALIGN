"""Train-fold-only error analysis for frozen high-resolution mel predictions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignmodel.joint.index import JointEvent
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.packed_data import PackedJointDataset, sha256_file


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _event(value: Mapping[str, Any]) -> JointEvent:
    return JointEvent(
        pitch=int(value["pitch"]),
        start=float(value.get("start", value.get("start_sec"))),
        end=float(value.get("end", value.get("end_sec"))),
        score_span=(
            tuple(int(item) for item in value["score_span"])
            if value.get("score_span") is not None
            else None
        ),
        relationship=str(value.get("relationship") or "match"),
        copy_pass=int(value.get("copy_pass") or 0),
        origin_relationship=value.get("origin_relationship"),
        rendered_index=value.get("rendered_index"),
        source_indices=tuple(
            int(item) for item in value.get("source_indices") or ()
        ),
        confidence=float(value.get("confidence", 1.0)),
    )


def _predicted_event(value: Mapping[str, Any]) -> JointEvent:
    return JointEvent(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        score_span=None,
        relationship="extra",
        confidence=float(value["confidence"]),
    )


def _lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
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


def _duration_bin(seconds: float) -> str:
    if seconds < 0.080:
        return "lt_80ms"
    if seconds < 0.120:
        return "80_to_120ms"
    if seconds < 0.180:
        return "120_to_180ms"
    return "ge_180ms"


def _confidence_bin(value: float) -> str:
    if value < 0.50:
        return "lt_0.50"
    if value < 0.65:
        return "0.50_to_0.65"
    if value < 0.80:
        return "0.65_to_0.80"
    if value < 0.90:
        return "0.80_to_0.90"
    return "ge_0.90"


def _rate_rows(
    support: Counter[str], errors: Counter[str], *, error_name: str
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "support": int(support[key]),
            error_name: int(errors[key]),
            f"{error_name}_rate": errors[key] / max(support[key], 1),
        }
        for key in sorted(support)
    }


def _prf(matched: int, predicted: int, target: int) -> dict[str, float | int]:
    precision = matched / max(predicted, 1)
    recall = matched / max(target, 1)
    return {
        "matched": matched,
        "predicted": predicted,
        "target": target,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        "count_ratio": predicted / max(target, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()

    ready = json.loads(args.ready_marker.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    prediction_hash = sha256_file(args.predictions)
    if prediction_hash != freeze["predictions_sha256"]:
        raise ValueError("Frozen prediction checksum mismatch")
    if freeze["release_pack_id"] != ready["hashes"]["pack_id"]:
        raise ValueError("Frozen predictions belong to another release")
    if (
        freeze.get("score_input")
        or freeze.get("target_column_read")
        or freeze.get("locked_test_materialized")
    ):
        raise ValueError("Inference-isolation manifest failed")
    predictions = {
        row["sample"]: row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if len(predictions) != int(ready["counts"][args.split]):
        raise ValueError(
            f"Diagnostic requires every audited {args.split} row"
        )

    manifest_path = Path(ready["paths"]["manifest"])
    if sha256_file(manifest_path) != ready["hashes"]["manifest_sha256"]:
        raise ValueError("Audited manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = {
        str(row["sample"]): row for row in manifest[args.split]
    }
    if set(predictions) != set(manifest_rows):
        raise ValueError("Frozen prediction IDs do not equal audited train IDs")

    duration_support: Counter[str] = Counter()
    duration_misses: Counter[str] = Counter()
    pitch_support: Counter[str] = Counter()
    pitch_misses: Counter[str] = Counter()
    relationship_support: Counter[str] = Counter()
    relationship_misses: Counter[str] = Counter()
    augmentation_counts: dict[str, list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    confidence_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    false_duration: Counter[str] = Counter()
    false_pitch: Counter[str] = Counter()
    boundary_false: Counter[str] = Counter()
    group_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    lcs_counts = [0, 0, 0]
    onset_counts = [0, 0, 0]
    predicted_splits = true_rearticulations = false_splits = 0
    target_rearticulations = 0

    with PackedJointDataset(
        Path(ready["paths"]["packed_root"]),
        manifest_sha256=ready["hashes"]["manifest_sha256"],
        verify_records=False,
        load_feature_arrays=False,
    ) as packed:
        packed_train = {
            item.sample: item
            for item in (
                packed[index] for index in packed.ordinals(args.split)
            )
        }
        for position, sample in enumerate(sorted(predictions), 1):
            row = manifest_rows[sample]
            prediction_notes = predictions[sample]["notes"]
            predicted = tuple(_predicted_event(note) for note in prediction_notes)
            target = tuple(
                _event(value)
                for value in packed_train[sample].target["target_events"]
            )
            pairs = pair_exact_pitch_onset(
                predicted, target, tolerance_sec=0.050
            )
            paired_predictions = {left for left, _right in pairs}
            paired_targets = {right for _left, right in pairs}
            pair_by_prediction = dict(pairs)

            correct = _lcs_length(
                [event.pitch for event in predicted],
                [event.pitch for event in target],
            )
            lcs_counts[0] += correct
            lcs_counts[1] += len(predicted)
            lcs_counts[2] += len(target)
            onset_counts[0] += len(pairs)
            onset_counts[1] += len(predicted)
            onset_counts[2] += len(target)

            for index, event in enumerate(target):
                duration = _duration_bin(event.end - event.start)
                pitch = str(event.pitch)
                relationship = event.relationship
                duration_support[duration] += 1
                pitch_support[pitch] += 1
                relationship_support[relationship] += 1
                if index not in paired_targets:
                    duration_misses[duration] += 1
                    pitch_misses[pitch] += 1
                    relationship_misses[relationship] += 1

            for index, (note, event) in enumerate(
                zip(prediction_notes, predicted)
            ):
                confidence = _confidence_bin(float(note["confidence"]))
                confidence_counts[confidence][1] += 1
                if index in paired_predictions:
                    confidence_counts[confidence][0] += 1
                    continue
                false_duration[_duration_bin(event.end - event.start)] += 1
                false_pitch[str(event.pitch)] += 1
                boundary_false[
                    "strong"
                    if max(
                        float(note.get("onset_strength", 0.0)),
                        float(note.get("boundary_strength", 0.0)),
                    )
                    >= 0.64
                    else "weak"
                ] += 1

            for left, right in zip(range(len(predicted) - 1), range(1, len(predicted))):
                first = predicted[left]
                second = predicted[right]
                if first.pitch != second.pitch or second.start - first.end > 0.100:
                    continue
                predicted_splits += 1
                target_left = pair_by_prediction.get(left)
                target_right = pair_by_prediction.get(right)
                is_true = (
                    target_left is not None
                    and target_right is not None
                    and target_right == target_left + 1
                    and target[target_left].pitch == target[target_right].pitch
                )
                if is_true:
                    true_rearticulations += 1
                else:
                    false_splits += 1
            target_rearticulations += sum(
                left.pitch == right.pitch
                for left, right in zip(target, target[1:])
            )

            keys = [
                *(f"augmentation:{value}" for value in row.get("error_types") or ()),
                f"source:{row['source']}",
                f"render:{row['audio_render']}",
            ]
            for key in keys:
                values = augmentation_counts[key]
                values[0] += len(pairs)
                values[1] += len(predicted)
                values[2] += len(target)
            for key in (f"source:{row['source']}", f"render:{row['audio_render']}"):
                values = group_counts[key]
                values[0] += correct
                values[1] += len(predicted)
                values[2] += len(target)
            if position == 1 or position % 250 == 0 or position == len(predictions):
                print(
                    f"diagnose={position}/{len(predictions)} sample={sample}",
                    flush=True,
                )

    metadata = json.loads(
        (
            args.predictions.parents[1]
            / "mel-cache-hop256-v1"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "align-mel-transcriber-train-diagnostics-v1",
        "split": args.split,
        "rows": len(predictions),
        "selection_uses_validation": args.split == "val",
        "selection_status": (
            "diagnostic_only_no_candidate_selection"
            if args.split == "val"
            else "train_only_candidate_selection_allowed"
        ),
        "timestamp_metrics_are_diagnostic_only": True,
        "inference_isolation": freeze,
        "integrity": {
            "predictions_sha256": prediction_hash,
            "manifest_sha256": ready["hashes"]["manifest_sha256"],
            "release_pack_id": ready["hashes"]["pack_id"],
            "cache_pack_id": metadata["pack_id"],
            "locked_test_touched": False,
        },
        "cache_and_pitch_policy": {
            "frontend": metadata["frontend_config"],
            "normalization": metadata["frontend_config"]["normalization"],
            "audio_pitch_spaces": sorted(
                {str(row["audio_pitch_space"]) for row in manifest_rows.values()}
            ),
            "effective_audio_transposes": sorted(
                {
                    int(row["effective_audio_transpose"])
                    for row in manifest_rows.values()
                }
            ),
            "target_pitch_space": "written",
            "model_midi_range": [52, 100],
        },
        "score_agnostic_pitch_sequence_lcs": _prf(*lcs_counts),
        "timestamp_diagnostic_50ms": _prf(*onset_counts),
        "misses": {
            "by_duration": _rate_rows(
                duration_support, duration_misses, error_name="misses"
            ),
            "by_pitch": _rate_rows(
                pitch_support, pitch_misses, error_name="misses"
            ),
            "by_relationship": _rate_rows(
                relationship_support, relationship_misses, error_name="misses"
            ),
        },
        "false_predictions": {
            "by_duration": dict(sorted(false_duration.items())),
            "by_pitch": dict(sorted(false_pitch.items(), key=lambda item: int(item[0]))),
            "by_boundary_strength": dict(sorted(boundary_false.items())),
        },
        "confidence_calibration": {
            key: {
                "matched_50ms": values[0],
                "predicted": values[1],
                "empirical_precision_50ms": values[0] / max(values[1], 1),
            }
            for key, values in sorted(confidence_counts.items())
        },
        "same_pitch_boundaries": {
            "predicted_split_count": predicted_splits,
            "predicted_split_rate": predicted_splits / max(onset_counts[1], 1),
            "true_rearticulation_count": true_rearticulations,
            "false_split_count": false_splits,
            "false_split_rate": false_splits / max(onset_counts[1], 1),
            "target_rearticulation_count": target_rearticulations,
            "target_rearticulation_rate": (
                target_rearticulations / max(onset_counts[2], 1)
            ),
        },
        "per_augmentation": {
            key: _prf(*values)
            for key, values in sorted(augmentation_counts.items())
        },
        "per_source_render": {
            key: _prf(*values) for key, values in sorted(group_counts.items())
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
