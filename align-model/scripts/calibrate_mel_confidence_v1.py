"""Select a mel note-confidence gate using frozen training rows only."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_v1_data import MelPackedCache

try:
    from numba import njit
except ImportError:
    njit = None


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


def _lcs_python(left: Sequence[int], right: Sequence[int]) -> int:
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


if njit is not None:
    _lcs_compiled = njit(cache=True)(_lcs_python)
else:
    _lcs_compiled = None


def _lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
    if _lcs_compiled is None:
        return _lcs_python(left, right)
    return int(
        _lcs_compiled(
            np.asarray(left, dtype=np.int16),
            np.asarray(right, dtype=np.int16),
        )
    )


def _metrics(matched: int, predicted: int, target: int) -> dict[str, Any]:
    precision = matched / max(predicted, 1)
    recall = matched / max(target, 1)
    return {
        "matched": matched,
        "predicted": predicted,
        "target": target,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "count_ratio": predicted / max(target, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.0, 0.80, 0.85, 0.90, 0.92),
    )
    args = parser.parse_args()

    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Confidence thresholds must be within [0, 1]")
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    prediction_hash = sha256_file(args.predictions)
    if prediction_hash != freeze["predictions_sha256"]:
        raise ValueError("Frozen prediction checksum mismatch")
    if freeze.get("split") != "train" or freeze.get("target_column_read"):
        raise ValueError("Calibration requires target-free frozen train inference")
    predictions = {
        row["sample"]: row["notes"]
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    cache = MelPackedCache(args.cache, deep=False)
    if cache.pack_id != freeze["cache_pack_id"]:
        raise ValueError("Frozen predictions belong to another mel cache")
    records = cache.records("train", include_targets=True)
    if len(records) != len(predictions) or {
        record.sample for record in records
    } != set(predictions):
        raise ValueError("Calibration does not cover every training row")
    counts = {threshold: [0, 0, 0] for threshold in thresholds}
    for position, record in enumerate(records, 1):
        notes = predictions[record.sample]
        target_pitch = [int(value["pitch"]) for value in record.target]
        for threshold in thresholds:
            predicted_pitch = [
                int(value["pitch"])
                for value in notes
                if float(value["confidence"]) >= threshold
            ]
            values = counts[threshold]
            values[0] += _lcs_length(predicted_pitch, target_pitch)
            values[1] += len(predicted_pitch)
            values[2] += len(target_pitch)
        if position == 1 or position % 500 == 0 or position == len(records):
            print(f"calibrate={position}/{len(records)}", flush=True)
    cache.close()

    candidates = [
        {"min_confidence": threshold, **_metrics(*counts[threshold])}
        for threshold in thresholds
    ]
    selected = max(
        candidates,
        key=lambda row: (
            float(row["f1"]),
            -abs(float(row["count_ratio"]) - 1.0),
            -float(row["min_confidence"]),
        ),
    )
    report = {
        "schema_version": "align-mel-confidence-calibration-v1",
        "selection_split": "train",
        "selection_metric": "score-agnostic pitch-sequence LCS F1",
        "timestamp_metric_used": False,
        "rows": len(records),
        "predictions_sha256": prediction_hash,
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "cache_pack_id": cache.pack_id,
        "thresholds_predeclared": list(thresholds),
        "candidates": candidates,
        "selected": selected,
        "validation_opened": False,
        "locked_test_touched": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps({"selected": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
