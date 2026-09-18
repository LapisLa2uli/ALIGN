"""Calibrate a score-free Basic-Pitch/Track-B event union on 53 rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignmodel.joint.index import JointEvent
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_v1_data import MelPackedCache


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)
CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80)
DEDUP_TOLERANCES = (0.030, 0.050, 0.080)


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


def _event(row: Mapping[str, Any], index: int) -> JointEvent:
    return JointEvent(
        pitch=int(row["pitch"]),
        start=float(row.get("start", row.get("start_sec"))),
        end=float(row.get("end", row.get("end_sec"))),
        score_span=None,
        relationship="extra",
        rendered_index=index,
    )


def _union(
    base: Sequence[Mapping[str, Any]],
    additions: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
    tolerance: float,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in base]
    for row in additions:
        if float(row.get("confidence", 1.0)) < confidence:
            continue
        if any(
            int(existing["pitch"]) == int(row["pitch"])
            and abs(float(existing["start"]) - float(row["start"]))
            <= tolerance
            for existing in output
        ):
            continue
        output.append(dict(row))
    return sorted(
        output,
        key=lambda row: (
            float(row["start"]),
            int(row["pitch"]),
            float(row["end"]),
        ),
    )


def _score(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    lcs_counts = [0, 0, 0]
    onset_counts = [0, 0, 0]
    for sample in sorted(targets):
        predicted = predictions[sample]
        target = targets[sample]
        lcs_counts[0] += _lcs(
            [int(row["pitch"]) for row in predicted],
            [int(row["pitch"]) for row in target],
        )
        lcs_counts[1] += len(predicted)
        lcs_counts[2] += len(target)
        pairs = pair_exact_pitch_onset(
            tuple(_event(row, index) for index, row in enumerate(predicted)),
            tuple(_event(row, index) for index, row in enumerate(target)),
            tolerance_sec=0.050,
        )
        onset_counts[0] += len(pairs)
        onset_counts[1] += len(predicted)
        onset_counts[2] += len(target)
    lcs = _prf(*lcs_counts)
    return {
        "pitch_sequence_lcs": lcs,
        "onset_50ms_diagnostic_only": _prf(*onset_counts),
        "rank": lcs["f1"]
        - 0.03 * abs(math.log(max(lcs["count_ratio"], 1e-5))),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--orn-cache", type=Path, required=True)
    parser.add_argument("--basic-pitch-report", type=Path, required=True)
    parser.add_argument("--trackb-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.release_manifest) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release mismatch")
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
    basic_report = json.loads(
        args.basic_pitch_report.read_text(encoding="utf-8")
    )
    basic_variant = next(
        row
        for row in basic_report["variants"]
        if int(row["variant"]) == int(basic_report["selected_variant"])
    )
    basic = basic_variant["predictions"]
    track_b = {
        row["sample"]: row["notes"]
        for row in (
            json.loads(line)
            for line in args.trackb_predictions.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    }
    cache = MelPackedCache(args.orn_cache, deep=False)
    records = cache.records("val", include_targets=True)
    targets = {record.sample: record.target for record in records}
    if set(targets) != set(basic) or set(basic) != set(track_b):
        raise ValueError("Frontend calibration populations differ")
    strategies: dict[str, dict[str, list[dict[str, Any]]]] = {
        "basic_pitch": basic,
        "track_b": track_b,
    }
    for confidence in CONFIDENCE_THRESHOLDS:
        for tolerance in DEDUP_TOLERANCES:
            strategies[
                f"trackb_plus_basic_c{confidence:.2f}_t{tolerance:.3f}"
            ] = {
                sample: _union(
                    track_b[sample],
                    basic[sample],
                    confidence=confidence,
                    tolerance=tolerance,
                )
                for sample in targets
            }
            strategies[
                f"basic_plus_trackb_c{confidence:.2f}_t{tolerance:.3f}"
            ] = {
                sample: _union(
                    basic[sample],
                    track_b[sample],
                    confidence=confidence,
                    tolerance=tolerance,
                )
                for sample in targets
            }
    reports = {
        name: _score(predictions, targets)
        for name, predictions in strategies.items()
    }
    selected_name, selected_report = max(
        reports.items(),
        key=lambda item: (
            item[1]["rank"],
            item[1]["pitch_sequence_lcs"]["f1"],
            item[0],
        ),
    )
    _atomic_json(
        args.output,
        {
            "schema_version": "align-orn-frontend-ensemble-calibration-v1",
            "release_manifest_sha256": sha256_file(args.release_manifest),
            "population": {"split": "calibration", "rows": len(targets)},
            "predeclared": {
                "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
                "dedup_tolerances_sec": list(DEDUP_TOLERANCES),
                "base_orders": ["track_b+basic_pitch", "basic_pitch+track_b"],
            },
            "reports": reports,
            "selection": {
                "metric": "score_agnostic_pitch_sequence_lcs",
                "selected": selected_name,
                "report": selected_report,
                "timestamp_diagnostic_not_used_for_selection": True,
            },
            "selected_predictions": strategies[selected_name],
            "input_hashes": {
                "basic_pitch_report": sha256_file(args.basic_pitch_report),
                "trackb_predictions": sha256_file(args.trackb_predictions),
            },
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    cache.close()
    print(
        json.dumps(
            {"selected": selected_name, **selected_report},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
