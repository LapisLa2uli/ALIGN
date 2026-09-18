"""Calibration-only polyphonic Basic Pitch diagnostic for ORN."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignmodel.joint.index import JointEvent
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    basic_pitch_cache_path,
    decode_basic_pitch_features,
    extract_sample_basic_pitch_features,
)
from alignmodel.transcription.mel_v1_data import MelPackedCache


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)
GRID = (
    BasicPitchDecodeConfig(),
    BasicPitchDecodeConfig(
        onset_threshold=0.40,
        frame_threshold=0.30,
        minimum_note_length_ms=30.0,
    ),
    BasicPitchDecodeConfig(
        onset_threshold=0.30,
        frame_threshold=0.20,
        minimum_note_length_ms=20.0,
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


def _event(value: Any, index: int) -> JointEvent:
    return JointEvent(
        pitch=int(value.pitch if hasattr(value, "pitch") else value["pitch"]),
        start=float(
            value.start
            if hasattr(value, "start")
            else value.get("start", value.get("start_sec"))
        ),
        end=float(
            value.end
            if hasattr(value, "end")
            else value.get("end", value.get("end_sec"))
        ),
        score_span=None,
        relationship="extra",
        rendered_index=index,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--orn-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.release_manifest) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
    rows = {
        row["sample"]: row
        for row in release["splits"]["development"]["calibration"]
    }
    target_cache = MelPackedCache(args.orn_cache, deep=False)
    records = target_cache.records("val", include_targets=True)
    if set(rows) != {record.sample for record in records}:
        raise ValueError("Basic Pitch calibration population mismatch")
    features = {}
    with resource_lease(
        args.resource_status,
        "gpu",
        track="orn-basic-pitch-v1-calibration",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(rows), "locked_test": False},
    ):
        for position, (sample, row) in enumerate(sorted(rows.items()), 1):
            features[sample] = extract_sample_basic_pitch_features(
                row["sample_dir"],
                cache_path=basic_pitch_cache_path(
                    args.feature_cache, row["sample_dir"], "orn-calibration"
                ),
            )
            if position == 1 or position % 10 == 0 or position == len(rows):
                print(f"basic-pitch={position}/{len(rows)}", flush=True)
    target = {record.sample: record.target for record in records}
    variants = []
    for index, config in enumerate(GRID):
        lcs_counts = [0, 0, 0]
        onset_counts = [0, 0, 0]
        predictions = {}
        for sample in sorted(rows):
            notes = decode_basic_pitch_features(features[sample], config)
            predictions[sample] = [
                {
                    "pitch": int(value.pitch),
                    "start": float(value.start),
                    "end": float(value.end),
                    "confidence": float(value.confidence),
                }
                for value in notes
            ]
            gold = target[sample]
            lcs_counts[0] += _lcs(
                [int(value.pitch) for value in notes],
                [int(value["pitch"]) for value in gold],
            )
            lcs_counts[1] += len(notes)
            lcs_counts[2] += len(gold)
            pairs = pair_exact_pitch_onset(
                tuple(_event(value, item) for item, value in enumerate(notes)),
                tuple(_event(value, item) for item, value in enumerate(gold)),
                tolerance_sec=0.050,
            )
            onset_counts[0] += len(pairs)
            onset_counts[1] += len(notes)
            onset_counts[2] += len(gold)
        lcs = _prf(*lcs_counts)
        variants.append(
            {
                "variant": index,
                "config": asdict(config),
                "pitch_sequence_lcs": lcs,
                "onset_50ms_diagnostic_only": _prf(*onset_counts),
                "rank": lcs["f1"]
                - 0.03 * abs(math.log(max(lcs["count_ratio"], 1e-5))),
                "predictions": predictions,
            }
        )
    selected = max(
        variants,
        key=lambda value: (
            value["rank"],
            value["pitch_sequence_lcs"]["f1"],
            -value["variant"],
        ),
    )
    _atomic_json(
        args.output,
        {
            "schema_version": "align-orn-basic-pitch-diagnostic-v1",
            "release_manifest_sha256": sha256_file(args.release_manifest),
            "population": {"split": "calibration", "rows": len(rows)},
            "predeclared_variants": len(GRID),
            "variants": variants,
            "selected_variant": selected["variant"],
            "selected_config": selected["config"],
            "selected_pitch_sequence": selected["pitch_sequence_lcs"],
            "selected_onset_50ms_diagnostic_only": selected[
                "onset_50ms_diagnostic_only"
            ],
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    target_cache.close()
    print(
        json.dumps(
            {
                "selected_variant": selected["variant"],
                "pitch_sequence": selected["pitch_sequence_lcs"],
                "onset_50ms_diagnostic_only": selected[
                    "onset_50ms_diagnostic_only"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
