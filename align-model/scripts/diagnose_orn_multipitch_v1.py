"""Calibration-only diagnosis of multi-pitch and Track-B candidate unions."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1_data import MelPackedCache
from alignmodel.transcription.ornament_multipitch_v1 import (
    MultiPitchConfig,
    MultiPitchDecodeConfig,
    OrnamentMultiPitchTranscriber,
    decode_multipitch_notes,
    infer_multipitch_probabilities,
)


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)
UNION_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


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


def _event(row: Mapping[str, Any], index: int):
    from alignmodel.joint.index import JointEvent

    return JointEvent(
        pitch=int(row["pitch"]),
        start=float(row.get("start", row.get("start_sec"))),
        end=float(row.get("end", row.get("end_sec"))),
        score_span=None,
        relationship="extra",
        rendered_index=index,
    )


def _union(
    track_b: Sequence[Mapping[str, Any]],
    multipitch: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in track_b]
    for row in multipitch:
        if float(row["confidence"]) < threshold:
            continue
        duplicate = any(
            int(existing["pitch"]) == int(row["pitch"])
            and abs(float(existing["start"]) - float(row["start"])) <= 0.050
            for existing in output
        )
        if not duplicate:
            output.append(dict(row))
    return sorted(
        output,
        key=lambda row: (
            float(row["start"]),
            int(row["pitch"]),
            float(row["end"]),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--orn-cache", type=Path, required=True)
    parser.add_argument("--trackb-predictions", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if sha256_file(args.release_manifest) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
    payload = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model = OrnamentMultiPitchTranscriber(
        MultiPitchConfig.from_dict(payload["model_config"])
    ).to(args.device)
    model.load_state_dict(payload["model_state_dict"])
    decode = MultiPitchDecodeConfig.from_dict(payload["decode_config"])
    cache = MelPackedCache(args.orn_cache, deep=False)
    records = cache.records("val", include_targets=True)
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
    if set(track_b) != {record.sample for record in records}:
        raise ValueError("Track-B calibration population mismatch")
    predictions = {}
    with resource_lease(
        args.resource_status,
        "gpu",
        track="ornament-multipitch-v1-calibration-diagnosis",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(records), "locked_test": False},
    ):
        for position, record in enumerate(records, 1):
            probability = infer_multipitch_probabilities(
                model,
                np.asarray(cache.mel(record), np.float32),
                args.device,
                batch_size=args.batch_size,
            )
            predictions[record.sample] = [
                value.to_dict()
                for value in decode_multipitch_notes(
                    probability,
                    midi_min=model.midi_min,
                    hop_sec=cache.frontend.hop_sec,
                    config=decode,
                )
            ]
            if position == 1 or position % 10 == 0 or position == len(records):
                print(f"diagnose={position}/{len(records)}", flush=True)
    strategies = {"track_b": track_b, "multipitch": predictions}
    for threshold in UNION_THRESHOLDS:
        strategies[f"union_conf_{threshold:.2f}"] = {
            sample: _union(
                track_b[sample],
                predictions[sample],
                threshold=threshold,
            )
            for sample in track_b
        }
    target_by_sample = {
        record.sample: list(record.target) for record in records
    }
    reports = {}
    for name, values in strategies.items():
        lcs_counts = [0, 0, 0]
        onset_counts = [0, 0, 0]
        for sample in sorted(values):
            predicted_rows = values[sample]
            target_rows = target_by_sample[sample]
            lcs_counts[0] += _lcs(
                [int(row["pitch"]) for row in predicted_rows],
                [int(row["pitch"]) for row in target_rows],
            )
            lcs_counts[1] += len(predicted_rows)
            lcs_counts[2] += len(target_rows)
            pairs = pair_exact_pitch_onset(
                tuple(_event(row, index) for index, row in enumerate(predicted_rows)),
                tuple(_event(row, index) for index, row in enumerate(target_rows)),
                tolerance_sec=0.050,
            )
            onset_counts[0] += len(pairs)
            onset_counts[1] += len(predicted_rows)
            onset_counts[2] += len(target_rows)
        reports[name] = {
            "score_agnostic_pitch_sequence": _prf(*lcs_counts),
            "onset_50ms_diagnostic_only": _prf(*onset_counts),
        }
    selected_name, selected = max(
        reports.items(),
        key=lambda item: (
            item[1]["score_agnostic_pitch_sequence"]["f1"]
            - 0.03
            * abs(
                math.log(
                    max(
                        item[1]["score_agnostic_pitch_sequence"]["count_ratio"],
                        1e-5,
                    )
                )
            ),
            item[1]["score_agnostic_pitch_sequence"]["f1"],
        ),
    )
    _atomic_json(
        args.output,
        {
            "schema_version": "align-orn-multipitch-calibration-diagnosis-v1",
            "release_manifest_sha256": sha256_file(args.release_manifest),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_decode": decode.to_dict(),
            "population": {"split": "calibration", "rows": len(records)},
            "strategies": reports,
            "selection": {
                "metric": "score_agnostic_pitch_sequence_lcs",
                "selected": selected_name,
                "selected_report": selected,
                "timestamp_diagnostic_not_used_for_selection": True,
            },
            "predictions": {
                name: values for name, values in strategies[selected_name].items()
            },
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    cache.close()
    print(json.dumps({"selected": selected_name, **selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
