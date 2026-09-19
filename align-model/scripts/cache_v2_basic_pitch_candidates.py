"""Freeze score-free Basic Pitch candidates for ORN v2 train only."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import calibrate_v2_acoustic_crf as acoustic
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    basic_pitch_cache_path,
    decode_basic_pitch_features,
    extract_sample_basic_pitch_features,
)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen ORN v2 release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("ORN v2 lockbox opening sentinel exists")
    rows = release["splits"]["development"]["train"]
    if len(rows) != 901:
        raise ValueError("Frozen train population mismatch")
    predictions = []
    decode = BasicPitchDecodeConfig()
    with resource_lease(
        args.resource_status,
        "gpu",
        track="orn-v2-basic-pitch-train-cache",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(rows), "locked_test": False},
    ):
        for position, row in enumerate(rows, 1):
            audio_path = Path(row["sample_dir"]) / "performance_audio.wav"
            if (
                sha256_file(audio_path)
                != row["source_hashes"]["performance_audio.wav"]
            ):
                raise ValueError(f"Train audio changed: {row['sample']}")
            features = extract_sample_basic_pitch_features(
                row["sample_dir"],
                cache_path=basic_pitch_cache_path(
                    args.feature_cache, row["sample_dir"], "orn-v2-train"
                ),
            )
            notes = decode_basic_pitch_features(features, decode)
            candidates = acoustic._identity_candidates(notes, features)
            predictions.append(
                {
                    "sample": row["sample"],
                    "audio_sha256": row["source_hashes"][
                        "performance_audio.wav"
                    ],
                    "candidates": [
                        {
                            "pitch": value.pitch,
                            "start": value.start,
                            "end": value.end,
                            "confidence": value.confidence,
                            "alternatives": value.alternatives,
                            "alternative_confidences": value.alternative_confidences,
                        }
                        for value in candidates
                    ],
                }
            )
            if position == 1 or position % 20 == 0 or position == len(rows):
                print(f"cache={position}/{len(rows)}", flush=True)
    _atomic_jsonl(args.output, predictions)
    freeze = {
        "schema_version": "align-orn-v2-basic-pitch-train-candidates-v1",
        "release_manifest_sha256": release_sha,
        "split": "train",
        "rows": len(predictions),
        "decode": vars(decode),
        "predictions": str(args.output.resolve()),
        "predictions_sha256": sha256_file(args.output),
        "score_input": False,
        "development_targets_read": False,
        "open_validation_read": False,
        "lockbox_targets_read": False,
    }
    acoustic._atomic_json(args.output.with_suffix(".freeze.json"), freeze)
    print(json.dumps(freeze, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
