"""Build and audit a high-recall Basic Pitch activation lattice on v2 train/cal."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

import calibrate_v2_acoustic_crf as acoustic
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    basic_pitch_cache_path,
    decode_basic_pitch_features,
    extract_sample_basic_pitch_features,
)
from alignmodel.transcription.basic_pitch_lattice_v1 import (
    SCHEMA_VERSION,
    ActivationCandidate,
    generate_activation_lattice,
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


def _atomic_jsonl_gz(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as stream:
                for row in rows:
                    stream.write(
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    stream.write(b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _pair(
    candidates: Sequence[ActivationCandidate],
    target: Sequence[Any],
    tolerance: float = 0.050,
) -> list[tuple[int, int]]:
    if not candidates or not target:
        return []
    weights = np.zeros((len(candidates), len(target)), np.float64)
    for candidate_index, candidate in enumerate(candidates):
        for target_index, event in enumerate(target):
            if candidate.pitch != event.pitch:
                continue
            delta = abs(candidate.start - event.start)
            if delta <= tolerance:
                weights[candidate_index, target_index] = 1.0 - delta
    rows, columns = linear_sum_assignment(-weights)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if weights[row, column] > 0.0
    ]


def _duration(seconds: float) -> str:
    if seconds < 0.080:
        return "lt_80ms"
    if seconds < 0.120:
        return "80_to_120ms"
    if seconds < 0.180:
        return "120_to_180ms"
    return "ge_180ms"


def _prf(matched: int, gold: int) -> dict[str, Any]:
    precision = 1.0 if matched else (1.0 if not gold else 0.0)
    recall = matched / max(gold, 1)
    return {
        "matched": matched,
        "selected_predictions": matched,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=1024)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen ORN v2 release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("ORN v2 lockbox opening sentinel exists")
    targets = {}
    target_artifact = release["artifacts"]["development_targets"]
    target_path = Path(target_artifact["path"])
    if sha256_file(target_path) != target_artifact["sha256"]:
        raise ValueError("Development target archive mismatch")
    with gzip.open(target_path, "rt", encoding="utf-8") as stream:
        targets = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }
    rows = [
        row
        for split in ("train", "calibration")
        for row in release["splits"]["development"][split]
    ]
    pools = []
    supervision = []
    total_candidates = 0
    total_target = 0
    total_matched = 0
    support: Counter[str] = Counter()
    matched_support: Counter[str] = Counter()
    exact_extra_identity = 0
    extra_support = 0
    decode = BasicPitchDecodeConfig()
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-v2-basic-pitch-activation-lattice",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(rows), "locked_test": False},
    ):
        for position, row in enumerate(rows, 1):
            cache_path = basic_pitch_cache_path(
                args.feature_cache,
                row["sample_dir"],
                f"orn-v2-{row['split']}",
            )
            features = extract_sample_basic_pitch_features(
                row["sample_dir"], cache_path=cache_path
            )
            standard = decode_basic_pitch_features(features, decode)
            candidates = generate_activation_lattice(
                features,
                standard,
                max_candidates=args.max_candidates,
            )
            target_row = targets[row["sample"]]
            index = ScoreEventIndex.from_musicxml(
                Path(row["sample_dir"]) / "verified_score.musicxml",
                target_row["lineage"],
            )
            pairs = _pair(candidates, index.rendered_events)
            pair_by_target = {target: candidate for candidate, target in pairs}
            pool_row = {
                "sample": row["sample"],
                "split": row["split"],
                "audio_sha256": row["source_hashes"]["performance_audio.wav"],
                "basic_pitch_cache": str(cache_path.resolve()),
                "basic_pitch_cache_sha256": sha256_file(cache_path),
                "standard_note_count": len(standard),
                "candidates": [value.to_dict() for value in candidates],
            }
            pools.append(pool_row)
            assignments = []
            prefix_complete = True
            for target_index, event in enumerate(index.rendered_events):
                kind = "copy" if event.is_copy else event.relationship
                keys = [
                    f"type:{kind}",
                    f"duration:{_duration(event.end - event.start)}",
                    (
                        "overlap:yes"
                        if any(
                            other_index != target_index
                            and other.start < event.end
                            and event.start < other.end
                            for other_index, other in enumerate(index.rendered_events)
                        )
                        else "overlap:no"
                    ),
                ]
                rendered_row = target_row["lineage"]["rendered_notes"][
                    int(event.rendered_index)
                ]
                keys.append(
                    f"origin:{rendered_row.get('extra_origin') or 'linked'}"
                )
                for key in keys:
                    support[key] += 1
                candidate_index = pair_by_target.get(target_index)
                if candidate_index is None:
                    prefix_complete = False
                    continue
                for key in keys:
                    matched_support[key] += 1
                if event.is_extra:
                    extra_support += 1
                    exact_extra_identity += int(prefix_complete)
                assignments.append(
                    {
                        "candidate_index": candidate_index,
                        "target_rendered_index": int(event.rendered_index),
                        "target_pitch": event.pitch,
                        "target_score_span": event.score_span,
                        "target_relationship": kind,
                        "target_copy_pass": event.copy_pass,
                        "onset_error_sec": abs(
                            candidates[candidate_index].start - event.start
                        ),
                    }
                )
            supervision.append(
                {
                    "sample": row["sample"],
                    "split": row["split"],
                    "candidate_pool_sha256": hashlib.sha256(
                        json.dumps(
                            pool_row["candidates"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "assignments": assignments,
                    "target_events": len(index.rendered_events),
                    "matched_events": len(pairs),
                    "full_sequence_covered": len(pairs)
                    == len(index.rendered_events),
                }
            )
            total_candidates += len(candidates)
            total_target += len(index.rendered_events)
            total_matched += len(pairs)
            if position == 1 or position % 20 == 0 or position == len(rows):
                print(
                    f"lattice={position}/{len(rows)} "
                    f"coverage={total_matched/max(total_target,1):.4f}",
                    flush=True,
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "candidate_pool.jsonl.gz"
    supervision_path = args.output_dir / "supervision.jsonl.gz"
    _atomic_jsonl_gz(pool_path, pools)
    _atomic_jsonl_gz(supervision_path, supervision)
    report_path = args.output_dir / "coverage_report.json"
    report = {
        "schema_version": f"{SCHEMA_VERSION}-coverage",
        "release_manifest_sha256": release_sha,
        "population": {
            "train": 901,
            "calibration": 64,
            "rows": len(rows),
        },
        "candidate_generation": {
            "score_input": False,
            "onset_floor": 0.04,
            "note_floors": [0.04, 0.08, 0.14],
            "fixed_duration_frames": [2, 4, 8],
            "max_onsets_per_pitch": 32,
            "max_candidates_per_row": args.max_candidates,
            "total_candidates": total_candidates,
        },
        "candidate_oracle": {
            **_prf(total_matched, total_target),
            "rows_with_full_sequence_coverage": sum(
                row["full_sequence_covered"] for row in supervision
            ),
            "exact_rendered_extra_prefix_identity": {
                "matched": exact_extra_identity,
                "support": extra_support,
                "recall": exact_extra_identity / max(extra_support, 1),
            },
            "by_stratum": {
                key: {
                    "matched": matched_support[key],
                    "support": count,
                    "recall": matched_support[key] / max(count, 1),
                }
                for key, count in sorted(support.items())
            },
        },
        "artifacts": {
            "candidate_pool": {
                "path": str(pool_path.resolve()),
                "sha256": sha256_file(pool_path),
            },
            "supervision": {
                "path": str(supervision_path.resolve()),
                "sha256": sha256_file(supervision_path),
            },
        },
        "timestamps": "used only to audit acoustic candidate support and create train labels; not a promotion metric",
        "open_validation_read": False,
        "lockbox_targets_read": False,
    }
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "candidate_oracle": report["candidate_oracle"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
