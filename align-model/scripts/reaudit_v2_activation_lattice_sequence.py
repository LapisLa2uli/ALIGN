"""Reaudit activation groups by official ordered pitch/event identity, not timestamps."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


@njit(cache=True)
def _lcs_pairs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((len(left) + 1, len(right) + 1), np.int32)
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                counts[i, j] = counts[i - 1, j - 1] + 1
            else:
                counts[i, j] = max(counts[i - 1, j], counts[i, j - 1])
    left_out = np.empty(counts[-1, -1], np.int32)
    right_out = np.empty(counts[-1, -1], np.int32)
    cursor = len(left_out) - 1
    i, j = len(left), len(right)
    while i and j:
        if left[i - 1] == right[j - 1]:
            left_out[cursor] = i - 1
            right_out[cursor] = j - 1
            cursor -= 1
            i -= 1
            j -= 1
        elif counts[i - 1, j] >= counts[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return left_out, right_out


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


def _atomic_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                for row in rows:
                    stream.write(
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":")
                        ).encode()
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


def _duration(seconds: float) -> str:
    if seconds < 0.080:
        return "lt_80ms"
    if seconds < 0.120:
        return "80_to_120ms"
    if seconds < 0.180:
        return "120_to_180ms"
    return "ge_180ms"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    if sha256_file(args.candidate_pool) != args.expected_pool_sha256:
        raise ValueError("Activation candidate pool mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    target_artifact = release["artifacts"]["development_targets"]
    with gzip.open(target_artifact["path"], "rt", encoding="utf-8") as stream:
        targets = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }
    with gzip.open(args.candidate_pool, "rt", encoding="utf-8") as stream:
        pools = [json.loads(line) for line in stream if line.strip()]
    supervision = []
    support: Counter[str] = Counter()
    matched_support: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-v2-activation-sequence-oracle",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(pools), "locked_test": False},
    ):
        for position, pool in enumerate(pools, 1):
            target_row = targets[pool["sample"]]
            release_row = next(
                row
                for row in release["splits"]["development"][pool["split"]]
                if row["sample"] == pool["sample"]
            )
            index = ScoreEventIndex.from_musicxml(
                Path(release_row["sample_dir"]) / "verified_score.musicxml",
                target_row["lineage"],
            )
            groups: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
            hop = 256 / 22050
            for candidate_index, candidate in enumerate(pool["candidates"]):
                groups[
                    (
                        int(candidate["pitch"]),
                        int(round(float(candidate["start"]) / hop)),
                    )
                ].append(candidate_index)
            ordered_groups = sorted(
                groups,
                key=lambda key: (
                    key[1],
                    key[0],
                    min(
                        float(pool["candidates"][value]["end"])
                        for value in groups[key]
                    ),
                ),
            )
            group_left, target_right = _lcs_pairs(
                np.asarray([key[0] for key in ordered_groups], np.int16),
                np.asarray(
                    [event.pitch for event in index.rendered_events], np.int16
                ),
            )
            assignments = []
            matched_targets = set()
            prefix_complete = True
            exact_extra_prefix = 0
            pair_by_target = {
                int(target_index): int(group_index)
                for group_index, target_index in zip(group_left, target_right)
            }
            for target_index, event in enumerate(index.rendered_events):
                rendered = target_row["lineage"]["rendered_notes"][
                    int(event.rendered_index)
                ]
                kind = "copy" if event.is_copy else event.relationship
                strata = (
                    f"type:{kind}",
                    f"duration:{_duration(event.end-event.start)}",
                    f"origin:{rendered.get('extra_origin') or 'linked'}",
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
                )
                support.update(strata)
                group_index = pair_by_target.get(target_index)
                if group_index is None:
                    prefix_complete = False
                    continue
                matched_targets.add(target_index)
                matched_support.update(strata)
                exact_extra_prefix += int(event.is_extra and prefix_complete)
                candidate_indices = groups[ordered_groups[group_index]]
                selected_candidate = min(
                    candidate_indices,
                    key=lambda value: (
                        abs(
                            float(pool["candidates"][value]["start"])
                            - event.start
                        ),
                        abs(
                            (
                                float(pool["candidates"][value]["end"])
                                - float(pool["candidates"][value]["start"])
                            )
                            - (event.end - event.start)
                        ),
                        -float(pool["candidates"][value]["confidence"]),
                    ),
                )
                assignments.append(
                    {
                        "candidate_group": group_index,
                        "candidate_indices": candidate_indices,
                        "selected_interval_candidate": selected_candidate,
                        "target_rendered_index": int(event.rendered_index),
                        "target_score_span": event.score_span,
                        "target_relationship": kind,
                        "target_copy_pass": event.copy_pass,
                    }
                )
            matched = len(matched_targets)
            totals["rows"] += 1
            totals["candidate_groups"] += len(ordered_groups)
            totals["target_events"] += len(index.rendered_events)
            totals["matched_events"] += matched
            totals["full_rows"] += int(matched == len(index.rendered_events))
            totals["exact_extra_prefix"] += exact_extra_prefix
            totals["extra_support"] += sum(
                event.is_extra for event in index.rendered_events
            )
            supervision.append(
                {
                    "sample": pool["sample"],
                    "split": pool["split"],
                    "candidate_groups": [
                        {
                            "pitch": key[0],
                            "onset_frame": key[1],
                            "candidate_indices": groups[key],
                        }
                        for key in ordered_groups
                    ],
                    "assignments": assignments,
                    "target_events": len(index.rendered_events),
                    "matched_events": matched,
                    "full_sequence_covered": matched
                    == len(index.rendered_events),
                }
            )
            if position == 1 or position % 25 == 0 or position == len(pools):
                print(
                    f"reaudit={position}/{len(pools)} "
                    f"coverage={totals['matched_events']/max(totals['target_events'],1):.4f}",
                    flush=True,
                )
    recall = totals["matched_events"] / max(totals["target_events"], 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supervision_path = args.output_dir / "sequence_supervision.jsonl.gz"
    _atomic_jsonl_gz(supervision_path, supervision)
    report = {
        "schema_version": "align-basic-pitch-activation-sequence-oracle-v1",
        "release_manifest_sha256": release_sha,
        "candidate_pool_sha256": sha256_file(args.candidate_pool),
        "population": {"train": 901, "calibration": 64, "rows": len(pools)},
        "totals": dict(totals),
        "candidate_oracle": {
            "precision": 1.0,
            "recall": recall,
            "f1": 2 * recall / (1 + recall),
            "rows_with_full_sequence_coverage": totals["full_rows"],
            "exact_rendered_extra_prefix_recall": totals[
                "exact_extra_prefix"
            ]
            / max(totals["extra_support"], 1),
            "by_stratum": {
                key: {
                    "matched": matched_support[key],
                    "support": value,
                    "recall": matched_support[key] / max(value, 1),
                }
                for key, value in sorted(support.items())
            },
        },
        "supervision": {
            "path": str(supervision_path.resolve()),
            "sha256": sha256_file(supervision_path),
        },
        "matching": "exclusive one-to-one LCS over unique pitch/onset proposal groups; interval variant chosen only after identity assignment",
        "timestamp_metrics_used_for_promotion": False,
        "open_validation_read": False,
        "lockbox_targets_read": False,
    }
    report_path = args.output_dir / "sequence_coverage_report.json"
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
