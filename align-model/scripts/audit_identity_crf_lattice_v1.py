"""Prove CRF gold-path coverage and exact identity round-trip on train/calibration."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignmodel.joint.identity_crf_v1 import (
    IdentityCandidate,
    build_identity_lattice,
    exact_identity_round_trip,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-release-sha256",
        default=EXPECTED_RELEASE_SHA256,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.release_manifest) != args.expected_release_sha256:
        raise ValueError("Frozen ORN release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
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
    if set(targets) != {row["sample"] for row in rows}:
        raise ValueError("Coverage population mismatch")
    failures = []
    summaries = []
    type_support: Counter[str] = Counter()
    split_support: Counter[str] = Counter()
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="ornament-identity-crf-v1-coverage",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(rows), "locked_test": False},
    ):
        for position, row in enumerate(rows, 1):
            target_row = targets[row["sample"]]
            lineage = target_row["lineage"]
            score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
            index = None
            try:
                index = ScoreEventIndex.from_musicxml(score_path, lineage)
                candidates = tuple(
                    IdentityCandidate(
                        event.pitch,
                        event.start,
                        event.end,
                        1.0,
                    )
                    for event in index.rendered_events
                )
                lattice = build_identity_lattice(
                    candidates,
                    index.events,
                    score_path,
                    targets=index.rendered_events,
                    target_deletions=index.deleted_event_indices,
                )
                round_trip = exact_identity_round_trip(lattice)
                if not round_trip["passed"]:
                    raise ValueError(f"round-trip failed: {round_trip}")
                summary = {
                    "sample": row["sample"],
                    "split": target_row["split"],
                    "score_events": len(index.events),
                    "rendered_events": len(index.rendered_events),
                    "deleted_events": len(index.deleted_event_indices),
                    "selected_hypotheses": len(lattice.hypotheses),
                    "gold_hypotheses": sum(
                        value.is_gold_compatible
                        for value in lattice.hypotheses
                    ),
                    "round_trip": round_trip,
                }
                summaries.append(summary)
                split_support[target_row["split"]] += 1
                type_support.update(
                    "copy" if event.is_copy else event.relationship
                    for event in index.rendered_events
                )
                type_support["missed_note"] += len(
                    index.deleted_event_indices
                )
            except Exception as exc:
                linked = (
                    [
                        {
                            "span": event.score_span,
                            "copy_pass": event.copy_pass,
                            "relationship": event.relationship,
                        }
                        for event in index.rendered_events
                        if event.score_span is not None
                    ]
                    if index is not None
                    else []
                )
                failures.append(
                    {
                        "sample": row["sample"],
                        "split": target_row["split"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "score_events": (
                            len(index.events) if index is not None else None
                        ),
                        "linked_events": len(linked),
                        "multi_span_events": sum(
                            value["span"][1] - value["span"][0] > 1
                            for value in linked
                        ),
                        "backward_identity_steps": sum(
                            right["span"][0] < left["span"][0]
                            and right["copy_pass"] == left["copy_pass"]
                            for left, right in zip(linked, linked[1:])
                        ),
                        "linked_examples": linked[:20],
                    }
                )
            if position == 1 or position % 25 == 0 or position == len(rows):
                print(
                    f"coverage={position}/{len(rows)} failures={len(failures)}",
                    flush=True,
                )
    report = {
        "schema_version": "align-ornament-identity-crf-coverage-v1",
        "release_manifest_sha256": sha256_file(args.release_manifest),
        "rows": len(rows),
        "split_support": dict(sorted(split_support.items())),
        "type_support": dict(sorted(type_support.items())),
        "covered_rows": len(summaries),
        "coverage_fraction": len(summaries) / max(len(rows), 1),
        "failures": failures,
        "failure_summary": {
            "rows_with_backward_identity_steps": sum(
                int(row.get("backward_identity_steps") or 0) > 0
                for row in failures
            ),
            "backward_identity_steps": sum(
                int(row.get("backward_identity_steps") or 0)
                for row in failures
            ),
            "rows_with_multi_span_events": sum(
                int(row.get("multi_span_events") or 0) > 0
                for row in failures
            ),
            "multi_span_events": sum(
                int(row.get("multi_span_events") or 0)
                for row in failures
            ),
            "no_gold_compatible_path": sum(
                "no gold-compatible" in row["error"] for row in failures
            ),
            "round_trip_failure": sum(
                "round-trip failed" in row["error"] for row in failures
            ),
            "by_split": {
                split: sum(row["split"] == split for row in failures)
                for split in ("train", "calibration")
            },
        },
        "per_row": summaries,
        "all_gold_paths_covered": not failures and len(summaries) == len(rows),
        "all_exact_identity_round_trips": not failures
        and all(row["round_trip"]["passed"] for row in summaries),
        "open_validation_read": False,
        "lockbox_targets_read": False,
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "coverage_fraction": report["coverage_fraction"],
                "failures": len(failures),
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
