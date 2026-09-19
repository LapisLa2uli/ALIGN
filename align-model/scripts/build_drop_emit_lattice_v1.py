"""Build DROP/EMIT supervision + prove gold-path coverage on train/cal only."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignmodel.joint.drop_emit_lattice_v1 import (
    SCHEMA_VERSION,
    build_drop_emit_lattice,
    exact_identity_round_trip,
    prove_gold_path_coverage,
    teacher_decode,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--sequence-supervision", type=Path, required=True)
    parser.add_argument("--expected-supervision-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args(argv)

    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    pool_sha = sha256_file(args.candidate_pool)
    if pool_sha != args.expected_pool_sha256:
        raise ValueError("Candidate pool mismatch")
    supervision_sha = sha256_file(args.sequence_supervision)
    if supervision_sha != args.expected_supervision_sha256:
        raise ValueError("Sequence supervision mismatch")

    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    target_artifact = release["artifacts"]["development_targets"]
    with gzip.open(target_artifact["path"], "rt", encoding="utf-8") as stream:
        targets = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }
    with gzip.open(args.candidate_pool, "rt", encoding="utf-8") as stream:
        pools = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }
    with gzip.open(args.sequence_supervision, "rt", encoding="utf-8") as stream:
        supervision = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }

    rows_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    acoustic_emits = 0
    template_inserts = 0
    dropped_groups = 0

    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-v2-drop-emit-lattice-coverage",
        command=[str(Path(__file__).resolve()), *map(str, argv or [])],
        metadata={"rows": len(pools), "locked_test": False},
    ):
        for split in ("train", "calibration"):
            for position, release_row in enumerate(
                release["splits"]["development"][split], 1
            ):
                sample = release_row["sample"]
                pool = pools[sample]
                truth = supervision[sample]
                target_row = targets[sample]
                index = ScoreEventIndex.from_musicxml(
                    Path(release_row["sample_dir"]) / "verified_score.musicxml",
                    target_row["lineage"],
                )
                lattice = build_drop_emit_lattice(
                    sample=sample,
                    split=split,
                    pool_candidates=pool["candidates"],
                    targets=index.rendered_events,
                    assignments=truth["assignments"],
                    target_deletions=index.deleted_event_indices,
                )
                coverage = prove_gold_path_coverage(lattice)
                round_trip = exact_identity_round_trip(lattice)
                if not coverage["passed"] or not round_trip["passed"]:
                    failures.append(
                        {
                            "sample": sample,
                            "split": split,
                            "coverage": coverage,
                            "round_trip": round_trip,
                        }
                    )
                else:
                    acoustic_emits += int(coverage["acoustic_emits"])
                    template_inserts += int(coverage["template_inserts"])
                    dropped_groups += int(coverage["dropped_groups"])
                teacher = (
                    [
                        {
                            "action": step.action,
                            "group_index": step.group_index,
                            "candidate_index": step.candidate_index,
                            "rendered_index": step.target_hint.rendered_index
                            if step.target_hint is not None
                            else None,
                            "relationship": (
                                step.target_hint.relationship
                                if step.target_hint is not None
                                else None
                            ),
                            "score_span": (
                                list(step.target_hint.score_span)
                                if step.target_hint is not None
                                and step.target_hint.score_span is not None
                                else None
                            ),
                            "copy_pass": (
                                step.target_hint.copy_pass
                                if step.target_hint is not None
                                else None
                            ),
                            "pitch": step.candidate.pitch,
                        }
                        for step in teacher_decode(lattice)
                    ]
                    if coverage["passed"]
                    else []
                )
                rows_out.append(
                    {
                        "sample": sample,
                        "split": split,
                        "groups": len(lattice.groups),
                        "targets": len(lattice.targets),
                        "coverage": coverage,
                        "round_trip": {
                            "passed": round_trip["passed"],
                            "events": round_trip.get("events"),
                        },
                        "teacher_path": teacher,
                        "assignments": truth["assignments"],
                        "pool_candidate_count": len(pool["candidates"]),
                    }
                )
                if position == 1 or position % 64 == 0 or position == len(
                    release["splits"]["development"][split]
                ):
                    print(
                        f"{split}={position}/"
                        f"{len(release['splits']['development'][split])} "
                        f"failures={len(failures)}",
                        flush=True,
                    )

    if failures:
        _atomic_json(args.output_dir / "COVERAGE_FAILURES.json", {"failures": failures})
        raise SystemExit(
            f"DROP/EMIT gold-path coverage failed on {len(failures)} rows; aborting"
        )

    supervision_path = args.output_dir / "supervision.jsonl.gz"
    _atomic_jsonl_gz(supervision_path, rows_out)
    report = {
        "schema_version": f"{SCHEMA_VERSION}-coverage",
        "status": "passed",
        "release_manifest_sha256": release_sha,
        "candidate_pool_sha256": pool_sha,
        "sequence_supervision_sha256": supervision_sha,
        "supervision_sha256": sha256_file(supervision_path),
        "population": {
            "train": 901,
            "calibration": 64,
            "rows": len(rows_out),
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
        "gold_path_coverage": 1.0,
        "exact_identity_round_trip": 1.0,
        "coverage_failures": 0,
        "totals": {
            "acoustic_emits": acoustic_emits,
            "template_inserts": template_inserts,
            "dropped_groups": dropped_groups,
            "target_events": sum(row["targets"] for row in rows_out),
        },
        "isolation": {
            "validation_targets_read": False,
            "lockbox_targets_read": False,
            "production_weights_mutated": False,
            "timestamps_used_for_promotion": False,
        },
    }
    report_path = args.output_dir / "coverage_report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": len(rows_out),
                "gold_path_coverage": 1.0,
                "exact_identity_round_trip": 1.0,
                "coverage_report_sha256": sha256_file(report_path),
                "supervision_sha256": report["supervision_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
