"""Calibration ceiling of the gold DROP/EMIT path through the frozen CRF.

Reads only the frozen train/calibration release. Validation and lockbox targets
stay closed. This is an upper bound, not a deployable score.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

import eval_orn_phase2_baseline_v1 as baseline
import train_drop_emit_lattice_v1 as trainer
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--sequence-supervision", type=Path, required=True)
    parser.add_argument("--crf-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-crf-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    if sha256_file(args.release_manifest) != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    if sha256_file(args.crf_checkpoint) != args.expected_crf_sha256:
        raise ValueError("Frozen CRF mismatch")

    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    target_path = release["artifacts"]["development_targets"]["path"]
    with gzip.open(target_path, "rt", encoding="utf-8") as stream:
        targets = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] == "calibration"
        }
    with gzip.open(args.candidate_pool, "rt", encoding="utf-8") as stream:
        pools = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] == "calibration"
        }
    with gzip.open(args.sequence_supervision, "rt", encoding="utf-8") as stream:
        supervision = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] == "calibration"
        }

    cal_rows = list(release["splits"]["development"]["calibration"])
    if args.max_rows:
        cal_rows = cal_rows[: args.max_rows]
    crf = trainer._load_crf(args.crf_checkpoint)
    device = torch.device("cpu")
    started = time.time()
    by_mode: dict[str, list[JointMetricSample]] = {
        "teacher_identity": [],
        "teacher_through_crf": [],
    }
    for position, release_row in enumerate(cal_rows, 1):
        sample = release_row["sample"]
        lattice, index, score_path = trainer._build_row(
            release_row, pools[sample], supervision[sample], targets[sample]
        )
        row_started = time.time()
        for mode in by_mode:
            predicted, deletions, _diagnostics = trainer._predict_row(
                model=None,
                crf=crf,
                lattice=lattice,
                index=index,
                score_path=score_path,
                mode=mode,
                device=device,
            )
            by_mode[mode].append(
                JointMetricSample(
                    predicted=predicted,
                    target=index.rendered_events,
                    source=release_row["leakage_group"],
                    predicted_deletions=deletions,
                    target_deletions=index.deleted_event_indices,
                    score_event_count=len(index.events),
                )
            )
        print(
            f"row={position}/{len(cal_rows)} seconds={time.time() - row_started:.2f}",
            flush=True,
        )

    reports = {
        mode: baseline._full_report(samples, seed=20260919, replicates=1000)
        for mode, samples in by_mode.items()
    }
    payload = {
        "schema_version": "align-drop-emit-teacher-ceiling-v1",
        "population": "calibration",
        "rows": len(cal_rows),
        "seconds": time.time() - started,
        "reports": reports,
        "note": (
            "teacher_identity uses gold rendered identities. "
            "teacher_through_crf is the ceiling if DROP/EMIT emits the gold path "
            "and the frozen CRF assigns identities. Neither is a deployable score."
        ),
        "open_validation_read": False,
        "lockbox_targets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for mode, report in reports.items():
        print(
            f"{mode} f1={report['f1']:.4f} p={report['precision']:.4f} "
            f"r={report['recall']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
