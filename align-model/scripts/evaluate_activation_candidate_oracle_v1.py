"""Calibration-only canonical oracle for the frozen activation candidate pool."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import calibrate_v2_acoustic_crf as acoustic
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_fast_v2 import fast_decode_identity_crf
from alignmodel.joint.identity_crf_v1 import IdentityCandidate, build_identity_lattice
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file


def _candidate(value: Mapping[str, Any]) -> IdentityCandidate:
    return IdentityCandidate(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        confidence=1.0,
        alternatives=tuple(int(item) for item in value["alternatives"]),
        alternative_confidences=tuple(
            float(item) for item in value["alternative_confidences"]
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--sequence-supervision", type=Path, required=True)
    parser.add_argument("--expected-supervision-sha256", required=True)
    parser.add_argument("--crf-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    if sha256_file(args.candidate_pool) != args.expected_pool_sha256:
        raise ValueError("Candidate pool mismatch")
    if sha256_file(args.sequence_supervision) != args.expected_supervision_sha256:
        raise ValueError("Sequence supervision mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    target_artifact = release["artifacts"]["development_targets"]
    with gzip.open(target_artifact["path"], "rt", encoding="utf-8") as stream:
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
    crf_payload = torch.load(args.crf_checkpoint, map_location="cpu", weights_only=False)
    crf = acoustic.OrnamentIdentityCRF(
        hidden=crf_payload["model_config"]["hidden"]
    )
    crf.load_state_dict(crf_payload["model_state_dict"])
    crf.eval()
    crf_samples = []
    identity_samples = []
    per_row = []
    for position, row in enumerate(
        release["splits"]["development"]["calibration"], 1
    ):
        sample = row["sample"]
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        index = ScoreEventIndex.from_musicxml(
            score_path, targets[sample]["lineage"]
        )
        pool = pools[sample]
        truth = supervision[sample]
        assignments = sorted(
            truth["assignments"],
            key=lambda value: int(value["target_rendered_index"]),
        )
        selected = [
            (
                assignment,
                _candidate(
                    pool["candidates"][
                        int(assignment["selected_interval_candidate"])
                    ]
                ),
            )
            for assignment in assignments
        ]
        selected.sort(key=lambda value: (value[1].start, value[1].pitch, value[1].end))
        candidates = tuple(value[1] for value in selected)
        lattice = build_identity_lattice(
            candidates,
            index.events,
            score_path,
            max_inference_hypotheses=16,
        )
        predicted, deletions, _diagnostics = fast_decode_identity_crf(crf, lattice)
        crf_samples.append(
            JointMetricSample(
                predicted=predicted,
                target=index.rendered_events,
                source=row["leakage_group"],
                predicted_deletions=deletions,
                target_deletions=index.deleted_event_indices,
                score_event_count=len(index.events),
            )
        )
        target_by_index = {
            int(event.rendered_index): event for event in index.rendered_events
        }
        oracle_events = []
        for output_index, (assignment, candidate) in enumerate(selected):
            target = target_by_index[int(assignment["target_rendered_index"])]
            oracle_events.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=target.score_span,
                    relationship=target.relationship,
                    copy_pass=target.copy_pass,
                    origin_relationship=target.origin_relationship,
                    rendered_index=output_index,
                    confidence=1.0,
                )
            )
        identity_samples.append(
            JointMetricSample(
                predicted=tuple(oracle_events),
                target=index.rendered_events,
                source=row["leakage_group"],
                predicted_deletions=index.deleted_event_indices,
                target_deletions=index.deleted_event_indices,
                score_event_count=len(index.events),
            )
        )
        per_row.append(
            {
                "sample": sample,
                "target_events": len(index.rendered_events),
                "selected_candidates": len(candidates),
                "full_sequence_covered": truth["full_sequence_covered"],
            }
        )
        if position == 1 or position % 16 == 0 or position == 64:
            print(f"oracle={position}/64", flush=True)
    report = {
        "schema_version": "align-activation-candidate-canonical-oracle-v1",
        "release_manifest_sha256": release_sha,
        "population": {"split": "calibration", "rows": 64},
        "candidate_identity_oracle": baseline._full_report(
            identity_samples,
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        ),
        "candidate_through_frozen_crf": baseline._full_report(
            crf_samples,
            seed=args.seed + 1,
            replicates=args.bootstrap_replicates,
        ),
        "per_row": per_row,
        "timestamps_used_for_selection": False,
        "open_validation_read": False,
        "lockbox_targets_read": False,
    }
    acoustic._atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "identity_oracle_f1": report["candidate_identity_oracle"]["f1"],
                "through_crf_f1": report["candidate_through_frozen_crf"]["f1"],
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
