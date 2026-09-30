"""Evaluate a frozen template-rescue candidate on tainted ORN v2 development data.

This is diagnostic only. ``open_validation`` was disqualified as held-out by
``OPEN_VALIDATION_CONTAMINATION.json``; the encrypted lockbox is not accessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import calibrate_v2_acoustic_crf as acoustic
import calibrate_v2_template_rescue as rescue
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    basic_pitch_cache_path,
    decode_basic_pitch_features,
    extract_sample_basic_pitch_features,
)


SCHEMA_VERSION = "align-orn-v2-template-rescue-development-diagnostic-v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--contamination-report", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("calibration", "open_validation"),
        default="open_validation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-grid", action="store_true")
    parser.add_argument("--grid-evidence", type=float, nargs="+")
    parser.add_argument("--grid-windows", type=float, nargs="+")
    parser.add_argument(
        "--grid-timing",
        choices=("global", "local"),
        nargs="+",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.release_manifest) != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    if sha256_file(args.candidate) != args.expected_candidate_sha256:
        raise ValueError("Frozen candidate mismatch")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("Frozen checkpoint mismatch")
    contamination = json.loads(
        args.contamination_report.read_text(encoding="utf-8")
    )
    if contamination["status"] != "open_validation_disqualified_as_held_out":
        raise ValueError("Missing open-validation disqualification")

    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if (
        candidate.get("mode") != "direct_template"
        or not candidate.get("frozen")
        or candidate["identity_crf_checkpoint_sha256"]
        != args.expected_checkpoint_sha256
    ):
        raise ValueError("Unexpected rescue candidate")
    model, payload = acoustic._load_model(args.checkpoint)
    if payload["data"]["release_manifest_sha256"] != args.expected_release_sha256:
        raise ValueError("Checkpoint/release mismatch")

    split = args.split
    rows = {
        str(row["sample"]): row
        for row in release["splits"]["development"][split]
    }
    targets = acoustic._targets(release, split)
    if set(rows) != set(targets):
        raise ValueError("Development population mismatch")
    decode = BasicPitchDecodeConfig(**candidate["base_decode"])
    contexts: dict[str, dict[str, Any]] = {}
    for position, sample in enumerate(sorted(rows), 1):
        row = rows[sample]
        sample_dir = Path(row["sample_dir"])
        score_path = sample_dir / "verified_score.musicxml"
        features = extract_sample_basic_pitch_features(
            sample_dir,
            cache_path=basic_pitch_cache_path(
                args.feature_cache,
                sample_dir,
                f"orn-v2-{split.replace('_', '-')}",
            ),
        )
        base = decode_basic_pitch_features(features, decode)
        index = ScoreEventIndex.from_musicxml(
            score_path, targets[sample]["lineage"]
        )
        base_candidates = acoustic._identity_candidates(base, features)
        baseline_sample, base_diagnostics = rescue._map(
            model, base_candidates, index, score_path
        )
        contexts[sample] = {
            "row": row,
            "features": features,
            "base": base,
            "index": index,
            "score_path": score_path,
            "base_diagnostics": base_diagnostics,
            "anchor_events": baseline_sample.predicted,
        }
        if position == 1 or position % 10 == 0 or position == len(rows):
            print(f"development={position}/{len(rows)}", flush=True)

    strategies = [
        (
            float(candidate["evidence_threshold"]),
            float(candidate["window_sec"]),
            "global",
        )
    ]
    if args.direct_grid:
        evidence_grid = (
            tuple(args.grid_evidence)
            if args.grid_evidence
            else rescue.DIRECT_EVIDENCE_THRESHOLDS
        )
        window_grid = (
            tuple(args.grid_windows)
            if args.grid_windows
            else rescue.WINDOWS_SEC[:2]
        )
        timing_grid = (
            tuple(args.grid_timing)
            if args.grid_timing
            else ("global", "local")
        )
        strategies = list(
            dict.fromkeys(
                (float(evidence), float(window), timing)
                for evidence in evidence_grid
                for window in window_grid
                for timing in timing_grid
            )
        )
    variants = []
    for variant_index, (evidence, window, timing) in enumerate(strategies):
        samples = []
        diagnostics = []
        for position, sample in enumerate(sorted(contexts), 1):
            context = contexts[sample]
            candidates = rescue._direct_rescue(
                context["base"],
                context["index"].events,
                context["score_path"],
                context["base_diagnostics"],
                context["features"],
                evidence_threshold=evidence,
                window_sec=window,
                anchor_events=(
                    context["anchor_events"] if timing == "local" else None
                ),
            )
            metric_sample, final_diagnostics = rescue._map(
                model,
                candidates,
                context["index"],
                context["score_path"],
            )
            samples.append(
                JointMetricSample(
                    predicted=metric_sample.predicted,
                    target=metric_sample.target,
                    source=context["row"]["leakage_group"],
                    predicted_deletions=metric_sample.predicted_deletions,
                    target_deletions=metric_sample.target_deletions,
                    score_event_count=metric_sample.score_event_count,
                )
            )
            diagnostics.append(
                {
                    "sample": sample,
                    "base_candidates": len(context["base"]),
                    "rescued_candidates": (
                        len(candidates) - len(context["base"])
                    ),
                    "decoder": final_diagnostics,
                }
            )
            if position == 1 or position % 20 == 0 or position == len(contexts):
                print(
                    f"variant={variant_index} rows={position}/{len(contexts)}",
                    flush=True,
                )
        variants.append(
            {
                "variant": variant_index,
                "evidence_threshold": evidence,
                "window_sec": window,
                "timing": timing,
                "rescued_candidates": sum(
                    row["rescued_candidates"] for row in diagnostics
                ),
                "official_note_wise": baseline._full_report(
                    samples,
                    seed=args.seed + variant_index,
                    replicates=args.bootstrap_replicates,
                ),
                "diagnostics": diagnostics,
            }
        )
    selected = max(
        variants,
        key=lambda value: (
            float(value["official_note_wise"]["f1"]),
            -int(value["variant"]),
        ),
    )
    report = selected["official_note_wise"]
    acoustic._atomic_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "rows": len(rows),
            "held_out": False,
            "held_out_disqualification": str(args.contamination_report),
            "release_manifest_sha256": args.expected_release_sha256,
            "candidate_sha256": args.expected_candidate_sha256,
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "direct_grid": args.direct_grid,
            "variants": variants,
            "selected_variant": selected["variant"],
            "official_note_wise": report,
            "lockbox_targets_read": False,
        },
    )
    print(
        json.dumps(
            {
                "f1": report["f1"],
                "precision": report["precision"],
                "recall": report["recall"],
                "bootstrap_95": report.get("bootstrap_95"),
                "held_out": False,
            },
            indent=2,
        ),
        flush=True,
    )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
