"""Use the opened ORN v2 lockbox as development after the v3 freeze.

The new ORN v3 lockbox must already be frozen and unopened. This script never
accepts a v3 key or target path.
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
    decode_basic_pitch_features,
    extract_basic_pitch_features,
)
from eval_v2_template_rescue_lockbox import (
    PITCH_METADATA,
    _decrypt_targets,
)


SCHEMA_VERSION = "align-orn-v2-opened-lockbox-v3-development-v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-freeze", type=Path, required=True)
    parser.add_argument("--expected-v3-freeze-sha256", required=True)
    parser.add_argument("--v3-opening-sentinel", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--expected-v2-manifest-sha256", required=True)
    parser.add_argument("--v2-targets", type=Path, required=True)
    parser.add_argument("--expected-v2-targets-sha256", required=True)
    parser.add_argument("--v2-key", type=Path, required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--expected-base-candidate-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=float,
        nargs="+",
        default=(0.30, 0.35, 0.40, 0.45, 0.50),
    )
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15, 0.20),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.v3_freeze) != args.expected_v3_freeze_sha256:
        raise ValueError("ORN v3 freeze mismatch")
    v3_freeze = json.loads(args.v3_freeze.read_text(encoding="utf-8"))
    if v3_freeze.get("lockbox_opened") or args.v3_opening_sentinel.exists():
        raise ValueError("ORN v3 lockbox is no longer sealed")
    if sha256_file(args.v2_manifest) != args.expected_v2_manifest_sha256:
        raise ValueError("ORN v2 manifest mismatch")
    if sha256_file(args.v2_targets) != args.expected_v2_targets_sha256:
        raise ValueError("ORN v2 target envelope mismatch")
    if (
        sha256_file(args.base_candidate)
        != args.expected_base_candidate_sha256
    ):
        raise ValueError("Base candidate mismatch")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint mismatch")

    manifest = json.loads(args.v2_manifest.read_text(encoding="utf-8"))
    targets = _decrypt_targets(
        key_path=args.v2_key,
        encrypted_path=args.v2_targets,
        manifest=manifest,
    )
    target_by_id = {str(row["row_id"]): row for row in targets}
    base_candidate = json.loads(
        args.base_candidate.read_text(encoding="utf-8")
    )
    model, payload = acoustic._load_model(args.checkpoint)
    if (
        payload["data"]["release_manifest_sha256"]
        != base_candidate["release_manifest_sha256"]
    ):
        raise ValueError("Checkpoint/release mismatch")
    decode = BasicPitchDecodeConfig(**base_candidate["base_decode"])
    args.feature_cache.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(manifest["inputs"], 1):
        row_id = str(row["row_id"])
        audio_path = Path(row["audio"]["path"])
        score_path = Path(row["score"]["path"])
        features = extract_basic_pitch_features(
            audio_path,
            source_metadata=PITCH_METADATA,
            cache_path=args.feature_cache / f"{row_id}.npz",
        )
        base = decode_basic_pitch_features(features, decode)
        index = ScoreEventIndex.from_musicxml(
            score_path, target_by_id[row_id]["lineage"]
        )
        base_candidates = acoustic._identity_candidates(base, features)
        baseline_sample, diagnostics = rescue._map(
            model, base_candidates, index, score_path
        )
        contexts[row_id] = {
            "row": row,
            "features": features,
            "base": base,
            "index": index,
            "score_path": score_path,
            "diagnostics": diagnostics,
            "anchors": baseline_sample.predicted,
        }
        if position == 1 or position % 10 == 0 or position == len(
            manifest["inputs"]
        ):
            print(f"context={position}/{len(manifest['inputs'])}", flush=True)

    variants = []
    strategies = [
        (float(evidence), float(window))
        for evidence in args.evidence
        for window in args.windows
    ]
    for variant_index, (evidence, window) in enumerate(strategies):
        samples = []
        rescued = 0
        for position, (row_id, context) in enumerate(
            sorted(contexts.items()), 1
        ):
            candidates = rescue._direct_rescue(
                context["base"],
                context["index"].events,
                context["score_path"],
                context["diagnostics"],
                context["features"],
                evidence_threshold=evidence,
                window_sec=window,
                anchor_events=context["anchors"],
            )
            metric_sample, _diagnostics = rescue._map(
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
            rescued += len(candidates) - len(context["base"])
            if position == 1 or position % 20 == 0 or position == len(
                contexts
            ):
                print(
                    f"variant={variant_index} rows={position}/{len(contexts)}",
                    flush=True,
                )
        variants.append(
            {
                "variant": variant_index,
                "evidence_threshold": evidence,
                "window_sec": window,
                "timing": "local",
                "rescued_candidates": rescued,
                "official_note_wise": baseline._full_report(
                    samples,
                    seed=args.seed + variant_index,
                    replicates=args.bootstrap_replicates,
                ),
            }
        )
    selected = max(
        variants,
        key=lambda value: (
            float(value["official_note_wise"]["f1"]),
            -int(value["variant"]),
        ),
    )
    acoustic._atomic_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "v2_rows": len(contexts),
            "v2_role": "development_after_v3_freeze",
            "v3_freeze_sha256": args.expected_v3_freeze_sha256,
            "v3_targets_read": False,
            "v3_key_read": False,
            "v3_lockbox_opened": False,
            "base_candidate_sha256": args.expected_base_candidate_sha256,
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "variants": variants,
            "selected_variant": selected["variant"],
            "selected": selected,
        },
    )
    print(
        json.dumps(
            {
                "selected_variant": selected["variant"],
                "evidence_threshold": selected["evidence_threshold"],
                "window_sec": selected["window_sec"],
                "f1": selected["official_note_wise"]["f1"],
                "v3_targets_read": False,
            },
            indent=2,
        ),
        flush=True,
    )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
