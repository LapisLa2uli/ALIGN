"""Tune Basic Pitch base decoding on opened v3 after the v4 freeze."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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


SCHEMA_VERSION = "align-orn-v3-base-decode-v4-development-v1"
DECODE_GRID = (
    (0.50, 0.40, 55.0),
    (0.55, 0.40, 55.0),
    (0.60, 0.40, 55.0),
    (0.65, 0.40, 55.0),
    (0.55, 0.45, 55.0),
    (0.60, 0.45, 55.0),
    (0.65, 0.45, 55.0),
    (0.55, 0.50, 55.0),
    (0.60, 0.50, 55.0),
    (0.65, 0.50, 55.0),
    (0.60, 0.45, 70.0),
    (0.65, 0.45, 70.0),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-freeze", type=Path, required=True)
    parser.add_argument("--expected-v4-freeze-sha256", required=True)
    parser.add_argument("--v4-opening-sentinel", type=Path, required=True)
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--expected-v3-manifest-sha256", required=True)
    parser.add_argument("--v3-targets", type=Path, required=True)
    parser.add_argument("--expected-v3-targets-sha256", required=True)
    parser.add_argument("--v3-key", type=Path, required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--expected-base-candidate-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.v4_freeze) != args.expected_v4_freeze_sha256:
        raise ValueError("ORN v4 freeze mismatch")
    v4 = json.loads(args.v4_freeze.read_text(encoding="utf-8"))
    if v4.get("lockbox_opened") or args.v4_opening_sentinel.exists():
        raise ValueError("ORN v4 is no longer sealed")
    expected = (
        (args.v3_manifest, args.expected_v3_manifest_sha256),
        (args.v3_targets, args.expected_v3_targets_sha256),
        (args.base_candidate, args.expected_base_candidate_sha256),
        (args.checkpoint, args.expected_checkpoint_sha256),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact mismatch: {path}")

    manifest = json.loads(args.v3_manifest.read_text(encoding="utf-8"))
    targets = _decrypt_targets(
        key_path=args.v3_key,
        encrypted_path=args.v3_targets,
        manifest=manifest,
    )
    target_by_id = {str(row["row_id"]): row for row in targets}
    base_candidate = json.loads(
        args.base_candidate.read_text(encoding="utf-8")
    )
    base_decode = BasicPitchDecodeConfig(**base_candidate["base_decode"])
    model, payload = acoustic._load_model(args.checkpoint)
    if (
        payload["data"]["release_manifest_sha256"]
        != base_candidate["release_manifest_sha256"]
    ):
        raise ValueError("Checkpoint/release mismatch")
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
        index = ScoreEventIndex.from_musicxml(
            score_path, target_by_id[row_id]["lineage"]
        )
        contexts[row_id] = {
            "row": row,
            "features": features,
            "index": index,
            "score_path": score_path,
        }
        if position == 1 or position % 20 == 0 or position == len(
            manifest["inputs"]
        ):
            print(f"context={position}/{len(manifest['inputs'])}", flush=True)

    variants = []
    for variant_index, (onset, frame, minimum_ms) in enumerate(DECODE_GRID):
        decode = replace(
            base_decode,
            onset_threshold=onset,
            frame_threshold=frame,
            minimum_note_length_ms=minimum_ms,
        )
        samples = []
        base_count = 0
        rescued_count = 0
        for position, (row_id, context) in enumerate(
            sorted(contexts.items()), 1
        ):
            base = decode_basic_pitch_features(
                context["features"], decode
            )
            base_count += len(base)
            base_candidates = acoustic._identity_candidates(
                base, context["features"]
            )
            baseline_sample, diagnostics = rescue._map(
                model,
                base_candidates,
                context["index"],
                context["score_path"],
            )
            candidates = rescue._direct_rescue(
                base,
                context["index"].events,
                context["score_path"],
                diagnostics,
                context["features"],
                evidence_threshold=0.40,
                window_sec=0.15,
                anchor_events=baseline_sample.predicted,
            )
            rescued_count += len(candidates) - len(base)
            metric_sample, _final = rescue._map(
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
            if position == 1 or position % 25 == 0 or position == len(
                contexts
            ):
                print(
                    f"variant={variant_index} rows={position}/{len(contexts)}",
                    flush=True,
                )
        variants.append(
            {
                "variant": variant_index,
                "onset_threshold": onset,
                "frame_threshold": frame,
                "minimum_note_length_ms": minimum_ms,
                "rescue_evidence_threshold": 0.40,
                "rescue_window_sec": 0.15,
                "base_candidate_count": base_count,
                "rescued_candidate_count": rescued_count,
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
            "v3_rows": len(contexts),
            "v3_role": "development_after_v4_freeze",
            "v4_freeze_sha256": args.expected_v4_freeze_sha256,
            "v4_targets_read": False,
            "v4_key_read": False,
            "v4_opened": False,
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
                "onset_threshold": selected["onset_threshold"],
                "frame_threshold": selected["frame_threshold"],
                "minimum_note_length_ms": selected[
                    "minimum_note_length_ms"
                ],
                "f1": selected["official_note_wise"]["f1"],
                "v4_targets_read": False,
            },
            indent=2,
        ),
        flush=True,
    )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
