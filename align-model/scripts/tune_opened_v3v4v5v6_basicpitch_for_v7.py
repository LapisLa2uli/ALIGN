"""Tune Basic Pitch + identity on opened v3–v6 after the sealed ORN v7 freeze.

Mel decode grids repeatedly failed sealed holdouts near 0.78–0.80. This
development switches the score-blind frontend to frozen Basic Pitch 0.4.0 and
grids precision-oriented decode/confidence/rescue settings. v7 targets/keys
are never read.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import calibrate_v2_acoustic_crf as acoustic
import calibrate_v2_template_rescue as rescue
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    FROZEN_DECODE_CONFIG,
    decode_basic_pitch_features,
    extract_basic_pitch_features,
)
from eval_v2_template_rescue_lockbox import PITCH_METADATA, _decrypt_targets


SCHEMA_VERSION = "align-orn-v3v4v5v6-basicpitch-identity-v7-development-v1"

# Precision-oriented around the historical near-miss BP baseline.
DECODE_GRID: tuple[dict[str, float], ...] = (
    {},
    {"onset_threshold": 0.55, "frame_threshold": 0.45},
    {"onset_threshold": 0.60, "frame_threshold": 0.45},
    {"onset_threshold": 0.60, "frame_threshold": 0.45, "minimum_note_length_ms": 70.0},
    {"onset_threshold": 0.65, "frame_threshold": 0.50},
)
CONFIDENCE_GRID = (0.0, 0.45, 0.55)
# None disables local rescue; higher evidence is more precision-stable after v3.
RESCUE_GRID: tuple[float | None, ...] = (None, 0.55, 0.65)


def _filter_notes(
    notes: Sequence[Any], minimum_confidence: float
) -> list[Any]:
    if minimum_confidence <= 0.0:
        return list(notes)
    return [
        note
        for note in notes
        if float(note.confidence) >= float(minimum_confidence)
    ]


def _load_population(
    *,
    name: str,
    manifest_path: Path,
    targets_path: Path,
    key_path: Path,
    cache_dir: Path,
) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = _decrypt_targets(
        key_path=key_path,
        encrypted_path=targets_path,
        manifest=manifest,
    )
    target_by_id = {str(row["row_id"]): row for row in targets}
    cache_dir.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(manifest["inputs"], 1):
        row_id = str(row["row_id"])
        audio_path = Path(row["audio"]["path"])
        score_path = Path(row["score"]["path"])
        features = extract_basic_pitch_features(
            audio_path,
            source_metadata=PITCH_METADATA,
            cache_path=cache_dir / f"{row_id}.npz",
        )
        index = ScoreEventIndex.from_musicxml(
            score_path, target_by_id[row_id]["lineage"]
        )
        contexts[f"{name}:{row_id}"] = {
            "population": name,
            "row": row,
            "features": features,
            "index": index,
            "score_path": score_path,
        }
        if (
            position == 1
            or position % 20 == 0
            or position == len(manifest["inputs"])
        ):
            print(
                f"context-{name}={position}/{len(manifest['inputs'])}",
                flush=True,
            )
    return contexts


def _rank(variant: Mapping[str, Any]) -> tuple[float, float, float, int]:
    metrics = variant["official_note_wise"]
    f1 = float(metrics["f1"])
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    precision_bonus = min(precision, 0.80)
    balance = -abs(precision - recall)
    return (f1, precision_bonus, balance, -int(variant["variant"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-freeze", type=Path, required=True)
    parser.add_argument("--expected-v7-freeze-sha256", required=True)
    parser.add_argument("--v7-opening-sentinel", type=Path, required=True)
    for version in ("v3", "v4", "v5", "v6"):
        parser.add_argument(f"--{version}-manifest", type=Path, required=True)
        parser.add_argument(
            f"--expected-{version}-manifest-sha256", required=True
        )
        parser.add_argument(f"--{version}-targets", type=Path, required=True)
        parser.add_argument(
            f"--expected-{version}-targets-sha256", required=True
        )
        parser.add_argument(f"--{version}-key", type=Path, required=True)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-identity-checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.v7_freeze) != args.expected_v7_freeze_sha256:
        raise ValueError("ORN v7 freeze mismatch")
    v7 = json.loads(args.v7_freeze.read_text(encoding="utf-8"))
    if v7.get("lockbox_opened") or args.v7_opening_sentinel.exists():
        raise ValueError("ORN v7 is no longer sealed")
    if (
        sha256_file(args.identity_checkpoint)
        != args.expected_identity_checkpoint_sha256
    ):
        raise ValueError("Identity checkpoint mismatch")

    identity_model = acoustic._load_model(args.identity_checkpoint)[0]
    contexts: dict[str, dict[str, Any]] = {}
    for name in ("v3", "v4", "v5", "v6"):
        contexts.update(
            _load_population(
                name=name,
                manifest_path=getattr(args, f"{name}_manifest"),
                targets_path=getattr(args, f"{name}_targets"),
                key_path=getattr(args, f"{name}_key"),
                cache_dir=args.cache_dir / name,
            )
        )

    variants = []
    variant_index = 0
    for decode_override in DECODE_GRID:
        decode = replace(FROZEN_DECODE_CONFIG, **decode_override)
        for minimum_confidence in CONFIDENCE_GRID:
            for rescue_threshold in RESCUE_GRID:
                samples = []
                note_count = 0
                rescued_count = 0
                for position, (_key, context) in enumerate(
                    sorted(contexts.items()), 1
                ):
                    base_notes = _filter_notes(
                        decode_basic_pitch_features(
                            context["features"], decode
                        ),
                        minimum_confidence,
                    )
                    note_count += len(base_notes)
                    base_candidates = acoustic._identity_candidates(
                        base_notes, context["features"]
                    )
                    if rescue_threshold is None:
                        metric_sample, _diagnostics = rescue._map(
                            identity_model,
                            base_candidates,
                            context["index"],
                            context["score_path"],
                        )
                    else:
                        baseline_sample, diagnostics = rescue._map(
                            identity_model,
                            base_candidates,
                            context["index"],
                            context["score_path"],
                        )
                        candidates = rescue._direct_rescue(
                            base_notes,
                            context["index"].events,
                            context["score_path"],
                            diagnostics,
                            context["features"],
                            evidence_threshold=float(rescue_threshold),
                            window_sec=0.15,
                            anchor_events=baseline_sample.predicted,
                        )
                        rescued_count += len(candidates) - len(base_notes)
                        metric_sample, _final = rescue._map(
                            identity_model,
                            candidates,
                            context["index"],
                            context["score_path"],
                        )
                    samples.append(
                        JointMetricSample(
                            predicted=metric_sample.predicted,
                            target=metric_sample.target,
                            source=(
                                f"{context['population']}:"
                                f"{context['row']['leakage_group']}"
                            ),
                            predicted_deletions=(
                                metric_sample.predicted_deletions
                            ),
                            target_deletions=metric_sample.target_deletions,
                            score_event_count=metric_sample.score_event_count,
                        )
                    )
                    if (
                        position == 1
                        or position % 50 == 0
                        or position == len(contexts)
                    ):
                        print(
                            f"variant={variant_index} "
                            f"rows={position}/{len(contexts)} "
                            f"override={decode_override} "
                            f"conf={minimum_confidence} "
                            f"rescue={rescue_threshold}",
                            flush=True,
                        )
                report = baseline._full_report(
                    samples,
                    seed=args.seed + variant_index,
                    replicates=args.bootstrap_replicates,
                )
                variants.append(
                    {
                        "variant": variant_index,
                        "decode_overrides": decode_override,
                        "decode_config": asdict(decode),
                        "minimum_confidence": minimum_confidence,
                        "rescue_evidence_threshold": rescue_threshold,
                        "rescue_window_sec": (
                            None if rescue_threshold is None else 0.15
                        ),
                        "transcribed_note_count": note_count,
                        "rescued_candidate_count": rescued_count,
                        "official_note_wise": report,
                    }
                )
                print(
                    json.dumps(
                        {
                            "variant": variant_index,
                            "decode_overrides": decode_override,
                            "minimum_confidence": minimum_confidence,
                            "rescue_evidence_threshold": rescue_threshold,
                            "f1": report["f1"],
                            "precision": report["precision"],
                            "recall": report["recall"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                variant_index += 1

    selected = max(variants, key=_rank)
    acoustic._atomic_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "development_rows": len(contexts),
            "development_populations": {
                name: sum(
                    1 for key in contexts if key.startswith(f"{name}:")
                )
                for name in ("v3", "v4", "v5", "v6")
            },
            "v7_freeze_sha256": args.expected_v7_freeze_sha256,
            "v7_targets_read": False,
            "v7_key_read": False,
            "v7_opened": False,
            "frontend": "frozen_basic_pitch_0.4.0",
            "selection_policy": (
                "maximize official note-wise F1; tie-break toward precision "
                "capped at 0.80 then balanced precision/recall"
            ),
            "variants": variants,
            "selected_variant": selected["variant"],
            "selected": selected,
        },
    )
    print(
        json.dumps(
            {
                "selected_variant": selected["variant"],
                "decode_overrides": selected["decode_overrides"],
                "minimum_confidence": selected["minimum_confidence"],
                "rescue_evidence_threshold": selected[
                    "rescue_evidence_threshold"
                ],
                "f1": selected["official_note_wise"]["f1"],
                "precision": selected["official_note_wise"]["precision"],
                "recall": selected["official_note_wise"]["recall"],
                "v7_targets_read": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
