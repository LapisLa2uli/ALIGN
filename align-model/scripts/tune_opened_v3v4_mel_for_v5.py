"""Tune mel/identity decode on opened v3+v4 after the sealed ORN v5 freeze.

v5 targets/keys are never read. Membership of the sealed holdout is unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import calibrate_v2_acoustic_crf as acoustic
import calibrate_v2_template_rescue as rescue
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_v1 import IdentityCandidate
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_v1 import (
    MelDecodeConfig,
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from eval_v2_template_rescue_lockbox import _decrypt_targets


SCHEMA_VERSION = "align-orn-v3v4-mel-transcriber-v5-development-v1"
CONFIDENCE_GRID = (0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80)
# Small decode perturbations around the frozen Track-B defaults, aimed at
# recovering match/copy recall without a new acoustic train.
DECODE_OVERRIDES: tuple[dict[str, float], ...] = (
    {},
    {"onset_threshold": 0.44, "voice_on": 0.48},
    {"onset_threshold": 0.42, "voice_on": 0.46, "min_note_sec": 0.035},
    {"boundary_threshold": 0.42, "strong_rearticulation": 0.58},
)


def _identity_candidates(notes: Sequence[Any]) -> tuple[IdentityCandidate, ...]:
    return tuple(
        IdentityCandidate(
            pitch=int(note.pitch),
            start=float(note.start),
            end=float(note.end),
            confidence=float(note.confidence),
            alternatives=tuple(map(int, note.pitch_candidates)),
            alternative_confidences=tuple(
                map(float, note.candidate_confidences)
            ),
        )
        for note in notes
    )


def _load_population(
    *,
    name: str,
    manifest_path: Path,
    targets_path: Path,
    key_path: Path,
    cache_dir: Path,
    mel_model: Any,
    frontend: Any,
    device: torch.device,
    batch_size: int,
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
        cache_path = cache_dir / f"{row_id}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            probabilities = {
                key: np.asarray(cached[key], np.float32)
                for key in cached.files
            }
        else:
            audio = load_audio_mono(
                Path(row["audio"]["path"]), frontend.sample_rate
            )
            mel, _normalization = extract_log_mel(
                audio, frontend, device=device
            )
            probabilities = infer_mel_probabilities(
                mel_model,
                np.asarray(mel, np.float32),
                device,
                window_frames=2048,
                overlap_frames=512,
                batch_size=batch_size,
            )
            np.savez_compressed(cache_path, **probabilities)
        score_path = Path(row["score"]["path"])
        index = ScoreEventIndex.from_musicxml(
            score_path, target_by_id[row_id]["lineage"]
        )
        contexts[f"{name}:{row_id}"] = {
            "population": name,
            "row": row,
            "probabilities": probabilities,
            "index": index,
            "score_path": score_path,
        }
        if position == 1 or position % 20 == 0 or position == len(
            manifest["inputs"]
        ):
            print(
                f"context-{name}={position}/{len(manifest['inputs'])}",
                flush=True,
            )
    return contexts


def _rank(variant: Mapping[str, Any]) -> tuple[float, float, float, int]:
    metrics = variant["official_note_wise"]
    f1 = float(metrics["f1"])
    recall = float(metrics["recall"])
    precision = float(metrics["precision"])
    # Prefer high F1, then recall ≥0.80 (v4 failure mode), then balanced P/R.
    recall_bonus = min(recall, 0.80)
    balance = -abs(precision - recall)
    return (f1, recall_bonus, balance, -int(variant["variant"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-freeze", type=Path, required=True)
    parser.add_argument("--expected-v5-freeze-sha256", required=True)
    parser.add_argument("--v5-opening-sentinel", type=Path, required=True)
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--expected-v3-manifest-sha256", required=True)
    parser.add_argument("--v3-targets", type=Path, required=True)
    parser.add_argument("--expected-v3-targets-sha256", required=True)
    parser.add_argument("--v3-key", type=Path, required=True)
    parser.add_argument("--v4-manifest", type=Path, required=True)
    parser.add_argument("--expected-v4-manifest-sha256", required=True)
    parser.add_argument("--v4-targets", type=Path, required=True)
    parser.add_argument("--expected-v4-targets-sha256", required=True)
    parser.add_argument("--v4-key", type=Path, required=True)
    parser.add_argument("--mel-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-mel-checkpoint-sha256", required=True)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-identity-checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.v5_freeze) != args.expected_v5_freeze_sha256:
        raise ValueError("ORN v5 freeze mismatch")
    v5 = json.loads(args.v5_freeze.read_text(encoding="utf-8"))
    if v5.get("lockbox_opened") or args.v5_opening_sentinel.exists():
        raise ValueError("ORN v5 is no longer sealed")
    expected = (
        (args.v3_manifest, args.expected_v3_manifest_sha256),
        (args.v3_targets, args.expected_v3_targets_sha256),
        (args.v4_manifest, args.expected_v4_manifest_sha256),
        (args.v4_targets, args.expected_v4_targets_sha256),
        (args.mel_checkpoint, args.expected_mel_checkpoint_sha256),
        (
            args.identity_checkpoint,
            args.expected_identity_checkpoint_sha256,
        ),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact mismatch: {path}")

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    mel_model, frontend, base_decode, mel_payload = load_mel_checkpoint(
        args.mel_checkpoint, device
    )
    if mel_payload["data"].get("locked_test_materialized"):
        raise ValueError("Mel checkpoint used locked test data")
    identity_model = acoustic._load_model(args.identity_checkpoint)[0]

    contexts: dict[str, dict[str, Any]] = {}
    contexts.update(
        _load_population(
            name="v3",
            manifest_path=args.v3_manifest,
            targets_path=args.v3_targets,
            key_path=args.v3_key,
            cache_dir=args.cache_dir / "v3",
            mel_model=mel_model,
            frontend=frontend,
            device=device,
            batch_size=args.batch_size,
        )
    )
    contexts.update(
        _load_population(
            name="v4",
            manifest_path=args.v4_manifest,
            targets_path=args.v4_targets,
            key_path=args.v4_key,
            cache_dir=args.cache_dir / "v4",
            mel_model=mel_model,
            frontend=frontend,
            device=device,
            batch_size=args.batch_size,
        )
    )

    variants = []
    variant_index = 0
    for override in DECODE_OVERRIDES:
        for minimum_confidence in CONFIDENCE_GRID:
            decode: MelDecodeConfig = replace(
                base_decode,
                min_confidence=float(minimum_confidence),
                **override,
            )
            samples = []
            note_count = 0
            for position, (_key, context) in enumerate(
                sorted(contexts.items()), 1
            ):
                notes = decode_mel_notes(
                    context["probabilities"],
                    midi_min=mel_model.config.midi_min,
                    hop_sec=frontend.hop_sec,
                    config=decode,
                )
                note_count += len(notes)
                metric_sample, _diagnostics = rescue._map(
                    identity_model,
                    _identity_candidates(notes),
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
                        predicted_deletions=metric_sample.predicted_deletions,
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
                        f"conf={minimum_confidence} override={override}",
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
                    "minimum_confidence": minimum_confidence,
                    "decode_overrides": override,
                    "decode_config": decode.to_dict(),
                    "transcribed_note_count": note_count,
                    "official_note_wise": report,
                }
            )
            print(
                json.dumps(
                    {
                        "variant": variant_index,
                        "minimum_confidence": minimum_confidence,
                        "decode_overrides": override,
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
                "opened_orn_v3": sum(
                    1 for key in contexts if key.startswith("v3:")
                ),
                "opened_orn_v4": sum(
                    1 for key in contexts if key.startswith("v4:")
                ),
            },
            "v5_freeze_sha256": args.expected_v5_freeze_sha256,
            "v5_targets_read": False,
            "v5_key_read": False,
            "v5_opened": False,
            "mel_checkpoint_sha256": args.expected_mel_checkpoint_sha256,
            "identity_checkpoint_sha256": (
                args.expected_identity_checkpoint_sha256
            ),
            "selection_policy": (
                "maximize official note-wise F1; tie-break toward recall "
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
                "minimum_confidence": selected["minimum_confidence"],
                "decode_overrides": selected["decode_overrides"],
                "f1": selected["official_note_wise"]["f1"],
                "precision": selected["official_note_wise"]["precision"],
                "recall": selected["official_note_wise"]["recall"],
                "v5_targets_read": False,
            },
            indent=2,
        ),
        flush=True,
    )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
