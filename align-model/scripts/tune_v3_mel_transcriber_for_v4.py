"""Evaluate frozen mel-transcriber candidates on opened v3 after v4 freeze."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

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
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from eval_v2_template_rescue_lockbox import _decrypt_targets


SCHEMA_VERSION = "align-orn-v3-mel-transcriber-v4-development-v1"
CONFIDENCE_GRID = (0.65, 0.70, 0.75, 0.80)


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
    parser.add_argument("--mel-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-mel-checkpoint-sha256", required=True)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-identity-checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
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
        (args.mel_checkpoint, args.expected_mel_checkpoint_sha256),
        (
            args.identity_checkpoint,
            args.expected_identity_checkpoint_sha256,
        ),
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
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(manifest["inputs"], 1):
        row_id = str(row["row_id"])
        cache_path = args.cache_dir / f"{row_id}.npz"
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
                batch_size=args.batch_size,
            )
            np.savez_compressed(cache_path, **probabilities)
        score_path = Path(row["score"]["path"])
        index = ScoreEventIndex.from_musicxml(
            score_path, target_by_id[row_id]["lineage"]
        )
        contexts[row_id] = {
            "row": row,
            "probabilities": probabilities,
            "index": index,
            "score_path": score_path,
        }
        if position == 1 or position % 20 == 0 or position == len(
            manifest["inputs"]
        ):
            print(f"context={position}/{len(manifest['inputs'])}", flush=True)

    variants = []
    for variant_index, minimum_confidence in enumerate(CONFIDENCE_GRID):
        decode = replace(
            base_decode, min_confidence=float(minimum_confidence)
        )
        samples = []
        note_count = 0
        for position, (row_id, context) in enumerate(
            sorted(contexts.items()), 1
        ):
            notes = decode_mel_notes(
                context["probabilities"],
                midi_min=mel_model.config.midi_min,
                hop_sec=frontend.hop_sec,
                config=decode,
            )
            note_count += len(notes)
            candidates = _identity_candidates(notes)
            metric_sample, _diagnostics = rescue._map(
                identity_model,
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
                "minimum_confidence": minimum_confidence,
                "decode_config": decode.to_dict(),
                "transcribed_note_count": note_count,
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
            "mel_checkpoint_sha256": (
                args.expected_mel_checkpoint_sha256
            ),
            "identity_checkpoint_sha256": (
                args.expected_identity_checkpoint_sha256
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
