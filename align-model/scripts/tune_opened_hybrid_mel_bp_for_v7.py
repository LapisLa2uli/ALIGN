"""Gap-fill mel notes with Basic Pitch on opened v3–v6 after the v7 freeze.

Pure Basic Pitch + identity stayed near F1 0.75 on this pool. The frozen mel
decode at confidence 0.75 remains the stronger frontend. This grid keeps those
mel notes and adds Basic Pitch notes that do not overlap them, then maps with
the frozen identity CRF. v7 targets and keys are never read.
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
from alignmodel.transcription.basic_pitch import (
    FROZEN_DECODE_CONFIG,
    decode_frozen_basic_pitch,
    extract_basic_pitch_features,
    sanitize_basic_pitch_notes,
)
from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from eval_v2_template_rescue_lockbox import PITCH_METADATA, _decrypt_targets


SCHEMA_VERSION = "align-orn-v3v4v5v6-hybrid-mel-bp-v7-development-v1"
# None keeps mel notes only. Other values are the maximum time overlap, in
# seconds, still treated as a gap that Basic Pitch may fill.
GAP_OVERLAP_SEC = (None, 0.02, 0.05)


def _candidates(
    notes: Sequence[Any], features: Any
) -> tuple[IdentityCandidate, ...]:
    output: list[IdentityCandidate] = []
    for note in notes:
        confidences = getattr(note, "candidate_confidences", ())
        pitches = getattr(note, "pitch_candidates", ())
        if confidences and pitches:
            output.append(
                IdentityCandidate(
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                    confidence=float(note.confidence),
                    alternatives=tuple(map(int, pitches)),
                    alternative_confidences=tuple(map(float, confidences)),
                )
            )
        else:
            output.extend(acoustic._identity_candidates((note,), features))
    return tuple(output)


def _overlap(left: Any, right: Any) -> float:
    return min(float(left.end), float(right.end)) - max(
        float(left.start), float(right.start)
    )


def _gap_fill(
    mel_notes: Sequence[Any],
    basic_pitch_notes: Sequence[Any],
    maximum_overlap_sec: float | None,
) -> list[Any]:
    if maximum_overlap_sec is None:
        return list(mel_notes)
    added = []
    for note in basic_pitch_notes:
        if any(
            _overlap(note, other) > maximum_overlap_sec
            for other in mel_notes
        ):
            continue
        added.append(note)
    return sorted(
        [*mel_notes, *added],
        key=lambda note: (float(note.start), int(note.pitch)),
    )


def _load_population(
    *,
    name: str,
    manifest_path: Path,
    targets_path: Path,
    key_path: Path,
    mel_cache_dir: Path,
    bp_cache_dir: Path,
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
    mel_cache_dir.mkdir(parents=True, exist_ok=True)
    bp_cache_dir.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(manifest["inputs"], 1):
        row_id = str(row["row_id"])
        mel_cache = mel_cache_dir / f"{row_id}.npz"
        if mel_cache.exists():
            cached = np.load(mel_cache)
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
            np.savez_compressed(mel_cache, **probabilities)
        features = extract_basic_pitch_features(
            Path(row["audio"]["path"]),
            source_metadata=PITCH_METADATA,
            cache_path=bp_cache_dir / f"{row_id}.npz",
        )
        score_path = Path(row["score"]["path"])
        index = ScoreEventIndex.from_musicxml(
            score_path, target_by_id[row_id]["lineage"]
        )
        contexts[f"{name}:{row_id}"] = {
            "population": name,
            "row": row,
            "probabilities": probabilities,
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
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    return (
        float(metrics["f1"]),
        min(precision, 0.80),
        -abs(precision - recall),
        -int(variant["variant"]),
    )


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
    parser.add_argument("--mel-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-mel-checkpoint-sha256", required=True)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-identity-checkpoint-sha256", required=True)
    parser.add_argument("--mel-cache-dir", type=Path, required=True)
    parser.add_argument("--bp-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
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
    if sha256_file(args.mel_checkpoint) != args.expected_mel_checkpoint_sha256:
        raise ValueError("Mel checkpoint mismatch")
    if (
        sha256_file(args.identity_checkpoint)
        != args.expected_identity_checkpoint_sha256
    ):
        raise ValueError("Identity checkpoint mismatch")

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
    decode = replace(
        base_decode, min_confidence=float(args.minimum_confidence)
    )
    identity_model = acoustic._load_model(args.identity_checkpoint)[0]
    contexts: dict[str, dict[str, Any]] = {}
    for name in ("v3", "v4", "v5", "v6"):
        contexts.update(
            _load_population(
                name=name,
                manifest_path=getattr(args, f"{name}_manifest"),
                targets_path=getattr(args, f"{name}_targets"),
                key_path=getattr(args, f"{name}_key"),
                mel_cache_dir=args.mel_cache_dir / name,
                bp_cache_dir=args.bp_cache_dir / name,
                mel_model=mel_model,
                frontend=frontend,
                device=device,
                batch_size=args.batch_size,
            )
        )

    prepared = []
    for key, context in sorted(contexts.items()):
        mel_notes = decode_mel_notes(
            context["probabilities"],
            midi_min=mel_model.config.midi_min,
            hop_sec=frontend.hop_sec,
            config=decode,
        )
        basic_pitch_notes = sanitize_basic_pitch_notes(
            decode_frozen_basic_pitch(context["features"]),
            context["features"],
            FROZEN_DECODE_CONFIG,
        )
        prepared.append((key, context, mel_notes, basic_pitch_notes))
        if len(prepared) == 1 or len(prepared) % 50 == 0 or len(
            prepared
        ) == len(contexts):
            print(f"decoded={len(prepared)}/{len(contexts)}", flush=True)

    variants = []
    for variant_index, maximum_overlap_sec in enumerate(GAP_OVERLAP_SEC):
        samples = []
        mel_count = 0
        added_count = 0
        for position, (_key, context, mel_notes, basic_pitch_notes) in enumerate(
            prepared, 1
        ):
            notes = _gap_fill(
                mel_notes, basic_pitch_notes, maximum_overlap_sec
            )
            mel_count += len(mel_notes)
            added_count += len(notes) - len(mel_notes)
            metric_sample, _diagnostics = rescue._map(
                identity_model,
                _candidates(notes, context["features"]),
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
                or position == len(prepared)
            ):
                print(
                    f"variant={variant_index} "
                    f"rows={position}/{len(prepared)} "
                    f"overlap={maximum_overlap_sec}",
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
                "maximum_overlap_sec": maximum_overlap_sec,
                "minimum_confidence": float(args.minimum_confidence),
                "mel_note_count": mel_count,
                "added_basic_pitch_note_count": added_count,
                "official_note_wise": report,
            }
        )
        print(
            json.dumps(
                {
                    "variant": variant_index,
                    "maximum_overlap_sec": maximum_overlap_sec,
                    "added_basic_pitch_note_count": added_count,
                    "f1": report["f1"],
                    "precision": report["precision"],
                    "recall": report["recall"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected = max(variants, key=_rank)
    acoustic._atomic_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "development_rows": len(contexts),
            "development_populations": {
                name: sum(1 for key in contexts if key.startswith(f"{name}:"))
                for name in ("v3", "v4", "v5", "v6")
            },
            "v7_freeze_sha256": args.expected_v7_freeze_sha256,
            "v7_targets_read": False,
            "v7_key_read": False,
            "v7_opened": False,
            "frontend": "mel_v1_conf_0.75_plus_basic_pitch_gap_fill",
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
                "maximum_overlap_sec": selected["maximum_overlap_sec"],
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
