"""Run ALIGN joint transcription/alignment and emit DataCreate JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.joint.infer import (
    build_gui_alignment_payload,
    candidate_configs_for_checkpoint,
    candidate_generation_for_checkpoint,
    infer_joint_sample,
)
from alignmodel.pipeline import run_pipeline
from alignmodel.types import PipelineConfig, pipeline_label_to_dict


def _note_first_payload(sample: Path, weights: Path, device: str) -> dict:
    state = run_pipeline(
        sample,
        stages={1, 2},
        config=PipelineConfig(
            weights_dir=None,
            alignment_weights_dir=str(weights),
            detect_intonation=False,
        ),
        device=device,
        weights_dir=None,
        alignment_weights_dir=weights,
    )
    events = []
    for event_index, pair in enumerate(
        sorted(state.rhythm_pairs, key=lambda value: value.perf_start)
    ):
        if pair.score_index < 0:
            continue
        score_note = state.score.notes[pair.score_index]
        repeated = any(
            repeat.repeat_start <= pair.perf_start < repeat.repeat_end
            for repeat in state.note_repetitions
        )
        events.append(
            {
                "id": f"aligned_{event_index:05d}",
                "note_id": f"note_{score_note.index:04d}",
                "sounding_index": score_note.index,
                "score_index": score_note.index,
                "is_rest": False,
                "pitch": None,
                "midi": score_note.pitch,
                "measure": score_note.measure,
                "duration_ql": score_note.ql_end - score_note.ql_start,
                "ref_start": score_note.start,
                "ref_end": score_note.end,
                "perf_start": pair.perf_start,
                "perf_end": pair.perf_end,
                "alignment_kind": pair.kind,
                "is_repetition": repeated,
            }
        )
    labels = []
    for label in state.labels:
        item = pipeline_label_to_dict(label)
        item["source"] = "auto"
        labels.append(item)
    return {
        "format_version": 2,
        "engine": "align-note-first",
        "sample_id": state.sample_id,
        "events": events,
        "labels": labels,
        "transcribed_notes": [
            {
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "confidence": note.confidence,
            }
            for note in state.transcribed_notes
        ],
        "note_mapping": state.note_mapping,
        "repetitions": [
            {
                "source_i0": item.source_i0,
                "source_i1": item.source_i1,
                "repeat_i0": item.repeat_i0,
                "repeat_i1": item.repeat_i1,
                "source_start": item.source_start,
                "source_end": item.source_end,
                "repeat_start": item.repeat_start,
                "repeat_end": item.repeat_end,
                "confidence": item.confidence,
            }
            for item in state.note_repetitions
        ],
        "summary": {
            "engine": "align-note-first",
            "backend": "contextual-note-aligner",
            "event_count": len(events),
            "transcribed_note_count": len(state.transcribed_notes),
            "mapped_note_count": sum(
                value is not None for value in state.note_mapping
            ),
            "repetition_count": len(state.note_repetitions),
            "candidate_count": len(labels),
            "sample_rate": state.sr,
            "hop_length": state.config.hop_length,
        },
    }


def _transcription_payload(
    sample: Path, checkpoint: Path | None = None
) -> dict:
    import torch

    from alignmodel.joint.candidates import (
        CANDIDATE_GENERATION_VERSION,
        HIGH_RECALL_DECODE_CONFIGS,
        basic_pitch_candidate_union,
    )
    from alignmodel.transcription.basic_pitch import (
        extract_sample_basic_pitch_features,
    )

    cache = sample / "basic_pitch_cache.npz"
    features = extract_sample_basic_pitch_features(sample, cache_path=cache)
    configs = HIGH_RECALL_DECODE_CONFIGS
    candidate_generation = CANDIDATE_GENERATION_VERSION
    minimum_confidence = 0.65
    if checkpoint is not None:
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        configs = candidate_configs_for_checkpoint(checkpoint_payload)
        candidate_generation = candidate_generation_for_checkpoint(
            checkpoint_payload
        )
        minimum_confidence = float(
            (checkpoint_payload.get("training") or {}).get(
                "minimum_candidate_confidence", minimum_confidence
            )
        )
    notes = basic_pitch_candidate_union(
        features,
        configs=configs,
        minimum_confidence=minimum_confidence,
    )
    return {
        "engine": "basic-pitch-candidate-union",
        "sample_id": sample.name,
        "transcribed_notes": [
            {
                "pitch": int(note.pitch),
                "start": float(note.start),
                "end": float(note.end),
                "confidence": float(note.confidence),
            }
            for note in notes
        ],
        "summary": {
            "engine": "basic-pitch-candidate-union",
            "candidate_generation": candidate_generation,
            "minimum_candidate_confidence": minimum_confidence,
            "transcribed_note_count": len(notes),
            "cache_path": str(cache),
        },
    }


def _joint_payload(sample: Path, checkpoint: Path, device: str) -> dict:
    result = infer_joint_sample(sample, checkpoint, device=device)
    from datacreate.melody import parse_sounding_notes

    return build_gui_alignment_payload(
        result, parse_sounding_notes(sample / "verified_score.musicxml")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Dump frozen Basic Pitch notes without score alignment",
    )
    args = parser.parse_args()
    if args.transcribe_only:
        payload = _transcription_payload(args.sample, args.checkpoint)
    elif args.checkpoint is not None:
        payload = _joint_payload(args.sample, args.checkpoint, args.device)
    elif args.weights is not None:
        payload = _note_first_payload(args.sample, args.weights, args.device)
    else:
        raise SystemExit("Provide --checkpoint (joint), --weights (legacy), or --transcribe-only")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
