"""Evaluate the transcription-first three-layer pipeline on procedural data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.eval_melodies import eval_sample, summarize_eval_rows
from alignmodel.pipeline import run_pipeline
from alignmodel.types import PipelineConfig, pipeline_label_to_dict


NOTE_FIRST_TYPES = {
    "repetition",
    "wrong_note",
    "extra_note",
    "missed_note",
    "rhythm_error",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-weights", type=Path, required=True)
    parser.add_argument("--stage-weights", type=Path, default=None)
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--stages", default="1,2,3")
    parser.add_argument(
        "--types",
        default=",".join(sorted(NOTE_FIRST_TYPES)),
        help="Comma-separated label types to score",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-repetition-model", action="store_true")
    parser.add_argument("--disable-contextual-aligner", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_rows = list(document.get(args.split) or [])
    stages = {
        int(value) for value in str(args.stages).split(",") if value.strip()
    }
    types = {
        value.strip()
        for value in str(args.types).split(",")
        if value.strip()
    }
    if args.max_samples:
        source_rows = source_rows[: args.max_samples]
    rows = []
    for index, raw in enumerate(source_rows, 1):
        row = dict(raw)
        if str(row.get("corpus") or row.get("root")) != "procedural12k":
            raise ValueError("Note-first evaluation rejected non-procedural row")
        sample = Path(row["sample_dir"])
        state = run_pipeline(
            sample,
            stages=stages,
            config=PipelineConfig(
                weights_dir=(
                    str(args.stage_weights) if args.stage_weights else None
                ),
                alignment_weights_dir=str(args.alignment_weights),
                detect_intonation=False,
                use_note_repetition_model=not args.disable_repetition_model,
                use_contextual_note_aligner=not args.disable_contextual_aligner,
            ),
            device=args.device,
            weights_dir=args.stage_weights,
            alignment_weights_dir=args.alignment_weights,
        )
        result = eval_sample(
            sample,
            pred_labels=[
                pipeline_label_to_dict(label) for label in state.labels
            ],
            types=types,
        )
        result["n_transcribed_notes"] = len(state.transcribed_notes)
        result["n_note_repetitions"] = len(state.note_repetitions)
        result["n_mapped_notes"] = sum(
            value is not None for value in state.note_mapping
        )
        rows.append(result)
        print(f"{args.split} {index}/{len(source_rows)}", flush=True)
    summary = summarize_eval_rows(rows)
    summary.update(
        {
            "procedural_only": True,
            "raw_derived_data_used": False,
            "split": args.split,
            "types": sorted(types),
            "stages": sorted(stages),
            "repetition_model_enabled": not args.disable_repetition_model,
            "contextual_aligner_enabled": not args.disable_contextual_aligner,
            "samples": rows,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "samples"},
            indent=2,
        )
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
