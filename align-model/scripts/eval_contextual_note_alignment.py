"""Evaluate Layer-1-aware score mapping on held-out procedural notes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.pipeline import run_pipeline
from alignmodel.transcription import match_notes
from alignmodel.types import PipelineConfig


def _gold(row: dict):
    document = json.loads(Path(row["note_map"]).read_text(encoding="utf-8"))
    clean_pitch = {
        int(note["clean_index"]): int(note["pitch_midi"])
        for note in document.get("clean_notes") or []
    }
    notes = []
    targets = []
    for note in sorted(
        document.get("rendered_notes") or [],
        key=lambda value: int(value["rendered_index"]),
    ):
        target = note.get("primary_clean_index")
        if (
            target is not None
            and str(note.get("relationship")) != "substitute"
            and int(target) in clean_pitch
        ):
            pitch = clean_pitch[int(target)]
        else:
            pitch = int(note["pitch_midi_written"]) - 2
        notes.append(
            {
                "pitch": pitch,
                "start": float(note["start_sec"]),
                "end": float(note["end_sec"]),
            }
        )
        targets.append(int(target) if target is not None else None)
    return notes, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-weights", type=Path, required=True)
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-contextual-aligner", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = list(document.get(args.split) or [])
    if args.max_samples:
        rows = rows[: args.max_samples]
    samples = []
    total_correct = total_assigned = total_positive = total_matches = 0
    for index, raw in enumerate(rows, 1):
        row = dict(raw)
        if str(row.get("corpus") or row.get("root")) != "procedural12k":
            raise ValueError("Contextual evaluation rejected non-procedural row")
        state = run_pipeline(
            Path(row["sample_dir"]),
            stages={1, 2},
            config=PipelineConfig(
                weights_dir=None,
                alignment_weights_dir=str(args.alignment_weights),
                detect_intonation=False,
                use_contextual_note_aligner=not args.disable_contextual_aligner,
            ),
            device=args.device,
            weights_dir=None,
            alignment_weights_dir=args.alignment_weights,
        )
        gold_notes, gold_targets = _gold(row)
        pairs = match_notes(state.transcribed_notes, gold_notes)
        pred_to_gold = {pred: gold for pred, gold in pairs}
        correct = assigned = 0
        for pred_index, predicted in enumerate(state.note_mapping):
            gold_index = pred_to_gold.get(pred_index)
            if gold_index is None:
                assigned += predicted is not None
                continue
            truth = gold_targets[gold_index]
            assigned += predicted is not None
            correct += (
                predicted is not None
                and truth is not None
                and predicted == truth
            )
        positive = sum(value is not None for value in gold_targets)
        total_correct += correct
        total_assigned += assigned
        total_positive += positive
        total_matches += len(pairs)
        samples.append(
            {
                "sample": state.sample_id,
                "n_correct": correct,
                "n_assigned": assigned,
                "n_positive": positive,
                "n_transcription_matches": len(pairs),
                "n_repetitions": len(state.note_repetitions),
            }
        )
        if index == 1 or index % 20 == 0 or index == len(rows):
            print(f"{args.split} {index}/{len(rows)}", flush=True)
    precision = total_correct / max(total_assigned, 1)
    recall = total_correct / max(total_positive, 1)
    report = {
        "procedural_only": True,
        "raw_derived_data_used": False,
        "contextual_aligner_enabled": not args.disable_contextual_aligner,
        "n_clips": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "n_correct": total_correct,
        "n_assigned": total_assigned,
        "n_positive": total_positive,
        "n_transcription_matches": total_matches,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
