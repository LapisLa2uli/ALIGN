"""Evaluate Layer-1-aware score mapping on held-out procedural notes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignmodel.pipeline import run_pipeline
from alignmodel.transcription import match_notes
from alignmodel.types import PipelineConfig


def _gold(row: dict):
    from alignmodel.stages.score_graph import build_score_graph

    sample = Path(str(row["sample_dir"]))
    path = Path(
        str(
            row.get("note_map")
            or Path(str(row["sample_dir"])) / "note_map.json"
        )
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    graph = build_score_graph(sample / "verified_score.musicxml")
    target_index_map = {
        source_index: graph_note.index
        for graph_note in graph.notes
        for source_index in (
            graph_note.source_note_indices or [graph_note.index]
        )
    }
    clean_pitch = {
        int(note["clean_index"]): int(note["pitch_midi"])
        for note in document.get("clean_notes") or []
    }
    rendered = sorted(
        document.get("rendered_notes") or [],
        key=lambda value: int(value["rendered_index"]),
    )
    offsets = [
        int(note["pitch_midi_written"])
        - clean_pitch[int(note["primary_clean_index"])]
        for note in rendered
        if note.get("primary_clean_index") is not None
        and int(note["primary_clean_index"]) in clean_pitch
        and str(note.get("relationship")) in {"match", "copy"}
    ]
    correction = -int(round(float(np.median(offsets)))) if offsets else 0
    notes = []
    targets = []
    for note in rendered:
        target = note.get("primary_clean_index")
        if (
            target is not None
            and str(note.get("relationship")) != "substitute"
            and int(target) in clean_pitch
        ):
            pitch = clean_pitch[int(target)]
        else:
            pitch = int(note["pitch_midi_written"]) + correction
        notes.append(
            {
                "pitch": pitch,
                "start": float(note["start_sec"]),
                "end": float(note["end_sec"]),
            }
        )
        targets.append(
            target_index_map.get(int(target))
            if target is not None
            else None
        )
    return notes, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-weights", type=Path, required=True)
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-contextual-aligner", action="store_true")
    parser.add_argument(
        "--strategy",
        choices=("contextual", "deterministic", "multi_start", "revision"),
        default="contextual",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_root = str(
        (document.get("roots") or {}).get("procedural12k") or ""
    )
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
                note_alignment_strategy=(
                    "deterministic"
                    if args.disable_contextual_aligner
                    else args.strategy
                ),
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
        "dataset_root": dataset_root,
        "exclusive_manifest_dataset": True,
        "raw_derived_data_used": "raw" in dataset_root.lower(),
        "contextual_aligner_enabled": not args.disable_contextual_aligner,
        "alignment_strategy": (
            "deterministic"
            if args.disable_contextual_aligner
            else args.strategy
        ),
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
