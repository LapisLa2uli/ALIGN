"""Run the note-first ALIGN pipeline and emit DataCreate-compatible JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.pipeline import run_pipeline
from alignmodel.types import PipelineConfig, pipeline_label_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    state = run_pipeline(
        args.sample,
        stages={1, 2},
        config=PipelineConfig(
            weights_dir=None,
            alignment_weights_dir=str(args.weights),
            detect_intonation=False,
        ),
        device=args.device,
        weights_dir=None,
        alignment_weights_dir=args.weights,
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
    payload = {
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
