"""Decode cached CTC outputs with one or more decoders and report breakdowns.

Optionally writes the chosen decoder's notes (pitch, start, end) per clip for
end-to-end aligner evaluation.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from alignmodel.transcription.ctc_decode_v2 import (
    StructuredDecodeConfig,
    greedy_decode_notes,
    rich_decode,
    structured_decode_notes,
)
from realistic92_transcriber_breakdown import Breakdown, load_gold


_STATE: dict[str, Any] = {}


def _init(root: str, cache: str, midi_min: int) -> None:
    _STATE.update(root=Path(root), cache=Path(cache), midi_min=midi_min)


def _decode(outputs: dict[str, np.ndarray], spec: dict[str, Any], midi_min: int) -> list:
    if spec["decoder"] == "greedy":
        return greedy_decode_notes(outputs["ctc"], midi_min, float(spec.get("blank_scale", 1.0)))
    if spec["decoder"] == "rich":
        return rich_decode(outputs["ctc"], midi_min, **spec.get("config", {}))
    return structured_decode_notes(outputs, midi_min, StructuredDecodeConfig(**spec.get("config", {})))


def _primary(notes: list) -> list[tuple[int, int]]:
    if notes and isinstance(notes[0], dict):
        return [(note["pitch"], note["frame"]) for note in notes if not note["optional"]]
    return list(notes)


def _run(payload: tuple[str, list[dict[str, Any]]]) -> tuple[str, list[list[int]], list[list[tuple[int, int]]]]:
    name, specs = payload
    with np.load(_STATE["cache"] / f"{name}.npz") as data:
        outputs = {key: np.asarray(data[key], np.float32) for key in data.files}
    gold = load_gold(_STATE["root"], name)
    decoded = [_decode(outputs, spec, _STATE["midi_min"]) for spec in specs]
    return name, gold, decoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True, help="JSON list of decoder specs")
    parser.add_argument("--write-notes-for", type=int, help="index of spec whose notes are written")
    parser.add_argument("--notes-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    info = json.loads((args.cache / "cache_info.json").read_text(encoding="utf-8"))
    names = json.loads(args.names.read_text(encoding="utf-8"))
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    breakdowns = [Breakdown() for _ in specs]
    notes_out: dict[str, list[list[float]]] = {}
    hop = float(info["hop_sec"])
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(args.root), str(args.cache), int(info["midi_min"]))) as pool:
        for name, gold, decoded in pool.map(_run, [(name, specs) for name in names], chunksize=8):
            for breakdown, notes in zip(breakdowns, decoded):
                breakdown.add([pitch for pitch, _frame in _primary(notes)], gold)
            if args.write_notes_for is not None:
                chosen = decoded[args.write_notes_for]
                if chosen and isinstance(chosen[0], dict):
                    starts = [note["frame"] * hop for note in chosen]
                    notes_out[name] = [
                        [note["pitch"], round(start, 6),
                         round(max(start + 0.01, starts[k + 1] if k + 1 < len(starts) else start + 0.1), 6),
                         round(note["confidence"], 4), int(note["optional"]),
                         note["alternative_pitch"], round(note["alternative_confidence"], 4)]
                        for k, (note, start) in enumerate(zip(chosen, starts))
                    ]
                else:
                    starts = [frame * hop for _pitch, frame in chosen]
                    notes_out[name] = [
                        [pitch, round(start, 6),
                         round(max(start + 0.01, starts[k + 1] if k + 1 < len(starts) else start + 0.1), 6)]
                        for k, ((pitch, _frame), start) in enumerate(zip(chosen, starts))
                    ]
    reports = []
    for spec, breakdown in zip(specs, breakdowns):
        report = breakdown.report()
        reports.append({"spec": spec, **report})
        print(json.dumps({
            "spec": spec, "f1": round(report["f1"], 5), "precision": round(report["precision"], 5),
            "recall": round(report["recall"], 5),
            "lt80_recall": {k: v for k, v in report["recall_by"].items() if k in ("dur_lt50", "dur_50to80")},
            "repeat_recall": report["recall_by"].get("same_pitch_neighbor"),
            "fp": report["false_positives"],
        }), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cache": str(args.cache), "clips": len(names), "reports": reports},
                                      indent=2) + "\n", encoding="utf-8")
    if args.write_notes_for is not None and args.notes_output is not None:
        args.notes_output.parent.mkdir(parents=True, exist_ok=True)
        args.notes_output.write_text(json.dumps({"spec": specs[args.write_notes_for], "notes": notes_out}),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
