"""End-to-end official note-wise F1 from saved transcriptions (9.2 repaired gold).

Transcribed notes (pitch, start, end) per clip are aligned with the structured
DP aligner and scored with the official exclusive note-wise metric. Span-less
predicted events take the rendered index of the gold note they pair with
along the pitch-sequence LCS; unpaired notes never match.
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.packed_data import sha256_file
from alignmodel.joint.perfect_dp_aligner_v1 import PerfectDPCosts, align_perfect
from alignmodel.transcription.ctc_decode_v2 import lcs_pairs
from realistic92_aligner_common import load_clip, metric_sample


UNPAIRED_OFFSET = 1_000_000
_STATE: dict[str, Any] = {}


def _init(root: str, lineage_dir: str, costs: dict[str, Any], aligner: str) -> None:
    _STATE.update(root=Path(root), lineage_dir=Path(lineage_dir), costs=costs, aligner=aligner)


def align_notes(notes, clip, costs: dict[str, Any], aligner: str = "dp_v1"):
    if aligner == "dp_v1":
        result = align_perfect(notes, clip.index.events, clip.score_path, PerfectDPCosts(**costs))
        return result.events, result.deletions
    if aligner == "dp_v2":
        from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts, align_robust
        result = align_robust(notes, clip.index.events, clip.score_path, RobustDPCosts(**costs))
        return result.events, result.deletions
    raise ValueError(f"Unknown aligner {aligner}")


def _run(payload: tuple[str, list[list[float]]]) -> tuple[str, Any, str | None]:
    name, raw = payload
    clip = load_clip(_STATE["root"], name, _STATE["lineage_dir"])
    if _STATE["aligner"] == "dp_v1":
        notes = [(int(row[0]), float(row[1]), float(row[2])) for row in raw if not (len(row) > 4 and row[4])]
    else:
        notes = [tuple(row) for row in raw]
    try:
        events, deletions = align_notes(notes, clip, _STATE["costs"], _STATE["aligner"])
        gold_pitch = np.asarray([event.pitch for event in clip.rendered], np.int64)
        pred_pitch = np.asarray(
            [event.pitch for event in sorted(events, key=lambda value: value.rendered_index)], np.int64
        )
        pairs = (
            {int(i): int(j) for i, j in lcs_pairs(pred_pitch, gold_pitch)}
            if len(pred_pitch) and len(gold_pitch) else {}
        )
        events = [
            replace(event, rendered_index=pairs.get(event.rendered_index,
                                                   UNPAIRED_OFFSET + event.rendered_index))
            for event in events
        ]
        return name, metric_sample(clip, events, deletions), None
    except Exception as error:  # noqa: BLE001
        return name, metric_sample(clip, (), frozenset()), f"{type(error).__name__}: {str(error)[:160]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aligner-freeze", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--aligner", default="dp_v1")
    parser.add_argument("--costs", default="{}")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.aligner_freeze.read_text(encoding="utf-8"))
    notes = json.loads(args.notes.read_text(encoding="utf-8"))["notes"]
    names = [name for name in freeze["eligible"][args.split] if name in notes]
    if args.limit:
        names = names[:args.limit]
    costs_path = Path(args.costs)
    costs = json.loads(costs_path.read_text(encoding="utf-8")) if costs_path.is_file() else json.loads(args.costs)
    samples, errors = [], collections.Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(args.root), freeze["lineage_dir"], costs, args.aligner)) as pool:
        for name, sample, error in pool.map(_run, [(name, notes[name]) for name in names], chunksize=4):
            samples.append(sample)
            if error:
                errors[error] += 1
    report = baseline._full_report(samples, seed=20260926, replicates=1000)
    by_group = collections.defaultdict(list)
    for sample in samples:
        by_group[sample.source.split(":")[0]].append(sample)
    result = {
        "split": args.split, "clips": len(names), "aligner": args.aligner, "costs": costs,
        "notes_file": str(args.notes), "notes_sha256": sha256_file(args.notes),
        "f1": report["f1"], "precision": report["precision"], "recall": report["recall"],
        "bootstrap_95": report["bootstrap_95"],
        "per_type_f1": {key: value["f1"] for key, value in report["per_type"].items()},
        "by_group_f1": {key: baseline._aggregate(value)["f1"] for key, value in by_group.items()},
        "errors": dict(errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("clips", "f1", "precision", "recall", "per_type_f1", "by_group_f1", "errors")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
