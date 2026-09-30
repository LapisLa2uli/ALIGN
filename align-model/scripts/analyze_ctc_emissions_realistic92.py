"""Separate true vs false emissions of the greedy CTC decoder on cached val outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignmodel.transcription.ctc_decode_v2 import greedy_tokens, lcs_pairs, runs_from_tokens
from realistic92_transcriber_breakdown import load_gold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.cache / "cache_info.json").read_text(encoding="utf-8"))
    midi_min = int(info["midi_min"])
    names = json.loads(args.names.read_text(encoding="utf-8"))[:args.limit]
    rows = []
    for name in names:
        with np.load(args.cache / f"{name}.npz") as data:
            out = {key: np.asarray(data[key], np.float32) for key in data.files}
        gold = load_gold(args.root, name)
        runs = runs_from_tokens(greedy_tokens(out["ctc"], 0.3))
        pitches = np.asarray([midi_min + t - 1 for t, _f, _l in runs], np.int64)
        gold_pitch = np.asarray([g["pitch"] for g in gold], np.int64)
        matched = set(lcs_pairs(pitches, gold_pitch)[:, 0].tolist()) if len(runs) and len(gold) else set()
        attack = np.maximum.reduce((out["onset"], out["rearticulation"], out["boundary"]))
        for index, (token, first, last) in enumerate(runs):
            lo, hi = max(0, first - 2), min(len(attack), first + 3)
            previous = runs[index - 1] if index else None
            following = runs[index + 1] if index + 1 < len(runs) else None
            kind = "normal"
            if previous is not None and previous[0] == token:
                kind = "same_as_previous"
            elif previous is not None and following is not None and previous[0] == following[0] != token:
                kind = "aba_middle"
            gap_blank = (
                float(out["ctc"][previous[2]:first, 0].min()) if previous is not None and previous[2] < first else 1.0
            )
            rows.append({
                "kind": kind,
                "true": index in matched,
                "peak": float(out["ctc"][first:last, token].max()),
                "run_frames": last - first,
                "attack": float(attack[lo:hi].max()),
                "reart": float(out["rearticulation"][lo:hi].max()),
                "gap_frames": (first - previous[2]) if previous is not None else -1,
                "gap_blank_min": gap_blank,
                "interval": abs(int(token) - int(previous[0])) if previous is not None else -1,
            })
    args.output.write_text(json.dumps(rows), encoding="utf-8")
    for kind in ("same_as_previous", "aba_middle", "normal"):
        subset = [row for row in rows if row["kind"] == kind]
        true = [row for row in subset if row["true"]]
        false = [row for row in subset if not row["true"]]
        print(f"== {kind}: true={len(true)} false={len(false)}")
        for feature in ("peak", "attack", "reart", "gap_frames", "gap_blank_min", "run_frames"):
            q = lambda values: [round(float(v), 3) for v in np.percentile(values, [10, 25, 50, 75, 90])] if values else []
            print(f"  {feature:14s} true {q([r[feature] for r in true])}  false {q([r[feature] for r in false])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
