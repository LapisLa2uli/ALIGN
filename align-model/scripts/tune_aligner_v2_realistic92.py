"""Coordinate search of robust DP aligner (v2) costs on 9.2 val transcriptions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts, align_robust
from alignmodel.transcription.ctc_decode_v2 import lcs_pairs
from e2e_eval_realistic92 import UNPAIRED_OFFSET
from realistic92_aligner_common import load_clip, metric_sample


_STATE: dict[str, Any] = {"clips": {}}


def _init(root: str, lineage_dir: str) -> None:
    _STATE.update(root=Path(root), lineage_dir=Path(lineage_dir))


def _run(payload):
    name, raw, costs = payload
    clip = _STATE["clips"].get(name)
    if clip is None:
        clip = load_clip(_STATE["root"], name, _STATE["lineage_dir"])
        _STATE["clips"][name] = clip
    try:
        result = align_robust([tuple(row) for row in raw], clip.index.events, clip.score_path, RobustDPCosts(**costs))
        events = sorted(result.events, key=lambda value: value.rendered_index)
        gold = np.asarray([e.pitch for e in clip.rendered], np.int64)
        pred = np.asarray([e.pitch for e in events], np.int64)
        pairs = {int(i): int(j) for i, j in lcs_pairs(pred, gold)} if len(pred) and len(gold) else {}
        events = [replace(e, rendered_index=pairs.get(e.rendered_index, UNPAIRED_OFFSET + e.rendered_index))
                  for e in events]
        return metric_sample(clip, events, result.deletions)
    except Exception:  # noqa: BLE001
        return metric_sample(clip, (), frozenset())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aligner-freeze", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.aligner_freeze.read_text(encoding="utf-8"))
    notes = json.loads(args.notes.read_text(encoding="utf-8"))["notes"]
    names = [n for n in freeze["eligible"]["val"] if n in notes][::args.stride]
    grid = {
        "drop_confidence": [0.0, 0.3, 0.5, 0.7, 0.85],
        "drop_cost": [0.3, 0.5, 0.8],
        "optional_match": [0.15, 0.35, 0.6, 1e9],
        "alternative_match": [0.4, 0.6, 1.0, 1e9],
        "insert": [0.7, 1.0, 1.2],
        "delete": [0.8, 1.0, 1.2],
        "substitute": [1.1, 1.4, 1.7],
        "copy": [0.3, 0.5, 1.0],
    }
    current = asdict(RobustDPCosts())
    history = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(args.root), freeze["lineage_dir"])) as pool:
        def score(costs: dict[str, Any]) -> float:
            samples = list(pool.map(_run, [(n, notes[n], costs) for n in names], chunksize=8))
            return float(baseline._aggregate(samples)["f1"])
        best = score(current)
        print(json.dumps({"start": current, "f1": best}), flush=True)
        for round_index in range(args.rounds):
            for key, values in grid.items():
                for value in values:
                    if value == current[key]:
                        continue
                    trial = {**current, key: value}
                    f1 = score(trial)
                    history.append({"costs": trial, "f1": f1})
                    print(json.dumps({"round": round_index, key: value, "f1": round(f1, 5), "best": round(best, 5)}), flush=True)
                    if f1 > best + 1e-5:
                        best, current = f1, trial
    args.output.write_text(json.dumps({"clips": len(names), "best_f1": best, "best_costs": current,
                                       "history": history}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_f1": best, "best_costs": current}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
