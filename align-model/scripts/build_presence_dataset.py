"""Build note-presence verifier examples from dual-mel caches (train/val splits only).

Per clip (degraded render only): every planted missed note with played
neighbours is a negative; up to ``--positives`` played score notes are
positives (half drawn from notes shorter than 150 ms). Each example is centred
where an aligner would expect the note: interpolated by score position between
the forced-aligned onsets of the nearest played neighbours.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.presence_verifier_v1 import extract_window
from alignmodel.transcription.mel_v1_data import MelPackedCache

HOP = 256 / 22050
_STATE: dict[str, Any] = {}


def _init(payload):
    _STATE.update(payload)
    _STATE["cache_obj"] = MelPackedCache(Path(payload["cache"]), deep=False)


def _clip(record_index: int):
    cache = _STATE["cache_obj"]
    record = _STATE["records"][record_index]
    name = record.sample
    lineage = json.loads((Path(_STATE["lineage_dir"]) / f"{name}.json").read_text(encoding="utf-8"))
    index = ScoreEventIndex.from_lineage(lineage)
    events = index.events
    onset_by_rendered = {int(row["gold_index"]): float(row["start_sec"]) / HOP for row in record.target}
    played: dict[int, float] = {}
    for rendered_index, event in enumerate(index.rendered_events):
        if event.score_span is None or event.copy_pass:
            continue
        frame = onset_by_rendered.get(rendered_index)
        if frame is not None:
            played.setdefault(int(event.score_span[0]), frame)
    deleted = set(index.deleted_event_indices)
    order = sorted(played)
    rng = random.Random(f"{_STATE['seed']}:{name}")

    def neighbours(k: int):
        before = [s for s in order if s < k]
        after = [s for s in order if s > k]
        if not before or not after:
            return None
        p, q = before[-1], after[0]
        span = events[q].ql_start - events[p].ql_start
        if span <= 0:
            return None
        frames = played[q] - played[p]
        center = played[p] + (events[k].ql_start - events[p].ql_start) / span * frames
        duration = (events[k].ql_end - events[k].ql_start) / span * frames
        return center, duration, frames, p, q

    rows = []
    candidates = []
    for k in sorted(deleted):
        geometry = neighbours(k)
        if geometry is not None:
            candidates.append((k, 0, geometry))
    positives = [k for k in order if k not in deleted]
    short = [k for k in positives if (neighbours(k) or (0, 99, 0, 0, 0))[1] * HOP < 0.15]
    chosen = set(rng.sample(short, min(len(short), _STATE["positives"] // 2)))
    rest = [k for k in positives if k not in chosen]
    chosen |= set(rng.sample(rest, min(len(rest), _STATE["positives"] - len(chosen))))
    for k in sorted(chosen):
        geometry = neighbours(k)
        if geometry is not None:
            candidates.append((k, 1, geometry))
    if not candidates:
        return []
    mel = cache.mel(record).astype(np.float32)
    for k, label, (center, duration, frames, p, q) in candidates:
        rows.append((extract_window(mel, center).astype(np.float16), int(events[k].pitch), int(events[p].pitch),
                     int(events[q].pitch), float(duration), float(abs(frames)) * 0.05, int(label)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--positives", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    eligible = set(freeze["eligible"][args.split])
    cache = MelPackedCache(args.cache, deep=False)
    records = [record for record in cache.records(args.split, include_targets=True)
               if "#" not in record.sample and record.sample in eligible]
    if args.limit:
        records = records[:args.limit]
    payload = {"cache": str(args.cache), "records": records, "lineage_dir": freeze["lineage_dir"],
               "positives": args.positives, "seed": args.seed}
    windows, meta = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(payload,)) as pool:
        for position, rows in enumerate(pool.map(_clip, range(len(records)), chunksize=8), 1):
            for window, pitch, previous, following, duration, uncertainty, label in rows:
                windows.append(window)
                meta.append((pitch, previous, following, duration, uncertainty, label, position - 1))
            if position % 500 == 0:
                print(f"{position}/{len(records)} examples={len(windows)}", flush=True)
    meta_array = np.array(meta, np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, windows=np.stack(windows), pitch=meta_array[:, 0].astype(np.int16),
             previous=meta_array[:, 1].astype(np.int16), following=meta_array[:, 2].astype(np.int16),
             duration=meta_array[:, 3], uncertainty=meta_array[:, 4], label=meta_array[:, 5].astype(np.int8),
             clip=meta_array[:, 6].astype(np.int32))
    print(json.dumps({"clips": len(records), "examples": len(windows),
                      "negatives": int((meta_array[:, 5] == 0).sum()), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
