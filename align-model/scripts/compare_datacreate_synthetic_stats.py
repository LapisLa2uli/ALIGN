"""Compare DataCreate takes with synthetic datasets on performance statistics.

Uses the same frozen transcriber outputs for both (cached CTC -> rich decode),
so the numbers describe what the stack sees: onset rate, inter-onset
intervals, share of very short notes (transition blips), pitch range, take
length, silence share, plus the score coverage and error mix of DataCreate
takes (human labels).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from precision_harness_v4 import HOP, decode_rows

ALIGN = Path(__file__).resolve().parents[1]


def _stats(rows_by_clip: dict[str, list[list[float]]], voiced_by_clip: dict[str, np.ndarray]) -> dict:
    iois, pitches, rates, lengths, voiced_share, short_share = [], [], [], [], [], []
    for name, rows in rows_by_clip.items():
        primary = [row for row in rows if not row[4]]
        if len(primary) < 4:
            continue
        starts = np.array([row[1] for row in primary])
        ioi = np.diff(starts)
        iois.extend(ioi.tolist())
        pitches.extend(int(row[0]) for row in primary)
        voiced = voiced_by_clip[name]
        active = float((voiced > 0.5).sum() * HOP)
        lengths.append(len(voiced) * HOP)
        voiced_share.append(active / max(len(voiced) * HOP, 1e-6))
        rates.append(len(primary) / max(active, 1e-6))
        short_share.append(float((ioi < 0.06).mean()))
    iois = np.array(iois)
    return {
        "clips": len(rates),
        "onsets_per_voiced_second": {"p25": round(float(np.percentile(rates, 25)), 2),
                                     "median": round(float(np.median(rates)), 2),
                                     "p75": round(float(np.percentile(rates, 75)), 2)},
        "ioi_seconds": {q: round(float(np.percentile(iois, p)), 3) for q, p in
                        (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))},
        "share_ioi_lt_60ms": round(float((iois < 0.06).mean()), 4),
        "share_ioi_lt_100ms": round(float((iois < 0.10).mean()), 4),
        "clip_share_ioi_lt_60ms_median": round(float(np.median(short_share)), 4),
        "pitch_written": {q: int(np.percentile(pitches, p)) for q, p in
                          (("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95))},
        "clip_seconds": {"median": round(float(np.median(lengths)), 1), "p90": round(float(np.percentile(lengths, 90)), 1)},
        "voiced_share_median": round(float(np.median(voiced_share)), 3),
    }


def _load_cache(directory: Path, names, decoder) -> tuple[dict, dict]:
    rows, voiced = {}, {}
    for name in names:
        path = directory / f"{name}.npz"
        if not path.is_file():
            continue
        cache = np.load(path)
        ctc = cache["ctc"].astype(np.float32)
        rows[name] = decode_rows(ctc, 52, decoder)
        voiced[name] = cache["voiced"].astype(np.float32)
    return rows, voiced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads((ALIGN / "runs/realistic92-stack-v2/CANDIDATE_STACK_V3.json").read_text(encoding="utf-8"))
    decoder = candidate["decoder"]
    report = {}
    dc_cache = ALIGN / "runs/precision-v4/dc-cache"
    dc_names = sorted(path.stem for path in dc_cache.glob("*.npz"))
    report["datacreate"] = _stats(*_load_cache(dc_cache, dc_names, decoder))
    for dataset, cache in (("9.2_val", ALIGN / "runs/realistic92-stack-v2/cache-val-v3c"),
                           ("10.2_val", ALIGN / "runs/fast102-v1/cache-val-v3c")):
        names = sorted(path.stem for path in cache.glob("*.npz"))
        report[dataset] = _stats(*_load_cache(cache, names, decoder))

    samples = ALIGN.parent / "DataCreate" / "samples"
    scores = collections.Counter()
    segment_notes = []
    for sample in sorted(p for p in samples.iterdir() if p.is_dir()):
        meta_path = sample / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        segment = meta.get("score_segment") or {}
        scores[str(segment.get("score") or segment.get("source") or segment.get("score_id") or "?")] += 1
        if segment.get("note_count"):
            segment_notes.append(int(segment["note_count"]))
    report["datacreate_scores"] = dict(scores.most_common())
    labels = collections.Counter()
    takes_with_repetition = 0
    for sample in sorted(p for p in samples.iterdir() if p.is_dir()):
        document = json.loads((sample / "labels.json").read_text(encoding="utf-8"))
        manual = [label for label in document.get("labels") or [] if label.get("source") == "manual"]
        labels.update(label["type"] for label in manual)
        takes_with_repetition += any(label["type"] == "repetition" for label in manual)
    report["datacreate_manual_label_types"] = dict(labels.most_common())
    report["datacreate_takes_with_manual_repetition"] = takes_with_repetition
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
