"""Development loop: DP aligner on repaired 9.2 train clips with error breakdown."""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.perfect_dp_aligner_v1 import PerfectDPCosts, align_perfect
from realistic92_aligner_common import load_clip, metric_sample, perfect_transcription


_STATE: dict[str, Any] = {}


def _init(root: str, lineage_dir: str, costs: dict[str, Any]) -> None:
    _STATE.update(root=Path(root), lineage_dir=Path(lineage_dir), costs=PerfectDPCosts(**costs))


def _run(name: str) -> dict[str, Any]:
    clip = load_clip(_STATE["root"], name, _STATE["lineage_dir"])
    try:
        result = align_perfect(
            perfect_transcription(clip), clip.index.events, clip.score_path, _STATE["costs"]
        )
    except Exception as error:  # noqa: BLE001
        return {"name": name, "error": f"{type(error).__name__}: {error}",
                "sample": metric_sample(clip, (), frozenset())}
    sample = metric_sample(clip, result.events, result.deletions)
    errors = collections.Counter()
    gold_by_index = {event.rendered_index: event for event in clip.rendered}
    for event in result.events:
        gold = gold_by_index[event.rendered_index]
        gold_kind = gold.relationship
        if event.score_span == gold.score_span and event.copy_pass == gold.copy_pass:
            if event.relationship != gold.relationship:
                errors[f"type:{gold_kind}->{event.relationship}"] += 1
        else:
            errors[f"loc:{gold_kind}->{event.relationship}"] += 1
    predicted_del = set(result.deletions)
    gold_del = set(clip.index.deleted_event_indices)
    errors["del_false_pos"] += len(predicted_del - gold_del)
    errors["del_false_neg"] += len(gold_del - predicted_del)
    gold_copies = max((event.copy_pass for event in clip.rendered), default=0)
    return {"name": name, "sample": sample, "errors": dict(errors),
            "copies_pred": result.copies, "copies_gold": gold_copies}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lineage-dir", type=Path, required=True)
    parser.add_argument("--names", type=Path, help="JSON list of train clip names")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--costs", default="{}")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--show-worst", type=int, default=8)
    args = parser.parse_args()

    if args.names:
        names = json.loads(args.names.read_text(encoding="utf-8"))
    else:
        names = sorted(path.stem for path in args.lineage_dir.glob("*.json"))
    random.Random(args.seed).shuffle(names)
    names = names[:args.limit]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init,
        initargs=(str(args.root), str(args.lineage_dir), json.loads(args.costs)),
    ) as pool:
        rows = list(pool.map(_run, names, chunksize=2))
    samples = [row["sample"] for row in rows]
    report = baseline._full_report(samples, seed=1, replicates=200)
    totals = collections.Counter()
    for row in rows:
        totals.update(row.get("errors", {}))
    copy_confusion = collections.Counter(
        (row.get("copies_gold"), row.get("copies_pred")) for row in rows if "copies_pred" in row
    )
    by_group = collections.defaultdict(list)
    for sample in samples:
        by_group[sample.source.split(":")[0]].append(sample)
    print(json.dumps({
        "clips": len(rows),
        "seconds": round(time.perf_counter() - started, 1),
        "f1": report["f1"], "precision": report["precision"], "recall": report["recall"],
        "by_group_f1": {k: baseline._aggregate(v)["f1"] for k, v in by_group.items()},
        "per_type_f1": {k: round(v["f1"], 4) for k, v in report["per_type"].items()},
        "errors": dict(totals.most_common(20)),
        "copies_gold_pred": {f"{k[0]}->{k[1]}": v for k, v in copy_confusion.items()},
        "exceptions": [row["error"][:120] for row in rows if "error" in row][:5],
    }, indent=2))
    worst = sorted(
        rows, key=lambda row: baseline._aggregate([row["sample"]])["f1"]
    )[:args.show_worst]
    for row in worst:
        print(row["name"], round(baseline._aggregate([row["sample"]])["f1"], 3),
              row.get("errors"), row.get("copies_gold"), row.get("copies_pred"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
