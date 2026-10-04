"""Which gold notes were dropped from forced-aligned CTC training targets.

Training targets keep only notes the forced aligner placed; skipped gold notes
are absent from the target sequence. This tallies the skipped notes by the same
categories the error audit uses.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import zlib
from pathlib import Path


DURATION_BINS = ((0.0, 0.05, "dur_lt50"), (0.05, 0.08, "dur_50to80"), (0.08, 0.12, "dur_80to120"),
                 (0.12, 0.2, "dur_120to200"), (0.2, 1e9, "dur_ge200"))


def _tags(rows: list[dict], index: int) -> list[str]:
    row = rows[index]
    duration = float(row["end_sec"]) - float(row["start_sec"])
    pitch = int(row["pitch_midi_written"])
    tags = ["all"]
    for lo, hi, label in DURATION_BINS:
        if lo <= duration < hi:
            tags.append(label)
    relationship = str(row.get("relationship") or "extra")
    tags.append(f"rel_{relationship}")
    if relationship == "extra" and not row.get("performed_indices"):
        tags.append("ornament_renderer_only")
    same = (index > 0 and int(rows[index - 1]["pitch_midi_written"]) == pitch) or (
        index + 1 < len(rows) and int(rows[index + 1]["pitch_midi_written"]) == pitch
    )
    tags.append("same_pitch_neighbor" if same else "pitch_change")
    return tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    stats = [json.loads(line) for line in (args.cache / "alignment_stats.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    notes = sum(row["notes"] for row in stats)
    skipped = sum(row["skipped"] for row in stats)
    excluded = sum(not row["kept"] for row in stats)
    kept_rows = [row for row in stats if row["kept"]]
    kept_notes = sum(row["notes"] for row in kept_rows)
    kept_skipped = sum(row["skipped"] for row in kept_rows)
    clips_with_skip = sum(row["skipped"] > 0 for row in kept_rows)

    connection = sqlite3.connect(str(args.cache / "index.sqlite"))
    records = connection.execute(
        "SELECT sample, split, target FROM records WHERE split='train' AND sample NOT LIKE '%#clean'"
    ).fetchall()
    random.Random(args.seed).shuffle(records)
    records = records[:args.limit]
    support: collections.Counter[str] = collections.Counter()
    dropped: collections.Counter[str] = collections.Counter()
    for sample, _split, blob in records:
        targets = json.loads(zlib.decompress(blob))
        kept = {int(row["gold_index"]) for row in targets}
        rendered = json.loads((args.root / sample / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
        for index in range(len(rendered)):
            for tag in _tags(rendered, index):
                support[tag] += 1
                if index not in kept:
                    dropped[tag] += 1
    print(json.dumps({
        "cache": str(args.cache),
        "all_clips": len(stats),
        "excluded_clips_skip_fraction_gt_limit": excluded,
        "all_notes_skip_rate": round(skipped / max(notes, 1), 4),
        "kept_clips_note_skip_rate": round(kept_skipped / max(kept_notes, 1), 4),
        "kept_clips_with_any_skip": round(clips_with_skip / max(len(kept_rows), 1), 4),
        "sampled_train_clips": len(records),
        "dropped_from_targets_by_category": {
            tag: {"support": support[tag], "dropped": dropped[tag],
                  "rate": round(dropped[tag] / max(support[tag], 1), 4)}
            for tag in sorted(support)
        },
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
