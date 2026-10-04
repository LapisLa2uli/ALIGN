"""Break down wrong missed-note calls that survive a candidate gate."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--max-p", type=float, default=0.01)
    parser.add_argument("--max-voiced", type=float, default=0.95)
    parser.add_argument("--max-run", type=int, default=2)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    for dataset, data in report["datasets"].items():
        rows = [row for row in data["features"] if row["kind"] == "missed"
                and row["pitch_probability"] < args.max_p and row["slot_voiced"] < args.max_voiced
                and row["run_length"] <= args.max_run]
        wrong = [row for row in rows if row["credit"] < 1]
        counts = collections.Counter()
        for row in wrong:
            distance = row.get("gold_deletion_distance", 99)
            counts["near_gold_deletion_1" if distance == 1 else "near_gold_deletion_2to3" if distance <= 3 else "no_gold_deletion_nearby"] += 1
            counts["same_pitch_score_neighbor" if row.get("same_pitch_score_neighbor") else "pitch_change"] += 1
        right = [row for row in rows if row["credit"] >= 1]
        print(dataset, "kept", len(rows), "wrong", len(wrong), dict(counts),
              "right same-pitch", sum(1 for r in right if r.get("same_pitch_score_neighbor")))
        for row in wrong[:12]:
            print("  ", {key: (round(value, 3) if isinstance(value, float) else value) for key, value in row.items()
                         if key not in ("kind",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
