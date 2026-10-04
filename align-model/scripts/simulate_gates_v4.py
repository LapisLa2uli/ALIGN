"""Simulate extra/missed gate thresholds on harness feature dumps (tuning half only)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    for dataset, data in report["datasets"].items():
        per_type = data["per_type"]
        features = data["features"]
        extras = [row for row in features if row["kind"] == "extra"]
        missed = [row for row in features if row["kind"] == "missed"]
        predicted, gold = per_type["extra"]["predicted"], per_type["extra"]["gold"]
        credit_total = per_type["extra"]["precision"] * predicted
        missed_gold = per_type["missed_note"]["gold"]
        print(f"===== {dataset}: extra P={per_type['extra']['precision']} R={per_type['extra']['recall']} "
              f"missed P={per_type['missed_note']['precision']} R={per_type['missed_note']['recall']}")
        for ioi, confidence, same in itertools.product([0.0, 0.06, 0.08, 0.1, 0.12], [0.0, 0.95, 0.99], [False, True]):
            withheld = [row for row in extras if row["ioi"] < ioi or row["confidence"] < confidence
                        or (same and row["same_pitch_neighbor"])]
            kept = predicted - len(withheld)
            credit = credit_total - sum(row["credit"] for row in withheld)
            print(f"  extra withhold ioi<{ioi} conf<{confidence} same={same}: "
                  f"P={credit / max(kept, 1):.4f} R={credit / max(gold, 1):.4f} withheld={len(withheld)}")
        def level(row):
            value = row.get("level_drop_db")
            return value if value is not None and value == value else -99.0

        for edges in ([-99, 0, 2, 4, 6, 8, 10, 15, 99],):
            text = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                rows = [row for row in missed if lo <= level(row) < hi]
                if rows:
                    text.append(f"[{lo},{hi}) n={len(rows)} P={sum(r['credit'] for r in rows) / len(rows):.2f}")
            print("  level_drop_db:", " | ".join(text))
        for probability, voiced, run, drop, onset in itertools.product(
                [1.01, 0.01], [1.01, 0.95], [100, 3], [-99.0, 2.0, 4.0, 6.0, 8.0], [2.0, 0.6]):
            kept = [row for row in missed if row["pitch_probability"] < probability
                    and row["slot_voiced"] < voiced and row["run_length"] < run and level(row) >= drop
                    and not ((row.get("slot_onset") or 0) > onset)]
            credit = sum(row["credit"] for row in kept)
            if kept:
                print(f"  missed keep p<{probability} voiced<{voiced} run<{run} drop>={drop} onset<={onset}: "
                      f"P={credit / len(kept):.4f} R={credit / max(missed_gold, 1):.4f} kept={len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
