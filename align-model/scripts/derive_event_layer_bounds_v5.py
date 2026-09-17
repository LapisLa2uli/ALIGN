from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    widths = Counter()
    jumps = []
    backward = []
    forward = []
    pass_transitions = Counter()
    rows = events = 0
    with gzip.open(args.supervision, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            document = json.loads(line)
            rows += 1
            previous = None
            for event in document["events"]:
                events += 1
                location = event["canonical_location"]
                if location["kind"] != "score_span":
                    continue
                span = location["score_span"]
                widths[int(span[1]) - int(span[0])] += 1
                current = (int(span[1]) - 1, int(event["copy_pass"]))
                if previous is not None:
                    pass_transitions[(previous[1], current[1])] += 1
                    if previous[1] == current[1]:
                        delta = current[0] - previous[0]
                        jumps.append(delta)
                        (backward if delta < 0 else forward).append(delta)
                previous = current
    jump_array = np.asarray(jumps, dtype=np.int64)
    report = {
        "schema_version": "align-event-layer-bounds-v5",
        "source_supervision": str(args.supervision.resolve()),
        "source_sha256": _sha256(args.supervision),
        "train_rows": rows,
        "rendered_events": events,
        "allowed_span_widths": sorted(widths),
        "span_width_counts": {str(key): value for key, value in widths.items()},
        "minimum_within_pass_jump": min(jumps),
        "maximum_within_pass_jump": max(jumps),
        "jump_quantiles": {
            str(value): float(np.quantile(jump_array, value))
            for value in (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)
        },
        "backward_transitions": len(backward),
        "forward_transitions": len(forward),
        "pass_transitions": {
            f"{left}->{right}": count
            for (left, right), count in pass_transitions.items()
        },
        "frozen_from_train_only": True,
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
