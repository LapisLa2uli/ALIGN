from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = events = 0
    violations = []
    transitions = Counter()
    jumps = []
    widths = Counter()
    allowed_pass = {
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (1, 0),
        (2, 0),
    }
    with gzip.open(args.supervision, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            document = json.loads(line)
            rows += 1
            rendered = document["events"]
            if [event["rendered_event_id"] for event in rendered] != list(
                range(len(rendered))
            ):
                violations.append(
                    {"sample": document["sample"], "reason": "rendered IDs"}
                )
            if any(
                current["start_sec"] < previous["start_sec"]
                for previous, current in zip(rendered, rendered[1:])
            ):
                violations.append(
                    {"sample": document["sample"], "reason": "temporal order"}
                )
            previous = None
            for event in rendered:
                events += 1
                span = event["score_span"]
                copy_pass = int(event["copy_pass"])
                if (event["relationship"] == "copy") != (copy_pass > 0):
                    violations.append(
                        {
                            "sample": document["sample"],
                            "reason": "copy relationship/pass mismatch",
                        }
                    )
                if span is None:
                    continue
                widths[int(span[1]) - int(span[0])] += 1
                current = (int(span[1]) - 1, copy_pass)
                if previous is not None:
                    transitions[(previous[1], current[1])] += 1
                    if (previous[1], current[1]) not in allowed_pass:
                        violations.append(
                            {
                                "sample": document["sample"],
                                "reason": f"pass {previous[1]}->{current[1]}",
                            }
                        )
                    if previous[1] == current[1]:
                        jumps.append(current[0] - previous[0])
                previous = current
    report = {
        "schema_version": "align-monotonic-supervision-v3-audit",
        "rows": rows,
        "events": events,
        "violations": violations,
        "valid_rows": rows
        - len({row["sample"] for row in violations}),
        "pass_transitions": {
            f"{left}->{right}": count
            for (left, right), count in transitions.items()
        },
        "minimum_within_pass_jump": min(jumps),
        "maximum_within_pass_jump": max(jumps),
        "span_width_counts": {
            str(width): count for width, count in widths.items()
        },
        "maximum_span_width": max(widths),
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
