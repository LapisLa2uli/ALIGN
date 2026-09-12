"""Read-only audit of MIDI pitch space vs written MusicXML."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from synthpipeline.pitch_convention import audit_bundle_pitch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict] = []
    for root in args.root:
        if not root.exists():
            continue
        bundles = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "metadata.json").exists()
        )
        if args.max_samples:
            bundles = bundles[: args.max_samples]
        for index, bundle in enumerate(bundles, start=1):
            rows.append(audit_bundle_pitch(bundle))
            if index == 1 or index % 250 == 0 or index == len(bundles):
                print(f"{root}: {index}/{len(bundles)}", flush=True)

    summary = {
        "n": len(rows),
        "status": dict(Counter(row["status"] for row in rows)),
        "declared_space": dict(Counter(row["declared_space"] for row in rows)),
        "inferred_space": dict(Counter(str(row["inferred_space"]) for row in rows)),
        "midi_minus_xml": dict(
            Counter(str(row["midi_minus_xml"]) for row in rows)
        ),
        "mismatches": [row for row in rows if row["status"] == "mismatch"][:50],
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
