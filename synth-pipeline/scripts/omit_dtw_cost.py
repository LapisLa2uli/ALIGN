"""Remove dense DTW diagnostics from existing bundles, preserving other arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from datacreate.alignment_storage import omit_dtw_cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Append-only per-file SHA-256 audit (JSONL)")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"started_at": time.time(), "changed": 0, "already_compact": 0, "bytes_saved": 0}
    with args.out.open("a", encoding="utf-8", buffering=1) as log:
        for root in args.root:
            for index, path in enumerate(sorted(root.glob("synth_*/alignment.npz")), 1):
                row = omit_dtw_cost(path)
                log.write(json.dumps(row) + "\n")
                summary["changed" if row["changed"] else "already_compact"] += 1
                summary["bytes_saved"] += row["bytes_saved"]
                if index % 500 == 0:
                    print(f"{root.name}: {index}, freed {summary['bytes_saved']/1e9:.2f} GB", flush=True)
    summary.update(finished_at=time.time(), free_GiB=shutil.disk_usage(args.root[0]).free / 1024**3)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
