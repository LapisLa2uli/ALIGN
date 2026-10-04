"""Split, lineage-repair audit and aligner freeze for dataset family 11 (DataCreate-like pilot).

Split (fixed seed, decided before any model reads the audio):
* every clip of the unseen development score (intro_theme) -> val;
* procedural and training-score RawData clips -> 80% train / 10% val / 10%
  test, one group per clip.
The 9.2 val/test scores are excluded at generation time. Run after
repair_realistic92_lineage.py has written lineage for every clip; this script
then writes ALIGNER_FREEZE.json with test lineage hashes.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from alignmodel.joint.packed_data import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("split", "freeze"), required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "split.json"
    if args.stage == "split":
        if split_path.exists():
            raise FileExistsError(split_path)
        names = sorted(path.name for path in args.root.iterdir()
                       if path.is_dir() and (path / "performance_audio.wav").is_file()
                       and (path / "note_map.json").is_file())
        dev = [name for name in names if name.startswith("synth_intro_theme_")]
        rest = [name for name in names if name not in set(dev)]
        random.Random(args.seed).shuffle(rest)
        n_test = len(rest) // 10
        n_val = len(rest) // 10
        splits = {"train": sorted(rest[n_test + n_val:]), "val": sorted(rest[n_test:n_test + n_val] + dev),
                  "test": sorted(rest[:n_test])}
        split_path.write_text(json.dumps({
            "schema_version": "align-realistic92-transcriber-split-v1",
            "dataset_version": "11.2",
            "root": str(args.root),
            "seed": args.seed,
            "grouping": "one group per clip; every clip of the unseen dev score intro_theme is val",
            "score_groups": {"train": [], "val": ["intro_theme"], "test": []},
            "counts": {key: len(value) for key, value in splits.items()},
            "splits": splits,
        }, indent=1) + "\n", encoding="utf-8")
        print(split_path, sha256_file(split_path), {key: len(value) for key, value in splits.items()})
        return 0
    split = json.loads(split_path.read_text(encoding="utf-8"))
    repair = args.output_dir / "repair-v1"
    audit_path = repair / "REPAIR_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    lineage_dir = repair / "lineage"
    eligible = {key: [name for name in names if (lineage_dir / f"{name}.json").is_file()]
                for key, names in split["splits"].items()}
    freeze = {
        "schema_version": "align-dclike11-aligner-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "11.2",
        "root": str(args.root),
        "split_sha256": sha256_file(split_path),
        "repair_audit_sha256": sha256_file(audit_path),
        "lineage_dir": str(lineage_dir.resolve()),
        "counts": {key: len(value) for key, value in eligible.items()},
        "eligible": eligible,
        "test_lineage_sha256": {name: sha256_file(lineage_dir / f"{name}.json") for name in eligible["test"]},
    }
    path = args.output_dir / "ALIGNER_FREEZE.json"
    path.write_text(json.dumps(freeze, indent=1) + "\n", encoding="utf-8")
    print(path, sha256_file(path), freeze["counts"], {k: v for k, v in audit.items() if isinstance(v, (int, float, str))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
