#!/usr/bin/env python3
"""Package all baseline code; optionally include the existing labelled smoke data.

Extract into a fresh ALIGN clone, then run baselines/scripts/bootstrap.sh.
Full datasets/checkpoints/venvs are deliberately separate rsync transfers.
"""
import argparse
import hashlib
import os
from pathlib import Path
import tarfile

B = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    roots = [B / name for name in ("common", "configs", "docs", "envs", "patches", "scripts", "tests", "splits")]
    roots += [B / "README.md", B / ".gitignore"]
    if args.include_smoke:
        roots += [B / "data" / name for name in ("smoke_bundles", "smoke_align")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    excluded = {"__pycache__", ".pytest_cache", ".git", "polytune", "laddersym"}
    with tarfile.open(args.out, "w:gz") as archive:
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(root)
            files = [root] if root.is_file() else []
            if root.is_dir():
                for directory, dirs, names in os.walk(root):
                    dirs[:] = sorted(d for d in dirs if d not in excluded)
                    files.extend(Path(directory) / name for name in sorted(names)
                                 if not name.endswith((".pyc", ".log")))
            for path in files:
                archive.add(path, arcname=str(Path("baselines") / path.relative_to(B)))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(f"{args.out.resolve()} ({args.out.stat().st_size / 2**20:.1f} MiB)\nsha256: {digest}")


if __name__ == "__main__":
    main()
