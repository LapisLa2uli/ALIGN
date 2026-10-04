"""Prune superseded synthetic datasets on E:, keeping a few bundles and the config.

For each dataset folder, keeps ``--keep-per-source`` bundles of every source
prefix (procedural ``synth_gen`` and each RawData score), copies the generation
config(s) and generation logs into ``_config/``, writes ``PRUNED.json`` with the
removed bundle names, then deletes the other bundle directories.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "output_2k_rawdata": {"versions": ["3.0", "3.1"], "configs": ["rawdata_snippets_2k.yaml"],
                          "logs": []},
    "outputRaw_2": {"versions": ["5.0", "5.1"], "configs": ["multi_error_10k.yaml"],
                    "logs": ["generate_outputRaw_2.log", "generate_outputRaw_2_resume.log",
                             "generate_pause_status.txt", "resume_generate.ps1", "rerender_oscillator.log",
                             "strip_ornaments_rerender.log", "strip_ornaments_rerender_resume.log",
                             "generate_output_2.log", "generate_output_2_fix.log"]},
    "outputRaw_sf_10k": {"versions": ["6.0"], "configs": ["rawdata_sf_10k.yaml"],
                         "logs": ["generate_outputRaw_sf_10k.log", "generate_outputRaw_sf_10k_w16.log",
                                  "transpose_audit_run.log", "transpose_audit_faults.jsonl",
                                  "transpose_audit_summary.json", "transpose_errors_tight_run.log",
                                  "transpose_errors_tight.jsonl", "transpose_errors_tight.txt",
                                  "transpose_errors_tight_summary.json", "transpose_errors_tight_extract.txt"]},
    "outputRaw_orn_10k": {"versions": ["7.0", "7.1"],
                          "configs": ["mixed_orn_10k_random.yaml", "mixed_orn_10k_rawdata.yaml"],
                          "logs": ["generate_outputRaw_orn_10k.log", "generate_outputRaw_orn_10k_status.txt",
                                   "regenerate_orn_musesounds.log", "regenerate_orn_musesounds_w4.log",
                                   "regenerate_orn_musesounds_w10.log", "regenerate_orn_musesounds_w10_resume.log",
                                   "generate_orn_v5_lockbox.log", "generate_orn_v5_lockbox_extend.log",
                                   "generate_orn_v6_lockbox.log"]},
    "outputRaw_fast_1k": {"versions": ["8.0", "8.1"], "configs": ["fast_notes_1k.yaml"],
                          "logs": ["generate_fast_1k.log", "regenerate_fast_1k_musesounds.log"]},
}


def _source(name: str) -> str:
    return name.rsplit("_", 1)[0]


def _force_remove(function, path, _info) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def prune(root: Path, folder: str, spec: dict, keep_per_source: int, dry_run: bool) -> dict:
    base = root / folder
    if not base.is_dir() or base.is_symlink():
        raise SystemExit(f"Refusing: {base} is not a real directory")
    bundles = sorted(entry.name for entry in os.scandir(base)
                     if entry.is_dir() and entry.name.startswith("synth_"))
    by_source: dict[str, list[str]] = collections.defaultdict(list)
    for name in bundles:
        by_source[_source(name)].append(name)
    keep = sorted(name for names in by_source.values() for name in names[:keep_per_source])
    remove = [name for name in bundles if name not in set(keep)]
    config_dir = base / "_config"
    report = {
        "schema_version": "synth-dataset-pruned-v1",
        "pruned_utc": datetime.now(timezone.utc).isoformat(),
        "folder": str(base),
        "registry_versions": spec["versions"],
        "bundles_before": len(bundles),
        "bundles_kept": keep,
        "bundles_removed_count": len(remove),
        "bundles_removed": remove,
        "configs_copied": [],
        "logs_moved": [],
    }
    if dry_run:
        return {key: value for key, value in report.items() if key != "bundles_removed"}
    config_dir.mkdir(exist_ok=True)
    for name in spec["configs"]:
        source = REPO / "config" / name
        if source.is_file():
            shutil.copy2(source, config_dir / name)
            report["configs_copied"].append(name)
    for name in spec["logs"]:
        source = root / name
        if source.is_file():
            shutil.move(str(source), str(config_dir / name))
            report["logs_moved"].append(name)
    (base / "PRUNED.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    started = time.perf_counter()
    done = 0

    def _delete(name: str) -> None:
        shutil.rmtree(base / name, onerror=_force_remove)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(_delete, remove):
            done += 1
            if done % 2000 == 0:
                print(f"{folder}: removed {done}/{len(remove)} ({time.perf_counter() - started:.0f}s)", flush=True)
    print(f"{folder}: removed {len(remove)} bundles, kept {len(keep)}", flush=True)
    return {key: value for key, value in report.items() if key != "bundles_removed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/"))
    parser.add_argument("--keep-per-source", type=int, default=3)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = []
    for folder, spec in DATASETS.items():
        if args.only and folder not in args.only:
            continue
        results.append(prune(args.root, folder, spec, args.keep_per_source, args.dry_run))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
