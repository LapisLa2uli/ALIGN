"""Run the two seed-42 datasets with disk monitoring and a final inventory.

Run with the Python environment that provides synthpipeline and tinysoundfont.
The underlying generation commands retain the requested configurations,
counts, eight workers, SoundFont, seed, and skip-existing behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import soundfile as sf
import numpy as np

from datacreate.alignment_storage import alignment_storage_kind
from synthpipeline.pipeline import sample_is_complete


REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "synth-pipeline"
DATASETS = (
    ("procedural", "config/multi_error_10k.yaml", "output_10k_multi", 10000),
    ("raw_snippets", "config/rawdata_snippets_2k.yaml", "output_2k_rawdata", 2000),
)


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def inventory(root: Path, expected: int, *, detailed: bool = False) -> dict:
    complete = []
    partial = []
    ids = []
    for sample in sorted(root.glob("synth_*")):
        if not sample.is_dir():
            continue
        if sample_is_complete(sample):
            complete.append(sample)
            ids.append(int(sample.name.rsplit("_", 1)[1]))
        else:
            partial.append(sample.name)
    expected_ids = set(range(42, 42 + expected))
    result = {
        "root": str(root), "expected": expected, "complete": len(complete),
        "partial": partial, "missing_ids": sorted(expected_ids - set(ids)),
        "unexpected_ids": sorted(set(ids) - expected_ids),
        "duplicate_ids": [key for key, n in Counter(ids).items() if n > 1],
    }
    if not detailed:
        return result
    size = 0
    allocated = 0
    seconds = 0.0
    source_counts = Counter()
    error_counts = Counter()
    repeated = 0
    storage_counts = Counter()
    for sample in complete:
        metadata = json.loads((sample / "metadata.json").read_text())
        mapping = json.loads((sample / "note_map.json").read_text())
        labels = json.loads((sample / "labels.json").read_text())
        if metadata["schema_version"] != "1.2" or labels["schema_version"] != "1.2":
            raise ValueError(f"Unexpected annotation schema: {sample}")
        if not mapping.get("rendered_notes"):
            raise ValueError(f"Missing rendered note lineage: {sample}")
        if metadata.get("audio_render") != "soundfont_v1":
            raise ValueError(f"Unexpected renderer: {sample}")
        with np.load(sample / "alignment.npz", allow_pickle=False) as alignment:
            storage_counts[alignment_storage_kind(alignment)] += 1
        for name in ("performance", "reference"):
            info = sf.info(sample / f"{name}_audio.wav")
            if info.samplerate != 22050 or info.channels != 1 or info.frames <= 0:
                raise ValueError(f"Invalid audio format: {sample}/{name}")
            if name == "performance":
                seconds += info.duration
        source_counts[metadata["source"]] += 1
        error_counts.update(metadata.get("error_types", []))
        repeated += bool(metadata.get("repeated"))
        for path in sample.rglob("*"):
            if path.is_file():
                stat = path.stat()
                size += stat.st_size
                allocated += stat.st_blocks * 512
    result.update(
        bytes=size, allocated_bytes=allocated, GB=size / 1e9,
        GiB=size / 1024**3, performance_hours=seconds / 3600,
        sources=dict(source_counts), planted_error_counts=dict(error_counts),
        repeated_samples=repeated,
        alignment_storage=dict(storage_counts),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=15.0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "generation.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    env = dict(os.environ)
    env.update({key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"
    )})
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    config_hashes = {
        name: hashlib.sha256((SYNTH / config).read_bytes()).hexdigest()
        for name, config, _, _ in DATASETS
    }
    manifest_path = run / "generation_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["config_sha256"] != config_hashes:
            raise ValueError("Configurations changed since this generation run")
    else:
        write_json(manifest_path, {
            "started_at": time.time(), "python": sys.executable,
            "seed": 42, "workers": 8, "soundfont": "freepats",
            "config_sha256": config_hashes,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "datasets": [dict(name=n, config=c, output=o, count=k) for n,c,o,k in DATASETS],
        })
        for name, config, _, _ in DATASETS:
            shutil.copy2(SYNTH / config, run / f"{name}.yaml")
        (run / "implementation.patch").write_bytes(subprocess.check_output(
            ["git", "diff", "--", "synth-pipeline", "DataCreate"], cwd=REPO
        ))
    status = {"state": "starting", "supervisor_pid": os.getpid(), "datasets": {}}
    child = None
    try:
        for name, config, output, count in DATASETS:
            root = SYNTH / output
            root.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "synthpipeline.cli", "--config", config,
                       "generate", "--count", str(count), "--workers", "8",
                       "--soundfont", "freepats", "--seed", "42",
                       "--output", f"./{output}", "--skip-existing"]
            print(f"Starting {name}: {' '.join(command)}", flush=True)
            with (run / f"{name}.log").open("a") as log:
                child = subprocess.Popen(command, cwd=SYNTH, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                paused = False
                while child.poll() is None:
                    free = shutil.disk_usage(root).free / 1024**3
                    if free < args.minimum_free_gib and not paused:
                        os.killpg(child.pid, signal.SIGSTOP)
                        paused = True
                    elif free >= args.minimum_free_gib + 5 and paused:
                        os.killpg(child.pid, signal.SIGCONT)
                        paused = False
                    row = inventory(root, count)
                    status.update(state="paused_low_disk" if paused else "generating",
                                  active_dataset=name, child_pid=child.pid,
                                  available_GiB=free, updated_at=time.time())
                    status["datasets"][name] = row
                    write_json(run / "status.json", status)
                    print(f"{name}: {row['complete']}/{count}, free={free:.1f} GiB, "
                          f"state={status['state']}", flush=True)
                    time.sleep(args.poll_seconds)
                if child.returncode:
                    raise RuntimeError(f"{name} generation exited with code {child.returncode}")
            status["datasets"][name] = inventory(root, count, detailed=True)
            row = status["datasets"][name]
            if row["missing_ids"] or row["unexpected_ids"] or row["duplicate_ids"] or row["partial"]:
                raise RuntimeError(f"Incomplete {name} dataset; inspect status and generation log")
            child = None
        status.update(state="complete", active_dataset=None, child_pid=None,
                      updated_at=time.time(), available_GiB=shutil.disk_usage(REPO).free / 1024**3)
        write_json(run / "status.json", status)
        write_json(run / "dataset_summary.json", status["datasets"])
        print("Both datasets are complete and inventoried.", flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status.update(state="failed", error=str(exc), updated_at=time.time())
        write_json(run / "status.json", status)
        raise


if __name__ == "__main__":
    main()
