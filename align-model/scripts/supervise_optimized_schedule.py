from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import psutil

from alignmodel.training_resources import (
    DEFAULT_STATUS_PATH,
    _atomic_json,
    _read_status,
    _status_mutex,
)


FULL_ARGUMENTS = [
    "-u",
    "scripts/train_outputraw_full_pipeline.py",
    "--ready-marker",
    "runs/joint-outputraw-full-v1/DATA_READY.json",
    "--output-dir",
    "runs/joint-outputraw-full-v1/training-v1-optimized",
    "--hardware-profile",
    "runs/joint-outputraw-full-v1/profiles/architecture-cuda-optimized.json",
    "--actual-training-profile",
    "runs/joint-outputraw-full-v1/profiles/actual-training-cuda.json",
    "--initialize-checkpoint",
    "runs/joint-audit-v2/end-to-end-v2/weak-note-continuation-optimized/joint_decoder.pt",
    "--resume-checkpoint",
    "runs/joint-outputraw-full-v1/training-v1-optimized/last_checkpoint.pt",
    "--prepared-local-cache",
    "runs/joint-outputraw-full-v1/prepared-local-v1",
    "--prepared-max-open-shards",
    "64",
    "--device",
    "cuda",
    "--path-device",
    "cpu",
    "--local-checkpoint-every",
    "250",
    "--path-checkpoint-every",
    "25",
    "--checkpoint-every",
    "250",
    "--gradient-accumulation",
    "1",
    "--acoustic-epochs",
    "1",
    "--structure-epochs",
    "1",
    "--errors-epochs",
    "1",
    "--joint-epochs",
    "3",
    "--path-epochs",
    "1",
    "--path-samples-per-epoch",
    "1000",
    "--bootstrap-replicates",
    "1000",
]

CANDIDATE_ARGUMENTS = [
    "-u",
    "scripts/train_candidate_rescorer.py",
    "--manifest",
    "data-audit/2026-09-14-v2/split.json",
    "--basic-cache-root",
    "runs/joint-audit-v2/basic-pitch-cache",
    "--example-cache-path",
    "runs/joint-audit-v2/components/candidate-rescorer-v1/examples-floor-050.sqlite",
    "--output-dir",
    "runs/joint-audit-v2/components/candidate-rescorer-v1",
    "--epochs",
    "3",
    "--batch-candidates",
    "65536",
    "--learning-rate",
    "0.0003",
    "--hidden-dim",
    "64",
    "--dropout",
    "0.05",
    "--candidate-floor",
    "0.50",
    "--global-candidate-gate",
    "0.65",
    "--hard-negative-ratio",
    "1.0",
    "--short-weight-lt-80ms",
    "4",
    "--short-weight-lt-120ms",
    "3",
    "--short-weight-lt-180ms",
    "2",
    "--workers",
    "1",
    "--prefetch",
    "1",
    "--device",
    "cuda",
    "--seed",
    "365",
    "--checkpoint-every-clips",
    "250",
    "--packed-training-root",
    "runs/joint-audit-v2/components/candidate-rescorer-v1/packed-training-v1",
]


def _alive(pid: int | None) -> bool:
    return pid is not None and psutil.pid_exists(pid)


def _publish(
    path: Path,
    *,
    full_pid: int | None,
    candidate_pid: int | None,
    queue_state: str,
    restarts: int,
) -> None:
    updated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _status_mutex(path):
        status = _read_status(path)
        status["queue"] = [
            {
                "track": "candidate-rescorer-optimized",
                "state": queue_state,
                "blocked_by": (
                    "outputraw-full-training"
                    if queue_state == "queued"
                    else None
                ),
            }
        ]
        status["schedule"] = {
            "supervisor_pid": os.getpid(),
            "full_pipeline_pid": full_pid,
            "candidate_rescorer_pid": candidate_pid,
            "full_pipeline_restarts": restarts,
            "policy": "full-pipeline-then-candidate-rescorer",
            "updated_utc": updated_utc,
        }
        status["revision"] = int(status.get("revision", 0)) + 1
        status["updated_utc"] = updated_utc
        _atomic_json(path, status)


def _spawn(
    python: Path,
    arguments: Sequence[str],
    *,
    root: Path,
    log_name: str,
) -> subprocess.Popen[Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    environment["OMP_NUM_THREADS"] = "1"
    log_root = root / "runs" / "schedule-supervisor"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = (log_root / f"{log_name}.stdout.log").open("a", encoding="utf-8")
    stderr = (log_root / f"{log_name}.stderr.log").open("a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        [str(python), *arguments],
        cwd=root,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        creationflags=flags,
    )
    return process


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path)
    parser.add_argument("--resource-status", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--full-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    python = (
        args.python.resolve()
        if args.python is not None
        else (root / ".venv-amt-bench" / "Scripts" / "python.exe")
    )
    status_path = (
        args.resource_status
        if args.resource_status.is_absolute()
        else root / args.resource_status
    )
    full_output = root / "runs/joint-outputraw-full-v1/training-v1-optimized"
    candidate_output = (
        root / "runs/joint-audit-v2/components/candidate-rescorer-v1"
    )
    full_pid = args.full_pid
    candidate_pid = None
    restarts = 0
    _publish(
        status_path,
        full_pid=full_pid,
        candidate_pid=None,
        queue_state="queued",
        restarts=restarts,
    )
    while not (full_output / "report.json").is_file():
        if _alive(full_pid):
            time.sleep(args.poll_seconds)
            continue
        if (full_output / "PAUSE_REQUESTED.json").is_file():
            time.sleep(args.poll_seconds)
            continue
        if restarts >= args.max_restarts:
            raise RuntimeError("OutputRaw restart limit reached")
        process = _spawn(
            python,
            FULL_ARGUMENTS,
            root=root,
            log_name=f"outputraw-restart-{restarts + 1}",
        )
        full_pid = process.pid
        restarts += 1
        try:
            psutil.Process(full_pid).nice(psutil.HIGH_PRIORITY_CLASS)
            psutil.Process(full_pid).cpu_affinity([16, 17, 18, 19])
        except (psutil.Error, ValueError):
            pass
        _publish(
            status_path,
            full_pid=full_pid,
            candidate_pid=None,
            queue_state="queued",
            restarts=restarts,
        )
        time.sleep(args.poll_seconds)

    resume = candidate_output / "mid_epoch_checkpoint.pt"
    if not resume.is_file():
        resume = candidate_output / "pause_checkpoint.pt"
    candidate_arguments = [
        *CANDIDATE_ARGUMENTS,
        "--resume-checkpoint",
        str(resume),
    ]
    while not (candidate_output / "report.json").is_file():
        if _alive(candidate_pid):
            time.sleep(args.poll_seconds)
            continue
        process = _spawn(
            python,
            candidate_arguments,
            root=root,
            log_name="candidate-rescorer",
        )
        candidate_pid = process.pid
        _publish(
            status_path,
            full_pid=None,
            candidate_pid=candidate_pid,
            queue_state="running",
            restarts=restarts,
        )
        time.sleep(args.poll_seconds)
        if (candidate_output / "mid_epoch_checkpoint.pt").is_file():
            resume = candidate_output / "mid_epoch_checkpoint.pt"
            candidate_arguments = [
                *CANDIDATE_ARGUMENTS,
                "--resume-checkpoint",
                str(resume),
            ]
    _publish(
        status_path,
        full_pid=None,
        candidate_pid=None,
        queue_state="completed",
        restarts=restarts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
