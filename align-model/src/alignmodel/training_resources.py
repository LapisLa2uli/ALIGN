"""Atomic advisory resource ownership for repository training jobs.

The status document is intentionally human-readable, but updates are guarded
by a short-lived sidecar mutex and published with ``os.replace``.  Long-lived
ownership is represented by PID leases; dead leases are removed on every
update so an interrupted trainer cannot permanently reserve the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import psutil


STATUS_SCHEMA_VERSION = "align-training-resource-status-v1"
DEFAULT_STATUS_PATH = Path("runs/TRAINING_RESOURCE_STATUS.json")


class ResourceBusyError(RuntimeError):
    """Raised when a live process already owns an exclusive resource."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return False


@contextmanager
def _status_mutex(path: Path, timeout_seconds: float = 15.0) -> Iterator[None]:
    """Serialize read-modify-write cycles without locking the JSON itself."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "created_utc": _utc(),
                        "host": socket.gethostname(),
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            break
        except FileExistsError:
            stale = False
            try:
                owner = json.loads(lock_path.read_text(encoding="utf-8"))
                age = time.time() - lock_path.stat().st_mtime
                stale = age > 30.0 and not _pid_alive(int(owner.get("pid", -1)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                stale = time.time() - lock_path.stat().st_mtime > 30.0
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring resource status: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "revision": 0,
            "leases": {},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != STATUS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported training resource status: {path}")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _clean_leases(status: dict[str, Any]) -> None:
    leases = status.setdefault("leases", {})
    for resource, lease in list(leases.items()):
        if not isinstance(lease, Mapping) or not _pid_alive(
            int(lease.get("pid", -1))
        ):
            leases.pop(resource, None)


def claim_resource(
    path: Path,
    resource: str,
    *,
    track: str,
    command: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Claim one exclusive resource and return its unforgeable lease id."""

    path = Path(path)
    with _status_mutex(path):
        status = _read_status(path)
        _clean_leases(status)
        existing = status["leases"].get(resource)
        if existing is not None:
            raise ResourceBusyError(
                f"{resource} is owned by PID {existing['pid']} "
                f"({existing.get('track', 'unknown')})"
            )
        lease_id = str(uuid.uuid4())
        status["leases"][resource] = {
            "lease_id": lease_id,
            "pid": os.getpid(),
            "track": str(track),
            "command": list(command or ()),
            "claimed_utc": _utc(),
            "heartbeat_utc": _utc(),
            "metadata": dict(metadata or {}),
        }
        status["schema_version"] = STATUS_SCHEMA_VERSION
        status["revision"] = int(status.get("revision", 0)) + 1
        status["updated_utc"] = _utc()
        _atomic_json(path, status)
    return lease_id


def release_resource(path: Path, resource: str, lease_id: str) -> None:
    path = Path(path)
    with _status_mutex(path):
        status = _read_status(path)
        _clean_leases(status)
        lease = status["leases"].get(resource)
        if lease is not None and lease.get("lease_id") == lease_id:
            status["leases"].pop(resource, None)
            status["revision"] = int(status.get("revision", 0)) + 1
            status["updated_utc"] = _utc()
            _atomic_json(path, status)


@contextmanager
def resource_lease(
    path: Path,
    resource: str,
    *,
    track: str,
    command: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[str]:
    lease_id = claim_resource(
        path,
        resource,
        track=track,
        command=command,
        metadata=metadata,
    )
    try:
        yield lease_id
    finally:
        release_resource(path, resource, lease_id)


def _safe_process_value(function: Any, default: Any = None) -> Any:
    try:
        return function()
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return default


def _python_processes(repo_root: Path) -> list[psutil.Process]:
    root = os.path.normcase(str(repo_root.resolve()))
    result = []
    for process in psutil.process_iter(("pid", "name", "cmdline", "cwd")):
        try:
            if "python" not in str(process.info.get("name") or "").casefold():
                continue
            command = " ".join(process.info.get("cmdline") or ())
            cwd = str(process.info.get("cwd") or "")
            if root in os.path.normcase(command) or os.path.normcase(cwd).startswith(
                root
            ):
                result.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return result


def _job_role(command: str) -> tuple[str, str]:
    lowered = command.casefold()
    if "train_outputraw_full_pipeline.py" in lowered:
        return "outputraw_full_training", "training"
    if "train_candidate_rescorer.py" in lowered:
        return "candidate_rescorer", "preprocessing_or_training"
    if "prepare_joint_outputraw_data.py" in lowered and "--phase repack" in lowered:
        return "outputraw_packed_relayout", "repacking"
    if "prepare_joint_outputraw_data.py" in lowered and "--phase benchmark" in lowered:
        return "outputraw_loader_benchmark", "benchmarking"
    if "prepare_joint_outputraw_data.py" in lowered:
        return "outputraw_data_preparation", "preprocessing"
    if "train_" in lowered or "torchrun" in lowered:
        return "other_training", "training"
    return "python_support", "support"


def _nvidia_snapshot() -> dict[str, Any]:
    def numeric(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    result: dict[str, Any] = {"available": False, "compute_processes": []}
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        return result
    devices = []
    for line in gpu.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 8:
            devices.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "gpu_percent": numeric(values[2]),
                    "memory_percent": numeric(values[3]),
                    "vram_used_mb": numeric(values[4]),
                    "vram_total_mb": numeric(values[5]),
                    "power_watts": numeric(values[6]),
                    "temperature_c": numeric(values[7]),
                }
            )
    compute = []
    for line in processes.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 3:
            compute.append(
                {
                    "pid": int(values[0]),
                    "name": values[1],
                    "vram_mb": numeric(values[2]),
                }
            )
    result.update(
        {
            "available": bool(devices),
            "devices": devices,
            "compute_processes": compute,
        }
    )
    return result


def collect_resource_snapshot(
    repo_root: Path,
    *,
    sample_seconds: float = 1.0,
) -> dict[str, Any]:
    """Collect rate-based process and machine counters over one interval."""

    processes = _python_processes(repo_root)
    initial_io: dict[int, Any] = {}
    initial_faults: dict[int, int] = {}
    for process in processes:
        process.cpu_percent(None)
        initial_io[process.pid] = _safe_process_value(process.io_counters)
        memory = _safe_process_value(process.memory_info)
        initial_faults[process.pid] = int(
            getattr(memory, "num_page_faults", 0) if memory is not None else 0
        )
    disk_before = psutil.disk_io_counters()
    machine_cpu = psutil.cpu_percent(interval=max(0.1, sample_seconds))
    elapsed = max(sample_seconds, 0.1)
    disk_after = psutil.disk_io_counters()
    rows = []
    for process in processes:
        command_parts = _safe_process_value(process.cmdline, []) or []
        command = " ".join(command_parts)
        role, phase = _job_role(command)
        memory = _safe_process_value(process.memory_info)
        io_after = _safe_process_value(process.io_counters)
        io_before = initial_io.get(process.pid)
        affinity = _safe_process_value(process.cpu_affinity, []) or []
        faults_after = int(
            getattr(memory, "num_page_faults", 0) if memory is not None else 0
        )
        rows.append(
            {
                "pid": process.pid,
                "ppid": _safe_process_value(process.ppid),
                "role": role,
                "phase": phase,
                "command": command,
                "cpu_percent": _safe_process_value(process.cpu_percent, 0.0),
                "affinity": affinity,
                "threads": _safe_process_value(process.num_threads),
                "rss_mb": (
                    round(memory.rss / 1024**2, 1) if memory is not None else None
                ),
                "private_mb": (
                    round(getattr(memory, "private", 0) / 1024**2, 1)
                    if memory is not None
                    else None
                ),
                "page_faults_per_sec": max(
                    0.0,
                    (faults_after - initial_faults.get(process.pid, faults_after))
                    / elapsed,
                ),
                "read_mib_per_sec": (
                    max(0, io_after.read_bytes - io_before.read_bytes)
                    / 1024**2
                    / elapsed
                    if io_after is not None and io_before is not None
                    else None
                ),
                "write_mib_per_sec": (
                    max(0, io_after.write_bytes - io_before.write_bytes)
                    / 1024**2
                    / elapsed
                    if io_after is not None and io_before is not None
                    else None
                ),
                "status": _safe_process_value(process.status),
                "create_time": _safe_process_value(process.create_time),
            }
        )
    memory = psutil.virtual_memory()
    disk = {
        "read_mib_per_sec": None,
        "write_mib_per_sec": None,
        "read_ops_per_sec": None,
        "write_ops_per_sec": None,
        "busy_percent": None,
    }
    if disk_before is not None and disk_after is not None:
        busy_before = getattr(disk_before, "busy_time", None)
        busy_after = getattr(disk_after, "busy_time", None)
        disk = {
            "read_mib_per_sec": max(
                0, disk_after.read_bytes - disk_before.read_bytes
            )
            / 1024**2
            / elapsed,
            "write_mib_per_sec": max(
                0, disk_after.write_bytes - disk_before.write_bytes
            )
            / 1024**2
            / elapsed,
            "read_ops_per_sec": max(
                0, disk_after.read_count - disk_before.read_count
            )
            / elapsed,
            "write_ops_per_sec": max(
                0, disk_after.write_count - disk_before.write_count
            )
            / elapsed,
            "busy_percent": (
                min(
                    100.0,
                    max(0, busy_after - busy_before) / (elapsed * 10.0),
                )
                if busy_before is not None and busy_after is not None
                else None
            ),
        }
    return {
        "sample_seconds": elapsed,
        "machine": {
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "cpu_percent": machine_cpu,
            "ram_total_mb": round(memory.total / 1024**2, 1),
            "ram_available_mb": round(memory.available / 1024**2, 1),
            "ram_percent": memory.percent,
            "pagefile_percent": psutil.swap_memory().percent,
            "disk": disk,
            "gpu": _nvidia_snapshot(),
        },
        "python_processes": sorted(rows, key=lambda row: row["pid"]),
    }


def refresh_resource_status(
    path: Path,
    repo_root: Path,
    *,
    sample_seconds: float = 1.0,
    queue: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    snapshot = collect_resource_snapshot(repo_root, sample_seconds=sample_seconds)
    processes = snapshot["python_processes"]
    physical_gpu_pids = {
        int(row["pid"])
        for row in snapshot["machine"]["gpu"].get("compute_processes", [])
    }
    gpu_utilization = max(
        (
            float(device.get("gpu_percent") or 0.0)
            for device in snapshot["machine"]["gpu"].get("devices", [])
        ),
        default=0.0,
    )
    cuda_intent = [
        row
        for row in processes
        if row["role"] != "python_support"
        and (
            "--device cuda" in row["command"].casefold()
            or "--local-device cuda" in row["command"].casefold()
            or "--path-device cuda" in row["command"].casefold()
        )
    ]
    # The Windows venv launcher duplicates the command. Prefer its child.
    reserved = []
    for row in cuda_intent:
        child_duplicate = any(
            int(other.get("ppid") or -1) == int(row["pid"])
            and other["role"] == row["role"]
            for other in cuda_intent
        )
        if child_duplicate:
            continue
        reserved.append(row)
    heavy_candidates = [
        row
        for row in processes
        if row["role"]
        in {
            "candidate_rescorer",
            "outputraw_packed_relayout",
            "outputraw_loader_benchmark",
            "outputraw_data_preparation",
            "outputraw_full_training",
            "other_training",
        }
    ]
    heavy = [
        row
        for row in heavy_candidates
        if not any(
            int(other.get("ppid") or -1) == int(row["pid"])
            and other["role"] == row["role"]
            for other in heavy_candidates
        )
    ]
    with _status_mutex(Path(path)):
        status = _read_status(Path(path))
        _clean_leases(status)
        status.update(
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "revision": int(status.get("revision", 0)) + 1,
                "updated_utc": _utc(),
                "host": socket.gethostname(),
                "writer_pid": os.getpid(),
                "repository": str(Path(repo_root).resolve()),
                "publication": {
                    "atomic_replace": True,
                    "mutex": str(
                        Path(path).with_suffix(Path(path).suffix + ".lock")
                    ),
                    "dead_pid_leases_reaped": True,
                },
                "policy": {
                    "gpu_exclusive": True,
                    "max_simultaneous_gpu_trainers": 1,
                    "max_heavy_cpu_jobs_during_gpu_training": 1,
                    "recommended_loader_workers": 4,
                    "recommended_prefetch": 16,
                    "threads_per_loader_worker": 1,
                    "physical_cpu_reserve_for_os": 2,
                },
                "ownership": {
                    "gpu_physical_pids": sorted(physical_gpu_pids),
                    "gpu_reserved": [
                        {
                            "pid": row["pid"],
                            "role": row["role"],
                            "phase": (
                                "suspended_queued_gpu_context"
                                if row.get("status") == psutil.STATUS_STOPPED
                                else
                                "gpu_active"
                                if int(row["pid"]) in physical_gpu_pids
                                and gpu_utilization >= 5.0
                                else "gpu_context_idle_cpu_phase"
                                if int(row["pid"]) in physical_gpu_pids
                                else "gpu_reserved_preprocessing"
                            ),
                            "command": row["command"],
                        }
                        for row in reserved
                    ],
                    "heavy_cpu_pids": sorted(
                        {int(row["pid"]) for row in heavy}
                    ),
                    "waiting_or_stalled_pids": sorted(
                        int(row["pid"])
                        for row in heavy
                        if float(row.get("cpu_percent") or 0.0) < 1.0
                        and int(row["pid"]) not in physical_gpu_pids
                    ),
                    "paused_pids": sorted(
                        int(row["pid"])
                        for row in processes
                        if row.get("status") == psutil.STATUS_STOPPED
                    ),
                },
                "recommended_concurrency": {
                    "current_heavy_jobs": len(heavy),
                    "oversubscribed": len(heavy) > 2,
                    "action": (
                        "Run one GPU trainer plus at most one bounded CPU loader; "
                        "queue repacks, cache builds, and benchmarks."
                    ),
                },
                "queue": [dict(row) for row in queue],
                "snapshot": snapshot,
            }
        )
        _atomic_json(Path(path), status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--queue", action="append", default=[])
    args = parser.parse_args(argv)
    queue = [
        {"track": value, "state": "queued"}
        for value in args.queue
    ]
    status = refresh_resource_status(
        args.status,
        args.repo_root,
        sample_seconds=max(0.1, args.sample_seconds),
        queue=queue,
    )
    print(json.dumps(status["ownership"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
