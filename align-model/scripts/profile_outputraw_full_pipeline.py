from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from alignmodel.joint.lattice import FEATURE_DIM
from alignmodel.joint.outputraw_full import (
    FullJointPipelineModel,
    FullPipelineModelConfig,
)


def _gpu_processes() -> list[dict[str, Any]]:
    output = []
    ancestors = {
        process.pid
        for process in psutil.Process(os.getpid()).parents()
    }
    for process in psutil.process_iter(("pid", "name", "cmdline", "status")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        lowered = command.lower()
        if process.info.get("status") == psutil.STATUS_STOPPED:
            continue
        if (
            process.pid != os.getpid()
            and process.pid not in ancestors
            and "python" in str(process.info.get("name") or "").lower()
            and (
                "--device cuda" in lowered
                or "--local-device cuda" in lowered
                or "--path-device cuda" in lowered
                or "cuda:" in lowered
            )
        ):
            output.append(
                {
                    "pid": process.pid,
                    "name": process.info.get("name"),
                    "command": command,
                }
            )
    return output


def _nvidia_smi() -> dict[str, Any]:
    def numeric(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,"
                "utilization.memory,power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 6:
            rows.append(
                {
                    "name": values[0],
                    "memory_total_mb": numeric(values[1]),
                    "memory_used_mb": numeric(values[2]),
                    "gpu_utilization_percent": numeric(values[3]),
                    "memory_utilization_percent": numeric(values[4]),
                    "power_limit_watts": numeric(values[5]),
                }
            )
    return {"available": bool(rows), "devices": rows}


def _storage_profile(directory: Path, size_mb: int) -> dict[str, float | int]:
    directory.mkdir(parents=True, exist_ok=True)
    size = int(size_mb) * 1024 * 1024
    block = os.urandom(1024 * 1024)
    with tempfile.NamedTemporaryFile(
        dir=directory, suffix=".storage-profile", delete=False
    ) as handle:
        path = Path(handle.name)
        started = time.perf_counter()
        for _ in range(size_mb):
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
        write_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        read = 0
        with path.open("rb", buffering=4 * 1024 * 1024) as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                read += len(chunk)
        read_seconds = time.perf_counter() - started
    finally:
        path.unlink(missing_ok=True)
    return {
        "bytes": size,
        "write_seconds": write_seconds,
        "write_mib_per_second": size / (1024 * 1024) / write_seconds,
        "read_seconds": read_seconds,
        "read_mib_per_second": read / (1024 * 1024) / read_seconds,
    }


def _optimizer(
    model: torch.nn.Module,
    *,
    device: torch.device,
    fused: bool,
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": 1e-4}
    if fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _benchmark(
    *,
    device: torch.device,
    edges: int,
    iterations: int,
    amp_dtype: torch.dtype | None,
    fused: bool,
    compile_model: bool,
) -> dict[str, Any]:
    torch.manual_seed(365)
    model: torch.nn.Module = FullJointPipelineModel(
        FullPipelineModelConfig(hidden_dim=128, component_dim=64)
    ).to(device)
    compile_status = "disabled"
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compile_status = "enabled"
        except Exception as error:  # pragma: no cover - hardware dependent
            return {"ok": False, "error": f"torch.compile: {error}"}
    try:
        optimizer = _optimizer(model, device=device, fused=fused)
    except (RuntimeError, TypeError) as error:
        return {"ok": False, "error": f"optimizer: {error}"}
    host = torch.randn(edges, FEATURE_DIM, dtype=torch.float32)
    if device.type == "cuda":
        host = host.pin_memory()
        features = host.to(device, non_blocking=True)
    else:
        features = host.to(device)

    context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else torch.autocast(device_type=device.type, enabled=False)
    )
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with context:
                loss = model(features).float().square().mean()
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            with context:
                loss = model(features).float().square().mean()
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    except (RuntimeError, torch._dynamo.exc.TorchDynamoException) as error:
        return {"ok": False, "error": str(error)}
    return {
        "ok": True,
        "edges": edges,
        "iterations": iterations,
        "seconds": elapsed,
        "edges_per_second": edges * iterations / elapsed,
        "amp_dtype": str(amp_dtype).removeprefix("torch.")
        if amp_dtype is not None
        else "float32",
        "fused_optimizer": fused,
        "torch_compile": compile_status,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--storage-size-mb", type=int, default=64)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Diagnostic override; never use for production training.",
    )
    args = parser.parse_args()

    active_gpu_jobs = _gpu_processes()
    if (
        args.device == "cuda"
        and active_gpu_jobs
        and not args.allow_shared_gpu
    ):
        raise RuntimeError(
            "Refusing CUDA profile while another CUDA Python job is active: "
            f"{[row['pid'] for row in active_gpu_jobs]}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device(args.device)
    memory = psutil.virtual_memory()
    report: dict[str, Any] = {
        "schema_version": "align-outputraw-hardware-profile-v1",
        "created_unix": time.time(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "profile_cpu_threads": torch.get_num_threads(),
            "ram_total_bytes": memory.total,
            "ram_available_bytes": memory.available,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_smi": _nvidia_smi(),
            "cuda_job_snapshot_at_profile_time": active_gpu_jobs,
        },
        "storage": _storage_profile(
            args.output.parent, max(8, args.storage_size_mb)
        ),
        "device_profiled": args.device,
        "benchmarks": [],
        "locked_test_touched": False,
    }
    dtypes: list[torch.dtype | None] = [None]
    if device.type == "cuda":
        dtypes.extend([torch.float16, torch.bfloat16])
    elif hasattr(torch, "bfloat16"):
        dtypes.append(torch.bfloat16)
    for edges in (8192, 32768, 65536):
        for dtype in dtypes:
            fused_options = (False, True) if device.type == "cuda" else (False,)
            for fused in fused_options:
                report["benchmarks"].append(
                    _benchmark(
                        device=device,
                        edges=edges,
                        iterations=max(1, args.iterations),
                        amp_dtype=dtype,
                        fused=fused,
                        compile_model=False,
                    )
                )
    successful = [
        row for row in report["benchmarks"] if row.get("ok")
    ]
    if not successful:
        raise RuntimeError("No synthetic training configuration succeeded")
    best = max(successful, key=lambda row: row["edges_per_second"])
    compile_result = _benchmark(
        device=device,
        edges=int(best["edges"]),
        iterations=max(1, args.iterations),
        amp_dtype=(
            None
            if best["amp_dtype"] == "float32"
            else getattr(torch, best["amp_dtype"])
        ),
        fused=bool(best["fused_optimizer"]),
        compile_model=True,
    )
    report["benchmarks"].append(compile_result)
    if (
        compile_result.get("ok")
        and compile_result["edges_per_second"] > best["edges_per_second"]
    ):
        best = compile_result
    report["selected"] = best
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
