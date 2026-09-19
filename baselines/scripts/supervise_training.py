"""Launch both audited baselines and record their processes, logs and disk state."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

BASELINES = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def gpu_free_mib():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    return {int(a): int(b) for a, b in (line.split(",") for line in output.splitlines())}


def log_progress(path):
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = list(re.finditer(r"Epoch\s+(\d+):\s+\d+%\|[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[([^\r\n]*)", tail))
    if not matches:
        return {}
    match = matches[-1]
    result = dict(epoch=int(match[1]), batches_completed=int(match[2]), batches_total=int(match[3]))
    for name in ("train_loss", "val_loss"):
        loss = re.search(rf"(?:^|, )\b{name}=([\d.eE+-]+)", match[4])
        if loss:
            result[name] = float(loss[1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--polytune-gpu", type=int, default=0)
    parser.add_argument("--laddersym-gpu", type=int, default=4)
    parser.add_argument("--minimum-free-gib", type=float, default=30)
    args = parser.parse_args()
    data, run = args.data.resolve(), args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    split = data / "split.json"
    split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    config = json.loads((run / "run_manifest.json").read_text())
    if config["split_sha256"] != split_sha or config["data"] != str(data):
        raise ValueError("Dataset or split changed since the experiment manifest was recorded")
    jobs = {}
    for name, gpu, key in (
        ("polytune", args.polytune_gpu, "polytune"),
        ("laddersym", args.laddersym_gpu, "laddersym_prompted"),
    ):
        settings = config["settings"][key]
        batch, accumulation = settings["batch_size"], settings["grad_accum"]
        minimum = settings["min_free_gpu_MiB"]
        run_name = "synth_20260914_s365"
        output = BASELINES / "runs" / name / run_name
        if output.exists():
            raise FileExistsError(f"Training output exists; resume explicitly instead of overwriting: {output}")
        command = ["bash", str(BASELINES / "scripts" / f"{name}_train.sh"),
                   "--data", str(data), "--split-json", str(split), "--profile", "cuda",
                   "--epochs", "40", "--batch-size", str(batch), "--run-name", run_name]
        if name == "laddersym":
            command.append("--prompted")
        command += ["--", "num_rows_per_batch=1", f"grad_accum={accumulation}",
                    "dataloader.train.num_workers=4", "dataloader.val.num_workers=4",
                    "modelcheckpoint.save_top_k=1", "trainer.log_every_n_steps=10"]
        jobs[name] = dict(state="waiting_for_resources", gpu=gpu, batch_size=batch,
                          gradient_accumulation=accumulation, min_free_gpu_MiB=minimum,
                          command=command, output=str(output), log=str(run / f"{name}_training.log"))
    state = dict(supervisor_pid=os.getpid(), split_sha256=split_sha, state="starting", jobs=jobs)
    children, logs = {}, {}
    try:
        while True:
            free = shutil.disk_usage(data).free / 1024**3
            gpu_memory = gpu_free_mib()
            for name, job in jobs.items():
                child = children.get(name)
                if child is None and job["state"] == "waiting_for_resources":
                    if free < args.minimum_free_gib + 10:
                        continue
                    if gpu_memory[job["gpu"]] < job["min_free_gpu_MiB"]:
                        occupied = {j["gpu"] for j in jobs.values() if j.get("pid") and j["state"] in ("training", "paused_low_disk")}
                        candidates = [gpu for gpu, memory in gpu_memory.items()
                                      if gpu not in occupied and memory >= job["min_free_gpu_MiB"]]
                        if not candidates:
                            continue
                        job["gpu"] = max(candidates, key=gpu_memory.get)
                    gpu_memory[job["gpu"]] -= job["min_free_gpu_MiB"]
                    if hashlib.sha256(split.read_bytes()).hexdigest() != split_sha:
                        raise ValueError("Split changed before training launch")
                    env = dict(os.environ)
                    env.update(CUDA_VISIBLE_DEVICES=str(job["gpu"]), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
                    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
                    logs[name] = Path(job["log"]).open("x")
                    child = subprocess.Popen(job["command"], cwd=BASELINES.parent, env=env,
                                             stdout=logs[name], stderr=subprocess.STDOUT, start_new_session=True)
                    children[name] = child
                    job.update(state="training", pid=child.pid, started_at=time.time())
                if child is None:
                    continue
                code = child.poll()
                if code is not None:
                    if job["state"] not in ("complete", "failed"):
                        job.update(state="complete" if code == 0 else "failed", exit_code=code, finished_at=time.time())
                        logs[name].close()
                elif free < args.minimum_free_gib and job["state"] == "training":
                    os.killpg(child.pid, signal.SIGSTOP)
                    job["state"] = "paused_low_disk"
                elif free >= args.minimum_free_gib + 10 and job["state"] == "paused_low_disk":
                    os.killpg(child.pid, signal.SIGCONT)
                    job["state"] = "training"
                job["progress"] = log_progress(Path(job["log"]))
            state.update(updated_at=time.time(), free_GiB=free,
                         state="finished" if all(j["state"] in ("complete", "failed") for j in jobs.values()) else "active")
            write_json(run / "training_status.json", state)
            print(json.dumps({"free_GiB": round(free, 2), "jobs": {n: {k: j[k] for k in ("state", "gpu", "progress") if k in j} for n, j in jobs.items()}}), flush=True)
            if state["state"] == "finished":
                break
            time.sleep(20)
    except BaseException:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGCONT)
                os.killpg(child.pid, signal.SIGTERM)
        state.update(state="supervisor_failed", updated_at=time.time())
        write_json(run / "training_status.json", state)
        raise


if __name__ == "__main__":
    main()
