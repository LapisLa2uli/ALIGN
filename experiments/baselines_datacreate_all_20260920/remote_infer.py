"""Copy converted DataCreate audio to the GPU host and run frozen baseline inference."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RETRAIN = ROOT / "experiments" / "baselines_retrain_20260918"
DATA = ROOT / "baselines" / "data" / "datacreate_all_20260920"
REMOTE_ROOTS = (
    "/home/weixi/projects/ALIGN",
    "/root/ALIGN",
    "/srv/ALIGN",
)
SSH_OPTS = (
    "-4",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "StrictHostKeyChecking=accept-new",
)


def ssh(host: str, command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, host, command],
        check=check,
        capture_output=True,
        text=True,
    )


def wait_for_host(host: str, attempts: int, delay: int) -> None:
    for attempt in range(1, attempts + 1):
        result = ssh(host, "hostname && echo READY", check=False)
        if result.returncode == 0 and "READY" in (result.stdout or ""):
            print(result.stdout.strip(), flush=True)
            return
        print(
            f"attempt {attempt}/{attempts} failed: {((result.stderr or result.stdout) or '').strip()[:200]}",
            flush=True,
        )
        if attempt < attempts:
            time.sleep(delay)
    raise ConnectionError(f"{host} did not become reachable")


def find_remote_root(host: str) -> str:
    for root in REMOTE_ROOTS:
        result = ssh(
            host,
            f'test -f "{root}/experiments/baselines_retrain_20260918/protocol.json" && echo {root}',
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[-1]
    raise FileNotFoundError("ALIGN checkout with retrain experiment not found on host")


def scp(host: str, src: str, dst: str, recursive: bool = False) -> None:
    cmd = ["scp", *SSH_OPTS]
    if recursive:
        cmd.append("-r")
    cmd.extend([src, dst])
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="i-2.gpushare.com")
    parser.add_argument("--attempts", type=int, default=180)
    parser.add_argument("--delay", type=int, default=20)
    args = parser.parse_args()
    wait_for_host(args.host, args.attempts, args.delay)
    remote_root = find_remote_root(args.host)
    print("remote_root", remote_root, flush=True)
    scp(
        args.host,
        str(RETRAIN / "infer_datacreate.py"),
        f"{args.host}:{remote_root}/experiments/baselines_retrain_20260918/infer_datacreate.py",
    )
    remote_data = f"{remote_root}/baselines/data/datacreate_all_20260920"
    ssh(args.host, f'mkdir -p "{remote_root}/baselines/data"')
    scp(args.host, str(DATA), f"{args.host}:{remote_root}/baselines/data", recursive=True)
    remote_out = f"{remote_root}/experiments/baselines_datacreate_all_20260920/predictions"
    ssh(args.host, f'mkdir -p "{remote_out}"')
    for model in ("polytune", "laddersym"):
        py = f"{remote_root}/baselines/envs/{model}/bin/python"
        cmd = (
            f'cd "{remote_root}" && "{py}" '
            f"experiments/baselines_retrain_20260918/infer_datacreate.py "
            f'--model {model} --data "{remote_data}" --no-restore '
            f'--out "{remote_out}/{model}"'
        )
        print(cmd, flush=True)
        subprocess.run(["ssh", *SSH_OPTS, args.host, cmd], check=True)
    local_pred = HERE / "predictions"
    local_pred.mkdir(parents=True, exist_ok=True)
    scp(args.host, f"{args.host}:{remote_out}", str(HERE), recursive=True)
    print("predictions", local_pred, flush=True)


if __name__ == "__main__":
    main()
