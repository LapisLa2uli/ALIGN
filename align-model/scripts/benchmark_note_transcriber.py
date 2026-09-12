"""Benchmark real transcriber training batches without writing checkpoints."""

from __future__ import annotations

import argparse
import ctypes
import json
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from alignmodel.transcription.data import NoteCropDataset, load_split
from alignmodel.transcription.model import (
    NoteFrameNet,
    NoteFrameNetConfig,
    note_frame_loss,
)
from alignmodel.transcription.train import augment_mel_batch


def _cpu_times() -> tuple[int, int, int]:
    idle = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        raise ctypes.WinError()
    return idle.value, kernel.value, user.value


def _sample_utilization(
    stop: threading.Event,
    cpu_samples: list[float],
    gpu_samples: list[int],
) -> None:
    previous = _cpu_times()
    while not stop.wait(1.0):
        current = _cpu_times()
        idle_delta = current[0] - previous[0]
        total_delta = (current[1] - previous[1]) + (current[2] - previous[2])
        if total_delta > 0:
            cpu_samples.append(100.0 * (1.0 - idle_delta / total_delta))
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            gpu_samples.append(int(result.stdout.strip().splitlines()[0]))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        previous = current


def _percentile(values: list[float | int], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint; omit to initialize a fresh v2 model",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--crop-frames", type=int, default=768)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    examples = load_split(args.root, args.manifest, "train", seed=args.seed)
    dataset = NoteCropDataset(
        examples,
        crop_frames=args.crop_frames,
        training=True,
        augment=False,
        crops_per_clip=2,
    )
    loader_options: dict = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    if args.workers > 0:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=max(1, args.prefetch_factor),
        )
    loader = DataLoader(dataset, **loader_options)

    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = NoteFrameNet(
            NoteFrameNetConfig.from_dict(payload.get("model_config"))
        ).cuda()
        model.load_state_dict(payload["model"])
    else:
        model = NoteFrameNet(NoteFrameNetConfig()).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    iterator = iter(loader)

    def train_step() -> None:
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch_dev = {
            key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batch_dev["mel"] = augment_mel_batch(batch_dev["mel"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(batch_dev["mel"], batch_dev.get("f0"))
            loss, _ = note_frame_loss(outputs, batch_dev)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    cpu_samples: list[float] = []
    gpu_samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_utilization,
        args=(stop, cpu_samples, gpu_samples),
        daemon=True,
    )
    sampler.start()
    print(
        f"benchmark-start workers={args.workers} batch={args.batch_size}",
        flush=True,
    )
    started = time.perf_counter()
    steps = 0
    while True:
        train_step()
        steps += 1
        if steps % 20 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - started >= args.seconds:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=2.0)

    print(
        json.dumps(
            {
                "workers": args.workers,
                "batch_size": args.batch_size,
                "steps": steps,
                "seconds": elapsed,
                "samples_per_second": steps * args.batch_size / elapsed,
                "cpu_mean": float(np.mean(cpu_samples)) if cpu_samples else 0.0,
                "cpu_p90": _percentile(cpu_samples, 90),
                "gpu_mean": float(np.mean(gpu_samples)) if gpu_samples else 0.0,
                "gpu_p90": _percentile(gpu_samples, 90),
                "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "utilization_samples": min(len(cpu_samples), len(gpu_samples)),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
