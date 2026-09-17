"""Profile Track B frontend, packed loading, and candidate temporal models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import (
    MelFrontendConfig,
    MelNoteTranscriber,
    MelTranscriberConfig,
    extract_log_mel,
    load_audio_mono,
    make_mel_targets,
    mel_transcriber_loss,
)
from alignmodel.transcription.mel_v1_data import (
    MelCropDataset,
    MelPackedCache,
)


def _frontend_profile(rows: list[dict], device: str) -> list[dict]:
    results = []
    audited_audio = []
    for row in rows:
        path = Path(row["sample_dir"]) / "performance_audio.wav"
        expected = row["source_hashes"]["performance_audio.wav"]
        if sha256_file(path) != expected:
            raise ValueError(f"Audited audio hash mismatch: {row['sample']}")
        audited_audio.append(
            load_audio_mono(path, MelFrontendConfig().sample_rate)
        )
    for hop in (128, 256):
        config = MelFrontendConfig(hop_length=hop)
        # Exclude one-time library/kernel initialization from both candidates.
        extract_log_mel(audited_audio[0], config, device=device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        frames = samples = 0
        audio_seconds = 0.0
        started = time.perf_counter()
        for audio in audited_audio:
            mel, _ = extract_log_mel(audio, config, device=device)
            samples += 1
            frames += mel.shape[1]
            audio_seconds += len(audio) / config.sample_rate
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        results.append({
            "hop_length": hop,
            "hop_ms": 1000.0 * config.hop_sec,
            "clips": samples,
            "frames": frames,
            "seconds": elapsed,
            "rows_per_second": samples / max(elapsed, 1e-9),
            "audio_realtime_factor": audio_seconds / max(elapsed, 1e-9),
            "estimated_full_cache_minutes": (
                4902 / max(samples / max(elapsed, 1e-9), 1e-9) / 60.0
            ),
        })
    return results


def _batch(batch_size: int, frames: int, device: torch.device) -> dict:
    notes = [
        {"pitch": 60 + index % 8, "start_sec": index * 0.20,
         "end_sec": index * 0.20 + (0.07 if index % 3 == 0 else 0.18)}
        for index in range(max(1, int(frames * 0.0116 / 0.20)))
    ]
    target = make_mel_targets(
        notes, frames=frames, hop_sec=256 / 22050,
        midi_min=52, midi_max=100,
    )
    return {
        "mel": torch.randn(batch_size, 128, frames, device=device),
        **{
            key: torch.from_numpy(value).to(device).repeat(
                batch_size, *([1] * value.ndim)
            )
            for key, value in target.items()
        },
        "frame_mask": torch.ones(
            batch_size, frames, dtype=torch.bool, device=device
        ),
    }


def _model_profile(device: torch.device, frames: int) -> list[dict]:
    rows = []
    for temporal in ("tcn", "bigru"):
        for batch_size in (4, 8):
            for amp_name, amp_dtype in (
                ("bf16", torch.bfloat16),
                ("fp16", torch.float16),
            ):
                torch.cuda.empty_cache()
                model = MelNoteTranscriber(
                    MelTranscriberConfig(temporal_kind=temporal)
                ).to(device).train()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=3e-4, fused=device.type == "cuda"
                )
                batch = _batch(batch_size, frames, device)
                scaler = torch.amp.GradScaler(
                    "cuda", enabled=amp_name == "fp16"
                )
                status = "ok"
                try:
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(
                            device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda",
                        ):
                            output = model(batch["mel"])
                            loss, _ = mel_transcriber_loss(output, batch)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    steps = 5
                    for _ in range(steps):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(
                            device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda",
                        ):
                            output = model(batch["mel"])
                            loss, _ = mel_transcriber_loss(output, batch)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    throughput = steps * batch_size / elapsed
                    peak = (
                        torch.cuda.max_memory_allocated() / 1024**2
                        if device.type == "cuda" else 0.0
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    status = f"failed: {error}"
                    throughput = peak = 0.0
                rows.append({
                    "temporal_kind": temporal,
                    "batch_size": batch_size,
                    "amp": amp_name,
                    "compile": False,
                    "crop_frames": frames,
                    "rows_per_second": throughput,
                    "vram_mb": peak,
                    "estimated_epoch_minutes": (
                        8634 / max(throughput, 1e-9) / 60.0
                    ),
                    "status": status,
                })
                del model, optimizer, batch, scaler
    # Windows TorchInductor support varies by installation.  Probe it once and
    # report failure instead of silently enabling it for the full run.
    compile_result = {"compile": True, "status": "not_tested"}
    try:
        model = MelNoteTranscriber().to(device).train()
        compiled = torch.compile(model)
        batch = _batch(4, frames, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, _ = mel_transcriber_loss(compiled(batch["mel"]), batch)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        compile_result.update(status="ok", temporal_kind="tcn", batch_size=4)
    except Exception as error:
        compile_result.update(status=f"failed: {type(error).__name__}: {error}")
    rows.append(compile_result)
    return rows


def _loader_profile(cache_root: Path, workers: Sequence[int]) -> list[dict]:
    cache = MelPackedCache(cache_root)
    records = cache.records("train", include_targets=True)[:512]
    cache.close()
    rows = []
    for count in workers:
        dataset = MelCropDataset(
            cache_root, records, crop_frames=1024, epoch=1, seed=365,
            crops_per_clip=1,
        )
        options = {
            "batch_size": 8, "num_workers": count,
            "pin_memory": True, "shuffle": False,
        }
        if count:
            options.update(
                persistent_workers=True, prefetch_factor=4
            )
        loader = DataLoader(dataset, **options)
        started = time.perf_counter()
        seen = 0
        for batch_index, batch in enumerate(loader):
            seen += int(batch["mel"].shape[0])
            if batch_index >= 39:
                break
        elapsed = time.perf_counter() - started
        rows.append({
            "workers": count,
            "prefetch_factor": 4 if count else None,
            "rows": seen,
            "seconds": elapsed,
            "rows_per_second": seen / max(elapsed, 1e-9),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--frontend-clips", type=int, default=6)
    parser.add_argument("--crop-frames", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ready = json.loads(args.ready_marker.read_text(encoding="utf-8"))
    manifest_path = Path(ready["paths"]["manifest"])
    if sha256_file(manifest_path) != ready["hashes"]["manifest_sha256"]:
        raise ValueError("Audited manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows = list(manifest["train"][:args.frontend_clips])
    device = torch.device(args.device)
    with resource_lease(
        args.resource_status,
        "gpu",
        track="mel-transcriber-v1-profile",
        metadata={"lockbox_materialization": False},
    ):
        frontend = _frontend_profile(train_rows, args.device)
        models = _model_profile(device, args.crop_frames)
        loading = _loader_profile(args.cache, (0, 2, 4, 6)) if args.cache else []
    successful = [
        row for row in models
        if row.get("status") == "ok" and row.get("rows_per_second")
    ]
    selected = max(successful, key=lambda row: row["rows_per_second"])
    report = {
        "schema_version": "align-mel-transcriber-profile-v1",
        "ready_sha256": sha256_file(args.ready_marker),
        "manifest_sha256": ready["hashes"]["manifest_sha256"],
        "locked_test_materialized": False,
        "frontend": frontend,
        "packed_loading": loading,
        "model_training": models,
        "measured_selection": selected,
        "selection_policy": (
            "Fastest stable pilot, subject to full-cache loader and "
            "train-fold calibration confirmation"
        ),
        "report_sha256_basis": hashlib.sha256(
            json.dumps(models, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
