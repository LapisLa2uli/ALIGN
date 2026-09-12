"""Run one loaded Tsumugi checkpoint over frozen ALIGN manifest rows.

This helper is intentionally dependency-light and is executed with Tsumugi's
own uv-managed Python environment. Tsumugi has no stable public Python API, so
pin the source revision recorded in ``requirements-amt-benchmark.txt``. It
writes raw sounding-pitch note events; the common external benchmark applies
ALIGN pitch conventions and metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsumugi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(args.tsumugi_root.resolve()))
    import torch

    from instrument_agnostic_amt.amt.cli.infer import (
        _load_model_and_settings,
        run_inference,
    )
    from instrument_agnostic_amt.amt.inference.audio import load_audio

    device = torch.device(args.device)
    model, config, settings = _load_model_and_settings(
        args.checkpoint.resolve(),
        device=device,
        window_ms_override=None,
        stride_ms_override=None,
        track_batch_size_override=None,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = args.split or ["val", "test_id", "test_ood"]
    report = {
        "model": "tsumugi",
        "checkpoint": str(args.checkpoint),
        "sample_rate": int(config.sample_rate),
        "splits": {},
    }
    for split in splits:
        rows = list(manifest.get(split) or [])
        if args.max_samples:
            rows = rows[: args.max_samples]
        output_rows = []
        skipped = []
        for index, row in enumerate(rows, 1):
            sample = Path(row["sample_dir"])
            wav = sample / "performance_audio.wav"
            try:
                waveform = load_audio(
                    wav, target_sample_rate=int(config.sample_rate)
                )
                started = time.perf_counter()
                notes, stats, _extra = run_inference(
                    model=model,
                    waveform=waveform,
                    model_config=config,
                    settings=settings,
                    device=device,
                    amp_enabled=device.type == "cuda",
                    amp_dtype=torch.float16,
                    window_batch_size=1,
                    disable_tqdm=True,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                output_rows.append(
                    {
                        "sample": sample.name,
                        "sample_dir": str(sample),
                        "inference_sec": elapsed,
                        "stats": stats,
                        "notes": [
                            {
                                "pitch": int(note.pitch),
                                "start": float(note.start_sample)
                                / float(config.sample_rate),
                                "end": float(note.end_sample)
                                / float(config.sample_rate),
                                "confidence": 1.0,
                            }
                            for note in notes
                        ],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"sample": sample.name, "error": str(exc)})
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(
                    f"{split} {index}/{len(rows)} "
                    f"ok={len(output_rows)} skipped={len(skipped)}",
                    flush=True,
                )
        report["splits"][split] = {
            "rows": output_rows,
            "skipped": skipped,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
