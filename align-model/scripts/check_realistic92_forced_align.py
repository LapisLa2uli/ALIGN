"""Spot-check forced alignment of 9.2 gold pitch order on train clips."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from alignmodel.transcription.forced_align_v1 import forced_align
from alignmodel.transcription.mel_v1 import (
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    names = list(json.loads(args.split.read_text(encoding="utf-8"))["splits"]["train"])
    random.Random(args.seed).shuffle(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _decode, _ = load_mel_checkpoint(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names[:args.limit]:
        sample = args.root / name
        gold = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
        audio = load_audio_mono(sample / "performance_audio.wav", frontend.sample_rate)
        mel, _ = extract_log_mel(audio, frontend, device=device)
        probabilities = infer_mel_probabilities(
            model, np.asarray(mel, np.float32), device,
            window_frames=2048, overlap_frames=512, batch_size=4,
        )
        started = time.perf_counter()
        aligned, stats = forced_align(
            probabilities,
            [int(row["pitch_midi_written"]) for row in gold],
            midi_min=model.config.midi_min,
        )
        stats["seconds"] = time.perf_counter() - started
        stats["sample"] = name
        print(json.dumps(stats), flush=True)

        seconds = min(12.0, len(audio) / frontend.sample_rate)
        limit = int(seconds / frontend.hop_sec)
        figure, axis = plt.subplots(figsize=(14, 5))
        axis.imshow(
            probabilities["pitch"][:limit].T * probabilities["voiced"][:limit][None, :],
            origin="lower", aspect="auto", cmap="magma",
            extent=(0, limit * frontend.hop_sec, model.config.midi_min - 0.5,
                    model.config.midi_min + probabilities["pitch"].shape[1] - 0.5),
        )
        for note in aligned:
            start = note.start_frame * frontend.hop_sec
            if start > seconds:
                break
            axis.plot(
                [start, note.end_frame * frontend.hop_sec], [note.pitch, note.pitch],
                color="cyan", linewidth=2,
            )
            axis.axvline(start, color="cyan", alpha=0.25, linewidth=0.6)
        axis.set_title(f"{name}: forced alignment over mel-v1 voiced*pitch (first {seconds:.0f}s)")
        axis.set_xlabel("seconds")
        axis.set_ylabel("written MIDI")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{name}.png", dpi=90)
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
