"""Plot model heads around gold same-pitch repeats on a 9.2 val clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from alignmodel.transcription.forced_align_v1 import ForcedAlignConfig, forced_align
from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, decode, _ = load_mel_checkpoint(args.checkpoint, device)
    sample = args.root / args.sample
    gold = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
    pitches = [int(row["pitch_midi_written"]) for row in gold]
    audio = load_audio_mono(sample / "performance_audio.wav", frontend.sample_rate)
    mel, _ = extract_log_mel(audio, frontend, device=device)
    probabilities = infer_mel_probabilities(
        model, np.asarray(mel, np.float32), device,
        window_frames=2048, overlap_frames=512, batch_size=4,
    )
    aligned, _ = forced_align(
        probabilities, pitches, midi_min=model.config.midi_min,
        config=ForcedAlignConfig(onset_weight=1.0, use_boundary_heads=True),
    )
    notes = decode_mel_notes(
        probabilities, midi_min=model.config.midi_min, hop_sec=frontend.hop_sec,
        config=decode,
    )
    repeat = next(
        (note for previous, note in zip(aligned, aligned[1:])
         if previous.pitch == note.pitch and note.index == previous.index + 1),
        aligned[0],
    )
    center = repeat.start_frame * frontend.hop_sec
    lo = max(0.0, center - args.window_sec / 2)
    hi = lo + args.window_sec
    first, last = int(lo / frontend.hop_sec), int(hi / frontend.hop_sec)
    times = np.arange(first, last) * frontend.hop_sec

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    top.imshow(
        np.asarray(mel)[:, first:last], origin="lower", aspect="auto", cmap="magma",
        extent=(lo, hi, 0, mel.shape[0]),
    )
    top.set_title(f"{args.sample} mel (cyan: forced-aligned gold starts, lime: decoded starts)")
    for note in aligned:
        start = note.start_frame * frontend.hop_sec
        if lo <= start <= hi:
            top.axvline(start, color="cyan", linewidth=0.8)
            top.text(start, mel.shape[0] - 8, str(note.pitch), color="cyan", fontsize=7)
    for note in notes:
        if lo <= note.start <= hi:
            top.axvline(note.start, color="lime", linewidth=0.8, linestyle="--")
    for key, color in (("voiced", "gray"), ("onset", "red"),
                       ("boundary", "orange"), ("rearticulation", "blue")):
        bottom.plot(times, probabilities[key][first:last], label=key, color=color, linewidth=1)
    for note in aligned:
        start = note.start_frame * frontend.hop_sec
        if lo <= start <= hi:
            bottom.axvline(start, color="cyan", linewidth=0.6, alpha=0.6)
    bottom.legend(loc="upper right")
    bottom.set_xlabel("seconds")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
