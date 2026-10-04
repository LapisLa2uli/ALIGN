"""Spectrogram views of predicted extra/missed notes on DataCreate takes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from alignmodel.melody import load_bundle_notes  # noqa: E402
from alignmodel.transcription.mel_v1 import load_audio_mono  # noqa: E402

SR = 22050


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def plot(sample: Path, alignment: dict, center_index: int, path: Path, title: str, audio_shift: int = 2) -> None:
    events = sorted(alignment["events"], key=lambda e: e["note_index"])
    center = events[center_index]
    t0, t1 = center["start"] - 0.6, center["start"] + 0.6
    audio = load_audio_mono(sample / "performance_audio.wav", SR)
    s0, s1 = max(0, int(t0 * SR)), min(len(audio), int(t1 * SR))
    segment = np.asarray(audio[s0:s1], np.float64)
    window, hop = 1024, 64
    frames = max(1, (len(segment) - window) // hop)
    taper = np.hanning(window)
    spec = np.stack([np.abs(np.fft.rfft(segment[k * hop:k * hop + window] * taper, n=4096)) for k in range(frames)], 1)
    freqs = np.fft.rfftfreq(4096, 1 / SR)
    keep = (freqs >= 120) & (freqs <= 4000)
    db = 20 * np.log10(spec[keep] + 1e-7)
    times = (np.arange(frames) * hop + window / 2) / SR + s0 / SR
    rms = np.array([np.sqrt(np.mean(segment[k * hop:k * hop + window] ** 2)) for k in range(frames)])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    axes[0].pcolormesh(times, freqs[keep], db, shading="auto", cmap="magma", vmin=db.max() - 65, vmax=db.max())
    axes[0].set_yscale("log")
    score = load_bundle_notes(sample)
    for k, event in enumerate(events):
        if event["end"] < t0 or event["start"] > t1:
            continue
        sounding = event["pitch"] - audio_shift
        color = {"match": "white", "copy": "lightgray", "substitute": "yellow", "extra": "red"}[event["relationship"]]
        if k == center_index:
            color = "cyan"
        axes[0].hlines(_hz(sounding), event["start"], event["end"], colors=color, linewidth=2.2)
        span = event.get("score_span")
        written = score[span[0]].pitch if span and span[0] < len(score) else None
        axes[0].text(event["start"], _hz(sounding) * 1.05, f"{event['pitch']}" + (f"/{written}" if written and written != event['pitch'] else ""),
                     color=color, fontsize=8)
    axes[0].set_title(title, fontsize=9)
    axes[1].plot(times, 20 * np.log10(rms + 1e-6), color="gray")
    axes[1].set_ylabel("dB")
    axes[1].set_xlabel("seconds")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=root / "DataCreate" / "samples")
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--take", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = args.samples / args.take
    alignment = json.loads((args.alignments / f"{args.take}.json").read_text(encoding="utf-8"))
    events = sorted(alignment["events"], key=lambda e: e["note_index"])
    shown = 0
    for k, event in enumerate(events):
        if event["relationship"] == "extra" and event["score_span"] is None and shown < args.limit:
            shown += 1
            plot(sample, alignment, k, args.output / f"{args.take}_extra_{shown}.png",
                 f"take {args.take}: predicted extra {event['pitch']} at {event['start']:.2f}s "
                 f"(cyan; white=match, yellow=substitute 'heard/written', red=other extras)")
    missed = sorted(alignment.get("missed_score_event_indices") or [])
    score = load_bundle_notes(sample)
    for number, index in enumerate(missed[:args.limit], 1):
        before = [k for k, e in enumerate(events) if e.get("score_span") and e["copy_pass"] == 0
                  and e["score_span"][1] <= index]
        if not before:
            continue
        k = before[-1]
        plot(sample, alignment, k, args.output / f"{args.take}_missed_{number}.png",
             f"take {args.take}: predicted missed score note {index} (written {score[index].pitch}) "
             f"after cyan note")
    print(f"plotted {shown} extras, {min(len(missed), args.limit)} missed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
