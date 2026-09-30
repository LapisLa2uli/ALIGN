"""Measure note_map timing agreement with 9.2 Muse Sounds audio (train split only).

Pairs mel-v1 notes with gold notes along the pitch-sequence LCS and reports
onset offsets and a per-clip linear time fit. Test clips are never read.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


def _lcs_pairs(left: list[int], right: list[int]) -> list[tuple[int, int]]:
    table = np.zeros((len(left) + 1, len(right) + 1), np.int32)
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            table[i, j] = (
                table[i - 1, j - 1] + 1 if left[i - 1] == right[j - 1]
                else max(table[i - 1, j], table[i, j - 1])
            )
    pairs = []
    i, j = len(left), len(right)
    while i and j:
        if left[i - 1] == right[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif table[i - 1, j] >= table[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))["splits"]
    names = list(split["train"])
    random.Random(args.seed).shuffle(names)
    names = names[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, decode, _ = load_mel_checkpoint(args.checkpoint, device)
    rows = []
    for name in names:
        sample = args.root / name
        gold = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))[
            "rendered_notes"
        ]
        audio = load_audio_mono(sample / "performance_audio.wav", frontend.sample_rate)
        mel, _ = extract_log_mel(audio, frontend, device=device)
        probabilities = infer_mel_probabilities(
            model, np.asarray(mel, np.float32), device,
            window_frames=2048, overlap_frames=512, batch_size=4,
        )
        notes = decode_mel_notes(
            probabilities, midi_min=model.config.midi_min,
            hop_sec=frontend.hop_sec, config=decode,
        )
        pairs = _lcs_pairs(
            [int(n.pitch) for n in notes],
            [int(g["pitch_midi_written"]) for g in gold],
        )
        if len(pairs) < 4:
            rows.append({"sample": name, "pairs": len(pairs)})
            continue
        pred_t = np.array([float(notes[i].start) for i, _ in pairs])
        gold_t = np.array([float(gold[j]["start_sec"]) for _, j in pairs])
        slope, intercept = np.polyfit(gold_t, pred_t, 1)
        residual = pred_t - (slope * gold_t + intercept)
        rows.append({
            "sample": name,
            "pairs": len(pairs),
            "pred_notes": len(notes),
            "gold_notes": len(gold),
            "audio_sec": len(audio) / frontend.sample_rate,
            "gold_last_end_sec": float(gold[-1]["end_sec"]),
            "median_offset_sec": float(np.median(pred_t - gold_t)),
            "p90_abs_offset_sec": float(np.percentile(np.abs(pred_t - gold_t), 90)),
            "fit_slope": float(slope),
            "fit_intercept_sec": float(intercept),
            "fit_p90_abs_residual_sec": float(np.percentile(np.abs(residual), 90)),
        })
        print(json.dumps(rows[-1]), flush=True)
    valid = [row for row in rows if "median_offset_sec" in row]
    summary = {
        "clips": len(rows),
        "valid": len(valid),
        "median_of_median_offset_sec": float(np.median([r["median_offset_sec"] for r in valid])),
        "median_p90_abs_offset_sec": float(np.median([r["p90_abs_offset_sec"] for r in valid])),
        "median_fit_slope": float(np.median([r["fit_slope"] for r in valid])),
        "median_fit_p90_abs_residual_sec": float(
            np.median([r["fit_p90_abs_residual_sec"] for r in valid])
        ),
        "median_audio_over_gold_duration": float(np.median([
            r["audio_sec"] / max(r["gold_last_end_sec"], 1e-6) for r in valid
        ])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
