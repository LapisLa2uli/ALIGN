"""Break down pitch-sequence LCS errors of a transcriber on the 9.2 val split."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
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
from diagnose_realistic92_timing import _lcs_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--override", default="{}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    names = json.loads(args.split.read_text(encoding="utf-8"))["splits"]["val"][:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, decode, _ = load_mel_checkpoint(args.checkpoint, device)
    decode = replace(decode, **json.loads(args.override))
    missed = collections.Counter()
    support = collections.Counter()
    extra = collections.Counter()
    predicted_total = 0
    missed_duration = []
    matched_duration = []
    for name in names:
        sample = args.root / name
        gold = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
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
        gold_pitch = [int(row["pitch_midi_written"]) for row in gold]
        pred_pitch = [int(note.pitch) for note in notes]
        pairs = _lcs_pairs(pred_pitch, gold_pitch)
        matched_gold = {j for _, j in pairs}
        matched_pred = {i for i, _ in pairs}
        predicted_total += len(pred_pitch)
        for j, row in enumerate(gold):
            labels = []
            relationship = str(row["relationship"])
            labels.append(
                "renderer_only" if relationship == "extra" and not row["performed_indices"]
                else relationship
            )
            repeat = (j > 0 and gold_pitch[j - 1] == gold_pitch[j]) or (
                j + 1 < len(gold) and gold_pitch[j + 1] == gold_pitch[j]
            )
            labels.append("same_pitch_neighbor" if repeat else "pitch_change")
            duration = float(row["end_sec"]) - float(row["start_sec"])
            labels.append(
                "dur_lt80" if duration < 0.08 else "dur_80_150" if duration < 0.15
                else "dur_150_300" if duration < 0.30 else "dur_ge300"
            )
            if not (52 <= gold_pitch[j] <= 100):
                labels.append("out_of_model_range")
            for label in labels:
                support[label] += 1
                if j not in matched_gold:
                    missed[label] += 1
            (matched_duration if j in matched_gold else missed_duration).append(duration)
        for i, note in enumerate(notes):
            if i in matched_pred:
                continue
            neighbors = [pred_pitch[k] for k in (i - 1, i + 1) if 0 <= k < len(pred_pitch)]
            if pred_pitch[i] in neighbors:
                extra["same_pitch_as_neighbor_pred"] += 1
            elif any(abs(pred_pitch[i] - value) in (12, 19, 24) for value in gold_pitch[max(0, i - 3):i + 4]):
                extra["octave_or_harmonic_like"] += 1
            else:
                extra["other"] += 1
            length = float(note.end) - float(note.start)
            extra["pred_dur_lt60" if length < 0.06 else "pred_dur_ge60"] += 1
    report = {
        "clips": len(names),
        "gold_support": dict(support),
        "missed": dict(missed),
        "miss_rate": {key: missed[key] / max(support[key], 1) for key in support},
        "extra_pred": dict(extra),
        "predicted_total": predicted_total,
        "median_missed_gold_duration": float(np.median(missed_duration)) if missed_duration else None,
        "median_matched_gold_duration": float(np.median(matched_duration)) if matched_duration else None,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
