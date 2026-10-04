"""Run-level and ornament-group diagnosis of v3 transcription misses, with spectrogram examples.

Same-pitch runs are measured as a whole, because which repeat an edit alignment
calls "missing" is arbitrary. Each internal boundary of a gold run is checked on
the forced CTC path: is it separated by a winning blank, and how deep is the
audio level dip there. Ornament expansions are grouped into consecutive runs to
see where inside a mordent/turn/trill the misses fall.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from alignmodel.transcription.ctc_decode_v2 import rich_decode  # noqa: E402
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint  # noqa: E402
from alignmodel.transcription.mel_v1 import load_audio_mono  # noqa: E402
from audit_stack_v3_val_errors import sequence_ops  # noqa: E402
from diagnose_stack_v3_error_causes import HOP, SR, _gold_rows, boundary_dip, forced_spans, rms_envelope  # noqa: E402


def _midi_hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def plot_example(path: Path, audio: np.ndarray, ctc: np.ndarray, spans, gold, j: int, midi_min: int,
                 audio_shift: int, title: str) -> None:
    left = max(0, j - 2)
    right = min(len(gold) - 1, j + 2)
    f0 = max(0, spans[left][0] - 12)
    f1 = min(ctc.shape[0], spans[right][1] + 12)
    s0, s1 = f0 * 256, f1 * 256 + 1024
    segment = np.asarray(audio[s0:s1], np.float64)
    window = 1024
    hop = 64
    frames = max(1, (len(segment) - window) // hop)
    taper = np.hanning(window)
    spec = np.stack([
        np.abs(np.fft.rfft(segment[k * hop:k * hop + window] * taper, n=4096)) for k in range(frames)
    ], axis=1)
    freqs = np.fft.rfftfreq(4096, 1 / SR)
    keep = (freqs >= 100) & (freqs <= 4000)
    db = 20 * np.log10(spec[keep] + 1e-6)
    times = (np.arange(frames) * hop + window / 2) / SR + s0 / SR
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    axes[0].pcolormesh(times, freqs[keep], db, shading="auto", cmap="magma", vmin=db.max() - 70, vmax=db.max())
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Hz (log)")
    colors = {j: "cyan"}
    for k in range(left, right + 1):
        sounding = gold[k]["pitch"] - audio_shift
        color = colors.get(k, "white")
        a, b = spans[k]
        for harmonic in (1, 2, 3):
            hz = _midi_hz(sounding) * harmonic
            axes[0].hlines(hz, a * HOP, max(b, a + 1) * HOP + 0.06, colors=color,
                           linestyles="-" if k == j else "--", linewidth=1.2 if k == j else 0.7)
        axes[0].text(a * HOP, _midi_hz(sounding) * 0.93, f"{gold[k]['pitch']}", color=color, fontsize=8)
    axes[0].set_title(title, fontsize=9)
    t = np.arange(f0, f1) * HOP
    token = int(np.clip(gold[j]["pitch"] - midi_min + 1, 1, ctc.shape[1] - 1))
    axes[1].plot(t, ctc[f0:f1, 0], color="gray", label="blank")
    axes[1].plot(t, ctc[f0:f1, token], color="cyan", label=f"missed note {gold[j]['pitch']}")
    for k in (j - 1, j + 1):
        if 0 <= k < len(gold):
            other = int(np.clip(gold[k]["pitch"] - midi_min + 1, 1, ctc.shape[1] - 1))
            axes[1].plot(t, ctc[f0:f1, other], linewidth=0.8, label=f"neighbour {gold[k]['pitch']}")
    axes[1].axvline(spans[j][0] * HOP, color="cyan", linestyle=":")
    axes[1].set_ylabel("CTC prob")
    axes[1].set_xlabel("seconds")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--figures-per-kind", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(Path(candidate["checkpoint"]), device)
    model.eval()
    midi_min = int(model.config.midi_min)
    decoder = candidate["decoder"]
    report: dict[str, Any] = {}
    for dataset_name in args.datasets:
        dataset = candidate["datasets"][dataset_name]
        root = Path(dataset["root"])
        freeze = json.loads(Path(dataset["aligner_freeze"]).read_text(encoding="utf-8"))
        names = list(freeze["eligible"]["val"])
        random.Random(args.seed).shuffle(names)
        names = names[:args.limit]
        c: collections.Counter[str] = collections.Counter()
        dips: dict[str, list[float]] = collections.defaultdict(list)
        figures: collections.Counter[str] = collections.Counter()
        for position, name in enumerate(names, 1):
            gold = _gold_rows(root, name)
            gold_pitch = [row["pitch"] for row in gold]
            metadata = json.loads((root / name / "metadata.json").read_text(encoding="utf-8"))
            audio_shift = int(metadata.get("effective_audio_transpose", 2))
            audio = load_audio_mono(root / name / "performance_audio.wav", SR)
            with torch.inference_mode():
                mel, _ = extract_dual_mel(audio, device)
                outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
            ctc = np.asarray(outputs["ctc"], np.float32)
            decoded = rich_decode(ctc, midi_min, **decoder)
            committed = [n for n in decoded if not n["optional"]]
            pred_pitch = [int(n["pitch"]) for n in committed]
            tokens = [int(np.clip(p - midi_min + 1, 1, ctc.shape[1] - 1)) for p in gold_pitch]
            spans = forced_spans(ctc, tokens)
            if spans is None:
                c["forced_infeasible"] += 1
                continue
            scaled = ctc.copy()
            scaled[:, 0] *= float(decoder["blank_scale"])
            winners = scaled.argmax(axis=1)
            rms = rms_envelope(audio, ctc.shape[0])
            ops = sequence_ops(pred_pitch, gold_pitch)
            deleted = {j for kind, _i, j in ops if kind == "del"}

            # Same-pitch runs.
            j = 0
            while j < len(gold):
                k = j
                while k + 1 < len(gold) and gold_pitch[k + 1] == gold_pitch[j]:
                    k += 1
                if k > j:
                    length = k - j + 1
                    lo, hi = spans[j][0] - 2, spans[k][1] + 2
                    emitted = sum(1 for n in committed if n["pitch"] == gold_pitch[j] and lo <= n["frame"] < hi)
                    deficit = max(0, length - emitted)
                    c[f"run|len{min(length, 5)}|gold"] += length
                    c[f"run|len{min(length, 5)}|deficit"] += deficit
                    c["run|boundaries"] += length - 1
                    for b in range(j, k):
                        gap_lo, gap_hi = spans[b][1], max(spans[b + 1][0], spans[b][1] + 1)
                        separated = bool((winners[gap_lo:gap_hi] == 0).any())
                        dip = boundary_dip(rms, spans[b], spans[b + 1], spans[b + 1][0])
                        gap_frames = spans[b + 1][0] - spans[b][1]
                        blank_peak = float(ctc[gap_lo:gap_hi, 0].max())
                        key = "separated" if separated else "merged"
                        c[f"run_boundary|{key}"] += 1
                        if np.isfinite(dip):
                            dips[f"dip|{key}"].append(dip)
                        dips[f"gap_frames|{key}"].append(gap_frames)
                        dips[f"blank_peak|{key}"].append(blank_peak)
                        prev_dur = gold[b]["duration"]
                        c[f"run_boundary|{key}|first_note_{'lt120' if prev_dur < 0.12 else 'ge120'}"] += 1
                        if not separated and figures["merged_repeat"] < args.figures_per_kind and dip > 0.6:
                            figures["merged_repeat"] += 1
                            plot_example(args.figures / f"{dataset_name}_merged_repeat_{figures['merged_repeat']}.png",
                                         audio, ctc, spans, gold, b + 1, midi_min, audio_shift,
                                         f"{dataset_name} {name}: repeated {gold_pitch[b]} merged; audio dip ratio {dip:.2f}")
                j = k + 1

            # Ornament-expansion groups.
            j = 0
            while j < len(gold):
                if gold[j]["subtype"] != "ornament_expansion":
                    j += 1
                    continue
                k = j
                while k + 1 < len(gold) and gold[k + 1]["subtype"] == "ornament_expansion":
                    k += 1
                length = k - j + 1
                group = list(range(j, k + 1))
                misses = [g for g in group if g in deleted]
                pitches = gold_pitch[j:k + 1]
                alternating = length >= 3 and len(set(pitches)) == 2 and all(
                    pitches[t] != pitches[t + 1] for t in range(length - 1))
                shape = "trill_like" if alternating else f"len{min(length, 4)}"
                c[f"orn|{shape}|groups"] += 1
                c[f"orn|{shape}|notes"] += length
                c[f"orn|{shape}|missed"] += len(misses)
                dur = float(np.median([gold[g]["duration"] for g in group]))
                dur_key = "lt50" if dur < 0.05 else "50to80" if dur < 0.08 else "ge80"
                c[f"orn_dur|{dur_key}|notes"] += length
                c[f"orn_dur|{dur_key}|missed"] += len(misses)
                for g in misses:
                    place = "first" if g == j else "last" if g == k else "middle"
                    c[f"orn|{shape}|missed_at_{place}"] += 1
                    peak = float(ctc[max(0, spans[g][0] - 2):spans[g][1] + 2, tokens[g]].max())
                    if peak < 0.05 and figures[f"orn_{dur_key}"] < args.figures_per_kind:
                        figures[f"orn_{dur_key}"] += 1
                        plot_example(args.figures / f"{dataset_name}_missed_ornament_{dur_key}_{figures[f'orn_{dur_key}']}.png",
                                     audio, ctc, spans, gold, g, midi_min, audio_shift,
                                     f"{dataset_name} {name}: missed ornament note {gold_pitch[g]} "
                                     f"({gold[g]['duration'] * 1000:.0f} ms in note_map), group {pitches}")
                j = k + 1

            # Short score notes (not ornaments, not repeats) with no evidence.
            for g in sorted(deleted):
                row = gold[g]
                if row["subtype"] != "score_note" or row["duration"] >= 0.08:
                    continue
                if (g > 0 and gold_pitch[g - 1] == row["pitch"]) or (g + 1 < len(gold) and gold_pitch[g + 1] == row["pitch"]):
                    continue
                peak = float(ctc[max(0, spans[g][0] - 2):spans[g][1] + 2, tokens[g]].max())
                if peak < 0.05 and figures["short_note"] < args.figures_per_kind:
                    figures["short_note"] += 1
                    plot_example(args.figures / f"{dataset_name}_missed_short_note_{figures['short_note']}.png",
                                 audio, ctc, spans, gold, g, midi_min, audio_shift,
                                 f"{dataset_name} {name}: missed score note {row['pitch']} ({row['duration'] * 1000:.0f} ms)")
            if position % 40 == 0:
                print(f"{dataset_name} {position}/{len(names)}", flush=True)
        report[dataset_name] = {
            "counts": dict(sorted(c.items())),
            "distributions": {
                key: {"n": len(v), "p10": round(float(np.percentile(v, 10)), 3),
                      "p50": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3)}
                for key, v in sorted(dips.items()) if v
            },
            "figures": dict(figures),
        }
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
