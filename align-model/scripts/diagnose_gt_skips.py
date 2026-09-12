"""Why does ground-truth reconstruction reject most synth clips?"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from eval_note_alignment import align_symbolic, midi_notes, score_notes

ROOT = Path(r"E:\output_2k_rawdata")
N = 150
SEED = 365


def reason(sample: Path) -> tuple[str, dict]:
    info: dict = {}
    clean = score_notes(sample / "verified_score.musicxml")
    perf = score_notes(sample / "performance_score.musicxml")
    mid = sample / "performance_audio.mid"
    if not clean:
        return "no_clean_notes", info
    if not perf:
        return "no_perf_notes", info
    if not mid.exists():
        return "no_midi", info
    played = midi_notes(mid)
    if not played:
        return "empty_midi", info
    info["n_perf"] = len(perf)
    info["n_midi"] = len(played)
    info["delta"] = len(played) - len(perf)
    if len(played) != len(perf):
        return "count_mismatch", info
    perf_p = [n["pitch"] for n in perf]
    play_p = [n["pitch"] for n in played]
    if play_p != perf_p:
        diffs = {a - b for a, b in zip(play_p, perf_p)}
        info["n_distinct_offsets"] = len(diffs)
        if len(diffs) != 1:
            return "pitch_mismatch_varied", info
        shift = diffs.pop()
        info["shift"] = shift
        if abs(shift) > 2:
            return "pitch_mismatch_big_shift", info
        return "ok_shifted", info
    truth, _ = align_symbolic([n["pitch"] for n in clean], perf_p)
    if sum(1 for t in truth if t is not None) < 0.6 * len(perf_p):
        return "low_map_coverage", info
    return "ok", info


def main() -> None:
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "labels.json").exists())
    rng = random.Random(SEED)
    shuffled = list(dirs)
    rng.shuffle(shuffled)
    counts: Counter = Counter()
    deltas: list[int] = []
    rep_flag: Counter = Counter()
    for sample in shuffled[:N]:
        try:
            why, info = reason(sample)
        except Exception as exc:  # noqa: BLE001
            counts[f"error:{type(exc).__name__}"] += 1
            continue
        counts[why] += 1
        if why == "count_mismatch":
            deltas.append(info["delta"])
            labels = json.loads((sample / "labels.json").read_text(encoding="utf-8"))
            has_rep = any(
                lab.get("type") == "repetition" for lab in labels.get("labels") or []
            )
            rep_flag[has_rep] += 1
    print("reasons:", dict(counts))
    if deltas:
        deltas.sort()
        print(
            "count_mismatch delta: min",
            deltas[0],
            "median",
            deltas[len(deltas) // 2],
            "max",
            deltas[-1],
            "frac_midi_more",
            round(sum(1 for d in deltas if d > 0) / len(deltas), 3),
        )
        print("count_mismatch has_repetition:", dict(rep_flag))


if __name__ == "__main__":
    main()
