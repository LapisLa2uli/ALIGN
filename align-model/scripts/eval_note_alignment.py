"""Alignment-only benchmark: which written note does each played note belong to?

Ground truth is rebuilt from the synth bundle (clean score + performance score
+ rendered MIDI), including copy blocks for replayed spans. Two candidate
aligners are then scored on the same clips:

  dtw       - current path: alignment.npz windows per clean note
  symbolic  - proposed path: note list -> Needleman-Wunsch on pitch, plus
              copy-block resolution, under simulated transcription noise
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from music21 import converter, note

ROOT = Path(r"E:\output_2k_rawdata")
OUT = Path(r"D:\stuff\Audio Evaluation\ALIGN\align-model\runs\eval-note-alignment")
N_CLIPS = 60
SEED = 365
NOISE_RATES = (0.0, 0.05, 0.10, 0.20)
ONSET_JITTER_SEC = 0.03
COPY_MIN_LEN = 4
COPY_MAX_DIST = 0.35
ONSET_TOL_SEC = 0.15


# ---------------------------------------------------------------- score input


def score_notes(path: Path) -> list[dict]:
    """Sounding notes with seconds from the score's own tempo map."""
    parsed = converter.parse(str(path))
    flat = parsed.flatten()
    rows: list[dict] = []
    try:
        sec_map = list(flat.secondsMap)
    except Exception:
        sec_map = []
    for item in sec_map:
        el = item.get("element")
        if not isinstance(el, note.Note) or el.duration.isGrace:
            continue
        start = float(item.get("offsetSeconds", 0.0))
        end = float(item.get("endTimeSeconds", start))
        rows.append({"pitch": int(el.pitch.midi), "start": start, "end": max(end, start + 0.05)})
    rows.sort(key=lambda r: (r["start"], r["pitch"]))
    return rows


def midi_notes(path: Path) -> list[dict]:
    from synthpipeline.timing import midi_note_times

    return [
        {"pitch": p, "start": s, "end": e}
        for p, s, e in sorted(midi_note_times(path), key=lambda t: (t[1], t[0]))
    ]


# ------------------------------------------------------------------ alignment


def needleman_wunsch(a: list[int], b: list[int]) -> list[tuple[int | None, int | None]]:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up (a only), 2 left (b only)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    bt[1:, 0] = 1
    bt[0, 1:] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        row_prev = dp[i - 1]
        row = dp[i]
        bt_row = bt[i]
        for j in range(1, m + 1):
            diag = row_prev[j - 1] + (0 if ai == b[j - 1] else 1)
            up = row_prev[j] + 1
            left = row[j - 1] + 1
            best, code = diag, 0
            if up < best:
                best, code = up, 1
            if left < best:
                best, code = left, 2
            row[j] = best
            bt_row[j] = code
    i, j = n, m
    path: list[tuple[int | None, int | None]] = []
    while i > 0 or j > 0:
        code = bt[i][j]
        if i > 0 and j > 0 and code == 0:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or code == 1):
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    path.reverse()
    return path


def _edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cur[j] = min(prev[j - 1] + (0 if ai == bj else 1), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def _insertion_runs(path: list[tuple[int | None, int | None]]) -> list[list[int]]:
    runs: list[list[int]] = []
    cur: list[int] = []
    for ci, pj in path:
        if ci is None and pj is not None:
            cur.append(pj)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _best_clean_window(run_pitches: list[int], clean: list[int]) -> tuple[int, float] | None:
    """Where in the clean score does this inserted run restate, if anywhere?"""
    length = len(run_pitches)
    best: tuple[int, float] | None = None
    for width in range(max(COPY_MIN_LEN, length - 2), min(len(clean), length + 2) + 1):
        for start in range(0, len(clean) - width + 1):
            dist = _edit_distance(run_pitches, clean[start : start + width]) / max(length, 1)
            if best is None or dist < best[1]:
                best = (start, dist)
    return best


def align_symbolic(
    clean_pitches: list[int],
    played_pitches: list[int],
    *,
    resolve_copies: bool = True,
) -> tuple[list[int | None], dict]:
    """Return predicted clean index per played note (None = extra)."""
    path = needleman_wunsch(clean_pitches, played_pitches)
    pred: list[int | None] = [None] * len(played_pitches)
    for ci, pj in path:
        if ci is not None and pj is not None:
            pred[pj] = ci
    stats = {"n_copy_blocks": 0, "copy_notes": 0}
    if resolve_copies:
        for run in _insertion_runs(path):
            if len(run) < COPY_MIN_LEN:
                continue
            run_pitches = [played_pitches[j] for j in run]
            found = _best_clean_window(run_pitches, clean_pitches)
            if found is None or found[1] > COPY_MAX_DIST:
                continue
            start, _dist = found
            stats["n_copy_blocks"] += 1
            sub = needleman_wunsch(
                clean_pitches[start : start + len(run)], run_pitches
            )
            for ci, pj in sub:
                if ci is not None and pj is not None:
                    pred[run[pj]] = start + ci
                    stats["copy_notes"] += 1
    return pred, stats


def align_dtw(sample: Path, played: list[dict], n_clean: int) -> list[int | None] | None:
    """Assign each played note to the clean note whose DTW window covers it."""
    try:
        from datacreate.note_alignment import build_note_alignment

        events = [
            ev
            for ev in build_note_alignment(sample).get("events") or []
            if not ev.get("is_rest")
        ]
    except Exception:
        return None
    if len(events) != n_clean or not events:
        return None
    starts = np.array([float(ev["perf_start"]) for ev in events])
    ends = np.array([float(ev["perf_end"]) for ev in events])
    pred: list[int | None] = []
    for pn in played:
        t = float(pn["start"])
        inside = np.flatnonzero((starts <= t + 1e-6) & (ends >= t - 1e-6))
        if inside.size:
            pred.append(int(inside[0]))
        else:
            pred.append(int(np.argmin(np.abs(starts - t))))
    return pred


# ------------------------------------------------------------- ground truth


def build_ground_truth(sample: Path) -> dict | None:
    clean = score_notes(sample / "verified_score.musicxml")
    perf = score_notes(sample / "performance_score.musicxml")
    mid_path = sample / "performance_audio.mid"
    if not clean or not perf or not mid_path.exists():
        return None
    played = midi_notes(mid_path)
    if not played:
        return None

    perf_p = [n["pitch"] for n in perf]
    play_p = [n["pitch"] for n in played]
    if len(play_p) != len(perf_p):
        return None
    shift = 0
    if play_p != perf_p:
        diffs = {a - b for a, b in zip(play_p, perf_p)}
        if len(diffs) != 1:
            return None
        shift = diffs.pop()
        if abs(shift) > 2:
            return None
        play_p = [p - shift for p in play_p]
        if play_p != perf_p:
            return None

    clean_p = [n["pitch"] for n in clean]
    truth, stats = align_symbolic(clean_p, perf_p, resolve_copies=True)
    n_mapped = sum(1 for t in truth if t is not None)
    if n_mapped < 0.6 * len(perf_p):
        return None
    return {
        "clean": clean,
        "clean_pitches": clean_p,
        "played": [
            {"pitch": p, "start": played[i]["start"], "end": played[i]["end"]}
            for i, p in enumerate(play_p)
        ],
        "truth": truth,
        "n_copy_blocks": stats["n_copy_blocks"],
        "copy_notes": stats["copy_notes"],
    }


# ------------------------------------------------------------ noisy frontend


def simulate_transcription(
    played: list[dict], truth: list[int | None], rate: float, rng: random.Random
) -> tuple[list[dict], list[int | None]]:
    """Drop / insert / mis-pitch notes; keep the truth label on survivors."""
    out_notes: list[dict] = []
    out_truth: list[int | None] = []
    for pn, tv in zip(played, truth):
        r = rng.random()
        if r < rate / 3:
            continue  # missed by the transcriber
        pitch = pn["pitch"]
        if rate / 3 <= r < 2 * rate / 3:
            pitch = pitch + rng.choice((-2, -1, 1, 2, 12, -12))
        jitter = rng.gauss(0.0, ONSET_JITTER_SEC)
        out_notes.append(
            {
                "pitch": pitch,
                "start": max(0.0, pn["start"] + jitter),
                "end": max(0.05, pn["end"] + jitter),
            }
        )
        out_truth.append(tv)
        if 2 * rate / 3 <= r < rate:
            out_notes.append(
                {
                    "pitch": pitch + rng.choice((-2, -1, 1, 2)),
                    "start": pn["start"] + 0.5 * max(0.05, pn["end"] - pn["start"]),
                    "end": pn["end"],
                }
            )
            out_truth.append("spurious")
    order = sorted(range(len(out_notes)), key=lambda i: out_notes[i]["start"])
    return [out_notes[i] for i in order], [out_truth[i] for i in order]


# ------------------------------------------------------------------- scoring


def score_pred(
    pred: list[int | None], truth: list, n_true_total: int
) -> dict:
    """Accuracy over real played notes; spurious notes counted as precision loss."""
    real = [(p, t) for p, t in zip(pred, truth) if t != "spurious"]
    correct = sum(1 for p, t in real if p == t)
    spurious = sum(1 for p, t in zip(pred, truth) if t == "spurious" and p is not None)
    return {
        "recall": correct / max(n_true_total, 1),
        "precision": correct / max(len(pred), 1),
        "n_correct": correct,
        "n_scored": len(real),
        "n_spurious_bound": spurious,
    }


def analyze(sample: Path, rng: random.Random, *, with_dtw: bool = True) -> dict | None:
    gt = build_ground_truth(sample)
    if gt is None:
        return None
    clean_p = gt["clean_pitches"]
    played = gt["played"]
    truth = gt["truth"]
    n_true = len(played)

    row: dict = {
        "sample": sample.name,
        "n_clean": len(clean_p),
        "n_played": n_true,
        "has_copy": gt["n_copy_blocks"] > 0,
        "copy_notes": gt["copy_notes"],
    }

    for rate in NOISE_RATES:
        if rate == 0.0:
            notes, tvals = played, truth
        else:
            notes, tvals = simulate_transcription(played, truth, rate, rng)
        pred, _stats = align_symbolic(clean_p, [n["pitch"] for n in notes])
        row[f"sym_{int(rate * 100)}"] = score_pred(pred, tvals, n_true)

    pred_nocopy, _ = align_symbolic(clean_p, [n["pitch"] for n in played], resolve_copies=False)
    row["sym_0_nocopy"] = score_pred(pred_nocopy, truth, n_true)

    if with_dtw:
        dtw_pred = align_dtw(sample, played, len(clean_p))
        row["dtw"] = score_pred(dtw_pred, truth, n_true) if dtw_pred is not None else None
        if dtw_pred is not None:
            onset_err = []
            try:
                from datacreate.note_alignment import build_note_alignment

                events = [
                    ev
                    for ev in build_note_alignment(sample).get("events") or []
                    if not ev.get("is_rest")
                ]
                first_seen: dict[int, float] = {}
                for pn, tv in zip(played, truth):
                    if isinstance(tv, int) and tv not in first_seen:
                        first_seen[tv] = pn["start"]
                for ci, t in first_seen.items():
                    if ci < len(events):
                        onset_err.append(abs(float(events[ci]["perf_start"]) - t))
            except Exception:
                onset_err = []
            row["dtw_onset_mae"] = float(np.mean(onset_err)) if onset_err else None
            row["dtw_onset_within_tol"] = (
                float(np.mean([e <= ONSET_TOL_SEC for e in onset_err])) if onset_err else None
            )
    return row


def _agg(rows: list[dict], key: str, field: str = "recall") -> float | None:
    vals = [r[key][field] for r in rows if r.get(key)]
    return round(float(np.mean(vals)), 4) if vals else None


def main() -> None:
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "labels.json").exists())
    rng = random.Random(SEED)
    shuffled = list(dirs)
    rng.shuffle(shuffled)

    rows: list[dict] = []
    skipped = 0
    for i, sample in enumerate(shuffled, start=1):
        if len(rows) >= N_CLIPS:
            break
        try:
            row = analyze(sample, random.Random(SEED + i))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {sample.name}: {exc}", flush=True)
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"{len(rows)}/{N_CLIPS} scanned={i} skipped={skipped}", flush=True)

    with_copy = [r for r in rows if r["has_copy"]]
    no_copy = [r for r in rows if not r["has_copy"]]
    summary = {
        "data": str(ROOT),
        "n_clips": len(rows),
        "n_skipped_for_gt": skipped,
        "n_with_copy": len(with_copy),
        "n_without_copy": len(no_copy),
        "mean_n_clean": round(float(np.mean([r["n_clean"] for r in rows])), 2),
        "mean_n_played": round(float(np.mean([r["n_played"] for r in rows])), 2),
        "accuracy_recall": {
            "dtw": _agg(rows, "dtw"),
            "symbolic_noise_0": _agg(rows, "sym_0"),
            "symbolic_noise_0_no_copy_handling": _agg(rows, "sym_0_nocopy"),
            "symbolic_noise_5": _agg(rows, "sym_5"),
            "symbolic_noise_10": _agg(rows, "sym_10"),
            "symbolic_noise_20": _agg(rows, "sym_20"),
        },
        "accuracy_recall_with_copy": {
            "dtw": _agg(with_copy, "dtw"),
            "symbolic_noise_0": _agg(with_copy, "sym_0"),
            "symbolic_noise_0_no_copy_handling": _agg(with_copy, "sym_0_nocopy"),
            "symbolic_noise_10": _agg(with_copy, "sym_10"),
        },
        "accuracy_recall_without_copy": {
            "dtw": _agg(no_copy, "dtw"),
            "symbolic_noise_0": _agg(no_copy, "sym_0"),
            "symbolic_noise_10": _agg(no_copy, "sym_10"),
        },
        "dtw_onset_mae_sec": round(
            float(np.mean([r["dtw_onset_mae"] for r in rows if r.get("dtw_onset_mae")])), 4
        )
        if any(r.get("dtw_onset_mae") for r in rows)
        else None,
        "dtw_onset_within_150ms": round(
            float(
                np.mean(
                    [
                        r["dtw_onset_within_tol"]
                        for r in rows
                        if r.get("dtw_onset_within_tol") is not None
                    ]
                )
            ),
            4,
        )
        if any(r.get("dtw_onset_within_tol") is not None for r in rows)
        else None,
        "copy_block_clips": Counter(r["has_copy"] for r in rows).get(True, 0),
        "samples": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
