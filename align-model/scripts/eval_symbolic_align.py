"""Probe: synth score-to-score mapping vs DTW, and rhythm from spacing."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from music21 import converter, note

from datacreate.melody import is_repeated_pass, parse_sounding_notes
from synthpipeline.timing import midi_note_times

ROOT = Path(r"D:\stuff\Audio Evaluation\ALIGN\synth-pipeline\output_2k_rawdata")
OUT = Path(r"D:\stuff\Audio Evaluation\ALIGN\align-model\runs\eval-symbolic-align")
N = 80
SEED = 365


def _notes_sec(path: Path) -> list[dict]:
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
        measure = el.getContextByClass("Measure")
        rows.append(
            {
                "pitch": int(el.pitch.midi),
                "start": start,
                "end": max(end, start + 0.05),
                "measure": int(measure.number) if measure and measure.number is not None else None,
            }
        )
    if rows:
        rows.sort(key=lambda r: (r["start"], r["pitch"]))
        return rows
    return [
        {
            "pitch": n.pitch,
            "start": n.start,
            "end": n.end,
            "measure": n.measure,
        }
        for n in parse_sounding_notes(path)
    ]


def _nw(a: list[int], b: list[int]) -> list[tuple[int | None, int | None]]:
    """Needleman–Wunsch. Match 0, substitute 1, indel 1. Returns (i, j) or None."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diag, 1 up (del b), 2 left (ins a)
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            best = match
            code = 0
            if delete < best:
                best, code = delete, 1
            if insert < best:
                best, code = insert, 2
            dp[i][j] = best
            bt[i][j] = code
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


def _first_pass_labels(sample: Path) -> list[dict]:
    doc = json.loads((sample / "labels.json").read_text(encoding="utf-8"))
    return [lab for lab in doc.get("labels") or [] if not is_repeated_pass(lab)]


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def _log_ratio(a: float, b: float, eps: float = 1e-4) -> float | None:
    if a < eps or b < eps:
        return None
    return abs(math.log(a / b))


def _suffix_repeat_len(pitches: list[int], min_len: int = 4) -> int:
    """Longest suffix that also occurs earlier. 0 if none."""
    n = len(pitches)
    best = 0
    for length in range(min(n // 2, 32), min_len - 1, -1):
        tail = pitches[-length:]
        head = pitches[:-length]
        for i in range(0, len(head) - length + 1):
            if head[i : i + length] == tail:
                return length
        if best:
            break
    return best


def _perturb(pitches: list[int], rng: random.Random, rate: float) -> list[int]:
    out: list[int] = []
    for p in pitches:
        r = rng.random()
        if r < rate / 3:
            continue
        if r < 2 * rate / 3:
            out.append(p)
            out.append(max(48, min(84, p + rng.choice((-2, -1, 1, 2)))))
            continue
        if r < rate:
            out.append(max(48, min(84, p + rng.choice((-2, -1, 1, 2)))))
            continue
        out.append(p)
    return out


def _dtw_onset_mae(sample: Path, clean: list[dict], pairs: list[tuple[int, int]]) -> float | None:
    align = sample / "alignment.npz"
    if not align.exists() or not pairs:
        return None
    try:
        from datacreate.note_alignment import build_note_alignment

        events = [
            ev
            for ev in build_note_alignment(sample).get("events") or []
            if not ev.get("is_rest")
        ]
    except Exception:
        return None
    if len(events) != len(clean):
        return None
    errs = []
    for ci, pj_start in pairs:
        if ci < 0 or ci >= len(events):
            continue
        errs.append(abs(float(events[ci]["perf_start"]) - float(pj_start)))
    if not errs:
        return None
    return sum(errs) / len(errs)


def analyze_sample(sample: Path, rng: random.Random, *, want_dtw: bool = False) -> dict | None:
    verified = sample / "verified_score.musicxml"
    perf_score = sample / "performance_score.musicxml"
    if not (verified.exists() and perf_score.exists()):
        return None
    clean = _notes_sec(verified)
    perf = _notes_sec(perf_score)
    if not clean or not perf:
        return None
    labels = _first_pass_labels(sample)
    gold_types = [lab.get("type") for lab in labels]
    rep = next((lab for lab in labels if lab.get("type") == "repetition"), None)
    split = float(rep["start_time"]) if rep else None
    if split is None:
        first = perf
        replay = []
    else:
        first = [n for n in perf if n["start"] < split - 0.02]
        replay = [n for n in perf if n["start"] >= split - 0.02]
        if not first:
            first, replay = perf, []

    path = _nw([n["pitch"] for n in clean], [n["pitch"] for n in first])
    n_match = n_sub = n_ins = n_del = 0
    matched: list[tuple[int, dict, dict]] = []
    for ci, pj in path:
        if ci is not None and pj is not None:
            if clean[ci]["pitch"] == first[pj]["pitch"]:
                n_match += 1
                matched.append((ci, clean[ci], first[pj]))
            else:
                n_sub += 1
        elif ci is None:
            n_ins += 1
        else:
            n_del += 1

    gold_n = Counter(gold_types)
    unexplained_sub = max(0, n_sub - gold_n.get("wrong_note", 0))
    unexplained_ins = max(0, n_ins - gold_n.get("extra_note", 0))
    unexplained_del = max(0, n_del - gold_n.get("missed_note", 0))

    ioi_flags = 0
    dur_flags = 0
    both_flags = 0
    for k in range(1, len(matched)):
        prev_c, prev_p = matched[k - 1][1], matched[k - 1][2]
        cur_c, cur_p = matched[k][1], matched[k][2]
        ioi_c = cur_c["start"] - prev_c["start"]
        ioi_p = cur_p["start"] - prev_p["start"]
        dur_c = cur_c["end"] - cur_c["start"]
        dur_p = cur_p["end"] - cur_p["start"]
        ioi = _log_ratio(ioi_p, ioi_c)
        dur = _log_ratio(dur_p, dur_c)
        ioi_hit = ioi is not None and ioi > 0.22
        dur_hit = dur is not None and dur > 0.22
        if ioi_hit:
            ioi_flags += 1
        if dur_hit:
            dur_flags += 1
        if ioi_hit or dur_hit:
            both_flags += 1

    gold_rhythm = [lab for lab in labels if lab.get("type") == "rhythm_error"]
    rhythm_comments = [str(lab.get("comment") or "") for lab in gold_rhythm]
    kinds = []
    for comment in rhythm_comments:
        for key in (
            "late start",
            "early start",
            "late end",
            "early end",
            "sudden tempo",
            "uneven",
        ):
            if key in comment:
                kinds.append(key)
                break
        else:
            kinds.append("other")

    flagged_idx = set()
    for k in range(len(matched)):
        _ci, cur_c, cur_p = matched[k]
        dur = _log_ratio(cur_p["end"] - cur_p["start"], cur_c["end"] - cur_c["start"])
        ioi = None
        if k > 0:
            prev_c, prev_p = matched[k - 1][1], matched[k - 1][2]
            ioi = _log_ratio(cur_p["start"] - prev_p["start"], cur_c["start"] - prev_c["start"])
        if (dur is not None and dur > 0.22) or (ioi is not None and ioi > 0.22):
            flagged_idx.add(_ci)

    rhythm_hits = 0
    for lab in gold_rhythm:
        part = lab.get("score_part") or {}
        i0 = part.get("start_note_index")
        i1 = part.get("end_note_index")
        if i0 is None:
            continue
        core = set(range(int(i0), int(i1) + 1)) if i1 is not None else {int(i0)}
        if flagged_idx & core:
            rhythm_hits += 1

    noisy = _perturb([n["pitch"] for n in first], rng, 0.12)
    noisy_path = _nw([n["pitch"] for n in clean], noisy)
    noisy_sub = sum(
        1
        for ci, pj in noisy_path
        if ci is not None and pj is not None and clean[ci]["pitch"] != noisy[pj]
    )
    noisy_ins = sum(1 for ci, pj in noisy_path if ci is None)
    noisy_del = sum(1 for ci, pj in noisy_path if pj is None)

    dtw_mae = None
    if want_dtw:
        dtw_pairs = [(ci, p["start"]) for ci, _c, p in matched]
        dtw_mae = _dtw_onset_mae(sample, clean, dtw_pairs)

    midi_n = 0
    midi_path = sample / "performance_audio.mid"
    if midi_path.exists():
        midi_n = len(midi_note_times(midi_path))

    unsup_rep = _suffix_repeat_len([n["pitch"] for n in perf])
    return {
        "sample": sample.name,
        "n_clean": len(clean),
        "n_perf": len(perf),
        "n_first": len(first),
        "n_replay": len(replay),
        "n_match": n_match,
        "n_sub": n_sub,
        "n_ins": n_ins,
        "n_del": n_del,
        "unexplained_sub": unexplained_sub,
        "unexplained_ins": unexplained_ins,
        "unexplained_del": unexplained_del,
        "gold_types": gold_types,
        "has_rep": bool(rep),
        "unsup_rep_len": unsup_rep,
        "unsup_rep_ok": bool(rep) == (unsup_rep >= 4),
        "ioi_flags": ioi_flags,
        "dur_flags": dur_flags,
        "spacing_flags": both_flags,
        "n_gold_rhythm": len(gold_rhythm),
        "rhythm_hits": rhythm_hits,
        "rhythm_kinds": kinds,
        "noisy_ops": noisy_sub + noisy_ins + noisy_del,
        "clean_ops": n_sub + n_ins + n_del,
        "midi_n": midi_n,
        "midi_vs_score": abs(midi_n - len(perf)),
        "dtw_onset_mae": dtw_mae,
        "match_rate": n_match / max(len(clean), 1),
    }


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / max(len(xs), 1), 4)


def main() -> None:
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "labels.json").exists())
    rng = random.Random(SEED)
    shuf = list(dirs)
    rng.shuffle(shuf)
    holdout = shuf[:N]
    rows = []
    for i, sample in enumerate(holdout, start=1):
        try:
            row = analyze_sample(
                sample, random.Random(SEED + i), want_dtw=(i <= 20 or i % 5 == 0)
            )
        except Exception as exc:
            print(f"skip {sample.name}: {exc}", flush=True)
            continue
        if row:
            rows.append(row)
        if i == 1 or i % 10 == 0:
            print(f"{i}/{len(holdout)} ok={len(rows)}", flush=True)

    type_counts = Counter()
    for row in rows:
        type_counts.update(row["gold_types"])
    kind_hits = defaultdict(lambda: [0, 0])
    for row in rows:
        hit = row["rhythm_hits"] > 0 if row["n_gold_rhythm"] else False
        for kind in row["rhythm_kinds"] or ["none"]:
            kind_hits[kind][1] += 1
            if hit:
                kind_hits[kind][0] += 1

    dtw = [r["dtw_onset_mae"] for r in rows if r["dtw_onset_mae"] is not None]
    summary = {
        "n": len(rows),
        "data": str(ROOT),
        "mean_n_clean": _mean([r["n_clean"] for r in rows]),
        "mean_n_perf": _mean([r["n_perf"] for r in rows]),
        "mean_n_first": _mean([r["n_first"] for r in rows]),
        "mean_n_replay": _mean([r["n_replay"] for r in rows]),
        "mean_match_rate": _mean([r["match_rate"] for r in rows]),
        "mean_sub": _mean([r["n_sub"] for r in rows]),
        "mean_ins": _mean([r["n_ins"] for r in rows]),
        "mean_del": _mean([r["n_del"] for r in rows]),
        "mean_unexplained_edits": _mean(
            [r["unexplained_sub"] + r["unexplained_ins"] + r["unexplained_del"] for r in rows]
        ),
        "frac_has_rep": _mean([1.0 if r["has_rep"] else 0.0 for r in rows]),
        "frac_unsup_rep_ok": _mean([1.0 if r["unsup_rep_ok"] else 0.0 for r in rows]),
        "mean_ioi_flags": _mean([r["ioi_flags"] for r in rows]),
        "mean_dur_flags": _mean([r["dur_flags"] for r in rows]),
        "mean_spacing_flags": _mean([r["spacing_flags"] for r in rows]),
        "rhythm_recall": _mean(
            [
                r["rhythm_hits"] / r["n_gold_rhythm"]
                for r in rows
                if r["n_gold_rhythm"]
            ]
        ),
        "n_with_rhythm": sum(1 for r in rows if r["n_gold_rhythm"]),
        "mean_clean_ops": _mean([r["clean_ops"] for r in rows]),
        "mean_noisy_ops": _mean([r["noisy_ops"] for r in rows]),
        "mean_midi_vs_score": _mean([r["midi_vs_score"] for r in rows]),
        "mean_dtw_onset_mae": _mean(dtw) if dtw else None,
        "n_dtw": len(dtw),
        "gold_type_counts": dict(type_counts),
        "rhythm_kind_hit_over_n": {k: {"hit": v[0], "n": v[1]} for k, v in kind_hits.items()},
        "samples": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
