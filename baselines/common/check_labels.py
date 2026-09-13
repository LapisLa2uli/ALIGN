#!/usr/bin/env python
"""Cross-check the converted label MIDIs against the source bundle.

For each track (default: first 3 of the manifest, or ``--ids``):

  (i)  |correct| + |extra| == #notes in performance_audio.mid (tied notes merged)
       and every (onset, pitch) of correct+extra matches a performance-MIDI note
       within ``--tol`` seconds (default 3 ms);
  (ii) |correct| + |removed| == #reference_notes entries minus tie continuations
       (== reference notes with cls correct + cls missed).

Usage::

    $B/envs/polytune/bin/python \
        $B/common/check_labels.py \
        --root $B/data/smoke_align [--n 3] [--ids id1 id2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
from pathlib import Path
from prepare_dataset import output_paths


def load_notes(path: str) -> list[tuple[float, float, int]]:
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes]
    return sorted(notes)


def match(a: list[tuple[float, float, int]], b: list[tuple[float, float, int]], tol: float):
    """Greedy one-to-one match on (onset, pitch). Returns (n_matched, unmatched_a, max_dt)."""
    used = np.zeros(len(b), dtype=bool)
    b_on = np.array([x[0] for x in b]) if b else np.zeros(0)
    b_p = np.array([x[2] for x in b]) if b else np.zeros(0, dtype=int)
    unmatched = []
    max_dt = 0.0
    for on, off, p in a:
        cand = np.where((~used) & (b_p == p) & (np.abs(b_on - on) <= tol))[0]
        if len(cand) == 0:
            unmatched.append((round(on, 4), p))
            continue
        j = cand[np.argmin(np.abs(b_on[cand] - on))]
        used[j] = True
        max_dt = max(max_dt, abs(b_on[j] - on))
    return len(a) - len(unmatched), unmatched, max_dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=BUNDLE_ROOT",
                    help="Override source bundle locations after migration; repeat for multi/raw")
    ap.add_argument("--tol", type=float, default=0.003)
    args = ap.parse_args()

    with open(os.path.join(args.root, "manifest.json")) as f:
        manifest = json.load(f)
    tracks = manifest["tracks"]
    eligible = sorted(tracks)
    audited = Path(args.root) / "split.audited.json"
    if audited.exists():
        sd = json.loads(audited.read_text())
        eligible = sorted(Path(v).name.replace(".midi", "") for v in sd["midi_filename"].values())
    ids = args.ids or eligible[:args.n]

    all_ok = True
    for tid in ids:
        rec = tracks[tid]
        roots = dict(spec.split("=", 1) for spec in args.set)
        bundle = str(Path(roots[rec["set"]]) / Path(rec["bundle"]).name) if rec["set"] in roots else rec["bundle"]
        out = output_paths(Path(args.root).resolve(), tid)
        correct = load_notes(out["correct"])
        extra = load_notes(out["extra"])
        removed = load_notes(out["removed"])

        # ---- (i) performance MIDI ----
        perf = load_notes(os.path.join(bundle, "performance_audio.mid"))
        ce = sorted(correct + extra)
        n_match, unmatched, max_dt = match(ce, perf, args.tol)
        ok_i = len(ce) == len(perf) and not unmatched

        # ---- (ii) reference notes ----
        with open(os.path.join(bundle, "note_labels.json")) as f:
            nl = json.load(f)
        ref = nl["reference_notes"]
        n_ref = len(ref) - sum(1 for r in ref if r.get("tie_prev"))
        n_ref_correct = sum(1 for r in ref if r["cls"] == "correct" and not r.get("tie_prev"))
        n_ref_missed = sum(1 for r in ref if r["cls"] == "missed" and not r.get("tie_prev"))
        ok_ii = len(correct) + len(removed) == n_ref

        # also compare against the reference MIDI note count for good measure
        ref_mid = load_notes(os.path.join(bundle, "reference_audio.mid"))

        all_ok &= ok_i and ok_ii
        print(f"{tid}  (set={rec['set']}, source={rec['source']}, split={rec['split']})")
        print(f"  correct={len(correct)} extra={len(extra)} removed={len(removed)}  label_stats={rec.get('label_stats')}")
        print(f"  (i)  |correct|+|extra| = {len(ce)}  vs performance_audio.mid notes = {len(perf)}; "
              f"matched {n_match}/{len(ce)} within {args.tol*1000:.0f} ms (max |dt| = {max_dt*1000:.2f} ms)"
              f"{'' if not unmatched else '  UNMATCHED: ' + str(unmatched[:10])}  -> {'OK' if ok_i else 'FAIL'}")
        print(f"  (ii) |correct|+|removed| = {len(correct) + len(removed)}  vs reference_notes minus ties = {n_ref} "
              f"(ref cls correct={n_ref_correct}, missed={n_ref_missed}; reference_audio.mid notes={len(ref_mid)})"
              f"  -> {'OK' if ok_ii else 'FAIL'}")
    print("\nALL OK" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
