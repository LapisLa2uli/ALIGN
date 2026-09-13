#!/usr/bin/env python
"""Load a converted DATA_ROOT with the *official* Polytune or LadderSym Dataset
class and print shapes + decoded target tokens.

Must be run from the repo root with the matching venv, e.g.::

    cd $B/Polytune
    MPLBACKEND=Agg ../envs/polytune/bin/python ../common/verify_loaders.py \
        --flavor polytune --root ../data/smoke_align --split train

    cd $B/LadderSym
    MPLBACKEND=Agg ../envs/laddersym/bin/python ../common/verify_loaders.py \
        --flavor laddersym --root ../data/smoke_align --split train

Besides printing, it checks that the error-class tokens in the first 2.048 s
segment of every requested item match the notes in the label MIDIs
(error_0 == 1135 Extra, error_1 == 1136 Missed, error_2 == 1137 Correct after
the +3 special-token offset).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.getcwd())  # import the repo's dataset/ + contrib/ packages

import numpy as np  # noqa: E402

import pretty_midi  # noqa: E402
import torch  # noqa: E402

from dataset.dataset_2_random import Dataset, collate_fn  # noqa: E402

NUM_SPECIAL = 3  # pad, eos, unk  (vocab.num_special_tokens())
EOS = 1
SEG_SECONDS = 2.048
CLASS_NAMES = {0: "Extra(1135)", 1: "Missed(1136)", 2: "Correct(1137)"}
CLASS_DIR = {0: "extra_notes", 1: "removed_notes", 2: "correct_notes"}


def decode(ds, row: torch.Tensor) -> list[str]:
    names = []
    for t in row.tolist():
        if t == EOS:
            names.append("<eos>")
            break
        if t < 0:
            break
        names.append(ds.get_token_name(t - NUM_SPECIAL))
    return names


def onsets_from_tokens(names: list[str]) -> Counter:
    """Return Counter{(error_class, pitch)} of note ONSETS.

    error_class and velocity are *state* tokens: with is_randomize_tokens=False the
    loader's run-length encoder drops a state token when it repeats the current
    state, so a ``pitch`` may follow ``error_k`` directly and inherit the last
    velocity.  Everything up to and including the leading ``tie`` token is the
    active-notes preamble (error/pitch pairs, no velocity) and is skipped.
    """
    out: Counter = Counter()
    # NOTE: do not skip the preamble -- its error_k tokens set the RLE state, so the
    # first error token after `tie` may have been dropped as redundant.  Preamble
    # pitches never count as onsets because no velocity token precedes them.
    cur_err = None
    cur_vel = 0
    for n in names:
        if n.startswith("error_"):
            cur_err = int(n.split("_")[1])
        elif n.startswith("velocity_"):
            cur_vel = int(n.split("_")[1])
        elif n.startswith("pitch_"):
            if cur_vel > 0:
                out[(cur_err, int(n.split("_")[1]))] += 1
        elif n == "<eos>":
            break
    return out


FPS = 125          # 16 kHz / hop 128
STEPS_PER_SEC = 100
MEL_LENGTH = 256


def expected_onsets(root: str, track_id: str, frame0: int) -> Counter:
    """Onsets the encoder puts into the segment of frames [frame0, frame0+256).

    run_length_encoding.encode_and_index_events quantises event times to
    ``round(t * 100)`` steps and a segment starting at frame j0 spans steps
    ``floor(0.8*j0) <= s < floor(0.8*(j0+256))`` (verified empirically; note the
    step at the right edge, 2.04 s for j0=0, is already excluded).
    """
    import math

    step_lo = math.floor(frame0 * STEPS_PER_SEC / FPS)
    step_hi = math.floor((frame0 + MEL_LENGTH) * STEPS_PER_SEC / FPS)
    out: Counter = Counter()
    for cls, sub in CLASS_DIR.items():
        p = os.path.join(root, "label", sub, track_id, "MIDI", f"{track_id}.mid")
        pm = pretty_midi.PrettyMIDI(p)
        for inst in pm.instruments:
            for n in inst.notes:
                s = round(n.start * STEPS_PER_SEC)
                if step_lo <= s < step_hi:
                    out[(cls, n.pitch)] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", choices=["polytune", "laddersym"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--items", type=int, default=2, help="how many dataset items to pull")
    ap.add_argument("--print-tokens", type=int, default=80)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    kw = dict(
        root_dir=root,
        split_json_path=os.path.join(root, "split.audited.json" if os.path.isfile(os.path.join(root, "split.audited.json")) else "split.json"),
        split=args.split,
        mel_length=256,
        event_length=1024,
        num_rows_per_batch=2,
        split_frame_length=2000,
        is_deterministic=True,
        is_randomize_tokens=False,
        is_random_alignment_shift_augmentation=False,
        shuffle=False,
    )
    if args.flavor == "laddersym":
        kw["use_prompt"] = True
        kw["audio_filename"] = "mix.wav"
    ds = Dataset(**kw)
    print(f"[{args.flavor}] split={args.split} len(ds)={len(ds)}")
    print("error_class codec range:", ds.codec.event_type_range("error_class"), "(+3 -> 1135..1137 in targets)")

    n_fail = 0
    for idx in range(min(args.items, len(ds))):
        row = ds.df[idx]
        track_id = os.path.normpath(row["extra_notes_midi"]).split(os.sep)[-3]
        item = ds[idx]
        print(f"\n--- item {idx}: {track_id}  ({len(item)}-tuple)")
        for name, t in zip(
            ["mistake_inputs", "score_inputs", "targets", "prompts", "prompts_attention_mask"], item
        ):
            print(f"  {name:<22} {tuple(t.shape)} {t.dtype}")
        targets = item[2]
        for r in range(targets.shape[0]):
            names = decode(ds, targets[r])
            tok_counts = Counter(int(t) for t in targets[r].tolist() if t in (1135, 1136, 1137))
            # deterministic: chunk r starts at frame 2000 r (split_frame_length) and
            # covers performance seconds [16 r, 16 r + 2.048)
            frame0 = 2000 * r
            t0 = frame0 / FPS
            got = onsets_from_tokens(names)
            exp = expected_onsets(root, track_id, frame0)
            ok = got == exp
            n_fail += int(not ok)
            print(f"  row {r}: window [{t0:.3f},{t0 + SEG_SECONDS:.3f}) s  n_tokens={len(names)}  "
                  f"error-token ids seen={dict(sorted(tok_counts.items()))}")
            print(f"    onsets from tokens : {sorted((CLASS_NAMES[c], p, k) for (c, p), k in got.items())}")
            print(f"    onsets in label MIDIs: {sorted((CLASS_NAMES[c], p, k) for (c, p), k in exp.items())}")
            print(f"    MATCH: {ok}")
            if r == 0:
                print(f"    tokens[:{args.print_tokens}]: {names[:args.print_tokens]}")
        if len(item) == 5:
            prompt_names = decode(ds, item[3][0])
            print(f"  prompt row 0: n={len(prompt_names)} tokens[:40]: {prompt_names[:40]}")
            print(f"  prompt mask row 0 sum: {int(item[4][0].sum())}")

    batch = collate_fn([ds[i] for i in range(min(2, len(ds)))])
    print("\ncollate_fn ->", [tuple(b.shape) for b in batch])
    print(f"\nRESULT: {'OK' if n_fail == 0 else f'{n_fail} segment(s) mismatched'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
