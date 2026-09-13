#!/usr/bin/env python
"""eval_bridge.py -- evaluate official Polytune / LadderSym predictions on ALIGN bundles.

Runs with the pinned baseline venvs (numpy 1.26, pretty_midi 0.2.9/0.2.10, mir_eval 0.7)
and with the project venv (numpy 2.x, pretty_midi 0.2.11, mir_eval 0.8.2).  Dependencies:
numpy, pretty_midi (+ its hard dependency mido), mir_eval, json.

Two evaluations are produced for a directory of predicted MIDIs
``<pred_dir>/<track_id>/mix.mid`` (the layout written by test_polytune.py /
test_laddersym.py under ``<hydra_run_dir>/<exp_tag_name>/``):

A. Note-level metrics (the papers' protocol).  Ground truth comes from the bundle's
   ``note_labels.json``:
       correct = performance_notes[cls == "correct"]
       extra   = performance_notes[cls == "extra"]   (sub wrong | inserted | repeat)
       missing = missed_notes[copy == 0]             (sub rest | wrong)
   Notes with ``tie_prev`` are merged into the preceding note (music21 renders a tie chain
   as one MIDI/audio note).  Pitches are ``sounding_pitch``.  Each class is scored with
   ``mir_eval.transcription.precision_recall_f1_overlap(onset_tolerance=0.05,
   offset_ratio=None)`` (onset + pitch, offsets ignored) and reported as
     * micro average (TP / n_pred / n_gt pooled over all pieces),
     * mean of per-piece P/R/F1 (mir_eval convention: a piece where either side is empty
       scores 0 -- this is what the official evaluate_errors.py averages), and
     * mean of per-piece P/R/F1 restricted to pieces with >= 1 GT note of the class,
   plus a class-agnostic "all" row (the official script's "Onset F1").

   Pitch convention.  ``--pitch-mode hz`` (default) passes pitches to mir_eval in Hz via
   ``mir_eval.util.midi_to_hz`` so the 50-cent tolerance means +-0.5 semitone.  The task
   brief suspected that the official evaluate_errors.py passes raw MIDI numbers; we checked
   both pinned envs: ``note_seq.sequences_lib.sequence_to_valued_intervals`` (note-seq 0.0.3
   and 0.0.5) ends with ``pretty_midi.note_number_to_hz(pitches)``, so the official script,
   as installed here, also compares in Hz -- ``--official-pitch-mode`` therefore selects
   ``hz``.  ``--pitch-mode raw-midi`` reproduces the raw-MIDI-number variant, in which
   1200*|log2(p_est/p_ref)| <= 50 accepts +-1 semitone for any pitch >= 35 and +-2
   semitones for pitch >= 69 (verified: ref 60 / est 61 -> F1 1.0; ref 69 / est 71 -> 1.0).

   Predicted class per note.  The decoder writes error classes as separate tracks in the
   order Extra, Missing, Correct with NO names (note_seq additionally emits a leading empty
   instrument for tempo/time-signature events), and absent classes are simply not written,
   which shifts the positions the official script zips against.  We resolve classes by
   (1) track name when it is one of extra / missing(missed, removed) / correct (a patched
   inference wrapper may add names), else (2) track order Extra, Missing, Correct, after
   dropping the note_seq empty leading instrument -- and we WARN whenever fewer or more than
   three tracks exist because the order is then ambiguous.  Track structure is read with
   mido (pretty_midi drops empty tracks on read, which is precisely the failure mode).
   ``--class-source order`` reproduces the official positional zip over pretty_midi
   instruments exactly; the ``official_replica`` block in the JSON is always computed
   that way for cross-checking against the official stdout.

B. Span-level 6-type metrics (our task).  Predicted notes are mapped to error spans and
   scored against ``labels.json`` with the errdet protocol (greedy 1-1 onset matching at
   50/100/200 ms; see the vendored functions below, credited to errdet/metrics.py).
   Mapping rules (all thresholds are CLI flags):
     1. a predicted Missed note and a predicted Extra note whose onsets are within
        --tau-wrong (0.10 s) and that are each other's nearest partner -> one wrong_note
        span [extra.onset, extra.offset];
     2. every remaining Missed -> missed_note [onset, offset];
     3. a run of >= --rep-min-run (3) consecutive remaining Extra notes with gaps
        < --rep-gap (0.35 s; optionally max'ed with --rep-gap-ioi-mult x median IOI) -- the
        gap is inter-onset by default (--rep-gap-mode onset, the specified rule) or, with
        --rep-gap-mode rest, the silence from the previous Extra note's offset to the next
        onset, which is tempo-independent and avoids fragmenting slow repeated passes --
        whose pitch sequence is a contiguous subsequence (LCS ratio >= --rep-lcs, 0.8) of
        the predicted notes (correct + extra) played in the preceding --rep-window-mult
        (2x) window -> one repetition span over the run.  "Consecutive" means no predicted
        Correct note lies between two notes of the run (--no-rep-break-on-correct disables
        this).  Up to --rep-trim (3) leading/trailing notes are shaved off a run when that
        strictly raises the LCS ratio (an inserted Extra note played right before a repeat
        would otherwise shift the span onset).  Multi-copy repeats are handled by also
        matching the run against periodic tilings of the material immediately preceding it;
        adjacent repetition spans closer than --rep-merge-gap (2.0 s) are merged.
     4. every remaining Extra -> extra_note [onset, offset];
     5. rhythm_error and intonation_error are never emitted (undetectable with three note
        classes); their rows are kept so the table shows the gap (recall 0).
   Recommended setting on ALIGN data: ``--rep-gap-mode rest --rep-gap 2.0`` (with the
   specified onset/0.35 s default, repeated passes in slow music fragment into runs whose
   2x windows no longer reach the material they repeat; on the 16-bundle oracle the
   repetition F1@50ms is 33% with the default and 71% with the recommended setting, and
   extra_note precision rises from 8% to 51% because fewer repeated notes leak out).
   ``--rep-window-min`` sets a floor (s) on the preceding window for short runs.
   Ground-truth spans default to ALL labels of the six types (first pass + repeated passes,
   as errdet does); ``--gold-first-pass-only`` drops labels whose comment marks a repeated
   pass (DataCreate melody.is_repeated_pass semantics).  Real-recording bundles without
   note_labels.json get part B only.
   Pieces: with --ids-from, every id of the split is scored and a missing prediction counts as
   EMPTY (zero recall, warned) -- unlike evaluate_errors.py, which skips such pieces; without it
   the ids come from <pred_dir>, bundles under --bundles that lack a prediction are reported in a
   warning and --all-bundles scores them as empty too.  An unreadable mix.mid is warned about and
   scored as empty instead of aborting the run.

Oracle helper:  ``--make-oracle <bundle_root> [<root2> ...] --out <dir>`` writes each
bundle's ground-truth classes as a 3-track MIDI (tracks named Extra, Missing, Correct;
``--oracle-unnamed`` omits names, ``--oracle-shift S`` shifts all times, ``--oracle-drop-class
C`` omits one track) for self-tests of the bridge.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pretty_midi
import mir_eval
import mir_eval.transcription
import mir_eval.util

try:  # mido is a hard dependency of pretty_midi, but keep the bridge usable without it
    import mido
except Exception:  # pragma: no cover
    mido = None

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
NOTE_CLASSES = ["extra", "missing", "correct"]          # official track order
CLASS_TRACK_NAMES = {"extra": "Extra", "missing": "Missing", "correct": "Correct"}
_NAME_SYNONYMS = {
    "extra": {"extra", "extra_notes", "extra notes", "extranotes", "extra_note"},
    "missing": {"missing", "missed", "removed", "missing_notes", "missed_notes", "removed_notes",
                "missing notes", "missed notes", "removed notes", "missing_note", "missed_note"},
    "correct": {"correct", "correct_notes", "correct notes", "correct_note"},
}
# errdet/data.py ERROR_TYPES (order preserved) + intonation_error as the 6th type
ERRDET_ERROR_TYPES = ["wrong_note", "missed_note", "extra_note", "rhythm_error", "repetition"]
SPAN_TYPES = ERRDET_ERROR_TYPES + ["intonation_error"]
UNDETECTABLE_TYPES = ("rhythm_error", "intonation_error")
TOLS = (0.05, 0.1, 0.2)
NOTE_ONSET_TOL = 0.05
PITCH_TOL_CENTS = 50.0
MIN_DUR = 0.01
VELOCITY = 90


# --------------------------------------------------------------------------------------
# span metric -- vendored verbatim from errdet/metrics.py
# (errdet/metrics.py imports errdet/data.py, which needs torch; only ERROR_TYPES became a
#  parameter so the 6-type list can be passed.  Logic is otherwise unchanged.)
# --------------------------------------------------------------------------------------
def match(pred, gt, tol):
    """pred/gt: lists of (start,end). Greedy 1-1 matching by onset distance."""
    used = set(); tp = 0
    for ps, pe in sorted(pred):
        best, bj = None, -1
        for j, (gs, ge) in enumerate(gt):
            if j in used: continue
            dd = abs(ps - gs)
            if dd <= tol and (best is None or dd < best):
                best, bj = dd, j
        if bj >= 0:
            used.add(bj); tp += 1
    return tp, len(pred), len(gt)


def prf(tp, np_, ng):
    p = tp / np_ if np_ else 0.0; r = tp / ng if ng else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(preds, gts, tols=(0.05, 0.1, 0.2), types=None):
    """preds/gts: dict sample_id -> list of (type, start, end). Returns nested dict."""
    if types is None:
        types = SPAN_TYPES
    res = {}
    for tol in tols:
        res[tol] = {}
        for typ in list(types) + ["any"]:
            tp = np_ = ng = 0
            for sid in gts:
                g = [(s, e) for t, s, e in gts[sid] if typ == "any" or t == typ]
                p = [(s, e) for t, s, e in preds.get(sid, []) if typ == "any" or t == typ]
                if typ == "any":   # collapse duplicates at identical spans
                    g = sorted(set(g)); p = sorted(set(p))
                a, b, c = match(p, g, tol); tp += a; np_ += b; ng += c
            res[tol][typ] = dict(zip(("P", "R", "F1"), prf(tp, np_, ng)), n_pred=np_, n_gt=ng)
    return res


def fmt(res):
    lines = []
    for tol, d in res.items():
        lines.append(f"-- onset tolerance {int(tol*1000)} ms")
        for typ, m in d.items():
            lines.append(f"  {typ:14s} P={m['P']*100:5.1f} R={m['R']*100:5.1f} F1={m['F1']*100:5.1f}  (pred {m['n_pred']}, gt {m['n_gt']})")
    return "\n".join(lines)
# ---------------------------------------------------------------- end of vendored code


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
class Warner:
    def __init__(self, quiet=False):
        self.messages = []
        self.quiet = quiet

    def __call__(self, msg):
        self.messages.append(msg)
        if not self.quiet:
            print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_repeated_pass(label):
    """Copy of DataCreate/src/datacreate/melody.py::is_repeated_pass."""
    comment = str(label.get("comment") or "")
    if "repeated pass" in comment:
        return True
    if "first pass" in comment:
        return False
    return bool(re.search(r"\(pass \d+\)", comment))


def _fix_span(onset, offset):
    onset = float(onset); offset = float(offset)
    if offset < onset + MIN_DUR:
        offset = onset + MIN_DUR
    return onset, offset


# --------------------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------------------
def gt_notes_from_note_labels(nl, warn=None, tag=""):
    """note_labels.json -> {class: [(onset, offset, sounding_pitch), ...]} (sorted)."""
    from prepare_dataset import classes_from_note_labels
    classes, _ = classes_from_note_labels(nl)
    return {key: sorted((max(0.0, n["start"]), max(n["end"], max(0.0, n["start"]) + 0.01), n["pitch"])
                        for n in classes["removed" if key == "missing" else key])
            for key in NOTE_CLASSES}


def gt_spans_from_labels(lb, first_pass_only=False, types=SPAN_TYPES):
    """labels.json -> [(type, start, end), ...] restricted to `types`."""
    labels = lb["labels"] if isinstance(lb, dict) else lb
    out = []
    for l in labels:
        if l.get("type") not in types:
            continue
        if first_pass_only and is_repeated_pass(l):
            continue
        out.append((l["type"], float(l["start_time"]), float(l["end_time"])))
    return out


# --------------------------------------------------------------------------------------
# predicted MIDI loading + class resolution
# --------------------------------------------------------------------------------------
def class_from_name(name):
    key = (name or "").strip().lower()
    if not key:
        return None
    for cls, syn in _NAME_SYNONYMS.items():
        if key in syn:
            return cls
    return None


def _midi_track_structure(path):
    """mido pass: ordered instrument tracks incl. EMPTY ones -> [{name, n_notes}]."""
    mf = mido.MidiFile(path)
    tracks = []
    for ti, tr in enumerate(mf.tracks):
        name = None; n_on = 0; prog = False
        for msg in tr:
            if msg.type == "track_name" and name is None:
                name = msg.name
            elif msg.type == "program_change":
                prog = True
            elif msg.type == "note_on" and msg.velocity > 0:
                n_on += 1
        if prog or n_on or name:          # skip the pure tempo/meta track
            tracks.append(dict(index=ti, name=name or "", n_notes=n_on))
    return tracks


def read_pred_midi_tracks(path, warn, use_mido=True):
    """Return (tracks, pm_instruments, n_dropped): tracks = ordered [{name, notes}] with
    empty tracks preserved when mido could be used; pm_instruments = pretty_midi's view
    (non-empty instruments only, as the official script sees them)."""
    pm = pretty_midi.PrettyMIDI(path)
    inst = []
    n_dropped = 0
    for ins in pm.instruments:
        notes = []
        for n in ins.notes:
            if n.end <= n.start:          # official note_seq path drops zero-length notes
                n_dropped += 1
                continue
            notes.append((float(n.start), float(n.end), int(n.pitch)))
        inst.append(dict(name=ins.name or "", notes=sorted(notes)))
    tracks = None
    if use_mido and mido is not None:
        try:
            struct = _midi_track_structure(path)
        except Exception as e:  # pragma: no cover
            warn(f"{path}: mido failed ({e}); using pretty_midi instruments only")
            struct = None
        if struct is not None:
            note_tracks = [t for t in struct if t["n_notes"] > 0]
            if len(note_tracks) == len(inst):
                tracks = []; k = 0
                for t in struct:
                    if t["n_notes"] > 0:
                        tracks.append(dict(name=t["name"] or inst[k]["name"], notes=inst[k]["notes"]))
                        k += 1
                    else:
                        tracks.append(dict(name=t["name"], notes=[]))
            else:
                warn(f"{path}: {len(note_tracks)} note-bearing MIDI tracks vs {len(inst)} pretty_midi "
                     f"instruments; using pretty_midi instruments only")
    if tracks is None:
        tracks = [dict(t) for t in inst]
    return tracks, inst, n_dropped


def assign_classes(tracks, class_source, warn, tag):
    """tracks: ordered [{name, notes}] -> ({class: notes}, info)."""
    by_class = {c: [] for c in NOTE_CLASSES}
    info = dict(n_tracks=len(tracks), track_names=[t["name"] for t in tracks],
                track_sizes=[len(t["notes"]) for t in tracks], source=None, ambiguous=False)
    if class_source in ("auto", "name"):
        named = OrderedDict(); unnamed = []
        for i, t in enumerate(tracks):
            c = class_from_name(t["name"])
            if c is None:
                unnamed.append(i)
            else:
                named.setdefault(c, []).append(i)
        if named:
            info["source"] = "name"
            for c, idxs in named.items():
                if len(idxs) > 1:
                    warn(f"{tag}: {len(idxs)} tracks named {c!r}; merging their notes")
                for i in idxs:
                    by_class[c].extend(tracks[i]["notes"])
            unnamed_with_notes = [i for i in unnamed if tracks[i]["notes"]]
            if unnamed_with_notes:
                remaining = [c for c in NOTE_CLASSES if c not in named]
                if len(unnamed_with_notes) == len(remaining):
                    warn(f"{tag}: {len(unnamed_with_notes)} unnamed note-bearing track(s) filled "
                         f"positionally into {remaining}")
                    for i, c in zip(unnamed_with_notes, remaining):
                        by_class[c].extend(tracks[i]["notes"])
                else:
                    n_ign = sum(len(tracks[i]["notes"]) for i in unnamed_with_notes)
                    warn(f"{tag}: ignoring {n_ign} notes in {len(unnamed_with_notes)} unnamed "
                         f"track(s) (named tracks cover {list(named)})")
                    info["ambiguous"] = True
            for c in by_class:
                by_class[c].sort()
            return by_class, info
        if class_source == "name":
            warn(f"{tag}: no recognised track names; all predicted notes ignored")
            info["source"] = "none"
            return by_class, info
    # ---- positional fallback (official semantics: Extra, Missing, Correct) ----
    info["source"] = "order"
    tr = list(tracks)
    if class_source == "auto" and len(tr) >= 2 and not tr[0]["notes"] and not tr[0]["name"]:
        tr = tr[1:]                       # note_seq's empty leading instrument (tempo holder)
        info["dropped_leading_empty"] = True
    if len(tr) != 3:
        info["ambiguous"] = True
        sizes = [len(t["notes"]) for t in tr]
        warn(f"{tag}: {len(tr)} unnamed track(s) (sizes {sizes}) instead of 3 -> class order "
             f"Extra/Missing/Correct is AMBIGUOUS; assigning positionally like the official script")
    for c, t in zip(NOTE_CLASSES, tr):
        by_class[c].extend(t["notes"])
    if len(tr) > 3:
        n_ign = sum(len(t["notes"]) for t in tr[3:])
        warn(f"{tag}: ignoring {n_ign} notes in {len(tr) - 3} surplus track(s)")
    for c in by_class:
        by_class[c].sort()
    return by_class, info


def load_pred(path, class_source, warn, tag):
    if path is None or not os.path.exists(path):
        return {c: [] for c in NOTE_CLASSES}, dict(n_tracks=0, source="missing", ambiguous=False), []
    try:
        tracks, inst, n_dropped = read_pred_midi_tracks(path, warn, use_mido=(class_source != "order"))
    except Exception as e:  # corrupt / non-MIDI file: the official script skips it; we score it as empty
        warn(f"{tag}: unreadable prediction {path} ({type(e).__name__}: {e}); counted as EMPTY prediction")
        return {c: [] for c in NOTE_CLASSES}, dict(n_tracks=0, source="unreadable", ambiguous=False), []
    if n_dropped:
        warn(f"{tag}: dropped {n_dropped} zero-length predicted note(s)")
    by_class, info = assign_classes(tracks, class_source, warn, tag)
    info["n_dropped_zero_length"] = n_dropped
    return by_class, info, inst


# --------------------------------------------------------------------------------------
# note-level metrics
# --------------------------------------------------------------------------------------
def _arrays(notes, pitch_mode):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[s, e] for s, e, _ in notes], dtype=float)
    p = np.array([p for _, _, p in notes], dtype=float)
    if pitch_mode == "hz":
        p = mir_eval.util.midi_to_hz(p)
    return iv, p


def note_prf(gt, pred, pitch_mode, onset_tol=NOTE_ONSET_TOL):
    """mir_eval onset+pitch P/R/F1 (offset ignored) plus the TP count used for pooling."""
    ri, rp = _arrays(gt, pitch_mode)
    ei, ep = _arrays(pred, pitch_mode)
    res = dict(n_gt=int(len(rp)), n_pred=int(len(ep)))
    if len(rp) == 0 or len(ep) == 0:      # mir_eval convention: empty side -> all zeros
        res.update(P=0.0, R=0.0, F1=0.0, tp=0)
        return res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P, R, F, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ri, rp, ei, ep, onset_tolerance=onset_tol, pitch_tolerance=PITCH_TOL_CENTS, offset_ratio=None)
        matching = mir_eval.transcription.match_notes(
            ri, rp, ei, ep, onset_tolerance=onset_tol, pitch_tolerance=PITCH_TOL_CENTS, offset_ratio=None)
    res.update(P=float(P), R=float(R), F1=float(F), tp=int(len(matching)))
    return res


def _mean_block(per_piece, key_filter=None):
    """per_piece: list of dict(P,R,F1,n_gt,n_pred) -> mean P/R/F1 (+ n_pieces)."""
    rows = [r for r in per_piece if key_filter is None or key_filter(r)]
    if not rows:
        return dict(P=None, R=None, F1=None, n_pieces=0)
    return dict(P=float(np.mean([r["P"] for r in rows])), R=float(np.mean([r["R"] for r in rows])),
                F1=float(np.mean([r["F1"] for r in rows])), n_pieces=len(rows))


def _micro_block(per_piece):
    tp = sum(r["tp"] for r in per_piece); npred = sum(r["n_pred"] for r in per_piece)
    ngt = sum(r["n_gt"] for r in per_piece)
    P, R, F = prf(tp, npred, ngt)
    return dict(P=P, R=R, F1=F, tp=tp, n_pred=npred, n_gt=ngt)


# --------------------------------------------------------------------------------------
# note -> span mapping
# --------------------------------------------------------------------------------------
def lcs_len(a, b):
    """LCS length, bit-parallel (Crochemore et al. 2001); exact, O(|b| * |a|/wordsize)."""
    if not a or not b:
        return 0
    if a == b:
        return len(a)
    masks = {}
    for i, x in enumerate(a):
        masks[x] = masks.get(x, 0) | (1 << i)
    full = (1 << len(a)) - 1
    V = full
    for y in b:
        U = V & masks.get(y, 0)
        V = ((V + U) | (V - U)) & full
    return len(a) - bin(V).count("1")


def best_repeat_ratio(run, prior):
    """Max LCS ratio between the run's pitch sequence and (a) any contiguous window of the
    preceding material of the same length, (b) periodic tilings of the material immediately
    preceding the run (multi-copy repeats).  Returns a value in [0, 1]."""
    n = len(run)
    if not prior or n == 0:
        return 0.0
    best = 0.0
    for i in range(0, len(prior) - n + 1):
        w = prior[i:i + n]
        if w == run:
            return 1.0
        best = max(best, lcs_len(run, w) / n)
    for m in range(1, min(len(prior), n) + 1):
        cell = prior[-m:]
        tiled = (cell * ((n + m - 1) // m))[:n]
        if tiled == run:
            return 1.0
        best = max(best, lcs_len(run, tiled) / n)
    return best


def notes_to_spans(pred, p, diag=None):
    """pred: {class: [(onset, offset, pitch)]} -> [(type, start, end), ...]."""
    extra = sorted(pred["extra"]); missing = sorted(pred["missing"]); correct = sorted(pred["correct"])
    spans = []
    used_e, used_m = set(), set()
    # (1) wrong note = mutual-nearest Missed/Extra pair within tau_wrong
    if extra and missing:
        eo = np.array([e[0] for e in extra]); mo = np.array([m[0] for m in missing])
        D = np.abs(eo[:, None] - mo[None, :])
        nn_e = D.argmin(axis=1); nn_m = D.argmin(axis=0)
        for i, j in enumerate(nn_e):
            if nn_m[j] == i and D[i, j] <= p["tau_wrong"]:
                spans.append(("wrong_note", extra[i][0], extra[i][1]))
                used_e.add(i); used_m.add(int(j))
    # (2) remaining Missed -> missed_note
    for j, m in enumerate(missing):
        if j not in used_m:
            spans.append(("missed_note", m[0], m[1]))
    # (3) repetition runs among the remaining Extra notes
    rem = [i for i in range(len(extra)) if i not in used_e]
    played = sorted(correct + extra)
    played_on = np.array([n[0] for n in played]) if played else np.zeros(0)
    played_pitch = [n[2] for n in played]
    gap_thr = p["rep_gap"]
    if p.get("rep_gap_ioi_mult", 0) > 0 and len(played) > 1:
        iois = np.diff(played_on); iois = iois[iois > 1e-6]
        if len(iois):
            gap_thr = max(gap_thr, p["rep_gap_ioi_mult"] * float(np.median(iois)))
    rest_mode = p.get("rep_gap_mode", "onset") == "rest"
    break_on_correct = p.get("rep_break_on_correct", True)
    rem_set = set(rem)
    seq = sorted([(n[0], n[1], "e", i) for i, n in enumerate(extra) if i in rem_set] +
                 [(n[0], n[1], "c", -1) for n in correct])
    runs, cur = [], []
    for s0, _, kind, i in seq:
        if kind == "c":                   # a Correct note between two Extras ends the run
            if break_on_correct and cur:
                runs.append(cur); cur = []
            continue
        if cur:
            prev = extra[cur[-1]]
            gap = s0 - (prev[1] if rest_mode else prev[0])
            if gap >= gap_thr:
                runs.append(cur); cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    rep_spans, consumed = [], set()
    run_log = []
    min_run = p["rep_min_run"]; k_trim = int(p.get("rep_trim", 0))
    for run in runs:
        if len(run) < min_run:
            continue
        t0 = extra[run[0]][0]; t1 = max(extra[i][1] for i in run); D = max(t1 - t0, 1e-3)
        lo = t0 - max(p["rep_window_mult"] * D, p.get("rep_window_min", 0.0))
        prior = [played_pitch[k] for k in range(len(played)) if lo <= played_on[k] < t0]
        R = [extra[i][2] for i in run]
        ratio = best_repeat_ratio(R, prior)
        lead = trail = 0
        if k_trim > 0 and ratio < 1.0:
            best = ratio
            for a in range(0, min(k_trim, len(run) - min_run) + 1):
                for b in range(0, min(k_trim, len(run) - min_run - a) + 1):
                    if a == 0 and b == 0:
                        continue
                    sub_R = R[a:len(R) - b]
                    sub_prior = prior + R[:a]          # trimmed leading notes become prior material
                    r = best_repeat_ratio(sub_R, sub_prior)
                    if r > best + 1e-9:
                        best, lead, trail = r, a, b
            ratio = best
        core = run[lead:len(run) - trail]
        t0c = extra[core[0]][0]; t1c = max(extra[i][1] for i in core)
        ok = ratio >= p["rep_lcs"]
        run_log.append(dict(start=round(t0c, 4), end=round(t1c, 4), n=len(core), trimmed=[lead, trail],
                            n_prior=len(prior), ratio=round(ratio, 3), accepted=ok))
        if ok:
            rep_spans.append([t0c, t1c]); consumed.update(core)
    rep_spans.sort()
    merged = []
    for s, e in rep_spans:
        if merged and s - merged[-1][1] < p["rep_merge_gap"]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    spans.extend(("repetition", s, e) for s, e in merged)
    # (4) remaining Extra -> extra_note
    for i in rem:
        if i not in consumed:
            spans.append(("extra_note", extra[i][0], extra[i][1]))
    if diag is not None:
        diag.update(gap_mode=("rest" if rest_mode else "onset"), gap_thr=gap_thr, n_runs=len(runs),
                    runs=run_log, n_wrong_pairs=len(used_e), n_rep_spans=len(merged))
    spans.sort(key=lambda t: (t[1], t[2], t[0]))
    return spans


# --------------------------------------------------------------------------------------
# id / bundle discovery
# --------------------------------------------------------------------------------------
def ids_from_split(split_json, split):
    d = load_json(split_json)
    fn = d["midi_filename"]; sp = d["split"]
    ids = []
    for k, path in fn.items():
        if sp.get(k) == split:
            base = os.path.basename(path).replace(".midi", "")   # official loader semantics
            ids.append((base, os.path.dirname(path)))
    return ids


def find_bundle(track_id, roots, set_name=None):
    cands = [track_id]
    if set_name and track_id.startswith(set_name + "_"):     # prepare_dataset collision prefix
        cands.append(track_id[len(set_name) + 1:])
    matches = []
    for r in roots:
        for c in cands:
            d = os.path.join(r, c)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "labels.json")):
                matches.append(os.path.realpath(d))
    matches = sorted(set(matches))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous bundle id {track_id}: {matches}")
    return matches[0] if matches else None



# --------------------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------------------
def run_eval(args):
    warn = Warner(quiet=args.quiet)
    roots = [os.path.abspath(r) for r in args.bundles]
    pred_dir = os.path.abspath(args.pred_dir)
    if args.ids_from:
        id_list = ids_from_split(args.ids_from, args.split)
    else:
        pred_ids = [d for d in sorted(os.listdir(pred_dir)) if os.path.isfile(os.path.join(pred_dir, d, "mix.mid"))]
        id_list = [(d, None) for d in pred_ids]
        covered = {os.path.realpath(b) for b in (find_bundle(d, roots) for d in pred_ids) if b}
        unpredicted = []
        for r in roots:
            for d in sorted(os.listdir(r)):
                bd = os.path.join(r, d)
                if os.path.exists(os.path.join(bd, "labels.json")) and os.path.realpath(bd) not in covered:
                    unpredicted.append(d)
        if unpredicted and args.all_bundles:
            id_list += [(d, None) for d in unpredicted if d not in pred_ids]
        elif unpredicted:
            warn(f"{len(unpredicted)} bundle(s) under {roots} have no prediction in {pred_dir} and are NOT "
                 f"evaluated (pass --all-bundles or --ids-from to count them as empty predictions): {unpredicted[:10]}")
    if not id_list:
        sys.exit("no track ids to evaluate (empty pred dir / split)")
    map_params = dict(tau_wrong=args.tau_wrong, rep_gap=args.rep_gap, rep_gap_mode=args.rep_gap_mode,
                      rep_gap_ioi_mult=args.rep_gap_ioi_mult, rep_break_on_correct=not args.no_rep_break_on_correct,
                      rep_trim=args.rep_trim, rep_window_min=args.rep_window_min,
                      rep_min_run=args.rep_min_run, rep_lcs=args.rep_lcs,
                      rep_window_mult=args.rep_window_mult, rep_merge_gap=args.rep_merge_gap)

    per_piece = OrderedDict()
    note_rows = {c: [] for c in NOTE_CLASSES + ["all"]}
    official_rows = {"Onset": [], "Track 0": [], "Track 1": [], "Track 2": []}
    span_preds, span_gts = OrderedDict(), OrderedDict()
    missing_pred, missing_bundle, no_note_labels, unreadable_pred = [], [], [], []
    pred_span_counts = {t: 0 for t in SPAN_TYPES}

    for tid, set_name in id_list:
        bdir = find_bundle(tid, roots, set_name)
        if bdir is None:
            raise FileNotFoundError(f"{tid}: no unambiguous bundle found under {roots}")
        ppath = os.path.join(pred_dir, tid, "mix.mid")
        if not os.path.exists(ppath):
            missing_pred.append(tid); warn(f"{tid}: no prediction at {ppath}; counted as EMPTY prediction")
            ppath = None
        pred, pinfo, pm_inst = load_pred(ppath, args.class_source, warn, tid)
        if pinfo.get("source") == "unreadable":
            unreadable_pred.append(tid)
        rec = OrderedDict(bundle=bdir, pred=pinfo, n_pred_notes={c: len(pred[c]) for c in NOTE_CLASSES})
        # ---- A: note level ----
        nlp = os.path.join(bdir, "note_labels.json")
        if os.path.exists(nlp) and not args.skip_note_level:
            nl = load_json(nlp)
            chk = nl.get("check", {})
            if chk and not (chk.get("perf", {}).get("ok", True) and chk.get("ref", {}).get("ok", True)):
                warn(f"{tid}: note_labels.json check flags a MIDI/label mismatch ({chk})")
            gt = gt_notes_from_note_labels(nl, warn, tid)
            rec["n_gt_notes"] = {c: len(gt[c]) for c in NOTE_CLASSES}
            rec["note_level"] = {}
            for c in NOTE_CLASSES:
                r = note_prf(gt[c], pred[c], args.pitch_mode)
                note_rows[c].append(r); rec["note_level"][c] = r
            r = note_prf(sum((gt[c] for c in NOTE_CLASSES), []), sum((pred[c] for c in NOTE_CLASSES), []), args.pitch_mode)
            note_rows["all"].append(r); rec["note_level"]["all"] = r
            # official replica: positional zip over pretty_midi instruments, Hz pitches; the official
            # script SKIPS pieces whose estimate is missing or unreadable, so they are left out here
            # (they still count as empty predictions in the micro / per-piece blocks above)
            if pinfo.get("source") not in ("missing", "unreadable"):
                official_rows["Onset"].append(note_prf(sum((gt[c] for c in NOTE_CLASSES), []),
                                                       sum((t["notes"] for t in pm_inst), []), "hz"))
                for k, (c, t) in enumerate(zip(NOTE_CLASSES, pm_inst)):
                    official_rows[f"Track {k}"].append(note_prf(gt[c], t["notes"], "hz"))
        else:
            if not os.path.exists(nlp):
                no_note_labels.append(tid)
        # ---- B: span level ----
        if not args.skip_span_level:
            lb = load_json(os.path.join(bdir, "labels.json"))
            gts = gt_spans_from_labels(lb, first_pass_only=args.gold_first_pass_only)
            diag = {}
            preds = notes_to_spans(pred, map_params, diag)
            span_gts[tid] = gts; span_preds[tid] = preds
            for t, _, _ in preds:
                pred_span_counts[t] += 1
            rec["span_level"] = dict(n_gt_spans=len(gts), n_pred_spans=len(preds), mapping=diag,
                                     pred_spans=[[t, round(s, 4), round(e, 4)] for t, s, e in preds])
        per_piece[tid] = rec

    out = OrderedDict()
    out["config"] = OrderedDict(pred_dir=pred_dir, bundles=roots, ids_from=args.ids_from, split=args.split,
                                pitch_mode=args.pitch_mode, class_source=args.class_source,
                                gold=("first_pass_only" if args.gold_first_pass_only else "all_labels"),
                                note_onset_tolerance=NOTE_ONSET_TOL, pitch_tolerance_cents=PITCH_TOL_CENTS,
                                span_tolerances=list(TOLS), mapping=map_params)
    out["n_pieces"] = len(per_piece)
    out["n_pieces_with_note_labels"] = len(note_rows["all"])
    out["pieces_without_note_labels"] = no_note_labels
    out["pieces_missing_prediction"] = missing_pred
    out["pieces_unreadable_prediction"] = unreadable_pred
    out["pieces_missing_bundle"] = missing_bundle
    out["n_ambiguous_track_order"] = sum(1 for r in per_piece.values() if r["pred"].get("ambiguous"))
    out["class_source_counts"] = dict(_count(r["pred"].get("source") for r in per_piece.values()))
    if note_rows["all"]:
        nl_out = OrderedDict(pitch_mode=args.pitch_mode, micro=OrderedDict(), per_piece_mean=OrderedDict(),
                             per_piece_mean_gt_nonempty=OrderedDict())
        for c in NOTE_CLASSES + ["all"]:
            nl_out["micro"][c] = _micro_block(note_rows[c])
            nl_out["per_piece_mean"][c] = _mean_block(note_rows[c])
            nl_out["per_piece_mean_gt_nonempty"][c] = _mean_block(note_rows[c], lambda r: r["n_gt"] > 0)
        nl_out["official_replica"] = OrderedDict(
            note="mean over pieces of per-piece mir_eval P/R/F1 with predicted classes taken by track "
                 "POSITION over pretty_midi instruments (Extra, Missing, Correct), Hz pitches -- what "
                 "evaluate_errors.py prints as 'Onset F1' and 'Track i F1'; a track key is only averaged "
                 "over pieces where that track exists; pieces with a missing/unreadable prediction are "
                 "skipped here (as the official script does) but scored as empty in the blocks above")
        for k, rows in official_rows.items():
            nl_out["official_replica"][k] = _mean_block(rows)
        out["note_level"] = nl_out
    else:
        out["note_level"] = None
    if span_gts:
        res = evaluate(span_preds, span_gts, tols=TOLS, types=SPAN_TYPES)
        out["span_level"] = OrderedDict(
            gold=out["config"]["gold"], n_pieces=len(span_gts),
            pred_span_counts=pred_span_counts,
            gt_span_counts=dict(_count(t for g in span_gts.values() for t, _, _ in g)),
            never_predicted=list(UNDETECTABLE_TYPES),
            results=OrderedDict((str(tol), res[tol]) for tol in TOLS),
            errdet_fmt=fmt(res))
    else:
        out["span_level"] = None
    out["per_piece"] = per_piece
    out["warnings"] = warn.messages
    return out


def _count(it):
    d = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return sorted(d.items())


# --------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------
def _pct(x):
    return "   -  " if x is None else f"{100 * x:6.1f}"


def print_report(out, verbose=False):
    print(f"\n== eval_bridge: {out['n_pieces']} piece(s)  [{out['n_pieces_with_note_labels']} with note_labels.json, "
          f"{len(out['pieces_missing_prediction'])} missing / {len(out.get('pieces_unreadable_prediction', []))} unreadable prediction(s), "
          f"{out['n_ambiguous_track_order']} with ambiguous track order]  class source: {out['class_source_counts']}")
    nl = out.get("note_level")
    if nl:
        print(f"\n-- Note level (onset {int(NOTE_ONSET_TOL * 1000)} ms, pitch {int(PITCH_TOL_CENTS)} cents, "
              f"pitch-mode={nl['pitch_mode']}); micro = pooled over pieces, mean = per-piece mean")
        print(f"  {'class':9s} {'micro-P':>7s} {'micro-R':>7s} {'micro-F1':>8s} | {'meanF1':>7s} {'meanF1(gt>0)':>12s} | "
              f"{'TP':>6s} {'n_pred':>7s} {'n_gt':>6s}")
        for c in NOTE_CLASSES + ["all"]:
            m = nl["micro"][c]; a = nl["per_piece_mean"][c]; b = nl["per_piece_mean_gt_nonempty"][c]
            print(f"  {c:9s} {_pct(m['P']):>7s} {_pct(m['R']):>7s} {_pct(m['F1']):>8s} | {_pct(a['F1']):>7s} "
                  f"{_pct(b['F1']):>12s} | {m['tp']:6d} {m['n_pred']:7d} {m['n_gt']:6d}")
        rep = nl["official_replica"]
        print("  official-replica (positional, per-piece mean F1): " +
              ", ".join(f"{k}={_pct(rep[k]['F1']).strip()}% (n={rep[k]['n_pieces']})" for k in rep if k != "note"))
    else:
        print("\n-- Note level: n/a (no bundle with note_labels.json)")
    sl = out.get("span_level")
    if sl:
        r = sl["results"]
        print(f"\n-- Span level (errdet protocol, gold={sl['gold']}, {sl['n_pieces']} pieces); "
              f"F1 in %; rhythm_error/intonation_error are never predicted")
        print(f"  {'type':17s} {'F1@50':>6s} {'F1@100':>7s} {'F1@200':>7s} | {'P@50':>6s} {'R@50':>6s} | {'n_pred':>6s} {'n_gt':>5s}")
        for t in SPAN_TYPES + ["any"]:
            a, b, c = r["0.05"][t], r["0.1"][t], r["0.2"][t]
            print(f"  {t:17s} {_pct(a['F1']):>6s} {_pct(b['F1']):>7s} {_pct(c['F1']):>7s} | {_pct(a['P']):>6s} "
                  f"{_pct(a['R']):>6s} | {a['n_pred']:6d} {a['n_gt']:5d}")
        if verbose:
            print(sl["errdet_fmt"])
    else:
        print("\n-- Span level: n/a")
    if out["warnings"]:
        print(f"\n{len(out['warnings'])} warning(s) (see JSON 'warnings')")


# --------------------------------------------------------------------------------------
# oracle helper
# --------------------------------------------------------------------------------------
def make_oracle(bundle_roots, out_dir, shift=0.0, unnamed=False, drop_class=None, quiet=False):
    warn = Warner(quiet=quiet)
    n = 0
    for root in bundle_roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            nlp = os.path.join(d, "note_labels.json")
            if not os.path.isdir(d) or not os.path.exists(nlp):
                continue
            tid = os.path.basename(d.rstrip("/"))
            gt = gt_notes_from_note_labels(load_json(nlp), warn, tid)
            pm = pretty_midi.PrettyMIDI(resolution=960)
            for c in NOTE_CLASSES:
                if c == drop_class:
                    continue
                ins = pretty_midi.Instrument(program=0, is_drum=False, name="" if unnamed else CLASS_TRACK_NAMES[c])
                for s, e, p in gt[c]:
                    s2 = s + shift; e2 = e + shift
                    if s2 < 0:
                        warn(f"{tid}: note shifted to negative time; clamped to 0"); s2 = 0.0
                    if e2 < s2 + MIN_DUR:
                        e2 = s2 + MIN_DUR
                    ins.notes.append(pretty_midi.Note(velocity=VELOCITY, pitch=int(p), start=s2, end=e2))
                pm.instruments.append(ins)
            od = os.path.join(out_dir, tid); os.makedirs(od, exist_ok=True)
            pm.write(os.path.join(od, "mix.mid"))
            n += 1
    print(f"wrote {n} oracle MIDI(s) to {out_dir} (shift={shift}, unnamed={unnamed}, drop_class={drop_class})")
    return n


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", help="directory with <track_id>/mix.mid predictions")
    ap.add_argument("--bundles", action="append", default=[], help="ALIGN bundle root (repeatable)")
    ap.add_argument("--ids-from", help="<DATA_ROOT>/split.json; evaluate the ids of --split")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--out", help="results JSON path (eval mode) or output dir (--make-oracle)")
    ap.add_argument("--pitch-mode", choices=["hz", "raw-midi"], default="hz",
                    help="hz: pitches -> Hz before mir_eval (default; also what the installed official "
                         "script does).  raw-midi: pass MIDI numbers as 'Hz' (accepts +-1 semitone).")
    ap.add_argument("--official-pitch-mode", action="store_true",
                    help="use the official evaluate_errors.py pitch convention -- verified to be Hz via "
                         "note_seq.sequence_to_valued_intervals in both pinned envs, so this selects 'hz'")
    ap.add_argument("--class-source", choices=["auto", "name", "order"], default="auto",
                    help="auto: track names, else positional with warning; order: official positional zip")
    ap.add_argument("--gold-first-pass-only", action="store_true",
                    help="drop labels.json entries marked '(repeated pass)' / '(pass N)'")
    ap.add_argument("--tau-wrong", type=float, default=0.10, help="Missed+Extra pairing window (s)")
    ap.add_argument("--rep-gap", type=float, default=0.35, help="max gap between consecutive Extra notes of a run (s)")
    ap.add_argument("--rep-gap-mode", choices=["onset", "rest"], default="onset",
                    help="onset: inter-onset gap (specified rule); rest: previous offset -> next onset "
                         "(tempo-independent; recommended, see module docstring)")
    ap.add_argument("--rep-gap-ioi-mult", type=float, default=0.0,
                    help="if > 0, run gap = max(--rep-gap, mult x median IOI of predicted notes)")
    ap.add_argument("--rep-min-run", type=int, default=3, help="minimum Extra notes in a repetition run")
    ap.add_argument("--no-rep-break-on-correct", action="store_true",
                    help="do not end a run at an intervening predicted Correct note")
    ap.add_argument("--rep-trim", type=int, default=3,
                    help="max leading/trailing run notes to drop when that raises the LCS ratio (0 = off)")
    ap.add_argument("--rep-lcs", type=float, default=0.8, help="min LCS ratio vs preceding material")
    ap.add_argument("--rep-window-mult", type=float, default=2.0, help="preceding window = mult x run duration")
    ap.add_argument("--rep-window-min", type=float, default=0.0, help="minimum preceding window length (s)")
    ap.add_argument("--rep-merge-gap", type=float, default=2.0, help="merge repetition spans closer than this (s)")
    ap.add_argument("--all-bundles", action="store_true",
                    help="without --ids-from: also evaluate every bundle under --bundles that has no prediction "
                         "(counted as an EMPTY prediction); by default such bundles are only reported in a warning")
    ap.add_argument("--skip-note-level", action="store_true")
    ap.add_argument("--skip-span-level", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="also print the errdet-style span table")
    ap.add_argument("--quiet", action="store_true", help="do not echo warnings to stderr")
    ap.add_argument("--make-oracle", nargs="+", metavar="BUNDLE_ROOT",
                    help="write ground-truth 3-track MIDIs for these bundle roots into --out and exit")
    ap.add_argument("--oracle-shift", type=float, default=0.0, help="shift oracle note times by S seconds")
    ap.add_argument("--oracle-unnamed", action="store_true", help="oracle tracks without names (like the decoder)")
    ap.add_argument("--oracle-drop-class", choices=NOTE_CLASSES, default=None, help="omit this oracle track")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.make_oracle:
        if not args.out:
            sys.exit("--make-oracle needs --out <dir>")
        make_oracle(args.make_oracle, os.path.abspath(args.out), shift=args.oracle_shift,
                    unnamed=args.oracle_unnamed, drop_class=args.oracle_drop_class, quiet=args.quiet)
        return 0
    if not args.pred_dir or not args.bundles:
        sys.exit("eval mode needs --pred-dir and at least one --bundles (or use --make-oracle)")
    if args.official_pitch_mode:
        args.pitch_mode = "hz"
        print("note: --official-pitch-mode -> pitch-mode=hz (the installed evaluate_errors.py converts MIDI "
              "numbers to Hz inside note_seq.sequence_to_valued_intervals; use --pitch-mode raw-midi for the "
              "raw-MIDI-number variant)", file=sys.stderr)
    out = run_eval(args)
    print_report(out, verbose=args.verbose)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
