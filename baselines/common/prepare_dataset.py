#!/usr/bin/env python
"""Convert ALIGN bundles into the MAESTRO-E style layout read by the official
Polytune (AAAI 2025) and LadderSym (ICLR 2026) dataset loaders.

Input: one or more bundle roots (``--set NAME=<bundle_root>``).  A bundle is a
directory ``<root>/<track_id>/`` holding ``performance_audio.wav``,
``reference_audio.wav``, ``reference_audio.mid``, ``labels.json`` and (unless
``--real-test``) ``note_labels.json``.

Output (``--out DATA_ROOT``)::

    DATA_ROOT/label/extra_notes/<id>/MIDI/<id>.mid     Extra   -> error_class 1
    DATA_ROOT/label/removed_notes/<id>/MIDI/<id>.mid   Missed  -> error_class 2
    DATA_ROOT/label/correct_notes/<id>/MIDI/<id>.mid   Correct -> error_class 3
    DATA_ROOT/mistake/<id>/mix.wav                     performance audio, 16 kHz mono PCM_16
    DATA_ROOT/score/<id>/mix.wav                       reference audio, 16 kHz mono PCM_16
    DATA_ROOT/score/<id>/mix.mid                       reference MIDI (LadderSym decoder prompt)
    DATA_ROOT/split.json                               {"midi_filename": {...}, "split": {...}}
    DATA_ROOT/manifest.json                            per-track bookkeeping + summary

Class semantics (note_labels.json -> Polytune/LadderSym classes):

    correct = performance_notes[cls == "correct"]
    extra   = performance_notes[cls == "extra"]      (sub wrong | inserted | repeat)
    removed = missed_notes[copy == 0]                (sub rest | wrong)

All label notes use ``sounding_pitch`` and live on the performance timeline.
Notes with ``tie_prev`` are merged into the preceding note (music21 renders a
tie chain as one MIDI/audio note).  Velocity is fixed to 90, minimum duration
0.01 s, program 71 (clarinet), 480 ticks per quarter.

Audio: the two ``mix.wav`` of a track are zero-padded to the SAME length
(``--no-equalize-audio`` disables this).  Both official Dataset classes assume
score and mistake audio share one timeline *and* length: ``_split_frame``
enumerates the 16 s training chunks by the *score* length (a shorter reference
silently drops the tail of the performance), and LadderSym slices its prompt
indices without padding (a reference shorter than 16 s raises IndexError for
random chunk starts).  The official inference handler pads both to the longer
one anyway, so equal lengths are the intended layout.

Bundles whose ``note_labels.json`` ``check.perf.ok``/``check.ref.ok`` is false
(labels inconsistent with the exported MIDI) are converted but left out of
``split.json`` unless ``--keep-failed-check`` is given.

Only numpy / scipy / soundfile / pretty_midi are needed; run it with the
Polytune venv::

    $B/envs/polytune/bin/python \
        $B/common/prepare_dataset.py \
        --out DATA_ROOT --set multi=/path/to/bundles [--set raw=/path/to/other] ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter, OrderedDict, defaultdict
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

TARGET_SR = 16000
PROGRAM = 71  # GM clarinet
VELOCITY = 90
MIN_DUR = 0.01
RESOLUTION = 480
INITIAL_TEMPO = 120.0

SPLITS = ("train", "validation", "test")
CLASS_DIRS = OrderedDict(
    [("extra", "extra_notes"), ("removed", "removed_notes"), ("correct", "correct_notes")]
)
REQUIRED_FILES = (
    "labels.json",
    "performance_audio.wav",
    "reference_audio.wav",
    "reference_audio.mid",
)


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #
def output_paths(out_root: Path, track_id: str) -> dict:
    return {
        "extra": out_root / "label" / "extra_notes" / track_id / "MIDI" / f"{track_id}.mid",
        "removed": out_root / "label" / "removed_notes" / track_id / "MIDI" / f"{track_id}.mid",
        "correct": out_root / "label" / "correct_notes" / track_id / "MIDI" / f"{track_id}.mid",
        "mistake_wav": out_root / "mistake" / track_id / "mix.wav",
        "score_wav": out_root / "score" / track_id / "mix.wav",
        "score_mid": out_root / "score" / track_id / "mix.mid",
    }


# --------------------------------------------------------------------------- #
# Bundle discovery
# --------------------------------------------------------------------------- #
def is_bundle(d: Path, real_test: bool) -> bool:
    if not d.is_dir():
        return False
    for f in REQUIRED_FILES:
        if not (d / f).is_file():
            return False
    if not real_test and not (d / "note_labels.json").is_file():
        raise ValueError(f"{d}: note_labels.json is required for supervised baselines. "
                         "Current ALIGN note_map.json is a different schema and does not "
                         "supply all missing-note times on the performance timeline. "
                         "Use the frozen labelled bundles or --real-test for inference only.")
    return True


def find_bundles(root: Path, real_test: bool, limit: int | None) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"bundle root does not exist: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    bundles = [d for d in dirs if is_bundle(d, real_test)]
    skipped = len(dirs) - len(bundles)
    if skipped:
        print(f"[{root}] skipped {skipped} dirs without the required files", flush=True)
    if limit is not None:
        bundles = bundles[:limit]
    return bundles


def validate_track_id(track_id: str) -> None:
    # ".mid"/".wav": the loaders build sibling paths with str.replace(".mid", ".wav")
    # (Polytune) and .replace(".wav", ".mid") (LadderSym) on the FULL path; "mix.":
    # audio is discovered with glob("mix.*"); ".midi": split.json ids are
    # basename.replace(".midi", "").
    bad = [".mid", ".wav", "mix.", os.sep, " "]
    for b in bad:
        if b in track_id:
            raise SystemExit(f"track id {track_id!r} contains {b!r}; the loaders key on it")


def validate_out_root(out_root: Path) -> None:
    for b in (".mid", ".wav"):
        if b in str(out_root):
            raise SystemExit(
                f"--out {out_root} contains {b!r}: the official loaders derive audio/MIDI "
                "paths with str.replace('.mid', '.wav') / ('.wav', '.mid') on the full path"
            )


def _tmp_path(path: Path) -> Path:
    """Temp name for atomic writes.  Hidden (leading dot) so the loaders' globs
    (``mix.*``, ``*.mid``) can never pick up a half-written file."""
    return path.parent / f".{path.name}.tmp"


def remove_stale_tmp(paths: dict) -> int:
    """Delete temp files an interrupted earlier run left next to the outputs.

    Covers the current hidden names and the ``mix.tmp.wav`` / ``mix.tmp.mid`` /
    ``<id>.tmp.mid`` names an earlier version of this script used -- those DO
    match the loaders' ``mix.*`` / ``*.mid`` globs and would shadow the real file.
    """
    n = 0
    for p in paths.values():
        for c in (_tmp_path(p), p.with_suffix(".tmp" + p.suffix)):
            if c.is_file():
                c.unlink()
                n += 1
    return n


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def load_audio_16k(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if sr != TARGET_SR:
        ratio = Fraction(TARGET_SR, int(sr))  # 22050 -> 16000 == 320/441
        y = resample_poly(y, ratio.numerator, ratio.denominator)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def write_wav16(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    sf.write(str(tmp), y, TARGET_SR, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def merge_ties(entries: list[dict], stats: Counter, tag: str) -> list[dict]:
    """Collapse tie chains: an entry with tie_prev extends the preceding entry.

    Entries are processed in list order (== ``index`` order for the JSON lists).
    Returns dicts with keys start, end, pitch, cls, sub, copy.
    """
    out: list[dict] = []
    for e in entries:
        pitch = int(e["sounding_pitch"])
        rec = {
            "start": float(e["onset"]),
            "end": float(e["offset"]),
            "pitch": pitch,
            "cls": e.get("cls"),
            "sub": e.get("sub"),
            "copy": int(e.get("copy", 0)),
        }
        if e.get("tie_prev") and out:
            prev = out[-1]
            prev["end"] = max(prev["end"], rec["end"])
            stats[f"{tag}_tie_merged"] += 1
            if prev["pitch"] != pitch:
                raise ValueError(f"{tag}: tied notes have different pitches")
            if prev["cls"] != rec["cls"]:
                stats[f"{tag}_tie_cls_conflict"] += 1
            continue
        out.append(rec)
    return out


def classes_from_note_labels(nl: dict) -> tuple[dict, Counter]:
    """Return {"correct": [...], "extra": [...], "removed": [...]} note lists."""
    if nl.get("schema_version") != "1.0" or not all(
        key in nl for key in ("performance_notes", "missed_notes", "reference_notes")
    ):
        raise ValueError("Expected note_labels.json schema 1.0, not note_map.json")
    stats: Counter = Counter()
    perf = merge_ties(nl.get("performance_notes", []), stats, "perf")
    missed = merge_ties(nl.get("missed_notes", []), stats, "missed")
    ref_entries = nl.get("reference_notes", [])
    stats["ref_entries"] = len(ref_entries)
    stats["ref_tie_prev"] = sum(1 for r in ref_entries if r.get("tie_prev"))

    classes = {"correct": [], "extra": [], "removed": []}
    for n in perf:
        if n["cls"] == "correct":
            classes["correct"].append(n)
        elif n["cls"] == "extra":
            classes["extra"].append(n)
            stats[f"extra_sub_{n['sub']}"] += 1
        else:
            raise ValueError(f"Unknown performance class: {n['cls']}")
    for n in missed:
        if n["copy"] == 0:
            classes["removed"].append(n)
            stats[f"removed_sub_{n['sub']}"] += 1
        else:
            stats["missed_in_repeat_ignored"] += 1
    for k in classes:
        classes[k].sort(key=lambda r: (r["start"], r["pitch"]))
    return classes, stats


def write_label_midi(path: Path, notes: list[dict], stats: Counter | None = None) -> int:
    """Write one label-class MIDI. Returns number of notes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(resolution=RESOLUTION, initial_tempo=INITIAL_TEMPO)
    inst = pretty_midi.Instrument(program=PROGRAM, is_drum=False, name="Clarinet")
    for n in notes:
        start = max(0.0, float(n["start"]))
        end = float(n["end"])
        if end < start + MIN_DUR:
            end = start + MIN_DUR
            if stats is not None:
                stats["min_dur_clamped"] += 1
        pitch = int(n["pitch"])
        if not 0 <= pitch <= 127:
            raise ValueError(f"pitch out of MIDI range: {pitch}")
        inst.notes.append(pretty_midi.Note(velocity=VELOCITY, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    tmp = _tmp_path(path)
    pm.write(str(tmp))
    os.replace(tmp, path)
    return len(inst.notes)


def copy_reference_midi(src: Path, dst: Path) -> dict:
    """Copy the reference MIDI, sanitising only if the official validator would
    reject it (start >= end or velocity == 0 raise in validate_note_sequence)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(str(src))
    n_notes = sum(len(i.notes) for i in pm.instruments)
    bad = 0
    for inst in pm.instruments:
        for n in inst.notes:
            fixed = False
            if n.end <= n.start:
                n.end = n.start + MIN_DUR
                fixed = True
            if n.velocity <= 0:
                n.velocity = VELOCITY
                fixed = True
            bad += int(fixed)
    if bad == 0:
        shutil.copyfile(src, dst)
    else:
        tmp = _tmp_path(dst)
        pm.write(str(tmp))
        os.replace(tmp, dst)
    return {"n_ref_midi_notes": n_notes, "ref_midi_sanitized": bad}


# --------------------------------------------------------------------------- #
# Per-bundle worker
# --------------------------------------------------------------------------- #
def process_bundle(job: dict) -> dict:
    set_name = job["set"]
    track_id = job["track_id"]
    bundle = Path(job["bundle"])
    out_root = Path(job["out_root"])
    real_test = job["real_test"]
    overwrite = job["overwrite"]
    equalize = job.get("equalize_audio", True)

    rec = {
        "track_id": track_id,
        "set": set_name,
        "bundle": str(bundle),
        "real_test": real_test,
    }
    try:
        meta = {}
        mp = bundle / "metadata.json"
        if mp.is_file():
            with open(mp) as f:
                meta = json.load(f)
        rec["source"] = str(meta.get("source") or ("real" if real_test else "unknown"))
        rec["seed"] = meta.get("seed")
        rec["repeated"] = meta.get("repeated")
        rec["error_types"] = meta.get("error_types")
        with open(bundle / "labels.json") as f:
            rec["n_span_labels"] = len(json.load(f).get("labels", []))

        paths = output_paths(out_root, track_id)
        n_stale = remove_stale_tmp(paths)
        if n_stale:
            rec["stale_tmp_removed"] = n_stale
        # source durations (header reads only)
        rec["perf_duration_s"] = round(sf.info(str(bundle / "performance_audio.wav")).duration, 3)
        rec["ref_duration_s"] = round(sf.info(str(bundle / "reference_audio.wav")).duration, 3)

        # Label notes --------------------------------------------------------
        stats: Counter = Counter()
        if real_test:
            classes = {"correct": [], "extra": [], "removed": []}
        else:
            with open(bundle / "note_labels.json") as f:
                nl = json.load(f)
            classes, stats = classes_from_note_labels(nl)
            from audit_supervision import issues_for_labels
            rec["supervision_issues"] = issues_for_labels(nl)
            rec["note_labels_check_ok"] = not rec["supervision_issues"]
        if not real_test and not any(classes.values()):
            raise ValueError("No supervised note targets; official training loaders need nonempty labels")
        rec["n_correct"] = len(classes["correct"])
        rec["n_extra"] = len(classes["extra"])
        rec["n_removed"] = len(classes["removed"])

        done = all(p.is_file() for p in paths.values())
        if done and equalize:
            # outputs from a run without equalisation: rewrite them
            done = (
                sf.info(str(paths["mistake_wav"])).frames
                == sf.info(str(paths["score_wav"])).frames
            )
        if done and not overwrite:
            for key in CLASS_DIRS:
                pm = pretty_midi.PrettyMIDI(str(paths[key]))
                actual = sorted((n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes)
                expected = sorted((max(0.0, n["start"]), max(n["end"], max(0.0, n["start"]) + MIN_DUR), n["pitch"])
                                  for n in classes[key])
                if len(actual) != len(expected) or any(
                    a[2] != e[2] or abs(a[0]-e[0]) > 0.003 or abs(a[1]-e[1]) > 0.003
                    for a, e in zip(actual, expected)
                ):
                    raise ValueError(f"Stale {key} labels for {track_id}; rerun with --overwrite")
            rec["status"] = "skipped_existing"
        else:
            perf = load_audio_16k(bundle / "performance_audio.wav")
            ref = load_audio_16k(bundle / "reference_audio.wav")
            if equalize:
                n = max(len(perf), len(ref))
                if len(perf) < n:
                    perf = np.pad(perf, (0, n - len(perf)))
                if len(ref) < n:
                    ref = np.pad(ref, (0, n - len(ref)))
            write_wav16(paths["mistake_wav"], perf)
            write_wav16(paths["score_wav"], ref)
            rec.update(copy_reference_midi(bundle / "reference_audio.mid", paths["score_mid"]))
            for cls_key in CLASS_DIRS:
                written = write_label_midi(paths[cls_key], classes[cls_key], stats)
                # round-trip check: the file the loaders will read has the notes we meant
                back = pretty_midi.PrettyMIDI(str(paths[cls_key]))
                n_back = sum(len(i.notes) for i in back.instruments)
                if n_back != written:
                    raise RuntimeError(
                        f"{cls_key}: wrote {written} notes but re-read {n_back}"
                    )
            rec["status"] = "written"

        rec["written_duration_s"] = round(sf.info(str(paths["mistake_wav"])).duration, 3)
        rec["written_score_duration_s"] = round(sf.info(str(paths["score_wav"])).duration, 3)
        # zero-padding actually present in the written files (also on the skip path)
        pad_perf = round(rec["written_duration_s"] - rec["perf_duration_s"], 3)
        pad_ref = round(rec["written_score_duration_s"] - rec["ref_duration_s"], 3)
        if pad_perf > 0.001:
            rec["perf_zero_padded_s"] = pad_perf
        if pad_ref > 0.001:
            rec["ref_zero_padded_s"] = pad_ref
        rec["label_stats"] = dict(stats)
        rec["outputs"] = {k: str(v) for k, v in paths.items()}
        return rec
    except Exception as e:  # report, do not kill the pool
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        return rec


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _split_sizes(n: int, val_frac: float, test_frac: float) -> tuple[int, int]:
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    if test_frac > 0 and n_test == 0 and n >= 3:
        n_test = 1
    if val_frac > 0 and n_val == 0 and n >= 3:
        n_val = 1
    # keep at least one training clip
    while n_test + n_val >= n and (n_test + n_val) > 0:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    return n_val, n_test


def assign_splits(
    records: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int,
    split_by_source_sets: list[str] | None,
) -> dict[str, str]:
    """Return {track_id: split}. Deterministic per set (seeded by seed + set name)."""
    by_set: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_set[r["set"]].append(r)

    out: dict[str, str] = {}
    for set_name in sorted(by_set):
        recs = sorted(by_set[set_name], key=lambda r: r["track_id"])
        if all(r["real_test"] for r in recs):
            for r in recs:
                out[r["track_id"]] = "test"
            continue

        rng = random.Random(f"{seed}:{set_name}")
        n = len(recs)
        n_val, n_test = _split_sizes(n, val_frac, test_frac)

        use_source = split_by_source_sets is not None and (
            len(split_by_source_sets) == 0 or set_name in split_by_source_sets
        )
        groups: dict[str, list[str]] = defaultdict(list)
        for r in recs:
            groups[r["source"]].append(r["track_id"])
        if use_source and (len(groups) < 3 or "unknown" in groups):
            raise ValueError(f"Source split for {set_name} needs >=3 known scores; "
                             "refusing to silently fall back to a leaking per-clip split")

        if use_source:
            order = sorted(groups)
            rng.shuffle(order)
            counts = {"validation": 0, "test": 0}
            target = {"validation": n_val, "test": n_test}
            for src in order:
                ids = groups[src]
                # pick the split with the largest remaining deficit; otherwise train
                deficits = {s: target[s] - counts[s] for s in ("test", "validation")}
                best = max(deficits, key=deficits.get)
                if deficits[best] > 0:
                    dest = best
                    counts[dest] += len(ids)
                else:
                    dest = "train"
                for tid in ids:
                    out[tid] = dest
        else:
            ids = [r["track_id"] for r in recs]
            rng.shuffle(ids)
            for i, tid in enumerate(ids):
                if i < n_test:
                    out[tid] = "test"
                elif i < n_test + n_val:
                    out[tid] = "validation"
                else:
                    out[tid] = "train"
    return out


def write_split_json(path: Path, records: list[dict], splits: dict[str, str]) -> None:
    recs = sorted(records, key=lambda r: (r["set"], r["track_id"]))
    midi_filename = OrderedDict()
    split = OrderedDict()
    for i, r in enumerate(recs):
        midi_filename[str(i)] = f"{r['set']}/{r['track_id']}.midi"
        split[str(i)] = splits[r["track_id"]]
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump({"midi_filename": midi_filename, "split": split}, f, indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarise(records: list[dict], splits: dict[str, str]) -> dict:
    table: dict = {}
    for r in records:
        key = (r["set"], splits[r["track_id"]])
        row = table.setdefault(
            key,
            {"n": 0, "perf_min": 0.0, "ref_min": 0.0, "correct": 0, "extra": 0, "removed": 0},
        )
        row["n"] += 1
        row["perf_min"] += r.get("perf_duration_s", 0.0) / 60.0
        row["ref_min"] += r.get("ref_duration_s", 0.0) / 60.0
        row["correct"] += r.get("n_correct", 0)
        row["extra"] += r.get("n_extra", 0)
        row["removed"] += r.get("n_removed", 0)
    return {f"{s}/{sp}": v for (s, sp), v in sorted(table.items())}


def print_summary(summary: dict, n_error: int) -> None:
    hdr = f"{'set/split':<24}{'clips':>7}{'perf min':>10}{'ref min':>9}{'correct':>9}{'extra':>8}{'removed':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    tot = {"n": 0, "perf_min": 0.0, "ref_min": 0.0, "correct": 0, "extra": 0, "removed": 0}
    for k, v in summary.items():
        print(
            f"{k:<24}{v['n']:>7}{v['perf_min']:>10.2f}{v['ref_min']:>9.2f}"
            f"{v['correct']:>9}{v['extra']:>8}{v['removed']:>9}"
        )
        for kk in tot:
            tot[kk] += v[kk]
    print("-" * len(hdr))
    print(
        f"{'TOTAL':<24}{tot['n']:>7}{tot['perf_min']:>10.2f}{tot['ref_min']:>9.2f}"
        f"{tot['correct']:>9}{tot['extra']:>8}{tot['removed']:>9}"
    )
    if n_error:
        print(f"\n!! {n_error} bundle(s) failed and were left out of split.json (see manifest.json)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_set(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--set expects NAME=<bundle_root>")
    name, root = spec.split("=", 1)
    name = name.strip()
    if not name or any(c in name for c in "/ ."):
        raise argparse.ArgumentTypeError(f"bad set name {name!r}")
    return name, Path(root).expanduser().resolve()


def build_jobs(args) -> list[dict]:
    per_set: list[tuple[str, list[Path]]] = []
    for name, root in args.set:
        bundles = find_bundles(root, args.real_test, args.limit)
        print(f"set {name!r}: {len(bundles)} bundle(s) in {root}", flush=True)
        per_set.append((name, bundles))

    # collision handling: same dir name in two sets -> prefix with set name
    name_count = Counter(b.name for _, bundles in per_set for b in bundles)
    jobs = []
    seen = set()
    for name, bundles in per_set:
        for b in bundles:
            tid = b.name if name_count[b.name] == 1 else f"{name}_{b.name}"
            validate_track_id(tid)
            if tid in seen:
                raise SystemExit(f"duplicate track id after prefixing: {tid}")
            seen.add(tid)
            jobs.append(
                {
                    "set": name,
                    "track_id": tid,
                    "bundle": str(b),
                    "out_root": str(args.out),
                    "real_test": args.real_test,
                    "overwrite": args.overwrite,
                    "equalize_audio": not args.no_equalize_audio,
                }
            )
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="DATA_ROOT to write")
    ap.add_argument("--set", action="append", required=True, type=parse_set, metavar="NAME=BUNDLE_ROOT")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=365)
    ap.add_argument(
        "--split-by-source",
        nargs="*",
        metavar="SET",
        default=None,
        help="group clips by metadata.json 'source' when splitting (all sets if no names given; "
        "sets with <3 distinct sources fall back to a per-clip split)",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="max bundles per set (sorted by name)")
    ap.add_argument(
        "--real-test",
        action="store_true",
        help="bundles without note_labels.json: write empty label MIDIs, everything -> split 'test'",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--no-equalize-audio",
        action="store_true",
        help="do NOT zero-pad the shorter of performance/reference audio to the longer one "
        "(the official loaders assume equal lengths; see module docstring)",
    )
    ap.add_argument(
        "--keep-failed-check",
        action="store_true",
        help="keep bundles whose note_labels.json check.perf.ok/check.ref.ok is false in split.json "
        "(default: convert them but leave them out of split.json)",
    )
    args = ap.parse_args(argv)

    args.out = args.out.expanduser().resolve()
    validate_out_root(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    jobs = build_jobs(args)
    if not jobs:
        raise SystemExit("no bundles found")
    print(f"{len(jobs)} track(s) -> {args.out}  (workers={args.workers})", flush=True)

    records: list[dict] = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(process_bundle, jobs, chunksize=1), 1):
                records.append(rec)
                if i % 50 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            records.append(process_bundle(job))
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)

    ok = [r for r in records if r.get("status") != "error"]
    bad = [r for r in records if r.get("status") == "error"]
    for r in bad:
        print(f"ERROR {r['set']}/{r['track_id']}: {r['error']}", file=sys.stderr, flush=True)

    failed_check = [r for r in ok if r.get("note_labels_check_ok") is False]
    if failed_check and not args.keep_failed_check:
        for r in failed_check:
            r["excluded"] = "failed_check"
            r["split"] = None
        included = [r for r in ok if r.get("note_labels_check_ok") is not False]
    else:
        included = ok
    if not included:
        raise SystemExit("no track left for split.json")

    splits = assign_splits(included, args.val_frac, args.test_frac, args.seed, args.split_by_source)
    for r in included:
        r["split"] = splits[r["track_id"]]
    write_split_json(args.out / "split.json", included, splits)

    summary = summarise(included, splits)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "out_root": str(args.out),
        "args": {
            "sets": {n: str(p) for n, p in args.set},
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "split_by_source": args.split_by_source,
            "real_test": args.real_test,
            "limit": args.limit,
            "equalize_audio": not args.no_equalize_audio,
            "keep_failed_check": args.keep_failed_check,
        },
        "target_sr": TARGET_SR,
        "audio": "16 kHz mono PCM_16; performance and reference zero-padded to equal length"
        if not args.no_equalize_audio
        else "16 kHz mono PCM_16; original lengths",
        "label_midi": {"program": PROGRAM, "velocity": VELOCITY, "min_dur_s": MIN_DUR, "resolution": RESOLUTION},
        "class_semantics": {
            "extra_notes": "performance_notes[cls=='extra'] (sub wrong|inserted|repeat) -> error_class 1",
            "removed_notes": "missed_notes[copy==0] (sub rest|wrong) -> error_class 2",
            "correct_notes": "performance_notes[cls=='correct'] -> error_class 3",
        },
        "summary": summary,
        "n_tracks": len(included),
        "n_errors": len(bad),
        "n_failed_check": len(failed_check),
        "excluded_failed_check": sorted(r["track_id"] for r in ok if r.get("excluded") == "failed_check"),
        "tracks": {r["track_id"]: {k: v for k, v in r.items() if k != "traceback"} for r in sorted(ok, key=lambda r: r["track_id"])},
        "errors": bad,
    }
    tmp = args.out / "manifest.tmp.json"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, args.out / "manifest.json")

    print_summary(summary, len(bad))
    if failed_check:
        ids = ", ".join(sorted(r["track_id"] for r in failed_check)[:5])
        more = "" if len(failed_check) <= 5 else ", ..."
        if args.keep_failed_check:
            print(f"\n!! {len(failed_check)} bundle(s) have note_labels.json check.ok == false; KEPT in split.json ({ids}{more})")
        else:
            print(f"\n!! {len(failed_check)} bundle(s) have note_labels.json check.ok == false -> converted but left out of "
                  f"split.json ({ids}{more}); use --keep-failed-check to include them")
    n_pad_ref = sum(1 for r in ok if r.get("ref_zero_padded_s"))
    n_pad_perf = sum(1 for r in ok if r.get("perf_zero_padded_s"))
    if n_pad_ref or n_pad_perf:
        print(f"audio equalised: reference zero-padded in {n_pad_ref} track(s) "
              f"({sum(r.get('ref_zero_padded_s', 0) for r in ok) / 60:.2f} min), "
              f"performance in {n_pad_perf} track(s) "
              f"({sum(r.get('perf_zero_padded_s', 0) for r in ok) / 60:.2f} min)")
    n_stale = sum(r.get("stale_tmp_removed", 0) for r in ok)
    if n_stale:
        print(f"removed {n_stale} stale temp file(s) from earlier interrupted runs")
    n_written = sum(1 for r in ok if r.get("status") == "written")
    n_skipped = sum(1 for r in ok if r.get("status") == "skipped_existing")
    print(f"\nwritten={n_written} skipped_existing={n_skipped} errors={len(bad)}  in {time.time() - t0:.1f}s")
    print(f"split.json   -> {args.out / 'split.json'}")
    print(f"manifest.json-> {args.out / 'manifest.json'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
