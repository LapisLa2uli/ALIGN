"""Require exported MIDI supervision to agree with actual SoundFont event timing."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from synth_supervision import MidiTimeline
import soundfile as sf
from tinysoundfont.midi import load, NoteOn, NoteOff


def rendered_notes(path):
    pending = defaultdict(deque)
    notes, serial = [], 0
    for event in load(str(path), persistent=False):
        action = event.action
        key = (getattr(event, "channel", 0), getattr(action, "key", None))
        if isinstance(action, NoteOn) and action.velocity > 0:
            pending[key].append((event.t, serial))
            serial += 1
        elif isinstance(action, NoteOff) or (isinstance(action, NoteOn) and action.velocity == 0):
            if pending[key]:
                start, index = pending[key].popleft()
                notes.append((index, key[1], float(start), float(event.t)))
    if any(pending.values()):
        raise ValueError(f"Unclosed renderer events: {path}")
    return sorted(notes)


def check_one(bundle):
    reasons, deltas = [], {}
    for name in ("performance", "reference"):
        midi = bundle / f"{name}_audio.mid"
        expected = MidiTimeline(midi).events
        actual = rendered_notes(midi)
        if len(expected) != len(actual) or any(row["pitch"] != event[1] for row, event in zip(expected, actual)):
            reasons.append(f"{name}_renderer_midi_note_mismatch")
            continue
        delta = max((max(abs(row["start"]-event[2]), abs(row["end"]-event[3]))
                     for row, event in zip(expected, actual)), default=0)
        deltas[name] = delta
        if delta > 0.003:
            reasons.append(f"{name}_renderer_midi_timing_mismatch")
        duration = sf.info(bundle / f"{name}_audio.wav").duration
        if any(event[3] > duration + 0.003 for event in actual):
            reasons.append(f"{name}_renderer_event_outside_audio")
    duration = max(sf.info(bundle / f"{name}_audio.wav").duration for name in ("performance", "reference"))
    labels = json.loads((bundle / "note_labels.json").read_text())
    if any(row["onset"] >= duration or row["offset"] > duration + 0.003
           for row in labels["performance_notes"] + labels["missed_notes"]):
        reasons.append("target_outside_audio")
    return dict(set=bundle.parent.name, sample=bundle.name, reasons=reasons,
                maximum_timing_delta_seconds=deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bundles = sorted(p.parent for p in args.targets.glob("*/*/note_labels.json"))
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(check_one, bundles, chunksize=8), 1):
            rows.append(row)
            if not row["reasons"]:
                target = args.out / row["set"] / row["sample"]
                target.parent.mkdir(parents=True, exist_ok=True)
                source = (args.targets / row["set"] / row["sample"]).resolve()
                if target.is_symlink():
                    if target.resolve() != source:
                        raise ValueError(f"Unexpected source link: {target}")
                else:
                    target.symlink_to(source, target_is_directory=True)
            if i % 1000 == 0:
                print(f"Checked {i}/{len(bundles)}", flush=True)
    excluded = [row for row in rows if row["reasons"]]
    report = dict(checked=len(rows), accepted=len(rows)-len(excluded), excluded=len(excluded),
                  tolerance_seconds=0.003,
                  reasons=dict(Counter(reason for row in excluded for reason in row["reasons"])),
                  by_set={name: dict(accepted=sum(not r["reasons"] for r in rows if r["set"]==name),
                                    excluded=sum(bool(r["reasons"]) for r in rows if r["set"]==name))
                          for name in sorted({r["set"] for r in rows})},
                  samples=rows)
    (args.out / "render_timing_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
