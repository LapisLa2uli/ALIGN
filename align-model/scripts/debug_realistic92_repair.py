"""Show template/raw-MIDI disagreements for a 9.2 clip that fails lineage repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_training_data as audit
from alignmodel.joint.global_ornament_lineage_v2 import _align_exact, _base_hypothesis
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.ornament_mapper_v1 import expand_ornament_hypothesis, score_ornament_patterns


def main() -> int:
    sample = Path(r"E:\outputRaw_realistic_10k") / sys.argv[1]
    original = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    midi = audit._midi_events(sample / "performance_audio.mid")
    midi["notes"] = sorted(
        midi["notes"],
        key=lambda note: (round(float(note["start"]), 6), -float(note["end"]), int(note["pitch"])),
    )
    shift = audit._inferred_midi_shift(original, midi, metadata)
    index = ScoreEventIndex.from_musicxml(sample / "performance_score.musicxml")
    patterns = score_ornament_patterns(sample / "performance_score.musicxml", index.events)
    template = expand_ornament_hypothesis(index.events, patterns, _base_hypothesis(index))
    raw = [int(row["pitch"]) + shift for row in midi["notes"]]
    pairs, raw_unmatched, template_unmatched, cost = _align_exact(raw, [u.pitch for u in template])
    print("shift", shift, "raw", len(raw), "template", len(template), "cost", cost)
    raw_bad, template_bad = set(raw_unmatched), set(template_unmatched)
    pair = dict(pairs)
    inverse = {t: r for r, t in pairs}
    rows = max(len(raw), len(template))
    for t, unit in enumerate(template):
        mark = "  <-- TEMPLATE UNMATCHED" if t in template_bad else ""
        r = inverse.get(t)
        raw_text = (
            f"raw[{r}]={raw[r]} t={midi['notes'][r]['start']:.3f}-{midi['notes'][r]['end']:.3f}"
            if r is not None else "-"
        )
        if template_bad and min(abs(t - b) for b in template_bad) <= 3:
            print(f"T{t} {unit.kind[:6]} {unit.pitch} score={unit.score_index} {unit.ornament_kind} | {raw_text}{mark}")
    for r in sorted(raw_bad):
        note = midi["notes"][r]
        print(f"RAW UNMATCHED raw[{r}]={raw[r]} t={note['start']:.3f}-{note['end']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
