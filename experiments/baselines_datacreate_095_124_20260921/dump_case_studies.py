"""Dump score, alignment, gold, and prediction timelines for clips 095 and 119."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(r"D:\stuff\Audio Evaluation\ALIGN")
SAMPLES = ROOT / "DataCreate" / "samples"
EVAL = ROOT / "align-model" / "runs" / "eval-datacreate-095-124-20260921"
PC = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_name(midi: int) -> str:
    return f"{PC[int(midi) % 12]}{int(midi) // 12 - 1}"


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def parse_score(path: Path) -> list[dict]:
    tree = ET.parse(path)
    notes = []
    idx = 0
    for measure in tree.iter():
        if local_name(measure.tag) != "measure":
            continue
        measure_number = measure.attrib.get("number", "?")
        for note in list(measure):
            if local_name(note.tag) != "note":
                continue
            tags = {local_name(child.tag): child for child in note}
            if "rest" in tags or "pitch" not in tags:
                continue
            if "tie" in tags and tags["tie"].attrib.get("type") == "stop":
                continue
            pitch = tags["pitch"]
            step = next(child.text for child in pitch if local_name(child.tag) == "step")
            octave = int(next(child.text for child in pitch if local_name(child.tag) == "octave"))
            alter = 0
            for child in pitch:
                if local_name(child.tag) == "alter" and child.text:
                    alter = int(float(child.text))
            semis = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
            midi = 12 * (octave + 1) + semis + alter
            notes.append(
                {
                    "i": idx,
                    "measure": measure_number,
                    "name": midi_name(midi),
                    "midi": midi,
                }
            )
            idx += 1
    return notes


def summarize_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for label in data.get("labels", []):
        part = label.get("score_part") or {}
        pitches = ",".join(midi_name(p) for p in (label.get("pitches") or [])[:12])
        extra = f" extra_copies={label.get('extra_copies')}" if label.get("extra_copies") else ""
        rpt = ""
        if label.get("repeats_label_range"):
            rng = label["repeats_label_range"]
            rpt = f" copies {rng['start_time']:.2f}-{rng['end_time']:.2f}s"
        lines.append(
            f"  {label['type']:13} {label['start_time']:7.2f}-{label['end_time']:7.2f}s  "
            f"m{label.get('measure_number')}  "
            f"core {part.get('core_start_note_index')}-{part.get('core_end_note_index')}  "
            f"span {part.get('start_note_index')}-{part.get('end_note_index')}  "
            f"[{pitches}]{extra}{rpt}"
        )
    return lines


def dump_sample(sample_id: str) -> None:
    sample = SAMPLES / sample_id
    print("=" * 88)
    print(f"SAMPLE {sample_id}")
    meta = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    print(f"score={meta.get('source_score')} measures={meta['score_segment']} duration_ratio={meta['score_location']['duration_ratio']:.3f}")
    notes = parse_score(sample / "verified_score.musicxml")
    print(f"score events: {len(notes)}")
    by_measure: dict[str, list[str]] = {}
    for note in notes:
        by_measure.setdefault(str(note["measure"]), []).append(f"{note['i']}:{note['name']}")
    for measure, items in by_measure.items():
        print(f"  m{measure}: " + " ".join(items))

    align = json.loads((sample / "note_alignment_mel_v1.json").read_text(encoding="utf-8"))
    rel = Counter(event["relationship"] for event in align["events"])
    copies = Counter(event.get("copy_pass", 0) for event in align["events"])
    print(f"alignment events={len(align['events'])} relationships={dict(rel)} copy_pass={dict(copies)}")
    print(f"deleted score events: {align.get('deleted_score_event_indices')}")
    print("alignment timeline:")
    for event in align["events"]:
        span = event.get("score_span")
        span_txt = f"{span[0]}-{span[1]}" if span else "—"
        score_names = ""
        if span:
            score_names = " ".join(notes[i]["name"] for i in range(span[0], min(span[1], len(notes))))
        print(
            f"  {event['start']:7.2f}-{event['end']:7.2f}  {midi_name(event['pitch']):4}  "
            f"{event['relationship']:10} copy={event.get('copy_pass', 0)}  "
            f"score[{span_txt}] {score_names}  conf={event.get('confidence', 0):.2f}"
        )

    print("GOLD labels.json:")
    print("\n".join(summarize_labels(sample / "labels.json")))
    print("AGENT labels_agent.json:")
    print("\n".join(summarize_labels(sample / "labels_agent.json")))
    print("EVAL predictions:")
    print("\n".join(summarize_labels(EVAL / "predictions" / f"{sample_id}.json")))
    print("RULES baseline:")
    print("\n".join(summarize_labels(EVAL / "rules-baseline" / f"{sample_id}.json")))


if __name__ == "__main__":
    dump_sample("095")
    dump_sample("119")
