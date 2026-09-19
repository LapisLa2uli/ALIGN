"""Refresh MIDI-facing timing while preserving generation-time note identities.

Only note_map.json is updated. Original maps are backed up before replacement;
scores, audio, error labels, mel features and alignment archives are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

from synthpipeline.note_map import attach_rendered_events, write_note_map
from synthpipeline.timing import midi_note_times


def prepare(sample: Path) -> tuple:
    original = (sample / "note_map.json").read_text()
    payload = json.loads(original)
    metadata = json.loads((sample / "metadata.json").read_text())
    if metadata.get("audio_render") != "soundfont_v1" or metadata.get("midi_pitch_space") != "sounding":
        raise ValueError(f"Expected SoundFont audio with sounding MIDI: {sample}")
    midi = sample / "performance_audio.mid"
    exact = midi_note_times(midi)
    if not exact:
        raise ValueError(f"No MIDI note events: {sample}")
    old = payload.get("rendered_notes", [])
    expected = [(pitch, round(start, 9), round(end, 9)) for pitch, start, end in exact]
    stored = [(r["pitch_midi_sounding"], r["start_sec"], r["end_sec"]) for r in old]
    if stored == expected:
        return sample, None, None
    attach_rendered_events(payload, midi,
                           sounding_transpose=int(metadata["sounding_transpose"]),
                           performed_score_path=sample / "performance_score.musicxml")
    # Lineage itself must stay exactly as captured during error injection.
    before = json.loads(original)
    for key in before.keys() - {"rendered_notes", "rendered_note_count", "render_validation"}:
        if before[key] != payload[key]:
            raise ValueError(f"Generation-time lineage changed: {sample}/{key}")
    return sample, original, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = [p for root in args.root for p in sorted(root.glob("synth_*")) if p.is_dir()]
    summary = {"checked": 0, "updated": 0, "old_events": 0, "new_events": 0,
               "old_unmapped": 0, "new_unmapped": 0}
    # Exclusive creation prevents overwriting the pre-repair backups on rerun.
    with (args.out_dir / "changes.jsonl").open("x") as log, gzip.open(
        args.out_dir / "original_maps.jsonl.gz", "xt", encoding="utf-8"
    ) as backup, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sample, original, payload in pool.map(prepare, samples, chunksize=8):
            summary["checked"] += 1
            if payload is not None:
                old = json.loads(original)
                backup.write(json.dumps({"path": str(sample), "original": original}) + "\n")
                backup.flush()
                write_note_map(sample / "note_map.json", payload)
                summary["updated"] += 1
                summary["old_events"] += old.get("rendered_note_count", 0)
                summary["new_events"] += payload["rendered_note_count"]
                summary["old_unmapped"] += old["render_validation"]["unmapped_performed_notes"]
                summary["new_unmapped"] += payload["render_validation"]["unmapped_performed_notes"]
                log.write(json.dumps({
                    "sample": str(sample),
                    "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "after_sha256": hashlib.sha256((sample / "note_map.json").read_bytes()).hexdigest(),
                    "before_events": old.get("rendered_note_count", 0),
                    "after_events": payload["rendered_note_count"],
                }) + "\n")
                log.flush()
            if summary["checked"] % 500 == 0:
                print(json.dumps(summary), flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
