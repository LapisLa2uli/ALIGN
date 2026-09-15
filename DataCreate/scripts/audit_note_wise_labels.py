"""Audit label documents for official canonical note-wise evaluability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datacreate.melody import canonical_note_location, parse_sounding_notes


def audit_sample(sample_dir: Path) -> dict[str, Any]:
    labels_path = sample_dir / "labels.json"
    score_path = sample_dir / "verified_score.musicxml"
    if not labels_path.is_file() or not score_path.is_file():
        return {
            "sample": sample_dir.name,
            "official_note_wise": "unavailable",
            "reason": "missing labels.json or verified_score.musicxml",
        }
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    notes = parse_sounding_notes(score_path)
    rows = []
    available = True
    for index, source in enumerate(document.get("labels") or []):
        label = dict(source)
        location = canonical_note_location(
            label, score_event_count=len(notes)
        )
        reasons = []
        if location is None:
            reasons.append("missing or invalid canonical score-event identity")
        part = label.get("score_part")
        if isinstance(part, dict) and location is not None:
            first = int(part["start_note_index"])
            last = int(part["end_note_index"])
            expected = [note.pitch for note in notes[first : last + 1]]
            pitches = label.get("pitches")
            if not isinstance(pitches, list) or [int(value) for value in pitches] != expected:
                reasons.append("pitches do not exactly validate the canonical range")
        if reasons:
            available = False
        rows.append(
            {
                "label_index": index,
                "type": label.get("type"),
                "canonical_location": location,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
    return {
        "sample": sample_dir.name,
        "schema_version": document.get("schema_version"),
        "official_note_wise": "available" if available else "unavailable",
        "score_event_count": len(notes),
        "labels": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        audit_sample(path)
        for path in sorted(args.samples_root.iterdir())
        if path.is_dir()
    ]
    output = {
        "schema_version": "datacreate-note-wise-label-audit-v1",
        "policy": (
            "read-only validation; this tool never fabricates or writes note identities"
        ),
        "samples": reports,
        "available": sum(
            row["official_note_wise"] == "available" for row in reports
        ),
        "unavailable": sum(
            row["official_note_wise"] != "available" for row in reports
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
