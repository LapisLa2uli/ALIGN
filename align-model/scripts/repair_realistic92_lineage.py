"""Repair 9.2 note_map lineage with the fail-closed global ornament reconstruction.

The shipped note_map maps some rendered MIDI notes to the wrong performed note
(out-of-order score spans, copy notes merged across passes). The ORN
reconstruction rebuilds rendered lineage from the performance MIDI and the
performance score's ornament template, and rejects any clip that does not
round-trip exactly. Eligibility depends only on target validity; no model
output is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_training_data as audit
from alignmodel.joint.global_ornament_lineage_v2 import (
    SCHEMA_VERSION as REPAIR_VERSION,
    _align_exact,
    _base_hypothesis,
    reconstruct_global_ornament_lineage,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.ornament_mapper_v1 import (
    expand_ornament_hypothesis,
    score_ornament_patterns,
)
from alignmodel.joint.packed_data import sha256_file


SCHEMA_VERSION = "align-realistic92-lineage-repair-v1"


def _order_simultaneous_onsets(
    notes: list[dict[str, Any]], performance_score: Path, shift: int
) -> list[dict[str, Any]]:
    """Order notes that share an onset to best match the performance template.

    music21 emits grace notes at the principal's onset, so MIDI order within
    a shared onset is arbitrary. Each shared-onset group is permuted
    independently; the exact-round-trip check still runs afterwards.
    """

    ordered = sorted(
        notes,
        key=lambda note: (round(float(note["start"]), 6), -float(note["end"]), int(note["pitch"])),
    )
    index = ScoreEventIndex.from_musicxml(performance_score)
    patterns = score_ornament_patterns(performance_score, index.events)
    template = [unit.pitch for unit in expand_ornament_hypothesis(
        index.events, patterns, _base_hypothesis(index)
    )]

    def cost(sequence: list[dict[str, Any]]) -> int:
        return _align_exact([int(note["pitch"]) + shift for note in sequence], template)[3]

    best_cost = cost(ordered)
    if best_cost == 0:
        return ordered
    groups: list[tuple[int, int]] = []
    start = 0
    for position in range(1, len(ordered) + 1):
        if (
            position == len(ordered)
            or round(float(ordered[position]["start"]), 6)
            != round(float(ordered[start]["start"]), 6)
        ):
            if position - start >= 2:
                groups.append((start, position))
            start = position
    for first, last in groups:
        if last - first > 4:
            continue
        for permutation in itertools.permutations(range(first, last)):
            if list(permutation) == list(range(first, last)):
                continue
            candidate = ordered[:first] + [ordered[k] for k in permutation] + ordered[last:]
            value = cost(candidate)
            if value < best_cost:
                ordered, best_cost = candidate, value
                if best_cost == 0:
                    return ordered
    return ordered


def _repair(payload: tuple[str, str, str]) -> dict[str, Any]:
    root, name, output = payload
    sample = Path(root) / name
    row: dict[str, Any] = {"sample": name}
    try:
        original = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))
        metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
        midi = audit._midi_events(sample / "performance_audio.mid")
        if midi is None or not midi.get("notes"):
            raise ValueError("performance MIDI unreadable or empty")
        midi = dict(midi)
        midi["notes"] = _order_simultaneous_onsets(
            midi["notes"], sample / "performance_score.musicxml",
            audit._inferred_midi_shift(original, midi, metadata),
        )
        shift = audit._inferred_midi_shift(original, midi, metadata)
        result = reconstruct_global_ornament_lineage(
            original,
            sample / "performance_score.musicxml",
            sample / "verified_score.musicxml",
            midi,
            written_shift=shift,
        )
        index = ScoreEventIndex.from_musicxml(
            sample / "verified_score.musicxml", result.lineage
        )
        destination = Path(output) / f"{name}.json"
        destination.write_text(json.dumps(result.lineage), encoding="utf-8")
        row.update({
            "eligible": True,
            "written_shift": shift,
            "rendered_notes": len(index.rendered_events),
            "score_events": len(index.events),
            "deleted_events": len(index.deleted_event_indices),
            "renderer_ornament_events": result.stats.get("renderer_ornament_events"),
            "lineage_sha256": sha256_file(destination),
            "note_map_sha256": sha256_file(sample / "note_map.json"),
        })
    except Exception as error:  # noqa: BLE001
        row.update({
            "eligible": False,
            "reason": f"{type(error).__name__}: {str(error)[:300]}",
        })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if sha256_file(args.split) != args.expected_split_sha256:
        raise ValueError("Frozen split mismatch")
    splits = json.loads(args.split.read_text(encoding="utf-8"))["splits"]
    lineage_dir = args.output_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (split_name, name)
        for split_name in ("train", "val", "test")
        for name in splits[split_name]
    ]
    if args.limit:
        jobs = jobs[:args.limit]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(
            _repair,
            [(str(args.root), name, str(lineage_dir)) for _, name in jobs],
            chunksize=8,
        )
        for position, ((split_name, _name), row) in enumerate(zip(jobs, results), 1):
            row["split"] = split_name
            rows.append(row)
            if position % 250 == 0 or position == len(jobs):
                eligible = sum(value["eligible"] for value in rows)
                print(f"repaired={position}/{len(jobs)} eligible={eligible}", flush=True)

    summary: dict[str, Any] = {}
    reasons: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split_name]
        summary[split_name] = {
            "clips": len(subset),
            "eligible": sum(row["eligible"] for row in subset),
            "procedural_eligible": sum(
                row["eligible"] and row["sample"].startswith("synth_gen_") for row in subset
            ),
            "rawdata_eligible": sum(
                row["eligible"] and not row["sample"].startswith("synth_gen_") for row in subset
            ),
        }
    for row in rows:
        if not row["eligible"]:
            key = row["reason"].split(":")[0] + ":" + row["reason"].split(":", 1)[1][:70]
            reasons[key] = reasons.get(key, 0) + 1
    audit_path = args.output_dir / "REPAIR_AUDIT.json"
    audit_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repair_policy": REPAIR_VERSION,
        "split_sha256": args.expected_split_sha256,
        "summary": summary,
        "top_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])[:25]),
        "model_outputs_read": False,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "top_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])[:10])},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
