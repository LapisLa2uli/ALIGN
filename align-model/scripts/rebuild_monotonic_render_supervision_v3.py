from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

import audit_training_data as audit
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.outputraw_full import verify_data_ready


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _monotonic_rows(
    document: dict, midi: dict, shift: int
) -> tuple[list[dict], dict]:
    performed = list(document["performed_notes"])
    raw = list(midi["notes"])
    left = [int(row["pitch"]) + shift for row in raw]
    right = [int(row["pitch_midi"]) for row in performed]
    rows, columns = len(left), len(right)
    cost = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    back = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    cost[:, 0] = np.arange(rows + 1)
    cost[0, :] = np.arange(columns + 1)
    back[1:, 0] = 2
    back[0, 1:] = 3
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            diagonal = cost[i - 1, j - 1] + (
                0 if left[i - 1] == right[j - 1] else 3
            )
            raw_unmatched = cost[i - 1, j] + 1
            performed_unmatched = cost[i, j - 1] + 1
            best = min(diagonal, raw_unmatched, performed_unmatched)
            cost[i, j] = best
            back[i, j] = (
                1
                if diagonal <= raw_unmatched and diagonal <= performed_unmatched
                else 2
                if raw_unmatched <= performed_unmatched
                else 3
            )
    assigned = {index: [] for index in range(rows)}
    i, j = rows, columns
    while i or j:
        operation = int(back[i, j])
        if operation == 1:
            if left[i - 1] == right[j - 1]:
                assigned[i - 1].append(j - 1)
            i -= 1
            j -= 1
        elif operation == 2:
            i -= 1
        elif operation == 3:
            j -= 1
        else:
            raise RuntimeError("Monotonic lineage backtrace failed")
    claimed = {
        performed_index
        for values in assigned.values()
        for performed_index in values
    }
    folded = 0
    for performed_index, item in enumerate(performed):
        if performed_index in claimed:
            continue
        pitch = int(item["pitch_midi"])
        candidates = [
            raw_index
            for raw_index, values in assigned.items()
            if values
            and left[raw_index] == pitch
            and min(abs(performed_index - value) for value in values) <= 4
        ]
        if candidates:
            target = min(
                candidates,
                key=lambda raw_index: min(
                    abs(performed_index - value)
                    for value in assigned[raw_index]
                ),
            )
            assigned[target].append(performed_index)
            assigned[target].sort()
            claimed.add(performed_index)
            folded += 1
    output = []
    for rendered_index, actual in enumerate(raw):
        indices = assigned[rendered_index]
        clean = list(
            dict.fromkeys(
                int(performed[index]["clean_index"])
                for index in indices
                if performed[index].get("clean_index") is not None
            )
        )
        relationships = [
            str(performed[index].get("relationship") or "extra")
            for index in indices
        ]
        output.append(
            {
                "rendered_index": rendered_index,
                "pitch_midi_sounding": int(actual["pitch"]),
                "pitch_midi_written": int(actual["pitch"]) + shift,
                "start_sec": round(float(actual["start"]), 9),
                "end_sec": round(
                    max(float(actual["end"]), float(actual["start"]) + 0.001),
                    9,
                ),
                "performed_indices": indices,
                "clean_indices": clean,
                "primary_clean_index": clean[0] if clean else None,
                "relationship": (
                    "copy"
                    if "copy" in relationships
                    else relationships[0]
                    if relationships
                    else "extra"
                ),
            }
        )
    mapped_order = [
        index for row in output for index in row["performed_indices"]
    ]
    return output, {
        "raw_midi_events": len(raw),
        "performed_notes": len(performed),
        "mapped_performed_notes": len(claimed),
        "unmapped_performed_notes": len(performed) - len(claimed),
        "folded_tied_notes": folded,
        "strictly_monotonic_performed_order": all(
            left < right for left, right in zip(mapped_order, mapped_order[1:])
        ),
        "backward_transfers": sum(
            right < left for left, right in zip(mapped_order, mapped_order[1:])
        ),
        "edit_cost": int(cost[rows, columns]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    manifest_path = Path(str(ready["paths"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    admitted = Counter()
    excluded = []
    totals = {"train": Counter(), "val": Counter()}
    destination = output / "train-monotonic-structural-supervision.jsonl.gz"
    with tempfile.NamedTemporaryFile(
        dir=output, suffix=".tmp", delete=False
    ) as raw_stream:
        temporary = Path(raw_stream.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for split in ("train", "val"):
                for row in manifest[split]:
                    sample_dir = Path(row["sample_dir"])
                    try:
                        note_map_path = sample_dir / "note_map.json"
                        if _sha256(note_map_path) != row["source_hashes"]["note_map.json"]:
                            raise ValueError("Raw note-map hash mismatch")
                        document = json.loads(
                            note_map_path.read_text(encoding="utf-8")
                        )
                        metadata = json.loads(
                            (sample_dir / "metadata.json").read_text(encoding="utf-8")
                        )
                        midi = audit._midi_events(
                            sample_dir / "performance_audio.mid"
                        )
                        if midi is None:
                            raise ValueError("Performance MIDI is unreadable")
                        shift = audit._inferred_midi_shift(
                            document, midi, metadata
                        )
                        rendered, stats = _monotonic_rows(
                            document, midi, shift
                        )
                        if stats["backward_transfers"]:
                            raise ValueError("Monotonic repair retained backward transfer")
                        rebuilt = dict(document)
                        rebuilt["rendered_notes"] = rendered
                        rebuilt["rendered_note_count"] = len(rendered)
                        index = ScoreEventIndex.from_musicxml(
                            sample_dir / "verified_score.musicxml", rebuilt
                        )
                        events = index.rendered_events
                        if [event.rendered_index for event in events] != list(
                            range(len(events))
                        ):
                            raise ValueError("Rendered event identities are not exclusive")
                        for name, value in stats.items():
                            if isinstance(value, int):
                                totals[split][name] += value
                        admitted[split] += 1
                        if split == "train":
                            stream.write(
                                json.dumps(
                                    {
                                        "schema_version": "align-monotonic-structural-supervision-v3",
                                        "sample": row["sample"],
                                        "events": [
                                            {
                                                "rendered_event_id": event.rendered_index,
                                                "pitch_written": event.pitch,
                                                "start_sec": event.start,
                                                "end_sec": event.end,
                                                "score_span": event.score_span,
                                                "source_indices": event.source_indices,
                                                "relationship": event.relationship,
                                                "copy_pass": event.copy_pass,
                                                "origin_relationship": event.origin_relationship,
                                            }
                                            for event in events
                                        ],
                                        "deleted_score_events": sorted(
                                            index.deleted_event_indices
                                        ),
                                        "repair_stats": stats,
                                        "source_hashes": row["source_hashes"],
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                    except Exception as error:
                        excluded.append(
                            {
                                "split": split,
                                "sample": row["sample"],
                                "reason": str(error),
                            }
                        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema_version": "align-monotonic-render-supervision-v3-report",
        "data_fingerprint": ready["hashes"]["pack_id"],
        "manifest_sha256": _sha256(manifest_path),
        "admitted": dict(admitted),
        "excluded": excluded,
        "totals": {
            split: dict(values) for split, values in totals.items()
        },
        "train_supervision": str(destination),
        "train_supervision_sha256": _sha256(destination),
        "validation_targets_materialized": False,
        "test_targets_materialized": False,
        "locked_test_touched": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
