from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import zlib
from collections import Counter
from pathlib import Path

from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload(row: dict) -> dict:
    connection = sqlite3.connect(
        f"file:{Path(row['target_db']).resolve().as_posix()}?mode=ro", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        saved = connection.execute(
            "SELECT source_hashes,payload FROM targets WHERE ordinal=?",
            (int(row["target_record"]),),
        ).fetchone()
    finally:
        connection.close()
    if saved is None:
        raise ValueError("Validated target record is missing")
    if json.loads(saved[0]) != row["source_hashes"]:
        raise ValueError("Validated target source hashes changed")
    value = json.loads(zlib.decompress(saved[1]))
    if value["source_hashes"] != row["source_hashes"]:
        raise ValueError("Validated payload source hashes changed")
    return value


def _representation(packed: object, provenance: dict) -> dict:
    example = packed.training_example()
    events = example.target_events
    if [event.rendered_index for event in events] != list(range(len(events))):
        raise ValueError("Rendered identities are not contiguous")
    if int(provenance["rendered_note_count"]) != len(events):
        raise ValueError("Packed/repaired rendered counts differ")
    copy_positions = {}
    replay_plans = []
    for copy_pass in sorted({event.copy_pass for event in events if event.is_copy}):
        indices = [
            index
            for index, event in enumerate(events)
            if event.copy_pass == copy_pass
        ]
        linked = [
            events[index].score_span
            for index in indices
            if events[index].score_span is not None
        ]
        source_span = (
            [min(span[0] for span in linked), max(span[1] for span in linked)]
            if linked
            else None
        )
        for position, rendered_index in enumerate(indices):
            copy_positions[rendered_index] = position
        final_copy = max(indices)
        resume = next(
            (
                event.score_span[0]
                for event in events[final_copy + 1 :]
                if not event.is_copy and event.score_span is not None
            ),
            source_span[1] if source_span is not None else -1,
        )
        replay_plans.append(
            {
                "copy_pass": copy_pass,
                "source_span": source_span,
                "rendered_event_indices": indices,
                "resume_event": int(resume),
                "copy_events": len(indices),
            }
        )
    rows = []
    for index, event in enumerate(events):
        score_pitches = (
            [example.score[value].pitch for value in range(*event.score_span)]
            if event.score_span is not None
            else []
        )
        rows.append(
            {
                "rendered_event_id": index,
                "pitch_written": event.pitch,
                "start_sec": event.start,
                "end_sec": event.end,
                "canonical_location": (
                    {
                        "kind": "score_span",
                        "score_span": list(event.score_span),
                        "score_pitches": score_pitches,
                        "copy_pass": event.copy_pass,
                    }
                    if event.score_span is not None
                    else {"kind": "rendered_extra", "rendered_event_id": index}
                ),
                "source_indices": list(event.source_indices),
                "relationship": event.relationship,
                "origin_relationship": event.origin_relationship,
                "state": "replay" if event.is_copy else "first_pass",
                "copy_pass": event.copy_pass,
                "copy_position": copy_positions.get(index),
            }
        )
    return {
        "schema_version": "align-canonical-structural-supervision-v2",
        "sample": packed.sample,
        "source": packed.source,
        "score_event_count": len(example.score),
        "rendered_event_count": len(events),
        "events": rows,
        "replay_plans": replay_plans,
        "omitted_score_events": sorted(example.target_deletions),
        "repair": provenance.get("render_repair"),
        "source_hashes": provenance["source_hashes"],
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
    repair = {"train": Counter(), "val": Counter()}
    admitted = Counter()
    excluded = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        by_sample = {
            dataset[ordinal].sample: ordinal
            for split in ("train", "val")
            for ordinal in dataset.ordinals(split)
        }
        destination = output / "train-structural-supervision.jsonl.gz"
        with tempfile.NamedTemporaryFile(
            dir=output, suffix=".tmp", delete=False
        ) as raw:
            temporary = Path(raw.name)
        try:
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                for split in ("train", "val"):
                    for row in manifest[split]:
                        try:
                            provenance = _payload(row)
                            document = _representation(
                                dataset[by_sample[row["sample"]]], provenance
                            )
                            policy = provenance.get("render_repair") or {}
                            for name, value in policy.items():
                                if isinstance(value, int):
                                    repair[split][name] += value
                            admitted[split] += 1
                            if split == "train":
                                stream.write(
                                    json.dumps(
                                        document,
                                        sort_keys=True,
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
        "schema_version": "align-structural-supervision-v2-build-report",
        "data_fingerprint": ready["hashes"]["pack_id"],
        "manifest_sha256": _sha256(manifest_path),
        "admitted": dict(admitted),
        "excluded": excluded,
        "repair_totals": {
            split: dict(values) for split, values in repair.items()
        },
        "train_supervision": str(destination),
        "train_supervision_sha256": _sha256(destination),
        "validation_targets_materialized": False,
        "test_targets_materialized": False,
        "locked_test_touched": False,
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
