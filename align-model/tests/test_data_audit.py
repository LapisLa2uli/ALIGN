from __future__ import annotations

import importlib.util
import gzip
import hashlib
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_training_data.py"
SPEC = importlib.util.spec_from_file_location("audit_training_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

from alignmodel.validated_targets import target_note_map


def _map_document() -> dict:
    clean = [
        {
            "clean_index": index,
            "deleted": False,
            "pitch_midi": 60,
            "pitch": "C4",
            "onset_ql": float(index),
            "duration_ql": 1.0,
            "measure": 1,
        }
        for index in range(2)
    ]
    performed = [
        {
            "performed_index": index,
            "clean_index": index,
            "relationship": "match",
            "origin_relationship": "match",
            "copy_pass": 0,
            "pitch_midi": 60,
            "pitch": "C4",
            "onset_ql": float(index),
            "duration_ql": 1.0,
            "measure": 1,
        }
        for index in range(2)
    ]
    return {
        "schema_version": "1.0",
        "kind": "synth_note_lineage",
        "clean_note_count": 2,
        "performed_note_count": 2,
        "clean_notes": clean,
        "performed_notes": performed,
        "deleted_clean_notes": [],
        "rendered_notes": [
            {
                "rendered_index": 0,
                "pitch_midi_sounding": 58,
                "pitch_midi_written": 60,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "performed_indices": [0, 1],
                "clean_indices": [0, 1],
                "primary_clean_index": 0,
                "relationship": "match",
            }
        ],
    }


def test_tied_render_event_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "note_map.json"
    path.write_text(json.dumps(_map_document()), encoding="utf-8")
    issues = audit.IssueLog()
    document, valid, stats = audit._validate_note_map(
        "sample",
        path,
        [60, 60],
        [60, 60],
        {
            "notes": [{"pitch": 58, "start": 0.0, "end": 2.0}],
            "duration_sec": 2.0,
            "pitchwheels": [],
        },
        {"midi_pitch_space": "sounding", "sounding_transpose": -2},
        issues,
    )
    assert document is not None
    assert valid
    assert stats["tied_notes_folded"] == 1
    assert issues.report() == []


def test_invalid_clean_index_is_rejected(tmp_path: Path) -> None:
    document = _map_document()
    document["clean_notes"][1]["clean_index"] = 3
    path = tmp_path / "note_map.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    issues = audit.IssueLog()
    _document, valid, _stats = audit._validate_note_map(
        "sample",
        path,
        [60, 60],
        [60, 60],
        None,
        {"midi_pitch_space": "sounding", "sounding_transpose": -2},
        issues,
    )
    assert not valid
    assert "clean_index_invalid" in {row["code"] for row in issues.report()}


def test_midi_pitch_space_is_inferred_from_performed_sequence() -> None:
    document = _map_document()
    written_midi = {
        "notes": [
            {"pitch": 60, "start": 0.0, "end": 1.0},
            {"pitch": 60, "start": 1.0, "end": 2.0},
        ]
    }
    sounding_midi = {
        "notes": [
            {"pitch": 58, "start": 0.0, "end": 1.0},
            {"pitch": 58, "start": 1.0, "end": 2.0},
        ]
    }
    assert audit._inferred_midi_shift(document, written_midi, {}) == 0
    assert audit._inferred_midi_shift(document, sounding_midi, {}) == 2


def test_sqlite_target_cache_loads_by_manifest_record(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    sample.mkdir()
    source_hashes = {"performance_audio.wav": hashlib.sha256(b"wav").hexdigest()}
    payload = {
        "ordinal": 0,
        "split": "train",
        "corpus": "test",
        "sample_dir": str(sample),
        "source_hashes": source_hashes,
        **_map_document(),
    }
    archive = tmp_path / "targets.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")
    database = tmp_path / "targets.sqlite"
    audit._write_sqlite_target_cache(archive, database)
    loaded = target_note_map(
        {
            "sample_dir": str(sample),
            "corpus": "test",
            "target_db": str(database),
            "target_record": 0,
            "source_hashes": source_hashes,
        }
    )
    assert loaded["rendered_notes"][0]["performed_indices"] == [0, 1]


def test_group_split_and_prefix_order_are_source_safe() -> None:
    records = []
    for source in range(12):
        for index in range(20):
            records.append(
                {
                    "sample": f"s{source}_{index}",
                    "sample_dir": f"/root/s{source}_{index}",
                    "corpus": "test",
                    "priority": 0,
                    "eligible": True,
                    "source": f"source-{source}",
                    "source_group": f"score:source-{source}",
                    "content_fingerprint": f"{source:02d}-{index:02d}",
                }
            )
    assigned, summary = audit._assign_groups(records, 365)
    memberships = {}
    for split, rows in assigned.items():
        for row in rows:
            memberships.setdefault(row["source_group"], set()).add(split)
    assert all(len(splits) == 1 for splits in memberships.values())
    assert all(assigned[split] for split in audit.SPLITS)
    assert summary["duplicate_rows_excluded"] == 0
    train_sources = {row["source_group"] for row in assigned["train"]}
    train_prefix = assigned["train"][: len(train_sources)]
    assert len({row["source_group"] for row in train_prefix}) == len(train_prefix)
