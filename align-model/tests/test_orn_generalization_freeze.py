from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_orn_generalization_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "freeze_orn_generalization_v1", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
freeze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeze
SPEC.loader.exec_module(freeze)


def _row(
    sample: str,
    *,
    provenance: str = "generated",
    source: str | None = None,
    verified: str | None = None,
    clean: str | None = None,
    near: str | None = None,
    lineage: str | None = None,
    audio: str | None = None,
) -> dict:
    return {
        "sample": sample,
        "provenance": provenance,
        "source_song_id": source or f"generated:{sample}",
        "source_score_sha256": (
            (source or sample) if provenance == "raw" else None
        ),
        "source_hashes": {
            "verified_score.musicxml": verified or f"verified-{sample}",
            "performance_audio.wav": audio or f"audio-{sample}",
        },
        "clean_fingerprint": clean or f"clean-{sample}",
        "near_lineage_hash": near or f"near-{sample}",
        "lineage_hash": lineage or f"lineage-{sample}",
        "audio_lineage_hash": f"audio-lineage-{sample}",
        "exposed_source": False,
        "outputraw_development_matches": [],
    }


def test_group_records_unions_raw_source_and_generated_lineage() -> None:
    rows = [
        _row("raw-a", provenance="raw", source="raw-source"),
        _row("raw-b", provenance="raw", source="raw-source"),
        _row("gen-a", verified="same-score"),
        _row("gen-b", verified="same-score"),
        _row("gen-c", audio="shared-audio"),
        _row("gen-d", audio="shared-audio"),
    ]
    groups = freeze._group_records(rows)
    memberships = {
        row["sample"]: row["leakage_group"]
        for members in groups.values()
        for row in members
    }
    assert memberships["raw-a"] == memberships["raw-b"]
    assert memberships["gen-a"] == memberships["gen-b"]
    assert memberships["gen-c"] == memberships["gen-d"]
    assert len(groups) == 3


def test_assign_groups_forces_exposed_and_small_raw_population_to_train() -> None:
    rows = [
        _row("exposed"),
        _row("raw", provenance="raw", source="raw-source"),
        *[_row(f"generated-{index:04d}") for index in range(1000)],
    ]
    rows[0]["exposed_source"] = True
    groups = freeze._group_records(rows)
    assigned, summary = freeze._assign_groups(groups, seed=123)
    train_samples = {row["sample"] for row in assigned["train"]}
    assert {"exposed", "raw"} <= train_samples
    assert not summary["raw_lockbox_allowed"]
    assert summary["raw_lockbox_limitation"]


def test_split_assignment_is_deterministic_and_identity_disjoint() -> None:
    rows = [_row(f"g-{index:04d}") for index in range(1000)]
    first, _summary = freeze._assign_groups(
        freeze._group_records(rows), seed=20260918
    )
    second, _summary = freeze._assign_groups(
        freeze._group_records(list(reversed(rows))), seed=20260918
    )
    first_ids = {
        split: {row["sample"] for row in values}
        for split, values in first.items()
    }
    second_ids = {
        split: {row["sample"] for row in values}
        for split, values in second.items()
    }
    assert first_ids == second_ids
    assert all(first_ids[split] for split in freeze.SPLITS)
    assert freeze._split_overlap(first)["passed"]


def test_lockbox_encryption_round_trip() -> None:
    key = bytes(range(32))
    rows = [{"row_id": "abc", "lineage": {"rendered_notes": [1, 2]}}]
    encrypted = freeze._encrypt_targets(rows, key)
    assert b"rendered_notes" not in encrypted
    assert freeze._decrypt_targets(encrypted, key) == rows
