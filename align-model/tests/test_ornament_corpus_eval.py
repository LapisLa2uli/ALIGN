from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "eval_ornament_corpus_v1.py"
)
SPEC = importlib.util.spec_from_file_location("eval_ornament_corpus_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
orn = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = orn
SPEC.loader.exec_module(orn)


def _row(
    sample: str,
    provenance: str,
    source_song_id: str,
    *,
    eligible: bool = True,
) -> dict:
    return {
        "sample": sample,
        "provenance": provenance,
        "source_song_id": source_song_id,
        "eligible": eligible,
    }


def test_equal_quotas_redistribute_exhausted_capacity() -> None:
    quotas = orn._equal_quotas({"a": 1, "b": 10, "c": 10}, 9, 17)
    assert sum(quotas.values()) == 9
    assert quotas["a"] == 1
    assert abs(quotas["b"] - quotas["c"]) <= 1


def test_selection_balances_provenance_then_source_song() -> None:
    rows = []
    for source in ("score:a", "score:b", "score:c"):
        rows.extend(
            _row(f"{source}-{index}", "raw_score", source)
            for index in range(20)
        )
    rows.extend(
        _row(
            f"generated-{index}",
            "random_generated",
            f"generated:{index:03d}",
        )
        for index in range(60)
    )
    selected = orn.select_balanced_manifest_rows(rows, limit=30, seed=123)
    assert len(selected) == 30
    assert sum(row["provenance"] == "raw_score" for row in selected) == 15
    assert (
        sum(row["provenance"] == "random_generated" for row in selected) == 15
    )
    raw_counts = {
        source: sum(row["source_song_id"] == source for row in selected)
        for source in ("score:a", "score:b", "score:c")
    }
    assert set(raw_counts.values()) == {5}
    assert len(
        {
            row["source_song_id"]
            for row in selected
            if row["provenance"] == "random_generated"
        }
    ) == 15
    assert selected == orn.select_balanced_manifest_rows(
        list(reversed(rows)), limit=30, seed=123
    )


def test_source_identity_distinguishes_raw_and_generated() -> None:
    assert orn._source_identity(
        {"source": "MozartClConcertoA", "source_score": "C:/scores/work.xml"},
        "a" * 64,
    ) == ("raw_score", "score:work", "raw-score:work")
    assert orn._source_identity({"source": "gen"}, "b" * 64) == (
        "random_generated",
        f"generated:{'b' * 64}",
        "random-generated",
    )


def test_duplicate_audio_is_excluded_before_sampling() -> None:
    rows = [
        {
            **_row("a", "raw_score", "score:a"),
            "content_fingerprint": "content-a",
            "hashes": {"performance_audio.wav": "same-audio"},
            "exclusion_reasons": [],
        },
        {
            **_row("b", "raw_score", "score:a"),
            "content_fingerprint": "content-b",
            "hashes": {"performance_audio.wav": "same-audio"},
            "exclusion_reasons": [],
        },
    ]
    counts = orn.exclude_duplicate_candidates(rows, seed=123)
    assert counts == {"content": 0, "performance_audio": 1}
    assert sum(row["eligible"] for row in rows) == 1
    excluded = next(row for row in rows if not row["eligible"])
    assert excluded["exclusion_reasons"][0]["code"] == (
        "duplicate_performance_audio"
    )


def test_ornament_stats_flag_polyphony_and_out_of_range() -> None:
    stats = orn._ornament_stats(
        {
            "performed_notes": [{"performed_index": 0}],
            "rendered_notes": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "pitch_midi_written": 60,
                    "performed_indices": [0],
                },
                {
                    "start_sec": 0.5,
                    "end_sec": 0.75,
                    "pitch_midi_written": 101,
                    "performed_indices": [],
                },
            ],
        }
    )
    assert stats["rendered_events"] == 2
    assert stats["unmapped_rendered_events"] == 1
    assert stats["max_simultaneous_rendered_events"] == 2
    assert stats["outside_track_b_written_pitch_range"] == 1


def test_transcription_extra_retains_exclusive_rendered_identity() -> None:
    candidate = orn.JointCandidate(60, 0.0, 0.5, 0.9)
    event = orn._candidate_events((candidate,))[0]
    assert event.relationship == "extra"
    assert event.rendered_index == 0
