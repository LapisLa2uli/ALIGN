from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "audit_renderer_provenance_v7.py"
SPEC = importlib.util.spec_from_file_location("audit_renderer_provenance_v7", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provenance
SPEC.loader.exec_module(provenance)


def _row(
    index: int,
    clean: int | None,
    *,
    copy_pass: int = 0,
    pitch: int = 60,
) -> dict:
    origin = "match" if clean is not None else "extra"
    return {
        "performed_index": index,
        "clean_index": clean,
        "pitch_midi": pitch,
        "duration_ql": 0.5,
        "relationship": "copy" if copy_pass else origin,
        "origin_relationship": origin,
        "copy_pass": copy_pass,
    }


class RendererProvenanceReplayTests(unittest.TestCase):
    def test_replay_source_and_resume_are_recovered(self) -> None:
        rows = [
            _row(0, 0, pitch=60),
            _row(1, None, pitch=61),
            _row(2, 1, pitch=62),
            _row(3, 0, copy_pass=1, pitch=60),
            _row(4, None, copy_pass=1, pitch=61),
            _row(5, 1, copy_pass=1, pitch=62),
            _row(6, 0, copy_pass=2, pitch=60),
            _row(7, None, copy_pass=2, pitch=61),
            _row(8, 1, copy_pass=2, pitch=62),
            _row(9, 2, pitch=64),
        ]
        result = provenance._validate_replay("sample", rows)
        self.assertEqual(result["maximum_copy_pass"], 2)
        self.assertEqual(len(result["replay_segments"]), 2)
        self.assertIsNone(result["replay_segments"][0]["resume_identity"])
        self.assertEqual(
            result["replay_segments"][1]["resume_identity"], ["clean", "2"]
        )

    def test_clean_identity_pass_must_be_consecutive(self) -> None:
        rows = [
            _row(0, 0),
            _row(1, 0, copy_pass=2),
        ]
        with self.assertRaisesRegex(ValueError, "expected 1"):
            provenance._validate_replay("sample", rows)

    def test_ambiguous_extra_only_source_fails_closed(self) -> None:
        rows = [
            _row(0, None, pitch=61),
            _row(1, None, pitch=61),
            _row(2, None, copy_pass=1, pitch=61),
        ]
        with self.assertRaisesRegex(ValueError, "source blocks"):
            provenance._validate_replay("sample", rows)


class RendererProvenanceOrderingTests(unittest.TestCase):
    def test_equal_onsets_form_explicit_temporal_group(self) -> None:
        rows = [{"start": 0.0}, {"start": 0.0}, {"start": 0.5}]
        assignment, groups = provenance._temporal_groups(rows, "start")
        self.assertEqual(assignment, [0, 0, 1])
        self.assertEqual(groups, [[0, 1], [2]])

    def test_packed_v2_canonical_location_is_normalized(self) -> None:
        row = {
            "canonical_location": {
                "kind": "score_span",
                "score_span": [3, 4],
            },
            "relationship": "copy",
            "copy_pass": 1,
            "source_indices": [3],
        }
        self.assertEqual(
            provenance._event_signature(row),
            ((3, 4), "copy", 1, (3,)),
        )


if __name__ == "__main__":
    unittest.main()
