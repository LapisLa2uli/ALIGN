from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from alignmodel.joint.index import JointEvent, ScoreEvent
from alignmodel.joint.candidates import (
    CANDIDATE_GENERATION_VERSION,
    LEGACY_CANDIDATE_GENERATION_VERSION,
)
from alignmodel.joint.infer import (
    JointSampleResult,
    build_gui_alignment_payload,
    candidate_configs_for_checkpoint,
    candidate_generation_for_checkpoint,
    match_sounding_index,
)
from alignmodel.joint.lattice import JointCandidate, LatticePath, LatticeStep, JointOperation


@dataclass(frozen=True)
class FakeSounding:
    index: int
    pitch: int
    ql_start: float
    ql_end: float
    start: float
    end: float
    measure: int | None = 1


def test_inference_reports_progress_at_real_stage_boundaries(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from alignmodel.joint import infer

    (tmp_path / "performance_audio.wav").touch()
    (tmp_path / "verified_score.musicxml").touch()
    progress = []
    def extract(*args, **kwargs):
        assert progress == ["transcriber"]
        return object()
    monkeypatch.setitem(sys.modules, "alignmodel.transcription.basic_pitch",
                        SimpleNamespace(extract_sample_basic_pitch_features=extract))
    monkeypatch.setattr(infer.ScoreEventIndex, "from_musicxml", lambda _: SimpleNamespace(events=[]))
    monkeypatch.setattr(infer, "load_joint_model", lambda *a, **kw: (None, None, {}))
    monkeypatch.setattr(infer, "basic_pitch_candidate_union", lambda *a, **kw: [])
    monkeypatch.setattr(infer, "add_score_repeat_hints", lambda notes, events: notes)
    class Lattice:
        def __init__(self, *args): pass
        def decode(self, *args):
            assert progress == ["transcriber", "aligner"]
            return SimpleNamespace(joint_events=lambda _: [])
    monkeypatch.setattr(infer, "SparseJointLattice", Lattice)
    result = infer.infer_joint_sample(tmp_path, tmp_path / "model.pt", progress=progress.append)
    assert result.sample_id == tmp_path.name


def _result(events: tuple[JointEvent, ...]) -> JointSampleResult:
    return JointSampleResult(
        sample_id="demo",
        candidates=(
            JointCandidate(60, 0.1, 0.5, 0.9),
            JointCandidate(62, 0.6, 1.0, 0.8),
            JointCandidate(84, 1.1, 1.3, 0.7),
        ),
        score=(
            ScoreEvent(0, 60, 0.0, 1.0, (0,), 1),
            ScoreEvent(1, 62, 1.0, 2.0, (1,), 1),
        ),
        path=LatticePath(
            steps=(
                LatticeStep(0, (0, 1), JointOperation.MATCH, None, None),
                LatticeStep(1, (1, 2), JointOperation.SUBSTITUTE, None, None),
                LatticeStep(2, None, JointOperation.EXTRA, None, None),
            ),
            trailing_deletions=(),
            score=1.5,
        ),
        events=events,
        checkpoint=Path("joint_decoder.pt"),
        minimum_candidate_confidence=0.65,
        cache_path=Path("cache.npz"),
    )


class JointInferGuiTests(unittest.TestCase):
    def test_checkpoint_schema_selects_matching_candidate_logic(self) -> None:
        v2 = {
            "training": {
                "frontend": {
                    "candidate_generation": CANDIDATE_GENERATION_VERSION
                }
            }
        }
        self.assertEqual(
            candidate_generation_for_checkpoint(v2),
            CANDIDATE_GENERATION_VERSION,
        )
        self.assertTrue(
            candidate_configs_for_checkpoint(v2)[0].adaptive_short_note_rescue
        )
        self.assertEqual(
            candidate_generation_for_checkpoint({}),
            LEGACY_CANDIDATE_GENERATION_VERSION,
        )
        self.assertFalse(
            candidate_configs_for_checkpoint({})[0].adaptive_short_note_rescue
        )

    def test_match_prefers_covering_sounding_note(self) -> None:
        sounding = [
            FakeSounding(0, 60, 0.0, 2.0, 0.0, 1.0),
            FakeSounding(1, 62, 2.0, 3.0, 1.0, 1.5),
        ]
        tied = ScoreEvent(0, 60, 1.0, 2.0, (0, 1), 1)
        self.assertEqual(match_sounding_index(tied, sounding), 0)

    def test_gui_payload_maps_kept_notes_and_drops_unlinked_extra_from_events(self) -> None:
        sounding = [
            FakeSounding(0, 60, 0.0, 1.0, 0.0, 0.5),
            FakeSounding(1, 62, 1.0, 2.0, 0.5, 1.0),
        ]
        payload = build_gui_alignment_payload(
            _result(
                (
                    JointEvent(60, 0.1, 0.5, (0, 1), "match", confidence=0.9),
                    JointEvent(64, 0.6, 1.0, (1, 2), "substitute", confidence=0.8),
                    JointEvent(84, 1.1, 1.3, None, "extra", confidence=0.7),
                )
            ),
            sounding,
        )
        self.assertEqual(payload["engine"], "align-joint")
        self.assertEqual(payload["note_mapping"], [0, 1, None])
        self.assertEqual(len(payload["transcribed_notes"]), 3)
        self.assertEqual([ev["score_index"] for ev in payload["events"]], [0, 1])
        self.assertEqual(payload["events"][0]["pitch"], "C4")
        self.assertEqual(payload["summary"]["mapped_note_count"], 2)
        self.assertEqual(payload["summary"]["ignored_note_count"], 0)
        self.assertEqual(
            payload["summary"]["candidate_generation"],
            LEGACY_CANDIDATE_GENERATION_VERSION,
        )

    def test_gui_payload_keeps_noise_extras_as_ignored(self) -> None:
        sounding = [
            FakeSounding(0, 60, 0.0, 1.0, 0.0, 0.5),
            FakeSounding(1, 62, 1.0, 2.0, 0.5, 1.0),
        ]
        result = JointSampleResult(
            sample_id="demo",
            candidates=(
                JointCandidate(60, 0.1, 0.5, 0.9),
                JointCandidate(84, 0.55, 0.7, 0.4),
                JointCandidate(62, 0.8, 1.1, 0.8),
            ),
            score=(
                ScoreEvent(0, 60, 0.0, 1.0, (0,), 1),
                ScoreEvent(1, 62, 1.0, 2.0, (1,), 1),
            ),
            path=LatticePath(
                steps=(
                    LatticeStep(0, (0, 1), JointOperation.MATCH, None, None),
                    LatticeStep(1, None, JointOperation.NOISE, None, None),
                    LatticeStep(2, (1, 2), JointOperation.MATCH, None, None),
                ),
                trailing_deletions=(),
                score=1.0,
            ),
            events=(
                JointEvent(60, 0.1, 0.5, (0, 1), "match", confidence=0.9),
                JointEvent(62, 0.8, 1.1, (1, 2), "match", confidence=0.8),
            ),
            checkpoint=Path("joint_decoder.pt"),
            minimum_candidate_confidence=0.65,
            cache_path=Path("cache.npz"),
        )
        payload = build_gui_alignment_payload(result, sounding)
        self.assertEqual(payload["note_mapping"], [0, None, 1])
        self.assertTrue(payload["transcribed_notes"][1]["ignored"])
        self.assertEqual(
            payload["transcribed_notes"][1]["ignored_reason"], "joint_noise"
        )
        self.assertFalse(payload["transcribed_notes"][0].get("ignored"))
        self.assertEqual(payload["summary"]["ignored_note_count"], 1)
        self.assertEqual(payload["summary"]["kept_note_count"], 2)
        self.assertEqual([ev["score_index"] for ev in payload["events"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
