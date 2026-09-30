from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from alignmodel.transcription.mel_v1 import MelNote


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "label_datacreate_agent_mel.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "label_datacreate_agent_mel", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
labeler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = labeler
SPEC.loader.exec_module(labeler)


def test_mel_candidates_expose_primary_and_alternative_pitches() -> None:
    notes = [
        MelNote(
            pitch=64,
            start=1.0,
            end=1.4,
            confidence=0.9,
            pitch_candidates=(64, 65),
            candidate_confidences=(0.8, 0.2),
            onset_strength=0.7,
            boundary_strength=0.75,
        )
    ]
    candidates = labeler.mel_candidates(notes)
    assert [candidate.pitch for candidate in candidates] == [64, 65]
    assert candidates[0].confidence == 0.9
    assert candidates[1].confidence == pytest.approx(0.18)
    assert candidates[0].acoustic_features == (0.7, 0.9, 0.9, 0.0, 0.8)


def test_agent_document_identifies_frozen_mel_pipeline() -> None:
    prediction = {
        "labels": [
            {
                "start_time": 1.0,
                "end_time": 1.2,
                "type": "wrong_note",
                "score_part": {
                    "start_note_index": 2,
                    "end_note_index": 4,
                    "core_start_note_index": 3,
                    "core_end_note_index": 3,
                    "start_measure": 2,
                    "end_measure": 2,
                },
                "note_ids": ["note_0002", "note_0003", "note_0004"],
            }
        ],
        "pipeline": {"sample_id": "001"},
    }
    document = labeler._agent_document(
        prediction,
        metadata={"mel_checkpoint_sha256": "abc", "training_performed": False},
    )
    assert document["annotator_id"] == labeler.ANNOTATOR_ID
    assert document["agent_labeling"]["method"] == labeler.METHOD
    assert (
        document["agent_labeling"]["transcriber"]
        == "frozen_align_mel_transcriber_v1"
    )
    assert document["agent_labeling"]["training_performed"] is False
    assert "mel transcriber" in document["labels"][0]["comment"]
