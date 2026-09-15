from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "label_datacreate_agent.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("label_datacreate_agent", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
labeler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = labeler
SPEC.loader.exec_module(labeler)


def test_agent_document_replaces_pipeline_source_and_annotator():
    document = {
        "schema_version": "1.2",
        "annotator_id": "frozen_error_heads_v3",
        "labels": [
            {
                "id": "error_head_v3_0000",
                "source": "frozen_error_heads_v3",
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
                    "pad_notes": 1,
                },
                "pitches": [64, 65, 67],
                "note_ids": ["note_0002", "note_0003", "note_0004"],
                "decoder": {"support": 1, "confidence": 0.9},
            }
        ],
        "pipeline": {"sample_id": "001"},
    }
    agent = labeler.agent_document_from_prediction(
        document, metadata={"policy": "hybrid_max_f1"}
    )
    assert agent["annotator_id"] == "cursor_agent_error_heads_v5"
    assert agent["labels"][0]["source"] == "agent"
    assert agent["labels"][0]["id"] == "agent_0000"
    assert agent["labels"][0]["note_id"] == "note_0003"
    assert agent["labels"][0]["core_note_ids"] == ["note_0003"]
    assert "decoder" not in agent["labels"][0]
    assert agent["agent_labeling"]["replaced_previous_agent_labels"] is True
    assert "agent" in agent["annotator_id"]
