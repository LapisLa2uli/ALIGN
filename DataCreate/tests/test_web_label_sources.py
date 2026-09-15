from __future__ import annotations

import json

from datacreate.config import PipelineConfig
from datacreate.models import LabelsDocument
from datacreate.web.app import LabelsPayload, _labels_path, create_app


def _document(annotator: str, source: str, label_id: str) -> dict:
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": annotator,
        "self_reported": [],
        "labels": [
            {
                "id": label_id,
                "source": source,
                "start_time": 0.1,
                "end_time": 0.2,
                "type": "wrong_note",
            }
        ],
    }


def test_label_source_paths_are_independent(tmp_path):
    sample = tmp_path / "001"
    sample.mkdir()
    _labels_path(sample, "human").write_text(
        json.dumps(_document("human", "manual", "human_1")),
        encoding="utf-8",
    )
    agent = _document("agent", "agent", "agent_1")
    agent["agent_labeling"] = {"method": "test"}
    _labels_path(sample, "agent").write_text(
        json.dumps(agent), encoding="utf-8"
    )
    human = json.loads(_labels_path(sample, "human").read_text())
    automatic = json.loads(_labels_path(sample, "agent").read_text())
    assert human["labels"][0]["id"] == "human_1"
    assert automatic["labels"][0]["id"] == "agent_1"
    assert _labels_path(sample, "human").name == "labels.json"
    assert _labels_path(sample, "agent").name == "labels_agent.json"


def test_agent_is_a_valid_distinct_label_source():
    document = LabelsDocument.model_validate(
        _document("agent", "agent", "agent_1")
    )
    assert document.labels[0].source == "agent"


def test_sample_routes_read_and_save_selected_source(tmp_path):
    sample = tmp_path / "001"
    sample.mkdir()
    (sample / "performance_audio.wav").write_bytes(b"wav")
    _labels_path(sample, "human").write_text(
        json.dumps(_document("human", "manual", "human_1")),
        encoding="utf-8",
    )
    agent = _document("agent", "agent", "agent_1")
    agent["agent_labeling"] = {"method": "test"}
    _labels_path(sample, "agent").write_text(
        json.dumps(agent), encoding="utf-8"
    )
    app = create_app(
        PipelineConfig(
            schema_version="1.1",
            paths={"samples_root": str(tmp_path)},
            taxonomy=["wrong_note"],
        )
    )
    endpoints = {
        route.name: route.endpoint
        for route in app.routes
        if getattr(route, "name", None)
        and getattr(route, "endpoint", None) is not None
    }
    loaded = endpoints["get_sample"]("001", "agent")
    assert loaded["labels"][0]["id"] == "agent_1"
    assert loaded["label_source"] == "agent"

    replacement = _document("agent-edited", "agent", "agent_2")
    result = endpoints["save_labels"](
        "001",
        LabelsPayload(
            labels=replacement["labels"],
            annotator_id="agent-edited",
        ),
        "agent",
    )
    assert result == {"status": "saved", "label_source": "agent"}
    assert json.loads(_labels_path(sample, "human").read_text())["labels"][0][
        "id"
    ] == "human_1"
    saved_agent = json.loads(_labels_path(sample, "agent").read_text())
    assert saved_agent["labels"][0]["id"] == "agent_2"
    assert saved_agent["agent_labeling"]["edited_in_gui"] is True
