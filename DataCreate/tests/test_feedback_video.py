import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("verovio")
pytest.importorskip("resvg_py")

from datacreate.feedback import prepare_report, add_score_locations
from datacreate.feedback_synthesis import prepare_examples, render_example
from datacreate.feedback_video import active_events, build_scenes, render_frame, WIDTH, HEIGHT


@pytest.fixture
def video_inputs(tmp_path, synthesis_inputs, fake_soundfont):
    def make(kind="extra_note"):
        synthesis_inputs(tmp_path)
        path = tmp_path / "note_alignment_v2.json"
        alignment = json.loads(path.read_text())
        if kind == "wrong_note":
            alignment["transcribed_notes"][1]["pitch"] = 63
        elif kind in {"missed_note", "all_missing"}:
            alignment["transcribed_notes"] = ([n for n in alignment["transcribed_notes"] if n["pitch"] != 62]
                                               if kind == "missed_note" else [])
            alignment["events"] = ([e for e in alignment["events"] if e["sounding_index"] != 1]
                                   if kind == "missed_note" else [])
        path.write_text(json.dumps(alignment))
        label = {"type": "missed_note" if kind == "all_missing" else kind, "source": "manual",
                 "score_event_indices": [0, 1, 2, 3] if kind == "all_missing" else [0, 1] if kind == "wrong_note" else [1],
                 "start_time": .2 if kind == "all_missing" else 1.1,
                 "end_time": 2.7 if kind == "all_missing" else 1.5}
        report = prepare_report([label])
        add_score_locations(report, tmp_path / "verified_score.musicxml")
        pairs = prepare_examples(report, tmp_path, tmp_path / "verified_score.musicxml", tmp_path / "reference_audio.mid")
        feedback = tmp_path / "feedback"
        feedback.mkdir()
        point = {"label_index": 0, "intro": "In bar two, listen to the reference.",
                 "performance_intro": "Now listen to your playing.", "feedback": "Practise this passage slowly."}
        for kind, key in [("reference", "reference_clip"), ("performance", "clip")]:
            point[key] = render_example(pairs[0][kind], feedback / f"{kind}.wav")
        (feedback / "playback_plan.json").write_text(json.dumps({"points": [point]}))
        (feedback / "report.json").write_text(json.dumps(report))
        output = tmp_path / "video"
        output.mkdir()
        return tmp_path, feedback, output, pairs
    return make


def test_cursor_uses_exact_half_open_audio_intervals():
    events = [{"start": .13, "end": .46, "ids": ["a"]}, {"start": .46, "end": .72, "ids": ["b"]}]
    assert active_events(events, .12) == []
    assert active_events(events, .13) == [events[0]]
    assert active_events(events, .46) == [events[1]]
    assert active_events(events, .72) == []


@pytest.mark.parametrize("kind,color,expected", [("extra_note", "extra", 1), ("wrong_note", "wrong", 1),
                                                  ("missed_note", "missing", 1), ("all_missing", "missing", 4)])
def test_notation_and_error_identities(video_inputs, kind, color, expected):
    sample, feedback, output, pairs = video_inputs(kind)
    scenes = build_scenes(sample, feedback, output)
    scene = scenes[0]
    notation = [n for p in scene["performance"] for n in p["notes"]]
    marked = [n for n in notation if n.get("error") == color]
    assert len(marked) == expected
    played = scene["performance_events"]
    assert [(n["start"], n["end"], n["pitch"]) for n in played] == [(n["start"], n["end"], n["pitch"]) for n in pairs[0]["performance"]["notes"]]
    assert not any(n.get("ghost") for n in played)
    xml = ET.parse(next((output / "score-assets").glob("*performance*.musicxml")))
    note_ids = [n.attrib["id"] for n in xml.findall(".//note") if "id" in n.attrib]
    assert len(note_ids) == len(set(note_ids)) == len(notation)
    assert len(xml.findall(".//measure")) == 1
    assert "bounding-box" not in next((output / "score-assets").glob("*performance*.svg")).read_text()
    segment = {"kind": "performance", "label_index": 0, "start_time": 1., "end_time": 10.}
    t = 1.+played[0]["start"]+.01 if played else 2.
    frame, state = render_frame(t, [segment], scenes, 10.)
    assert frame.size == (WIDTH, HEIGHT)
    assert state["active_ids"] == ([played[0]["id"]] if played else [])


def test_missing_narration_is_not_silently_omitted(video_inputs):
    sample, feedback, output, _ = video_inputs()
    path = feedback / "report.json"
    report = json.loads(path.read_text())
    report["labels"].append(report["labels"][0])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="every label"):
        build_scenes(sample, feedback, output)


def test_stale_alignment_is_rejected(video_inputs):
    sample, feedback, output, _ = video_inputs()
    with (sample / "note_alignment_v2.json").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="alignment changed"):
        build_scenes(sample, feedback, output)
