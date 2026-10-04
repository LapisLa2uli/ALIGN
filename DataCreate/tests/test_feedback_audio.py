from dataclasses import replace
import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from datacreate.feedback import FeedbackConfig, FeedbackError, run_feedback
from datacreate.feedback_audio import checked_clip, compose_audio, extract_clip, locate_excerpts


@pytest.fixture
def performance_labels(tmp_path):
    rate = 16000
    # Distinct regions make it possible to detect a wrong crop or a trim-offset error.
    time = np.arange(rate * 3) / rate
    audio = (0.1 * np.sin(2 * np.pi * (220 + 110 * (time >= 1)) * time)).astype("float32")
    recording = tmp_path / "trimmed.wav"
    sf.write(recording, audio, rate, subtype="FLOAT")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"audio_reference": "trimmed.wav", "labels": [
        {"type": "extra_note", "source": "manual", "measure_number": 2,
         "start_time": 1.0, "end_time": 1.5},
        {"type": "wrong_note", "source": "auto_rejected", "start_time": 100, "end_time": 101},
    ]}))
    return labels, recording, audio


def config():
    return replace(FeedbackConfig(), fish_local=True, fish_base_url="http://127.0.0.1:8081")


def speech_audio():
    data = io.BytesIO()
    rate = 22050
    wave = 0.08 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)
    sf.write(data, wave, rate, format="MP3")
    return data.getvalue()


def test_full_excerpt_sequence_crop_and_offline_retry(performance_labels, tmp_path, monkeypatch):
    labels, recording, original = performance_labels
    monkeypatch.setenv("API_302_KEY", "test-key")
    spoken = []
    calls = []
    plan = {"points": [{"label_index": 0, "intro": "In bar 2, listen to this passage.",
                        "feedback": "You added an extra note. Practise the connection slowly."}]}

    def handle(request):
        calls.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            row = json.loads(body["messages"][1]["content"])["report"]["labels"][0]
            assert row["excerpt_available"] is True and row["label_index"] == 0
            assert "start_time" not in row and "audio_reference" not in row
            assert "trimmed.wav" not in request.content.decode()
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(plan)}}]})
        spoken.append(body["text"])
        assert "Performance excerpt" not in body["text"]
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=speech_audio())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(labels, config=config(), client=client, output_dir=tmp_path / "first")
    assert calls == ["/v1/chat/completions", "/v1/tts", "/v1/tts"]
    assert spoken == ["In bar two, listen to this passage.", plan["points"][0]["feedback"]]
    clip, rate = sf.read(output / "excerpt-000.wav", dtype="float32")
    np.testing.assert_array_equal(clip, original[12000:28000])  # .25 s context on each side
    assert rate == 16000
    timeline = json.loads((output / "timeline.json").read_text())["segments"]
    assert [row["kind"] for row in timeline] == ["speech", "performance", "speech"]
    assert timeline[1]["source_start_time"] == 0.75
    assert timeline[1]["source_end_time"] == 1.75
    assert timeline[1]["start_time"] == pytest.approx(timeline[0]["end_time"] + .3)
    assert timeline[2]["start_time"] == pytest.approx(timeline[1]["end_time"] + .3)
    mixed, rate = sf.read(output / "feedback.mp3", always_2d=True)
    assert rate == 44100 and mixed.shape[1] == 2
    assert len(mixed) / rate == pytest.approx(timeline[-1]["end_time"], abs=.05)

    monkeypatch.delenv("API_302_KEY")
    calls.clear()
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        retry = run_feedback(output / "playback_plan.json", plan_input=True, config=config(),
                             client=client, output_dir=tmp_path / "retry")
    assert calls == ["/v1/tts", "/v1/tts"]
    assert (retry / "excerpt-000.wav").read_bytes() == (output / "excerpt-000.wav").read_bytes()
    assert json.loads((retry / "feedback.json").read_text())["status"] == "complete"
    with pytest.raises(FeedbackError, match="--plan"):
        run_feedback(output / "feedback.txt", text_input=True, config=config())


@pytest.mark.parametrize("points", [[], [{"label_index": 99, "intro": "A", "feedback": "B"}],
    [{"label_index": 0, "intro": "A", "feedback": "B"}] * 2,
    [{"label_index": 0, "intro": "", "feedback": "B"}]])
def test_invalid_plan_never_synthesized(performance_labels, tmp_path, monkeypatch, points):
    monkeypatch.setenv("API_302_KEY", "test-key")
    calls = []

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"points": points})}}]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FeedbackError):
            run_feedback(performance_labels[0], config=config(), client=client, output_dir=tmp_path / "bad")
    assert calls == ["/v1/chat/completions"]


def test_wrong_recording_times_fail_before_provider(performance_labels, tmp_path):
    labels, recording, _ = performance_labels
    document = json.loads(labels.read_text())
    document["labels"][0]["end_time"] = 10
    labels.write_text(json.dumps(document))
    with pytest.raises(FeedbackError, match="matching trimmed audio"):
        run_feedback(labels, config=config(), dry_run=True, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_clip_edges_and_tamper_detection(performance_labels, tmp_path):
    _, recording, _ = performance_labels
    report = {"labels": [{"start_time": 0., "end_time": .1}, {"start_time": 2.9, "end_time": 3.}]}
    clips = locate_excerpts(report, recording, .25)
    assert clips[0]["source_start_time"] == 0
    assert clips[1]["source_end_time"] == 3
    clip = extract_clip(recording, clips[0], tmp_path / "excerpt.wav")
    assert checked_clip(tmp_path, clip) == (tmp_path / "excerpt.wav").resolve()
    (tmp_path / "excerpt.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checked_clip(tmp_path, clip)
    with pytest.raises(ValueError, match="local"):
        checked_clip(tmp_path, {**clip, "file": "../excerpt.wav"})


def test_no_excerpt_option_keeps_plain_prompt(performance_labels, tmp_path):
    output = run_feedback(performance_labels[0], config=replace(config(), include_performance=False),
                          dry_run=True, output_dir=tmp_path / "plain")
    request = json.loads((output / "request.json").read_text())
    assert "excerpt_available" not in request["messages"][1]["content"]
    assert json.loads((output / "feedback.json").read_text())["performance_excerpts"] is False


def test_reference_then_performance_and_saved_plan_retry(performance_labels, tmp_path, monkeypatch):
    import mido
    from music21 import meter, note, stream

    labels, _, _ = performance_labels
    document = json.loads(labels.read_text())
    document["labels"][0]["score_part"] = {"start_note_index": 1, "end_note_index": 2, "pad_notes": 0}
    labels.write_text(json.dumps(document))
    score, part = stream.Score(), stream.Part()
    measure = stream.Measure(number=2)
    measure.append(meter.TimeSignature("4/4"))
    for pitch in [60, 62, 64, 65]:
        measure.append(note.Note(pitch, quarterLength=1))
    part.append(measure)
    score.append(part)
    score.write("musicxml", fp=tmp_path / "verified_score.musicxml")
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for pitch in [60, 62, 64, 65]:
        track.append(mido.Message("note_on", note=pitch, velocity=80))
        track.append(mido.Message("note_off", note=pitch, time=480))
    midi.save(tmp_path / "reference_audio.mid")
    reference = np.sin(2 * np.pi * 550 * np.arange(24000 * 3) / 24000).astype("float32") * .05
    sf.write(tmp_path / "reference_audio.wav", reference, 24000, subtype="FLOAT")
    plan = {"points": [{"label_index": 0, "intro": "In bar two, the score calls for this passage.",
                        "performance_intro": "Now listen to how you played it.",
                        "feedback": "You added a note. Practise the connection slowly."}]}
    monkeypatch.setenv("API_302_KEY", "test-key")
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith("completions"):
            row = json.loads(json.loads(request.content)["messages"][1]["content"])["report"]["labels"][0]
            assert row["score_location"]["phrase"] == "bar 2"
            assert row["reference_available"] is True
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(plan)}}]})
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=speech_audio())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(labels, config=config(), client=client, output_dir=tmp_path / "paired")
    timeline = json.loads((output / "timeline.json").read_text())["segments"]
    assert [x["kind"] for x in timeline] == ["speech", "reference", "speech", "performance", "speech"]
    ref, _ = sf.read(output / "reference-000.wav", dtype="float32")
    np.testing.assert_array_equal(ref, reference[12000:36000])
    assert len(calls) == 4
    calls.clear()
    monkeypatch.delenv("API_302_KEY")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        retry = run_feedback(output / "playback_plan.json", plan_input=True, config=config(),
                             client=client, output_dir=tmp_path / "retry-paired")
    assert calls == ["/v1/tts"] * 3
    assert (retry / "reference-000.wav").read_bytes() == (output / "reference-000.wav").read_bytes()
