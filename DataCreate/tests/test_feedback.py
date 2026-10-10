from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from datacreate.feedback import FeedbackConfig, FeedbackError, add_score_locations, main, prepare_report, prepare_speech_text, run_feedback


@pytest.fixture
def labels(tmp_path):
    path = tmp_path / "labels_agent.json"
    path.write_text(json.dumps({"schema_version": "1.2", "annotator_id": "private-id",
        "agent_labeling": {"checkpoint": "private/path"}, "labels": [
            {"type": "wrong_note", "source": "agent", "start_time": 1.0, "end_time": 1.2,
             "score_part": {"start_note_index": 2, "end_note_index": 6, "pad_notes": 2,
                            "core_start_note_index": 4, "core_end_note_index": 4},
             "pitches": [60, 62, 64, 65, 67]},
            {"type": "extra_note", "source": "auto_rejected"},
            {"type": "wrong_note", "source": "synthetic", "comment": "repeated pass 2"},
        ]}), encoding="utf-8")
    return path


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("API_302_KEY", "secret-llm")
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "secret-fish")
    monkeypatch.setenv("FISH_AUDIO_REFERENCE_ID", "voice-id")


def llm_response(text="The report flags a possible wrong note. Try this passage slowly."):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": text}}]})


@pytest.mark.parametrize("model", ["drama-3-preview", "s2.1-pro"])
def test_latest_hosted_fish_config_and_controls(tmp_path, credentials, monkeypatch, model):
    config = FeedbackConfig.load(Path(__file__).parents[1] / "config" / "feedback.fish.yaml")
    assert config.fish_model == "drama-3-preview" and not config.fish_local
    config = replace(config, fish_model=model)
    monkeypatch.setenv("SSSTOKEN_API_KEY", "")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    source = tmp_path / "teacher.txt"
    source.write_text("In bar 2, listen to the reference.")
    calls = []
    def handle(request):
        calls.append(request)
        assert str(request.url) == "https://api.fish.audio/v1/tts"
        assert request.headers["model"] == model
        assert request.headers["authorization"] == "Bearer secret-fish"
        body = json.loads(request.content)
        assert body["reference_id"] == "voice-id"
        assert body["text"] == "In bar two, listen to the reference."
        assert body["latency"] == "normal" and body["sample_rate"] == 44100
        assert body["prosody"] == {"speed": 1., "volume": 0, "normalize_loudness": True}
        assert body["condition_on_previous_chunks"] is True
        assert "references" not in body
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"ID3audio")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(source, config=config, text_input=True, client=client, output_dir=tmp_path / "run")
    manifest = json.loads((output / "feedback.json").read_text())
    assert manifest["tts_model"] == model and manifest["tts_preview"] == (model == "drama-3-preview")
    assert manifest["status"] == "complete" and len(calls) == 1


def test_unknown_fish_model_cannot_silently_fallback():
    with pytest.raises(FeedbackError, match="Unsupported hosted Fish model"):
        replace(FeedbackConfig(), fish_model="drama-3-typo").validate()


def test_drama_failure_does_not_retry_or_switch_models(tmp_path, credentials):
    source = tmp_path / "teacher.txt"
    source.write_text("In bar two.")
    calls = []
    def handle(request):
        calls.append(request.headers["model"])
        return httpx.Response(503, text="secret-fish")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FeedbackError, match="HTTP 503"):
            run_feedback(source, config=replace(FeedbackConfig(), fish_model="drama-3-preview"),
                         text_input=True, client=client, output_dir=tmp_path / "run")
    assert calls == ["drama-3-preview"]
    assert (tmp_path / "run" / "feedback.txt").is_file()


def test_full_pipeline_contract_and_artifacts(labels, tmp_path, credentials):
    calls = []
    progress = []
    audio = b"ID3\x04\x00\x00" + b"test-audio" * 50

    def handle(request):
        calls.append(request)
        payload = json.loads(request.content)
        if len(calls) == 1:
            assert str(request.url) == "https://api.302.ai/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer secret-llm"
            assert payload["model"] == "gpt-4o-mini"
            assert payload["stream"] is False
            report = json.loads(payload["messages"][1]["content"])["report"]
            assert report["label_count"] == 1
            assert report["labels"][0]["score_location"]["phrase"] == "the marked passage"
            assert set(report["labels"][0]) == {"type", "source", "score_location"}
            assert "private" not in request.content.decode()
            return llm_response()
        assert str(request.url) == "https://api.fish.audio/v1/tts"
        assert request.headers["authorization"] == "Bearer secret-fish"
        assert request.headers["model"] == "s2.1-pro"
        assert payload["reference_id"] == "voice-id"
        assert payload["format"] == "mp3"
        assert payload["text"].startswith("The report flags")
        assert "labels" not in payload
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=audio)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(labels, output_dir=tmp_path / "run", client=client, progress=progress.append)
    assert progress == ["labels", "narration", "speech"]
    assert len(calls) == 2
    assert (output / "feedback.mp3").read_bytes() == audio
    assert json.loads((output / "feedback.json").read_text())["status"] == "complete"
    for path in output.glob("*.json"):
        assert "secret-" not in path.read_text()
    assert not list(output.glob("*.part"))


def test_dry_run_requires_no_keys_and_makes_no_calls(labels, tmp_path, monkeypatch):
    for name in ("API_302_KEY", "FISH_AUDIO_API_KEY", "FISH_AUDIO_REFERENCE_ID"):
        monkeypatch.delenv(name, raising=False)
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Network call"))) as client:
        output = run_feedback(labels, output_dir=tmp_path / "preview", dry_run=True, client=client)
    assert json.loads((output / "feedback.json").read_text())["status"] == "dry_run"
    assert (output / "request.json").exists()
    assert not (output / "feedback.mp3").exists()


@pytest.mark.parametrize("metadata", [
    {"status": "alignment_uncertain"},
    {"summary": {"status": "alignment_uncertain"}},
    {"agent_labeling": {"status": "alignment_uncertain"}},
])
def test_uncertain_model_documents_cannot_become_empty_feedback(metadata):
    with pytest.raises(FeedbackError, match="not a zero-error result"):
        prepare_report({"labels": [], **metadata})


def test_spoken_budget_keeps_whole_points_and_audio_pairs():
    from datacreate.feedback import validate_playback_plan
    points = [{"label_index": i, "intro": "In bar two, listen to the score.",
               "performance_intro": "Now listen to your performance.",
               "feedback": " ".join(["Practice"] * 28) + ".",
               "reference_clip": {"file": f"reference-{i}.wav"},
               "clip": {"file": f"excerpt-{i}.wav"}} for i in range(3)]
    plan = validate_playback_plan({"points": points}, FeedbackConfig())
    assert len(plan["points"]) == 2
    assert plan["points"][1]["feedback"] == points[1]["feedback"]
    assert plan["points"][1]["clip"] == points[1]["clip"]
    assert plan["points"][1]["reference_clip"] == points[1]["reference_clip"]
    points[0]["feedback"] = "Practice " * 101
    with pytest.raises(FeedbackError, match="100-word"):
        validate_playback_plan({"points": points[:1]}, FeedbackConfig())


def test_audio_plan_prefers_distinct_passages_before_applying_word_budget():
    from datacreate.feedback import validate_playback_plan
    report = {"labels": [
        {"type": "wrong_note", "score_location": {"spans": [{"bar": 12}]}},
        {"type": "wrong_note", "score_location": {"spans": [{"bar": 12}]}},
        {"type": "repetition", "score_location": {"spans": [{"bar": 1}, {"bar": 3}]}},
    ]}
    points = [{"label_index": i, "intro": "Listen to the score.",
               "performance_intro": "Listen to your performance.",
               "feedback": "Practice " * 30} for i in range(3)]
    plan = validate_playback_plan({"points": points}, FeedbackConfig(), report, {0, 1, 2})
    assert [p["label_index"] for p in plan["points"]] == [0, 2]
    assert len(report["labels"]) == 3


def test_candidates_check_alignment_status_before_narration(tmp_path):
    source = tmp_path / "candidates.json"
    source.write_text('{"labels": []}')
    (tmp_path / "note_alignment_v2.json").write_text(json.dumps({
        "summary": {"status": "alignment_uncertain"}, "labels": []}))
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Network call"))) as client:
        with pytest.raises(FeedbackError, match="not a zero-error result"):
            run_feedback(source, output_dir=tmp_path / "run", client=client)
    assert not (tmp_path / "run").exists()


def test_missing_tts_key_fails_before_paid_llm_call(labels, tmp_path, credentials, monkeypatch):
    monkeypatch.delenv("FISH_AUDIO_API_KEY")
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Network call"))) as client:
        with pytest.raises(FeedbackError, match="FISH_AUDIO_API_KEY"):
            run_feedback(labels, output_dir=tmp_path / "run", client=client)
    assert not (tmp_path / "run").exists()


def test_failed_tts_preserves_text_and_retries_without_llm(labels, tmp_path, credentials, monkeypatch):
    calls = []

    def fail_tts(request):
        calls.append(request)
        return llm_response() if len(calls) == 1 else httpx.Response(401, text="secret-fish provider detail")

    output = tmp_path / "failed"
    with httpx.Client(transport=httpx.MockTransport(fail_tts)) as client:
        with pytest.raises(FeedbackError, match="HTTP 401") as error:
            run_feedback(labels, output_dir=output, client=client)
    assert "secret-fish" not in str(error.value)
    assert len(calls) == 2
    assert (output / "feedback.txt").exists()
    assert not (output / "feedback.mp3").exists()
    assert json.loads((output / "feedback.json").read_text())["status"] == "tts_failed"
    monkeypatch.delenv("API_302_KEY")

    def retry(request):
        assert request.url.path == "/v1/tts"
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"ID3audio")

    with httpx.Client(transport=httpx.MockTransport(retry)) as client:
        retried = run_feedback(output / "feedback.txt", text_input=True, output_dir=tmp_path / "retry", client=client)
    assert (retried / "feedback.mp3").exists()


@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"oops": True}), httpx.Response(200, json={"choices": []}),
    httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}),
    httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "cut off"}}]}),
    httpx.Response(200, text="not json"), llm_response("x" * 5001),
    httpx.Response(429, text="secret-llm"),
])
def test_llm_failure_never_calls_fish(labels, tmp_path, credentials, response):
    calls = []

    def handle(request):
        calls.append(request)
        return response

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FeedbackError):
            run_feedback(labels, output_dir=tmp_path / "run", client=client)
    assert len(calls) == 1
    assert json.loads((tmp_path / "run" / "feedback.json").read_text())["status"] == "llm_failed"


@pytest.mark.parametrize("mime,body", [("application/json", b'{"error":"bad voice"}'),
                                      ("audio/mpeg", b""), ("audio/mpeg", b"not mp3")])
def test_non_audio_never_saved_as_mp3(tmp_path, credentials, mime, body):
    source = tmp_path / "narration.txt"
    source.write_text("Practice this passage slowly.")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": mime}, content=body))) as client:
        with pytest.raises(FeedbackError):
            run_feedback(source, text_input=True, output_dir=tmp_path / "run", client=client)
    assert not (tmp_path / "run" / "feedback.mp3").exists()
    assert not list((tmp_path / "run").glob("*.part"))


def test_interrupted_audio_stream_is_cleaned(tmp_path, credentials):
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"ID3first chunk"
            raise httpx.ReadError("secret-fish should not leak")

    source = tmp_path / "speech.txt"
    source.write_text("Try again slowly.")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "audio/mpeg"}, stream=BrokenStream()))) as client:
        with pytest.raises(FeedbackError, match="network request failed") as error:
            run_feedback(source, text_input=True, output_dir=tmp_path / "run", client=client)
    assert "secret-fish" not in str(error.value)
    assert not list((tmp_path / "run").glob("*.part"))


@pytest.mark.parametrize("exception, expected", [
    (httpx.ConnectError, "Local Fish connection failed"),
    (httpx.ReadTimeout, "Local Fish speech request timed out"),
])
def test_local_fish_transport_error_preserves_plan_recovery(tmp_path, exception, expected):
    source = tmp_path / "speech.txt"
    source.write_text("Try the passage slowly.")
    config = replace(FeedbackConfig(), fish_local=True, fish_base_url="http://127.0.0.1:8081")
    def fail(request):
        raise exception("private connection details", request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(FeedbackError, match=expected) as error:
            run_feedback(source, text_input=True, config=config, output_dir=tmp_path / "run", client=client)
    assert "playback_plan.json with --plan" in str(error.value)
    assert "private connection details" not in str(error.value)
    assert (tmp_path / "run" / "feedback.txt").exists()
    assert not list((tmp_path / "run").glob("*.part"))


def test_text_only_needs_no_fish_credentials(labels, tmp_path, monkeypatch):
    monkeypatch.setenv("API_302_KEY", "key")
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    with httpx.Client(transport=httpx.MockTransport(lambda _: llm_response())) as client:
        output = run_feedback(labels, text_only=True, output_dir=tmp_path / "run", client=client)
    assert (output / "feedback.txt").exists()
    assert json.loads((output / "feedback.json").read_text())["status"] == "text_ready"


@pytest.mark.parametrize("document", [{}, {"events": []}, {"labels": None},
    {"labels": [None]}, {"labels": [{"type": "wrong_note", "start_time": 2, "end_time": 1}]},
    {"labels": [{"type": "wrong_note", "score_part": {"start_note_index": 5, "end_note_index": 2}}]},
    {"labels": [{"type": "wrong_note", "pitches": [128]}]},
])
def test_invalid_reports_rejected(document):
    with pytest.raises(FeedbackError):
        prepare_report(document)


def test_empty_report_is_explicit_and_does_not_invent_passages(tmp_path, monkeypatch):
    from datacreate.feedback import EMPTY_REPORT_NARRATION, PIPELINE_REVISION
    source = tmp_path / "candidates.json"
    source.write_text('{"labels": []}')
    monkeypatch.delenv("API_302_KEY", raising=False)
    calls = []
    config = FeedbackConfig(fish_local=True, fish_base_url="http://127.0.0.1:8081")
    def handle(request):
        calls.append(request.url.path)
        assert request.url.path == "/v1/tts"
        assert json.loads(request.content)["text"] == EMPTY_REPORT_NARRATION
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"ID3test-audio")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(source, config=config, client=client, output_dir=tmp_path / "run")
    manifest = json.loads((output / "feedback.json").read_text())
    assert calls == ["/v1/tts"]
    assert manifest["label_count"] == 0 and manifest["no_issues_marked"]
    assert manifest["excerpt_reason"] == "no_marked_issues"
    assert manifest["llm_model"] is None
    assert manifest["pipeline_revision"] == PIPELINE_REVISION
    assert not manifest["performance_excerpts"] and not manifest["reference_excerpts"]


def test_empty_unknown_source_and_repetition_semantics():
    assert prepare_report({"labels": []})["label_count"] == 0
    report = prepare_report([
        {"type": "repetition", "extra_copies": 2},
        {"type": "wrong_note", "comment": "first pass"},
        {"type": "wrong_note", "comment": "(pass 2)"},
    ])
    assert report["label_count"] == 2
    assert report["counts_by_source"] == {"unknown": 2}
    assert report["labels"][0]["extra_copies"] == 2


def test_identity_only_documents_and_conflicting_redundant_fields():
    report = prepare_report([
        {"type": "missed_note", "note_ids": ["note_0007"]},
        {"type": "wrong_note", "score_event_indices": [10, 11]},
        {"type": "wrong_note", "score_part": {"start_note_index": 1, "end_note_index": 3},
         "note_ids": ["note_0099"], "core_note_ids": ["note_0099"]},
    ])
    assert report["labels"][0]["note_ids"] == ["note_0007"]
    assert report["labels"][1]["score_event_indices"] == [10, 11]
    assert "note_ids" not in report["labels"][2]
    assert "core_note_ids" not in report["labels"][2]


def test_feedback_yaml_loads_and_rejects_secret_fields(tmp_path):
    path = tmp_path / "feedback.yaml"
    path.write_text("language: English\nllm_model: my-enabled-model\n")
    assert FeedbackConfig.load(path).llm_model == "my-enabled-model"
    path.write_text("api_key: do-not-store-here\n")
    with pytest.raises(FeedbackError, match="Unknown"):
        FeedbackConfig.load(path)


def test_output_not_overwritten_and_input_limit(labels, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FeedbackError, match="already exists"):
        run_feedback(labels, output_dir=output, dry_run=True)
    with pytest.raises(FeedbackError, match="max_input_chars"):
        run_feedback(labels, dry_run=True, config=replace(FeedbackConfig(), max_input_chars=10))


@pytest.mark.parametrize("url", ["http://api.302.ai/v1", "https://key@api.302.ai", "https://api.302.ai?key=secret"])
def test_insecure_config_rejected(url):
    with pytest.raises(FeedbackError):
        replace(FeedbackConfig(), llm_base_url=url).validate()


def test_cli_dry_run(labels, tmp_path, capsys):
    main(["--labels", str(labels), "--output", str(tmp_path / "preview"), "--dry-run"])
    assert "Feedback saved" in capsys.readouterr().out


def test_local_tts_without_cloud_credentials(tmp_path, monkeypatch):
    for name in ("API_302_KEY", "FISH_AUDIO_API_KEY", "FISH_AUDIO_REFERENCE_ID"):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / "speech.txt"
    source.write_text("Practice the marked passage slowly.")
    config = replace(FeedbackConfig(), fish_local=True, fish_base_url="http://127.0.0.1:8081",
                     fish_model="fish-speech-1.5")

    def handle(request):
        assert str(request.url) == "http://127.0.0.1:8081/v1/tts"
        assert "authorization" not in request.headers
        assert "model" not in request.headers
        assert json.loads(request.content)["reference_id"] is None
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"ID3audio")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(source, text_input=True, config=config, client=client, output_dir=tmp_path / "run")
    assert (output / "feedback.mp3").exists()
    assert json.loads((output / "feedback.json").read_text())["fish_local"] is True


@pytest.mark.parametrize("url", ["http://example.com:8081", "https://api.fish.audio", "http://127.0.0.1.evil.example"])
def test_local_mode_cannot_silently_send_to_remote_host(url):
    with pytest.raises(FeedbackError):
        replace(FeedbackConfig(), fish_local=True, fish_base_url=url).validate()


@pytest.fixture
def feedback_score(tmp_path):
    from music21 import meter, note, stream, tie

    score = stream.Score()
    part = stream.Part()
    first = stream.Measure(number=8)
    first.append(meter.TimeSignature("4/4"))
    first.append(note.Rest(quarterLength=1))
    first.append(note.Note("C4", quarterLength=1))
    first.append(note.Note("D4", quarterLength=1))
    held = note.Note("E4", quarterLength=1)
    held.tie = tie.Tie("start")
    first.append(held)
    second = stream.Measure(number=9)
    continuation = note.Note("E4", quarterLength=1)
    continuation.tie = tie.Tie("stop")
    second.append(continuation)
    for pitch in ("F4", "G4", "A4"):
        second.append(note.Note(pitch, quarterLength=1))
    part.append([first, second])
    score.append(part)
    path = tmp_path / "verified_score.musicxml"
    score.write("musicxml", fp=path)
    return path


def test_score_locations_use_core_and_reset_at_each_bar(feedback_score):
    report = prepare_report([
        {"type": "wrong_note", "score_part": {"start_note_index": 0, "end_note_index": 4,
         "pad_notes": 2, "core_start_note_index": 1, "core_end_note_index": 2}},
        {"type": "missed_note", "note_ids": ["note_0003"]},
        {"type": "extra_note", "score_part": {"start_note_index": 3, "end_note_index": 5,
         "pad_notes": 1}},
    ])
    add_score_locations(report, feedback_score)
    locations = [row["score_location"] for row in report["labels"]]
    assert locations[0] == {"phrase": "the 2nd to 3rd notes of bar 8", "scope": "core",
                            "spans": [{"bar": 8, "first_note": 2, "last_note": 3}]}
    # Rests and tied continuations do not add canonical sounding-note positions.
    assert locations[1]["phrase"] == "the 1st note of bar 9"
    assert locations[2]["phrase"] == "the 1st to 3rd notes of bar 9"
    assert locations[2]["scope"] == "context"


def test_score_locations_keep_disjoint_and_cross_bar_spans(feedback_score):
    report = prepare_report([{"type": "wrong_note", "score_event_indices": [3, 2, 0, 2]}])
    add_score_locations(report, feedback_score)
    assert report["labels"][0]["score_location"]["phrase"] == (
        "the 1st note of bar 8; the 3rd note of bar 8; the 1st note of bar 9")


def test_score_locations_without_score_never_invent_ordinals():
    report = prepare_report([
        {"type": "wrong_note", "measure_number": 8, "note_id": "note_0100"},
        {"type": "missed_note", "score_part": {"start_note_index": 100, "end_note_index": 104,
         "start_measure": 8, "end_measure": 9}},
        {"type": "wrong_note", "note_id": "note_0100"},
    ])
    add_score_locations(report)
    assert [row["score_location"]["phrase"] for row in report["labels"]] == [
        "bar 8", "bars 8 to 9", "the marked passage"]


def test_score_locations_reject_out_of_bounds(feedback_score):
    report = prepare_report([{"type": "wrong_note", "note_id": "note_0006"}])
    with pytest.raises(FeedbackError, match="beyond the supplied score"):
        add_score_locations(report, feedback_score)


def test_spoken_numbers_and_non_english_text():
    text = "Check the 7th and 8th notes of bar 2. Repeat the 21st note of bar 124."
    assert prepare_speech_text(text, "English") == (
        "Check the seventh and eighth notes of bar two. Repeat the twenty first note of bar one hundred twenty four.")
    assert prepare_speech_text("The 12th, 20th and 100th notes.", "en-US") == "The twelfth, twentieth and one hundredth notes."
    assert prepare_speech_text("3.14, -3, +3, id42, 1000000", "English") == "3.14, -3, +3, id42, 1000000"
    assert prepare_speech_text("小节 2", "Chinese") == "小节 2"


def test_saved_text_retry_speaks_and_saves_expanded_numbers(tmp_path):
    source = tmp_path / "original.txt"
    source.write_text("Check the 7th note of bar 2.")
    config = replace(FeedbackConfig(), fish_local=True, fish_base_url="http://127.0.0.1:8081")
    expected = "Check the seventh note of bar two."

    def handle(request):
        assert request.url.path == "/v1/tts"
        assert json.loads(request.content)["text"] == expected
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"ID3audio")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(source, text_input=True, config=config, output_dir=tmp_path / "run", client=client)
    assert (output / "feedback.txt").read_text().strip() == expected


@pytest.mark.parametrize("explicit_score", [False, True])
def test_dry_run_resolves_score_and_filters_outbound_fields(feedback_score, tmp_path, explicit_score):
    source = tmp_path / "labels.json"
    source.write_text(json.dumps([{"type": "wrong_note", "note_id": "note_0001",
        "start_time": 1.0, "end_time": 1.2, "comment": "technical details", "pitches": [62]}]))
    if explicit_score:
        feedback_score = feedback_score.rename(tmp_path / "custom.musicxml")
    output = run_feedback(source, dry_run=True, output_dir=tmp_path / "preview",
                          score_path=feedback_score if explicit_score else None)
    request = json.loads((output / "request.json").read_text())
    outbound = json.loads(request["messages"][1]["content"])["report"]["labels"][0]
    assert outbound["score_location"]["phrase"] == "the 2nd note of bar 8"
    assert set(outbound) == {"type", "source", "score_location"}
    local = json.loads((output / "report.json").read_text())["labels"][0]
    assert local["start_time"] == 1.0
    assert local["comment"] == "technical details"
    assert "score_sha256" in json.loads((output / "feedback.json").read_text())
