from dataclasses import replace
import json

import httpx
import pytest

from datacreate.feedback import FeedbackConfig, FeedbackError, run_feedback, synthesize_speech


@pytest.fixture
def qwen_config(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-qwen")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "test-workspace")
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "")
    monkeypatch.setenv("FISH_AUDIO_REFERENCE_ID", "")
    return replace(FeedbackConfig(), tts_provider="qwen")


def synthesis_response(**changes):
    return httpx.Response(200, json={"output": {"finish_reason": "stop", "audio": {
        "url": "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/test.mp3?signature=private"}},
        "usage": {"input_tokens": 20, "output_tokens": 100, "total_tokens": 120}, **changes})


@pytest.mark.parametrize("language,voice,text", [
    ("English", "Abby_v3.1", "In bar two, listen to the reference."),
    ("Chinese", "xieshurou_v3.1", "第二小节，请听示范。"),
])
def test_qwen_saved_text_contract_and_private_download(tmp_path, qwen_config, language, voice, text):
    source = tmp_path / "teacher.txt"
    source.write_text(text, encoding="utf-8")
    config = replace(qwen_config, language=language, qwen_voice=voice)
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "POST":
            assert str(request.url) == "https://test-workspace.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
            assert request.headers["authorization"] == "Bearer secret-qwen"
            body = json.loads(request.content)
            assert body["model"] == "qwen-audio-3.1-tts-flash"
            assert body["input"]["voice"] == voice
            assert body["input"]["text"] == text
            assert body["input"]["format"] == "mp3"
            assert body["input"]["instruction"] == config.qwen_instruction
            assert "X-DashScope-SSE" not in request.headers
            return synthesis_response()
        assert request.url.scheme == "https"
        assert request.url.query == b"signature=private"
        assert "authorization" not in request.headers and "cookie" not in request.headers
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"ID3audio")

    with httpx.Client(transport=httpx.MockTransport(handle), headers={"Cookie": "private-cookie"}) as client:
        output = run_feedback(source, config=config, text_input=True, output_dir=tmp_path / "run", client=client)
    assert len(calls) == 2
    assert (output / "feedback.mp3").read_bytes() == b"ID3audio"
    manifest = json.loads((output / "feedback.json").read_text())
    assert manifest["status"] == "complete" and manifest["tts_provider"] == "qwen"
    assert manifest["fish_model"] is None and not manifest["fish_local"]
    usage = json.loads((output / "feedback.tts.json").read_text())["usage"]
    assert usage == {"input_tokens": 20, "output_tokens": 100, "total_tokens": 120}
    for path in output.glob("*.json"):
        assert "secret-qwen" not in path.read_text() and "signature=" not in path.read_text()


@pytest.mark.parametrize("variable,value", [("DASHSCOPE_API_KEY", ""),
    ("DASHSCOPE_WORKSPACE_ID", ""), ("DASHSCOPE_WORKSPACE_ID", "https://wrong.example")])
def test_qwen_preflight_before_paid_llm(tmp_path, qwen_config, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    monkeypatch.setenv("API_302_KEY", "secret-llm")
    source = tmp_path / "labels.json"
    source.write_text('[{"type":"wrong_note"}]')
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Paid API call"))) as client:
        with pytest.raises(FeedbackError, match=variable):
            run_feedback(source, config=qwen_config, client=client)


@pytest.mark.parametrize("url", ["https://evil.example/audio.mp3", "https://127.0.0.1/audio.mp3",
    "https://aliyuncs.com.evil.example/audio.mp3", "https://key@result.aliyuncs.com/a.mp3"])
def test_qwen_rejects_unexpected_download_host(tmp_path, qwen_config, url):
    response = synthesis_response(output={"finish_reason": "stop", "audio": {"url": url}})
    calls = []
    def handle(request):
        calls.append(request)
        return response
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FeedbackError, match="unsupported audio URL"):
            synthesize_speech("In bar two.", tmp_path / "audio.mp3", qwen_config, client)
    assert len(calls) == 1


@pytest.mark.parametrize("response", [httpx.Response(401, text="secret-qwen"),
    httpx.Response(429, text="secret-qwen"), httpx.Response(200, text="invalid"),
    synthesis_response(output={"finish_reason": "length"}),
    synthesis_response(output={"finish_reason": "stop", "audio": {}}),
    synthesis_response(code="InvalidParameter", message="secret-qwen")])
def test_qwen_failure_preserves_text_without_leaking_details(tmp_path, qwen_config, response):
    source = tmp_path / "teacher.txt"
    source.write_text("In bar two.")
    output = tmp_path / "run"
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(FeedbackError) as error:
            run_feedback(source, text_input=True, config=qwen_config, output_dir=output, client=client)
    assert "secret-qwen" not in str(error.value)
    assert (output / "feedback.txt").is_file()
    assert json.loads((output / "feedback.json").read_text())["status"] == "tts_failed"
    assert not (output / "feedback.mp3").exists()
    assert not list(output.glob("*.part"))


def test_qwen_interrupted_download_preserves_existing_audio(tmp_path, qwen_config):
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"ID3partial"
            raise httpx.ReadError("private signed URL")
    def handle(request):
        if request.method == "POST":
            return synthesis_response()
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, stream=BrokenStream())
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"existing-audio")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FeedbackError, match="network request failed"):
            synthesize_speech("In bar two.", output, qwen_config, client)
    assert output.read_bytes() == b"existing-audio"
    assert not list(tmp_path.glob("*.part"))


def test_qwen_dry_run_needs_no_credentials(tmp_path, monkeypatch):
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_WORKSPACE_ID", "API_302_KEY"):
        monkeypatch.setenv(key, "")
    source = tmp_path / "labels.json"
    source.write_text('[{"type":"wrong_note"}]')
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Network call"))) as client:
        output = run_feedback(source, config=replace(FeedbackConfig(), tts_provider="qwen"), dry_run=True, client=client)
    assert json.loads((output / "feedback.json").read_text())["status"] == "dry_run"


def test_qwen_comparison_mix_and_plan_retry(tmp_path, qwen_config, monkeypatch, synthesis_inputs, fake_soundfont):
    """Real music synthesis/mixing, with only provider responses substituted."""
    from io import BytesIO
    import numpy as np
    import soundfile as sf
    synthesis_inputs(tmp_path)
    source = tmp_path / "labels.json"
    source.write_text(json.dumps({"labels": [{"type": "wrong_note", "source": "human",
        "start_time": .7, "end_time": 1.1,
        "score_part": {"start_note_index": 1, "end_note_index": 2, "pad_notes": 0}}]}))
    monkeypatch.setenv("API_302_KEY", "test-llm")
    spoken_audio = BytesIO()
    sf.write(spoken_audio, .1 * np.sin(2*np.pi*440*np.arange(16000)/16000), 16000, format="MP3")
    plan = {"points": [{"label_index": 0, "intro": "In bar two, listen to the reference.",
        "performance_intro": "Now listen to your playing.", "feedback": "Practise the connection slowly."}]}
    calls = []
    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(plan)}}]})
        if request.method == "POST":
            return synthesis_response()
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=spoken_audio.getvalue())
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        output = run_feedback(source, config=qwen_config, client=client, output_dir=tmp_path / "first")
        calls.clear()
        retry = run_feedback(output / "playback_plan.json", config=qwen_config, plan_input=True,
                             client=client, output_dir=tmp_path / "retry")
    assert len(calls) == 6 and not any("completions" in path for path in calls)
    timeline = json.loads((retry / "timeline.json").read_text())["segments"]
    assert [row["kind"] for row in timeline] == ["speech", "reference", "speech", "performance", "speech"]
    assert sf.info(retry / "feedback.mp3").duration > 3
    assert len(list(retry.glob("*.tts.json"))) == 3
    for name in ("reference-000.wav", "excerpt-000.wav"):
        assert (output / name).read_bytes() == (retry / name).read_bytes()
