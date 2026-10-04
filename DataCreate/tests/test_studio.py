from io import BytesIO
import json
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from datacreate.config import PipelineConfig
from datacreate.web import studio


def wav_bytes(seconds=1, channels=1):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * int(seconds * 8000) * channels)
    return buffer.getvalue()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    score = tmp_path / "piece.musicxml"
    score.write_text("<score-partwise/>")
    config = PipelineConfig(paths={"work_dir": str(tmp_path), "raw_data_score": str(score)})
    monkeypatch.delenv("ALIGN_FEEDBACK_CONFIG", raising=False)
    for key in ("OPENAI_API_KEY", "API_302_KEY", "FISH_AUDIO_API_KEY", "FISH_AUDIO_REFERENCE_ID"):
        monkeypatch.setenv(key, "test-value")
    calls = []

    class Pipeline:
        def __init__(self, config):
            self.root = config.resolved_path("samples_root")

        def create_sample(self, job_id, **kwargs):
            path = self.root / job_id
            path.mkdir(parents=True)
            return path

        def run_stage1_2(self, job): calls.append("score")
        def run_stage3(self, job): calls.append("reference")
        def run_stage4(self, job): calls.append("audio")
        def run_stage5(self, job):
            calls.append("alignment")
            (job / "candidates.json").write_text(json.dumps({"labels": [{"type": "wrong_note", "source": "auto"}]}))
        def run_stage7(self, job): calls.append("features")

    def feedback(source, *, output_dir, text_input, **kwargs):
        calls.append("speech_retry" if text_input else "feedback")
        if not text_input:
            assert source.name == "candidates.json"
            assert json.loads(source.read_text())["labels"][0]["source"] == "auto"
        output_dir.mkdir()
        (output_dir / "feedback.txt").write_text("Try the passage slowly.")
        (output_dir / "feedback.mp3").write_bytes(b"ID3test-audio")

    monkeypatch.setattr(studio, "make_pipeline", Pipeline)
    monkeypatch.setattr(studio, "run_feedback", feedback)
    app = FastAPI()
    app.include_router(studio.studio_router(config))
    return TestClient(app), calls, config, feedback


def submit(client, audio=None, score_id="piece.musicxml"):
    return client.post("/api/studio/takes", files={"audio": ("take.wav", wav_bytes() if audio is None else audio, "audio/wav")}, data={"score_id": score_id})


def test_pipeline_to_mp3_and_status_recovery(harness):
    client, calls, config, _ = harness
    assert client.get("/studio").status_code == 200
    settings = client.get("/api/studio/config").json()
    assert settings["ready"] and settings["scores"][0]["id"] == "piece.musicxml"
    response = submit(client)
    assert response.status_code == 202
    job_id = response.json()["id"]
    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "complete"
    assert state["narration"] == "Try the passage slowly."
    audio = client.get(state["audio_url"])
    assert audio.content == b"ID3test-audio"
    assert audio.headers["content-type"] == "audio/mpeg"
    assert calls == ["score", "reference", "audio", "alignment", "features", "feedback"]
    assert studio.StudioJobs(config).status(job_id)["status"] == "complete"
    assert client.post(f"/api/studio/takes/{job_id}/retry").status_code == 409


@pytest.mark.parametrize("audio", [b"invalid", wav_bytes(.5), wav_bytes(1, channels=2), wav_bytes()[:-5]], ids=["invalid", "short", "stereo", "truncated"])
def test_invalid_audio_releases_job_slot(harness, audio):
    client, calls, _, _ = harness
    assert submit(client, audio=audio).status_code == 422
    assert not calls
    assert submit(client).status_code == 202


def test_score_paths_are_allowlisted(harness):
    client, calls, _, _ = harness
    assert submit(client, score_id="../piece.musicxml").status_code == 422
    assert not calls
    response = client.post("/api/studio/takes", files={"audio": ("take.wav", wav_bytes()), "score": ("../../my-score.musicxml", b"<score-partwise/>")})
    assert response.status_code == 202
    assert client.get("/api/studio/takes/not-an-id").status_code == 404


def test_missing_credentials_does_not_start_pipeline(harness, monkeypatch):
    client, calls, _, _ = harness
    monkeypatch.delenv("OPENAI_API_KEY")
    assert client.get("/api/studio/config").json()["ready"] is False
    assert submit(client).status_code == 503
    assert not calls


def test_default_local_fish_needs_no_fish_credentials(harness, monkeypatch):
    client, _, config, _ = harness
    monkeypatch.delenv("FISH_AUDIO_API_KEY")
    monkeypatch.delenv("FISH_AUDIO_REFERENCE_ID")
    feedback = studio.StudioJobs(config).feedback_config()
    assert feedback.llm_base_url == "https://api.ssstoken.net/v1"
    assert feedback.llm_model == "gpt-6-luna"
    assert feedback.llm_api_key_env == "OPENAI_API_KEY"
    assert feedback.fish_local is True
    assert feedback.fish_base_url == "http://127.0.0.1:8081"
    assert feedback.fish_model == "fish-speech-1.5"
    assert client.get("/api/studio/config").json()["ready"] is True
    assert submit(client).status_code == 202


def test_explicit_hosted_config_still_requires_fish_key(harness, monkeypatch, tmp_path):
    client, _, _, _ = harness
    hosted = tmp_path / "hosted.yaml"
    hosted.write_text("fish_local: false\n")
    monkeypatch.setenv("ALIGN_FEEDBACK_CONFIG", str(hosted))
    monkeypatch.delenv("FISH_AUDIO_API_KEY")
    settings = client.get("/api/studio/config").json()
    assert settings["ready"] is False
    assert "FISH_AUDIO_API_KEY" in settings["message"]


def test_speech_failure_preserves_text_and_retries_without_alignment(harness, monkeypatch):
    client, calls, _, success = harness
    def failure(source, *, output_dir, **kwargs):
        output_dir.mkdir()
        (output_dir / "feedback.txt").write_text("Saved narration")
        raise RuntimeError("private provider details")
    monkeypatch.setattr(studio, "run_feedback", failure)
    job_id = submit(client).json()["id"]
    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "failed" and state["can_retry"]
    assert state["narration"] == "Saved narration"
    assert "private provider details" not in state["message"]
    assert client.get(f"/api/studio/takes/{job_id}/audio").status_code == 404
    monkeypatch.setattr(studio, "run_feedback", success)
    assert client.post(f"/api/studio/takes/{job_id}/retry").status_code == 202
    assert calls == ["score", "reference", "audio", "alignment", "features", "speech_retry"]
    assert client.get(f"/api/studio/takes/{job_id}").json()["status"] == "complete"


def test_busy_and_interrupted_jobs(harness):
    _, _, config, _ = harness
    jobs = studio.StudioJobs(config)
    job_id = "a" * 32
    jobs.directory(job_id).mkdir(parents=True)
    jobs.save(job_id, status="processing", stage="alignment")
    assert jobs.status(job_id)["status"] == "failed"
    jobs.reserve(job_id)
    assert jobs.status(job_id)["status"] == "processing"
    with pytest.raises(studio.HTTPException) as error:
        jobs.reserve("b" * 32)
    assert error.value.status_code == 409


def test_upload_limit(tmp_path):
    import asyncio
    upload = studio.UploadFile(BytesIO(b"too big"), filename="score.xml")
    with pytest.raises(studio.HTTPException) as error:
        asyncio.run(studio.save_upload(upload, tmp_path / "score.xml", 3))
    assert error.value.status_code == 413


def test_analysis_progress_reads_live_model_stage_and_does_not_fake_completion(harness):
    _, _, config, _ = harness
    jobs = studio.StudioJobs(config)
    job_id = "c" * 32
    jobs.directory(job_id).mkdir(parents=True)
    sample = jobs.root / "samples" / job_id
    sample.mkdir(parents=True)
    jobs.reserve(job_id)
    jobs.save(job_id, status="processing", stage="alignment")
    assert jobs.status(job_id)["analysis_progress"]["current"] == "transcriber"
    for step, completed in [("transcriber", 0), ("aligner", 1), ("labels", 2)]:
        (sample / "alignment_progress.json").write_text(json.dumps({"step": step}))
        progress = jobs.status(job_id)["analysis_progress"]
        assert progress["current"] == step
        assert progress["completed"] == completed
        assert progress["steps"][completed]["state"] == "active"
    # An interrupted atomic write must not break status polling.
    (sample / "alignment_progress.json").write_text("invalid")
    assert jobs.status(job_id)["analysis_progress"]["current"] == "transcriber"
    for step, completed in [("labels", 2), ("narration", 3), ("speech", 4)]:
        jobs.save(job_id, stage="feedback", analysis_step=step)
        assert jobs.status(job_id)["analysis_progress"]["completed"] == completed
    jobs.save(job_id, status="failed")
    progress = jobs.status(job_id)["analysis_progress"]
    assert progress["completed"] == 4
    assert progress["steps"][4]["state"] == "failed"
    jobs.save(job_id, status="complete", stage="complete")
    assert jobs.status(job_id)["analysis_progress"]["completed"] == 5
    assert all(s["state"] == "complete" for s in jobs.status(job_id)["analysis_progress"]["steps"])
