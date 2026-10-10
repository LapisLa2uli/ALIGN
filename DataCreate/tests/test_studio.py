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
    for key in ("SSSTOKEN_API_KEY", "API_302_KEY", "FISH_AUDIO_API_KEY", "FISH_AUDIO_REFERENCE_ID"):
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
        assert kwargs['all_labels'] is True
        calls.append("speech_retry" if text_input else "feedback")
        if not text_input:
            assert source.name == "candidates.json"
            assert json.loads(source.read_text())["labels"][0]["source"] == "auto"
        output_dir.mkdir()
        (output_dir / "feedback.txt").write_text("Try the passage slowly.")
        (output_dir / "feedback.mp3").write_bytes(b"ID3test-audio")
        if not text_input:
            (output_dir / "playback_plan.json").write_text('{"points": [{"label_index": 0}]}')

    def video(sample, feedback, output, **kwargs):
        calls.append("video")
        assert (feedback / "feedback.mp3").is_file()
        output.mkdir()
        (output / "feedback.mp4").write_bytes(b"test-video")

    monkeypatch.setattr(studio, "make_pipeline", Pipeline)
    monkeypatch.setattr(studio, "run_feedback", feedback)
    monkeypatch.setattr(studio, "render_feedback_video", video)
    app = FastAPI()
    app.include_router(studio.studio_router(config))
    return TestClient(app), calls, config, feedback


def submit(client, audio=None, score_id="piece.musicxml"):
    return client.post("/api/studio/takes", files={"audio": ("take.wav", wav_bytes() if audio is None else audio, "audio/wav")}, data={"score_id": score_id})


def test_dedicated_upload_page(harness):
    client, _, _, _ = harness
    page = client.get('/upload')
    assert page.status_code == 200
    assert page.headers['cache-control'] == 'no-store'
    assert '/static/studio-upload.js?v=' in page.text
    assert '/static/studio-upload.css?v=' in page.text
    assert 'id="performance-file"' in page.text
    assert 'id="score-choice"' in page.text
    assert 'id="progress-screen" hidden' in page.text
    assert 'id="success-screen" hidden' in page.text
    assert 'role="status">feedback uploaded!</p>' in page.text
    assert '<video id="feedback-video" controls playsinline preload="metadata"' in page.text


def test_pipeline_to_mp4_and_status_recovery(harness):
    client, calls, config, _ = harness
    page = client.get("/studio")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert '/static/studio.js?v=' in page.text
    assert '/static/studio.css?v=' in page.text
    assert '<video id="feedback-video"' in page.text
    assert 'download="align-feedback.mp4"' in page.text
    settings = client.get("/api/studio/config").json()
    assert settings["ready"] and settings["scores"][0]["id"] == "piece.musicxml"
    response = submit(client)
    assert response.status_code == 202
    job_id = response.json()["id"]
    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "complete"
    assert state['detailed_progress']['completed'] == 16
    assert state["narration"] == "Try the passage slowly."
    audio = client.get(state["audio_url"])
    assert audio.content == b"ID3test-audio"
    assert audio.headers["content-type"] == "audio/mpeg"
    video = client.get(state["video_url"])
    assert video.content == b"test-video"
    assert video.headers['content-type'] == 'video/mp4'
    assert video.headers['content-disposition'].startswith('inline;')
    ranged = client.get(state['video_url'], headers={'Range': 'bytes=0-3'})
    assert ranged.status_code == 206 and ranged.content == b'test'
    assert calls == ["score", "reference", "audio", "alignment", "features", "feedback", "video"]
    assert studio.StudioJobs(config).status(job_id)["status"] == "complete"
    assert client.post(f"/api/studio/takes/{job_id}/retry").status_code == 409


@pytest.fixture
def preview_takes(harness):
    client, _, config, _ = harness
    jobs = studio.StudioJobs(config)
    current, previous = 'a' * 32, 'b' * 32
    for job_id in (current, previous):
        directory = jobs.directory(job_id)
        directory.mkdir(parents=True)
        (directory / 'recording.wav').write_bytes(wav_bytes())
        sample = jobs.root / 'samples' / job_id
        sample.mkdir(parents=True)
        (sample / 'verified_score.musicxml').write_text('<score-partwise/>')
        jobs.save(job_id, status='complete', stage='complete', video_status='unavailable')
    video = jobs.directory(previous) / 'video' / 'feedback.mp4'
    video.parent.mkdir()
    video.write_bytes(b'previous-video')
    return client, jobs, current, previous


def test_upload_preview_reuses_video_for_same_pcm_and_score(preview_takes):
    import struct
    client, jobs, current, previous = preview_takes
    # Same samples, different non-audio WAV metadata must still match.
    path = jobs.directory(current) / 'recording.wav'
    original = path.read_bytes()
    extra = b'JUNK' + struct.pack('<I', 4) + b'test'
    path.write_bytes(original[:4] + struct.pack('<I', len(original) - 8 + len(extra))
                     + original[8:12] + extra + original[12:])
    state = client.get(f'/api/studio/takes/{current}').json()
    assert state['status'] == 'complete' and state['video_status'] == 'unavailable'
    assert 'video_url' not in state  # This take's own artifacts remain distinct.
    assert state['preview_video_source_id'] == previous
    video = client.get(state['preview_video_url'], headers={'Range': 'bytes=0-7'})
    assert video.status_code == 206 and video.content == b'previous'
    assert client.get(f'/api/studio/takes/{current}/video').status_code == 404
    # A newly rendered video always takes priority over the previous take.
    own_video = jobs.directory(current) / 'video' / 'feedback.mp4'
    own_video.parent.mkdir()
    own_video.write_bytes(b'new-video')
    state = client.get(f'/api/studio/takes/{current}').json()
    assert state['video_url'] == f'/api/studio/takes/{current}/video'
    assert 'preview_video_url' not in state


@pytest.mark.parametrize('change', ['audio', 'score', 'full_score', 'segment',
                                  'failed', 'uncertain', 'missing', 'empty', 'corrupt'])
def test_upload_preview_rejects_unrelated_or_unavailable_video(preview_takes, change):
    client, jobs, current, previous = preview_takes
    directory = jobs.directory(previous)
    sample = jobs.root / 'samples' / previous
    if change == 'audio':
        (directory / 'recording.wav').write_bytes(wav_bytes(2))
    elif change == 'score':
        (sample / 'verified_score.musicxml').write_text('<different-score/>')
    elif change == 'full_score':
        (sample / 'full_score.musicxml').write_text('<different-full-score/>')
    elif change == 'segment':
        (sample / 'metadata.json').write_text('{"score_segment":{"start_measure":12}}')
    elif change == 'failed':
        jobs.save(previous, status='failed')
    elif change == 'uncertain':
        (sample / 'note_alignment_v2.json').write_text('{"summary":{"status":"alignment_uncertain"}}')
    elif change == 'missing':
        (directory / 'video' / 'feedback.mp4').unlink()
    elif change == 'empty':
        (directory / 'video' / 'feedback.mp4').write_bytes(b'')
    elif change == 'corrupt':
        (directory / 'status.json').write_text('invalid json')
    state = client.get(f'/api/studio/takes/{current}').json()
    assert state['status'] == 'complete'
    assert 'preview_video_url' not in state


def test_upload_preview_refreshes_matches_without_restart(preview_takes):
    client, jobs, current, previous = preview_takes
    assert jobs.matching_video(current) == previous
    # Invalidate cached fingerprints when the file changes.
    (jobs.directory(previous) / 'recording.wav').write_bytes(wav_bytes(2))
    assert jobs.matching_video(current) is None
    (jobs.directory(previous) / 'recording.wav').write_bytes(wav_bytes())
    assert jobs.matching_video(current) == previous
    jobs.save(current, status='failed')
    state = client.get(f'/api/studio/takes/{current}').json()
    assert state['status'] == 'failed' and 'preview_video_url' not in state


def test_uncertain_alignment_stops_before_narration(harness, monkeypatch):
    client, calls, _, _ = harness
    original = studio.make_pipeline

    class UncertainPipeline(original):
        def run_stage5(self, job):
            super().run_stage5(job)
            (job / "candidates.json").write_text('{"labels": []}')
            (job / "note_alignment_v2.json").write_text(json.dumps({
                "labels": [], "summary": {"status": "alignment_uncertain",
                    "score_event_count": 583, "transcribed_note_count": 87},
                "diagnostics": {"match_fraction": .4138, "minimum_match_fraction": .45,
                                "missed_withheld": 481}}))

    monkeypatch.setattr(studio, "make_pipeline", UncertainPipeline)
    job_id = submit(client).json()["id"]
    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "failed"
    assert "not a zero-error result" in state["message"]
    assert state["assessment"]["withheld_count"] == 481
    assert state["assessment"]["score_event_count"] == 583
    assert not state["can_retry"]
    assert "feedback" not in calls
    assert state["analysis_progress"]["current"] == "labels"
    assert next(s for s in state["analysis_progress"]["steps"] if s['id']=='labels')["state"] == "failed"
    assert client.post(f"/api/studio/takes/{job_id}/retry").status_code == 409
    assert client.get(f"/api/studio/takes/{job_id}/audio").status_code == 404
    assert client.get(f"/api/studio/takes/{job_id}/video").status_code == 404


def test_old_uncertain_result_is_not_offered_as_success(harness):
    client, _, config, _ = harness
    job_id = submit(client).json()["id"]
    jobs = studio.StudioJobs(config)
    sample = jobs.root / "samples" / job_id
    (sample / "note_alignment_v2.json").write_text(json.dumps({
        "summary": {"status": "alignment_uncertain"}, "labels": []}))
    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "failed"
    assert "narration" not in state and "audio_url" not in state
    assert "video_url" not in state
    assert (jobs.directory(job_id) / "feedback" / "feedback.mp3").exists()


def test_empty_success_explains_withheld_candidates(harness):
    client, _, config, _ = harness
    job_id = submit(client).json()["id"]
    sample = studio.StudioJobs(config).root / "samples" / job_id
    (sample / "note_alignment_v2.json").write_text(json.dumps({
        "summary": {"status": "ok"}, "labels": [],
        "diagnostics": {"extras_withheld": 2, "match_fraction": .96}}))
    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "complete"
    assert state["assessment"]["label_count"] == 0
    assert "2 possible" in state["assessment"]["message"]


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
    monkeypatch.setenv("SSSTOKEN_API_KEY", "")
    assert client.get("/api/studio/config").json()["ready"] is False
    assert submit(client).status_code == 503
    assert not calls


@pytest.mark.parametrize("failure, expected", [
    ("LLM network request failed. private/path", "network connection"),
    ("LLM returned HTTP 401. private/path", "SSSTOKEN_API_KEY"),
    ("LLM returned HTTP 404. private/path", "model or API route"),
    ("Each feedback point needs spoken intro text. private/path", "incomplete audio plan"),
    ("Speech network request failed. private/path", "local Fish service"),
])
def test_provider_failure_has_actionable_safe_message(harness, monkeypatch, failure, expected):
    client, _, _, _ = harness
    def fail(*args, **kwargs):
        raise RuntimeError(failure)
    monkeypatch.setattr(studio, "run_feedback", fail)
    job_id = submit(client).json()["id"]
    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "failed" and state["can_retry"]
    assert expected in state["message"]
    assert "private/path" not in state["message"]


def test_default_hosted_fish_matches_interpretation(harness):
    client, _, config, _ = harness
    feedback = studio.StudioJobs(config).feedback_config()
    assert feedback.llm_base_url == "https://api.ssstoken.net/v1"
    assert feedback.llm_model == "gpt-6-luna"
    assert feedback.llm_api_key_env == "SSSTOKEN_API_KEY"
    assert feedback.fish_local is False
    assert feedback.fish_base_url == "https://api.fish.audio"
    assert feedback.fish_model == "drama-3-preview"
    assert feedback.fish_reference_id_env == "FISH_AUDIO_REFERENCE_ID"
    assert feedback.fish_speed == 1.0 and feedback.speech_min_wpm == 140
    assert feedback.include_performance is True
    assert client.get("/api/studio/config").json()["ready"] is True
    assert submit(client).status_code == 202
    settings = client.get('/api/studio/config').json()
    assert settings['speech'] == {'provider': 'fish', 'mode': 'hosted',
                                  'model': 'drama-3-preview', 'speech_min_wpm': 140}
    assert 'test-value' not in json.dumps(settings)


def test_explicit_local_fish_needs_no_fish_credentials(harness, monkeypatch):
    client, _, config, _ = harness
    monkeypatch.setenv('ALIGN_FEEDBACK_CONFIG', str(studio.WEB.parents[2] / 'config' / 'feedback.local.yaml'))
    monkeypatch.setenv('FISH_AUDIO_API_KEY', '')
    monkeypatch.setenv('FISH_AUDIO_REFERENCE_ID', '')
    assert studio.StudioJobs(config).feedback_config().fish_local
    assert client.get('/api/studio/config').json()['ready']


def test_explicit_hosted_config_still_requires_fish_key(harness, monkeypatch, tmp_path):
    client, _, _, _ = harness
    hosted = tmp_path / "hosted.yaml"
    hosted.write_text("fish_local: false\n")
    monkeypatch.setenv("ALIGN_FEEDBACK_CONFIG", str(hosted))
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "")
    settings = client.get("/api/studio/config").json()
    assert settings["ready"] is False
    assert "FISH_AUDIO_API_KEY" in settings["message"]
    error = studio.feedback_failure_message(ValueError("Fish Audio returned HTTP 402. private-key"))
    assert "API balance" in error and "local Fish" not in error and "private-key" not in error


def test_qwen_config_requires_qwen_credentials_only(harness, monkeypatch, tmp_path):
    client, _, _, _ = harness
    hosted = tmp_path / "qwen.yaml"
    hosted.write_text("tts_provider: qwen\n")
    monkeypatch.setenv("ALIGN_FEEDBACK_CONFIG", str(hosted))
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "")
    monkeypatch.setenv("FISH_AUDIO_REFERENCE_ID", "")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "")
    settings = client.get("/api/studio/config").json()
    assert not settings["ready"] and "DASHSCOPE_API_KEY" in settings["message"]
    assert "FISH_AUDIO" not in settings["message"]
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "test-workspace")
    assert client.get("/api/studio/config").json()["ready"]
    error = studio.feedback_failure_message(ValueError("Qwen Audio returned HTTP 401. private-key"))
    assert "DASHSCOPE_API_KEY" in error and "Fish" not in error and "private-key" not in error


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
    for step, completed in [("transcriber", 0), ("locator", 1), ("aligner", 2), ("labels", 3)]:
        (sample / "alignment_progress.json").write_text(json.dumps({"step": step}))
        progress = jobs.status(job_id)["analysis_progress"]
        assert progress["current"] == step
        assert progress["completed"] == completed
        assert progress["steps"][completed]["state"] == "active"
    # An interrupted atomic write must not break status polling.
    (sample / "alignment_progress.json").write_text("invalid")
    assert jobs.status(job_id)["analysis_progress"]["current"] == "transcriber"
    for step, completed in [("labels", 3), ("narration", 4), ("speech", 5)]:
        jobs.save(job_id, stage="feedback", analysis_step=step)
        assert jobs.status(job_id)["analysis_progress"]["completed"] == completed
    jobs.save(job_id, status="failed")
    progress = jobs.status(job_id)["analysis_progress"]
    assert progress["completed"] == 5
    assert progress["steps"][5]["state"] == "failed"
    jobs.save(job_id, status="complete", stage="complete")
    assert jobs.status(job_id)["analysis_progress"]["completed"] == 7
    assert all(s["state"] == "complete" for s in jobs.status(job_id)["analysis_progress"]["steps"])


@pytest.mark.parametrize("with_reference", [False, True], ids=["midi-only", "with-legacy-reference-wav"])
def test_studio_uses_real_feedback_pipeline_and_retries_saved_plan(harness, monkeypatch, with_reference, synthesis_inputs, fake_soundfont):
    """Only detection and provider HTTP are mocked; feedback/MP3 assembly are real."""
    from functools import partial
    import httpx
    import numpy as np
    import soundfile as sf
    from datacreate.feedback import run_feedback
    from datacreate.feedback_video import render_video
    # Exercise actual engraving, timing and MP4 encoding after the speech retry.
    monkeypatch.setattr(studio, "render_feedback_video", render_video)

    client, _, config, _ = harness
    base_pipeline = studio.make_pipeline

    class Pipeline(base_pipeline):
        def run_stage1_2(self, job):
            from music21 import stream, meter, note
            score, part = stream.Score(), stream.Part()
            measure = stream.Measure(number=2)
            measure.append(meter.TimeSignature("4/4"))
            for pitch in [60, 62, 64, 65]:
                measure.append(note.Note(pitch, quarterLength=1))
            part.append(measure)
            score.append(part)
            score.write("musicxml", fp=job / "verified_score.musicxml")

        def run_stage3(self, job):
            synthesis_inputs(job)
            if not with_reference:
                return
            import mido
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            for pitch in [60, 62, 64, 65]:
                track.append(mido.Message("note_on", note=pitch, velocity=80))
                track.append(mido.Message("note_off", note=pitch, time=480))
            midi.save(job / "reference_audio.mid")
            sf.write(job / "reference_audio.wav", .1 * np.sin(2 * np.pi * 330 * np.arange(48000) / 16000), 16000)

        def run_stage4(self, job):
            sf.write(job / "performance_audio.wav", .1 * np.sin(2 * np.pi * 220 * np.arange(48000) / 16000), 16000)

        def run_stage5(self, job):
            (job / "candidates.json").write_text(json.dumps({"labels": [{
                "type": "wrong_note", "source": "auto", "start_time": 1.0, "end_time": 1.5,
                "score_part": {"start_note_index": 1, "end_note_index": 2, "pad_notes": 0},
            }]}))
            # This must never be used as the prediction source.
            (job / "labels.json").write_text('{"labels": []}')

    monkeypatch.setattr(studio, "make_pipeline", Pipeline)
    speech = BytesIO()
    sf.write(speech, .1 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000), 16000, format="MP3")
    requests = []
    fail_speech = True
    point = {"label_index": 0, "intro": "In bar two, listen to this passage.",
             "feedback": "Check this possible wrong note and practise slowly."}
    point["performance_intro"] = "Now hear the transcription of your playing."

    def handle(request):
        requests.append(str(request.url))
        body = json.loads(request.content)
        if request.url.path.endswith("/chat/completions"):
            assert str(request.url) == "https://api.ssstoken.net/v1/chat/completions"
            assert body["model"] == "gpt-6-luna"
            report = json.loads(body["messages"][1]["content"])["report"]
            assert report["label_count"] == 1
            assert report["labels"][0]["excerpt_available"] is True
            assert report["labels"][0]["reference_available"] is True
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"points": [point]})}}]})
        assert str(request.url) == "https://api.fish.audio/v1/tts"
        assert request.headers['authorization'] == 'Bearer test-value'
        assert request.headers['model'] == 'drama-3-preview'
        assert body["reference_id"] == "test-value"
        assert body['prosody']['speed'] == 1.0
        if fail_speech:
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=speech.getvalue())

    with httpx.Client(transport=httpx.MockTransport(handle)) as provider:
        details = []
        def tracked_feedback(*args, detail_progress, **kwargs):
            def track(event):
                details.append(event)
                detail_progress(event)
            return run_feedback(*args, detail_progress=track, client=provider, **kwargs)
        monkeypatch.setattr(studio, "run_feedback", tracked_feedback)
        response = submit(client)
        assert response.status_code == 202
        job_id = response.json()["id"]
        state = client.get(f"/api/studio/takes/{job_id}").json()
        assert state["status"] == "failed" and state["can_retry"]
        assert state["analysis_progress"]["current"] == "speech"
        output = studio.StudioJobs(config).directory(job_id) / "feedback"
        assert (output / "playback_plan.json").is_file()
        assert (output / "excerpt-000.wav").is_file()
        fail_speech = False
        assert client.post(f"/api/studio/takes/{job_id}/retry").status_code == 202

    state = client.get(f"/api/studio/takes/{job_id}").json()
    assert state["status"] == "complete"
    assert {'prepare_feedback', 'write', 'examples', 'synthesize', 'mix'} <= {d['substep'] for d in details}
    voice_steps = [d for d in details if d['substep'] == 'synthesize' and 'total' in d]
    assert [d['completed'] for d in voice_steps[-3:]] == [0, 1, 2]
    assert all(d['total'] == 3 for d in voice_steps)
    assert sum(url.endswith("/chat/completions") for url in requests) == 1
    manifest = json.loads((output / "feedback.json").read_text())
    assert manifest["input_kind"] == "plan" and not manifest["fish_local"]
    assert manifest['tts_provider'] == 'fish' and manifest['tts_model'] == 'drama-3-preview'
    assert manifest["performance_excerpts"] is True
    assert manifest["reference_excerpts"] is True
    timeline = json.loads((output / "timeline.json").read_text())["segments"]
    expected = ["speech", "reference", "speech", "performance", "speech"]
    assert [segment["kind"] for segment in timeline] == expected
    levels = [segment["output_lufs"] for segment in timeline if segment["output_lufs"] is not None]
    assert max(levels) - min(levels) < 1.0
    for i, segment in enumerate(timeline):
        if segment["kind"] in {"reference", "performance"}:
            assert segment["reverb_mix"] == .08
            assert segment["silence_after_seconds"] == .5
            assert segment["start_time"] - timeline[i - 1]["end_time"] == pytest.approx(.5)
    assert state["feedback_details"]["pipeline_revision"] == studio.PIPELINE_REVISION
    assert state["feedback_details"]["tts_model"] == "drama-3-preview"
    delivered = client.get(state["audio_url"])
    assert delivered.content == (output / "feedback.mp3").read_bytes()
    decoded, rate = sf.read(BytesIO(delivered.content), always_2d=True)
    assert rate == 44100 and decoded.shape[1] == 2
    assert (output / 'report.json').is_file()
    video = client.get(state['video_url'])
    assert video.status_code == 200 and video.headers['content-type'] == 'video/mp4'
    assert b'ftyp' in video.content[:32]
    video_manifest = json.loads((output.parent / 'video' / 'video.json').read_text())
    assert video_manifest['status'] == 'complete'
    assert video_manifest['labels'] == [0]
    assert video_manifest['audio_resynthesized'] is False
    # A playable picture is insufficient: verify every speech and music segment
    # survives the MP4 mux at the original timing and loudness.
    import subprocess
    from datacreate.audio_utils import _find_ffmpeg
    raw = subprocess.run([
        _find_ffmpeg(), '-v', 'error', '-i', str(output.parent / 'video' / 'feedback.mp4'),
        '-vn', '-f', 'f32le', '-ac', '2', '-ar', str(rate), 'pipe:1',
    ], capture_output=True, check=True).stdout
    video_audio = np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
    assert len(video_audio) >= round(timeline[-1]['end_time'] * rate)
    for segment in timeline:
        start, end = round(segment['start_time'] * rate), round(segment['end_time'] * rate)
        expected, actual = decoded[start:end], video_audio[start:end]
        assert np.corrcoef(expected.ravel(), actual.ravel())[0, 1] > .98
        assert np.sqrt(np.mean(actual**2) / np.mean(expected**2)) == pytest.approx(1., rel=.1)


def test_video_failure_retry_preserves_audio_and_needs_no_provider(harness, monkeypatch):
    client, calls, config, _ = harness
    real = studio.render_feedback_video
    def failure(sample, feedback, output, **kwargs):
        output.mkdir()
        (output / 'feedback.part.mp4').write_bytes(b'partial')
        raise RuntimeError('private renderer details')
    monkeypatch.setattr(studio, 'render_feedback_video', failure)
    job_id = submit(client).json()['id']
    state = client.get(f'/api/studio/takes/{job_id}').json()
    assert state['status'] == 'failed' and state['can_retry']
    assert state['retry_kind'] == 'video'
    assert state['analysis_progress']['current'] == 'video'
    assert state['analysis_progress']['completed'] == 6
    assert 'private renderer' not in state['message']
    assert 'video_url' not in state
    assert client.get(state['audio_url']).content == b'ID3test-audio'
    assert client.get(f'/api/studio/takes/{job_id}/video').status_code == 404
    root = studio.StudioJobs(config).directory(job_id)
    original_audio = (root / 'feedback' / 'feedback.mp3').read_bytes()
    monkeypatch.setenv('SSSTOKEN_API_KEY', '')
    monkeypatch.setattr(studio, 'render_feedback_video', real)
    assert client.post(f'/api/studio/takes/{job_id}/retry').status_code == 202
    state = client.get(f'/api/studio/takes/{job_id}').json()
    assert state['status'] == 'complete' and state['video_url']
    assert calls.count('feedback') == 1 and calls.count('alignment') == 1
    assert (root / 'feedback' / 'feedback.mp3').read_bytes() == original_audio
    assert len(list(root.glob('video-*/feedback.part.mp4'))) == 1


def test_audio_only_feedback_completes_without_animation(harness, monkeypatch):
    client, calls, config, original = harness
    def audio_only(*args, output_dir, **kwargs):
        original(*args, output_dir=output_dir, **kwargs)
        (output_dir / 'playback_plan.json').unlink()
    monkeypatch.setattr(studio, 'run_feedback', audio_only)
    job_id = submit(client).json()['id']
    state = client.get(f'/api/studio/takes/{job_id}').json()
    assert state['status'] == 'complete' and state['audio_url']
    assert state['video_status'] == 'unavailable' and state['video_message']
    assert 'video_url' not in state and 'video' not in calls
    assert state['analysis_progress']['steps'][-1]['state'] == 'skipped'
    assert state['detailed_progress']['phases'][-1]['state'] == 'skipped'


def test_detailed_progress_tracks_live_units_failure_and_restart(harness):
    _, _, config, _ = harness
    jobs = studio.StudioJobs(config)
    job_id = 'e' * 32
    jobs.directory(job_id).mkdir(parents=True)
    jobs.reserve(job_id)
    jobs.save(job_id, status='processing', stage='reference')
    detail = jobs.status(job_id)['detailed_progress']
    assert detail['current'] == 'reference' and detail['completed'] == 1
    assert detail['phases'][0]['substeps'][0]['state'] == 'complete'
    jobs.save(job_id, stage='feedback', analysis_step='narration', progress_detail={
        'substep': 'write', 'completed': 1, 'total': 3, 'message': 'Writing point 2 of 3'})
    detail = jobs.status(job_id)['detailed_progress']
    assert detail['message'] == 'Writing point 2 of 3'
    assert detail['units'] == {'completed': 1, 'total': 3}
    jobs.save(job_id, stage='video', analysis_step='video', progress_detail={
        'substep': 'frames', 'completed': 48, 'total': 240, 'message': 'Rendering frames'})
    detail = jobs.status(job_id)['detailed_progress']
    assert detail['completed'] == 14 and detail['current'] == 'frames'
    assert detail['phases'][-1]['substeps'][-1]['state'] == 'pending'
    jobs.save(job_id, status='failed', message='Video failed')
    detail = jobs.status(job_id)['detailed_progress']
    assert detail['phases'][-1]['substeps'][1]['state'] == 'failed'
    assert detail['units']['completed'] == 48
    jobs.save(job_id, status='processing', progress_detail={})
    assert jobs.status(job_id)['detailed_progress']['current'] == 'engrave'
    jobs.release()
    detail = jobs.status(job_id)['detailed_progress']
    assert detail['status'] == 'failed' and 'restarted' in detail['message']


def keyed_submit(client, key='robot-session-001', *, audio=None, score=None, score_id='piece.musicxml'):
    files = {'audio': ('take.wav', wav_bytes() if audio is None else audio, 'audio/wav')}
    if score is not None:
        files['score'] = ('piece.musicxml', score, 'application/xml')
    return client.post('/api/studio/takes', files=files, data={'score_id':score_id},
                       headers={'Idempotency-Key':key})


def test_keyed_acceptance_replay_lookup_and_restart(harness):
    client, calls, config, _ = harness
    first = keyed_submit(client)
    assert first.status_code == 202 and not first.json()['replayed']
    result = first.json()
    completed_calls = list(calls)
    replay = keyed_submit(client)
    assert replay.status_code == 200 and replay.json()['id'] == result['id']
    assert replay.json()['replayed'] and calls == completed_calls
    assert client.get('/api/studio/requests/robot-session-001').json() == replay.json()
    assert client.get(result['status_url']).json()['status'] == 'complete'
    app = FastAPI(); app.include_router(studio.studio_router(config))
    restarted = TestClient(app)
    assert restarted.get('/api/studio/requests/robot-session-001').json()['id'] == result['id']
    assert keyed_submit(restarted).json()['id'] == result['id'] and calls == completed_calls
    settings = client.get('/api/studio/config').json()
    assert settings['default_score_id'] == 'piece.musicxml'
    assert settings['integration_contract'] == 'studio-robot-v1'


@pytest.mark.parametrize('change', ['audio', 'uploaded_score', 'selected_score'])
def test_key_conflict_never_reprocesses(harness, change):
    client, calls, config, _ = harness
    first = keyed_submit(client)
    completed_calls = list(calls)
    kwargs = {}
    if change == 'audio': kwargs['audio'] = wav_bytes(2)
    elif change == 'uploaded_score': kwargs['score'] = b'<different-score/>'
    else: config.resolved_path('raw_data_score').write_text('<changed-score/>')
    conflict = keyed_submit(client, **kwargs)
    assert conflict.status_code == 409
    assert conflict.json()['detail']['code'] == 'idempotency_conflict'
    assert calls == completed_calls
    assert client.get('/api/studio/requests/robot-session-001').json()['id'] == first.json()['id']


def test_keyed_busy_replay_and_restart_before_worker(harness, monkeypatch):
    client, calls, config, _ = harness
    monkeypatch.setattr(studio.StudioJobs, 'process', lambda *args, **kwargs: None)
    first = keyed_submit(client).json()
    # Lost response or worker not yet started: replay cannot schedule another run.
    assert keyed_submit(client).json()['id'] == first['id']
    busy = keyed_submit(client, key='new-key')
    assert busy.status_code == 409 and busy.json()['detail']['code'] == 'studio_busy'
    assert busy.headers['Retry-After'] == '2'
    assert client.get('/api/studio/requests/new-key').status_code == 404
    assert submit(client).status_code == 409  # GUI shares the same busy slot.
    app = FastAPI(); app.include_router(studio.studio_router(config))
    restarted = TestClient(app)
    assert restarted.get('/api/studio/requests/robot-session-001').json()['id'] == first['id']
    status = restarted.get(first['status_url']).json()
    assert status['status'] == 'failed' and not status['can_retry']
    assert 'restarted' in status['message']
    assert keyed_submit(restarted).json()['id'] == first['id']
    assert keyed_submit(restarted, key='new-key').status_code == 202
    assert calls == []


def test_key_replay_does_not_require_credentials(harness, monkeypatch):
    client, calls, _, _ = harness
    first = keyed_submit(client).json()
    monkeypatch.setattr(studio.StudioJobs, 'readiness', lambda self: {'ready':False, 'message':'Unavailable'})
    assert keyed_submit(client).json()['id'] == first['id']
    assert keyed_submit(client, key='new-key').status_code == 503
    assert client.get('/api/studio/requests/new-key').status_code == 404


def test_invalid_request_does_not_consume_key(harness):
    client, calls, _, _ = harness
    assert keyed_submit(client, audio=b'not a WAV').status_code == 422
    assert client.get('/api/studio/requests/robot-session-001').status_code == 404
    assert keyed_submit(client, score_id='not-a-score').status_code == 422
    assert keyed_submit(client, key='bad key').status_code == 422
    assert keyed_submit(client, key='a'*129).status_code == 422
    assert keyed_submit(client).status_code == 202


def test_keyed_score_is_frozen_and_raw_key_not_stored(harness, monkeypatch):
    client, _, config, _ = harness
    seen = []
    def process(self, job_id, score):
        config.resolved_path('raw_data_score').write_text('<changed/>')
        seen.append(score.read_bytes())
        self.release()
    monkeypatch.setattr(studio.StudioJobs, 'process', process)
    result = keyed_submit(client).json()
    assert seen == [b'<score-partwise/>']
    root = config.resolved_path('work_dir') / 'studio'
    assert b'robot-session-001' not in (root / 'requests.sqlite3').read_bytes()
    # Removing a job must not make its accepted key reusable.
    (root / 'jobs' / result['id'] / 'status.json').unlink()
    assert client.get('/api/studio/requests/robot-session-001').status_code == 410
    assert keyed_submit(client, score=b'<score-partwise/>').status_code == 410


def test_concurrent_duplicate_submissions_schedule_one_job(harness, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    client, _, _, _ = harness
    entered, release = Event(), Event()
    started = []
    def process(self, job_id, score):
        started.append(job_id); entered.set()
        assert release.wait(10)
        self.release()
    monkeypatch.setattr(studio.StudioJobs, 'process', process)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(keyed_submit, client)
        try:
            assert entered.wait(10)
            replay = keyed_submit(client)
            assert replay.status_code == 200 and replay.json()['id'] == started[0]
        finally:
            release.set()
        assert first.result().status_code == 202
    assert len(started) == 1


def test_acceptance_storage_failure_does_not_schedule_or_bind_key(harness, monkeypatch):
    client, calls, _, _ = harness
    original = studio.StudioJobs.save
    def failed_save(*args, **kwargs): raise OSError('simulated disk failure')
    monkeypatch.setattr(studio.StudioJobs, 'save', failed_save)
    with pytest.raises(OSError, match='disk failure'): keyed_submit(client)
    assert client.get('/api/studio/requests/robot-session-001').status_code == 404
    assert not calls
    monkeypatch.setattr(studio.StudioJobs, 'save', original)
    assert keyed_submit(client).status_code == 202


def test_acceptance_commit_failure_rolls_back_before_processing(harness, monkeypatch):
    from contextlib import contextmanager
    client, calls, config, _ = harness
    original = studio.request_store
    @contextmanager
    def failing_store(root):
        with original(root) as connection:
            yield connection
            raise OSError('simulated commit failure')
    monkeypatch.setattr(studio, 'request_store', failing_store)
    with pytest.raises(OSError, match='commit failure'): keyed_submit(client)
    assert not calls
    assert client.get('/api/studio/requests/robot-session-001').status_code == 404
    assert not list((config.resolved_path('work_dir') / 'studio' / 'jobs').iterdir())
    monkeypatch.setattr(studio, 'request_store', original)
    assert keyed_submit(client).status_code == 202


def test_no_default_for_multiple_scores_and_missing_score_rejected(harness):
    client, _, config, _ = harness
    root = config.resolved_path('raw_data_score').parent
    (root / 'second.musicxml').write_text('<score-partwise/>')
    config.paths['raw_data_score'] = str(root)
    assert client.get('/api/studio/config').json()['default_score_id'] is None
    assert keyed_submit(client, score_id='').status_code == 422
    assert client.get('/api/studio/requests/robot-session-001').status_code == 404


def test_hardware_48khz_wav_and_uploaded_score(harness):
    client, calls, _, _ = harness
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as recording:
        recording.setnchannels(1); recording.setsampwidth(2); recording.setframerate(48000)
        recording.writeframes(b'\0\0' * 48000)
    first = keyed_submit(client, audio=buffer.getvalue(), score=b'<score-partwise/>', score_id='')
    assert first.status_code == 202
    replay = keyed_submit(client, audio=buffer.getvalue())
    assert replay.status_code == 200 and replay.json()['id'] == first.json()['id']
    assert calls.count('alignment') == calls.count('feedback') == calls.count('video') == 1


def broadband_wav():
    import numpy as np
    buffer=BytesIO()
    with wave.open(buffer,'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(48000)
        audio.writeframes(np.random.default_rng(2026).normal(0,278,6*48000).astype('<i2').tobytes())
    return buffer.getvalue()


@pytest.mark.parametrize('keyed',[True,False])
def test_noise_rejected_before_pipeline_and_provider_calls(harness,keyed):
    client,calls,config,_=harness
    recording=broadband_wav()
    response=keyed_submit(client,audio=recording) if keyed else submit(client,audio=recording)
    assert response.status_code==202
    job_id=response.json()['id']
    status=client.get(f'/api/studio/takes/{job_id}').json()
    assert status['status']=='failed' and status['stage']=='input_quality'
    assert status['input_quality']['reason']=='stationary_broadband_noise'
    assert not status['can_retry'] and status['assessment'] is None
    assert status['detailed_progress']['current']=='input_quality'
    assert status['detailed_progress']['status']=='failed'
    assert 'record again' in status['message']
    assert not calls
    root=config.resolved_path('work_dir')/'studio'
    assert not (root/'samples'/job_id).exists()
    assert not (root/'jobs'/job_id/'feedback').exists()
    assert client.post(f'/api/studio/takes/{job_id}/retry').status_code==409
    assert client.get(f'/api/studio/takes/{job_id}/audio').status_code==404
    assert client.get(f'/api/studio/takes/{job_id}/video').status_code==404
    if keyed:
        assert client.get('/api/studio/requests/robot-session-001').json()['id']==job_id
        replay=keyed_submit(client,audio=recording)
        assert replay.status_code==200 and replay.json()['id']==job_id
        app=FastAPI();app.include_router(studio.studio_router(config));restarted=TestClient(app)
        assert keyed_submit(restarted,audio=recording).json()['id']==job_id
        assert restarted.get(f'/api/studio/takes/{job_id}').json()['input_quality']==status['input_quality']
    assert not calls
    # The failure releases the busy slot for the user's next recording.
    assert submit(client).status_code==202


def test_quality_rejection_never_exposes_old_artifacts(harness):
    client,calls,config,_=harness
    job_id=keyed_submit(client,audio=broadband_wav()).json()['id']
    root=config.resolved_path('work_dir')/'studio'
    sample=root/'samples'/job_id;sample.mkdir(parents=True)
    (sample/'candidates.json').write_text('{"labels":[]}')
    (sample/'note_alignment_v2.json').write_text('{"status":"ok"}')
    directory=root/'jobs'/job_id
    (directory/'feedback').mkdir();(directory/'video').mkdir()
    (directory/'feedback'/'feedback.mp3').write_bytes(b'ID3old')
    (directory/'feedback'/'feedback.txt').write_text('Old narration')
    (directory/'video'/'feedback.mp4').write_bytes(b'old')
    status=client.get(f'/api/studio/takes/{job_id}').json()
    assert status['assessment'] is None and not status['can_retry']
    assert not {'audio_url','video_url','preview_video_url','narration','feedback_details'} & status.keys()
    assert client.post(f'/api/studio/takes/{job_id}/retry').status_code==409
    assert not calls


def test_quality_check_progress_and_pass_are_persisted(harness,monkeypatch):
    from datacreate import input_quality
    client,_,config,_=harness
    original=input_quality.assess_recording
    def inspect(path):
        saved=json.loads((path.parent/'status.json').read_text())
        assert saved['stage']=='input_quality' and saved['status']=='processing'
        progress=studio.detailed_progress(saved,None)
        assert progress['phases'][0]['state']=='active'
        return original(path)
    monkeypatch.setattr(input_quality,'assess_recording',inspect)
    job_id=submit(client).json()['id']
    status=client.get(f'/api/studio/takes/{job_id}').json()
    assert status['status']=='complete'
    assert status['input_quality']['status']=='not_assessed' # 8 kHz legacy GUI fixture


def test_eligible_quiet_tone_proceeds_and_preserves_original_rate(harness):
    import numpy as np
    client,calls,_,_=harness
    buffer=BytesIO();sr=48000
    audio=(10*np.sin(2*np.pi*440*np.arange(sr*6)/sr)).astype('<i2')
    with wave.open(buffer,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(sr);wav.writeframes(audio.tobytes())
    response=keyed_submit(client,audio=buffer.getvalue())
    assert response.status_code==202
    state=client.get(response.json()['status_url']).json()
    assert state['status']=='complete' and state['input_quality']['status']=='passed'
    assert state['input_quality']['sample_rate']==48000
    assert calls.count('alignment')==calls.count('feedback')==1
