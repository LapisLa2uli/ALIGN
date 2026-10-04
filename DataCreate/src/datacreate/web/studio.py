"""Local practice UI: bounded recording jobs -> configured detector -> spoken feedback."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4
import wave

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from datacreate.config import PipelineConfig
from datacreate.feedback import FeedbackConfig, run_feedback

WEB = Path(__file__).resolve().parent
MAX_SECONDS = 300
ANALYSIS_STEPS = [
    ("transcriber", "Transcriber", "Transcribing the notes in your performance"),
    ("aligner", "Aligner", "Matching your performance to the score"),
    ("labels", "Label analysis", "Analyzing possible issues and score locations"),
    ("narration", "Narration", "Writing your spoken feedback"),
    ("speech", "Speech audio", "Synthesizing the feedback MP3"),
]


def analysis_progress(state, sample):
    """Stage milestones, never an estimate of elapsed time or model completion."""
    stage = state.get("stage")
    step = state.get("analysis_step")
    if stage == "alignment":
        step = "transcriber"
        try:
            reported = json.loads((sample / "alignment_progress.json").read_text())["step"]
            if reported in {"transcriber", "aligner", "labels"}:
                step = reported
        except (OSError, ValueError, KeyError):
            pass
    elif stage == "features":
        step = "labels"
    elif stage == "feedback" and not step:
        step = "narration"
    keys = [key for key, _, _ in ANALYSIS_STEPS]
    finished = state["status"] == "complete"
    index = len(keys) if finished else keys.index(step) if step in keys else -1
    if index < 0:
        return None
    return {"current": None if finished else keys[index],
            "completed": index, "total": len(keys),
            "message": "Your feedback is ready" if finished else ANALYSIS_STEPS[index][2],
            "steps": [{"id": key, "label": label,
                       "state": "complete" if i < index else (
                           "failed" if state["status"] == "failed" else "active") if i == index else "pending"}
                      for i, (key, label, _) in enumerate(ANALYSIS_STEPS)]}


def make_pipeline(config):
    # Loading rendering/ML dependencies should not delay the studio or status API.
    from datacreate.pipeline import DataCreatePipeline
    return DataCreatePipeline(config)


class StudioJobs:
    def __init__(self, config: PipelineConfig):
        self.config = deepcopy(config)
        self.root = (config.resolved_path("work_dir") or Path("work")) / "studio"
        self.config.paths["samples_root"] = str(self.root / "samples")
        self.lock = Lock()
        self.active: str | None = None

    def feedback_config(self):
        path = os.environ.get("ALIGN_FEEDBACK_CONFIG")
        # The practice studio uses the local Fish service by default. Hosted
        # speech is available only through an explicit configuration override.
        return FeedbackConfig.load(
            Path(path) if path else WEB.parents[2] / "config" / "feedback.local.yaml"
        )

    def readiness(self):
        try:
            config = self.feedback_config()
            required = [config.llm_api_key_env]
            if not config.fish_local:
                required += [config.fish_api_key_env, config.fish_reference_id_env]
            missing = [key for key in required if not os.environ.get(key, "").strip()]
            return {"ready": not missing, "message": "Missing server environment: " + ", ".join(missing) if missing else "Ready for your next take", "instrument": config.instrument}
        except (ValueError, OSError):
            return {"ready": False, "message": "Check the server's ALIGN_FEEDBACK_CONFIG file.", "instrument": "B-flat clarinet"}

    def directory(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Take not found")
        return self.root / "jobs" / job_id

    def save(self, job_id, **changes):
        directory = self.directory(job_id)
        path = directory / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {"id": job_id}
        state.update(changes)
        temporary = directory / "status.tmp"
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(path)

    def status(self, job_id):
        directory = self.directory(job_id)
        try:
            state = json.loads((directory / "status.json").read_text())
        except FileNotFoundError:
            raise HTTPException(404, "Take not found") from None
        if state["status"] == "processing" and self.active != job_id:
            state.update(status="failed", message="The server restarted. Start a new take, or retry feedback if analysis finished.")
        sample = self.root / "samples" / job_id
        state["can_retry"] = state["status"] == "failed" and (sample / "candidates.json").exists()
        state["analysis_progress"] = analysis_progress(state, sample)
        if state["status"] == "processing" and state["analysis_progress"]:
            state["message"] = state["analysis_progress"]["message"]
        text = directory / "feedback" / "feedback.txt"
        if text.exists():
            state["narration"] = text.read_text(encoding="utf-8")
        if state["status"] == "complete":
            state["audio_url"] = f"/api/studio/takes/{job_id}/audio"
        return state

    def reserve(self, job_id):
        with self.lock:
            if self.active:
                raise HTTPException(409, "Another take is processing. Please wait for it to finish.")
            self.active = job_id

    def release(self):
        with self.lock:
            self.active = None

    def process(self, job_id, score=None, retry=False):
        directory = self.directory(job_id)
        stage = "Preparing your score"
        try:
            config = self.feedback_config()
            sample = self.root / "samples" / job_id
            if not retry:
                pipeline = make_pipeline(self.config)
                job = pipeline.create_sample(job_id, score=score, performance=directory / "recording.wav")
                stages = [("score", "Preparing your score", pipeline.run_stage1_2),
                          ("reference", "Rendering the reference performance", pipeline.run_stage3),
                          ("audio", "Preparing your recording", pipeline.run_stage4),
                          ("alignment", "Aligning notes and identifying possible issues", pipeline.run_stage5),
                          ("features", "Building mel spectrograms", pipeline.run_stage7)]
                for key, stage, action in stages:
                    self.save(job_id, status="processing", stage=key, message=stage)
                    action(job)
            stage = "Creating spoken feedback"
            self.save(job_id, status="processing", stage="feedback", analysis_step="labels", message=stage)
            output = directory / "feedback"
            # Preserve failed attempts; a speech retry reuses the saved narration.
            source = sample / "candidates.json"
            text_input = False
            plan_input = False
            if output.exists():
                archived = directory / f"feedback-{uuid4().hex}"
                output.rename(archived)
                if (archived / "playback_plan.json").exists():
                    source, plan_input = archived / "playback_plan.json", True
                elif (archived / "feedback.txt").exists():
                    source, text_input = archived / "feedback.txt", True
            def feedback_progress(step):
                self.save(job_id, analysis_step=step)
            run_feedback(source, config=config, output_dir=output, text_input=text_input,
                         plan_input=plan_input, progress=feedback_progress)
            self.save(job_id, status="complete", stage="complete", message="Your feedback is ready")
        except Exception:
            logging.getLogger(__name__).exception("Studio take %s failed at %s", job_id, stage)
            self.save(job_id, status="failed", message=f"{stage} failed. Check the server log and provider configuration.")
        finally:
            self.release()


async def save_upload(upload: UploadFile, target: Path, limit: int):
    total = 0
    try:
        with target.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "Upload is too large")
                stream.write(chunk)
        if not total:
            raise HTTPException(422, "The uploaded file is empty")
    finally:
        await upload.close()


def validate_recording(path):
    try:
        with wave.open(str(path), "rb") as recording:
            duration = recording.getnframes() / recording.getframerate()
            if (recording.getnchannels() != 1 or recording.getsampwidth() != 2
                    or not 8000 <= recording.getframerate() <= 96000 or not 1 <= duration <= MAX_SECONDS + 1):
                raise ValueError()
            expected = recording.getnframes() * 2
            if len(recording.readframes(recording.getnframes())) != expected:
                raise ValueError()
    except (wave.Error, EOFError, ValueError, ZeroDivisionError):
        raise HTTPException(422, "Record 1–300 seconds of mono PCM audio before submitting.") from None


def studio_router(config: PipelineConfig):
    router = APIRouter()
    jobs = StudioJobs(config)

    def scores():
        root = config.resolved_path("raw_data_score")
        if not root or not root.exists():
            return []
        return sorted(p for p in (root.iterdir() if root.is_dir() else [root]) if p.suffix.lower() in {".musicxml", ".xml", ".mxl"})

    @router.get("/studio", response_class=HTMLResponse)
    def studio():
        return HTMLResponse((WEB / "templates" / "studio.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @router.get("/api/studio/config")
    def settings():
        return {**jobs.readiness(), "max_seconds": MAX_SECONDS,
                "scores": [{"id": p.name, "name": p.stem.replace("_", " ")} for p in scores()]}

    @router.post("/api/studio/takes", status_code=202)
    async def create_take(background: BackgroundTasks, audio: UploadFile = File(...),
                          score: UploadFile | None = File(None), score_id: str = Form("")):
        ready = jobs.readiness()
        if not ready["ready"]:
            raise HTTPException(503, ready["message"])
        job_id = uuid4().hex
        jobs.reserve(job_id)
        directory = jobs.directory(job_id)
        try:
            directory.mkdir(parents=True)
            await save_upload(audio, directory / "recording.wav", 60_000_000)
            validate_recording(directory / "recording.wav")
            if score and score.filename:
                suffix = Path(score.filename).suffix.lower()
                if suffix not in {".musicxml", ".xml", ".mxl"}:
                    raise HTTPException(422, "Choose a MusicXML (.musicxml, .xml, .mxl) score")
                score_path = directory / f"score{suffix}"
                await save_upload(score, score_path, 10_000_000)
            else:
                score_path = next((p for p in scores() if p.name == score_id), None)
                if score_path is None:
                    raise HTTPException(422, "Select or upload a score first")
            jobs.save(job_id, status="processing", stage="queued", message="Your take is queued")
            background.add_task(jobs.process, job_id, score_path)
            return {"id": job_id}
        except Exception:
            # Remove only this request's known upload files on validation failure.
            for name in ("recording.wav", "score.musicxml", "score.xml", "score.mxl"):
                (directory / name).unlink(missing_ok=True)
            jobs.release()
            raise
        finally:
            await audio.close()
            if score:
                await score.close()

    @router.get("/api/studio/takes/{job_id}")
    def take_status(job_id: str):
        return jobs.status(job_id)

    @router.post("/api/studio/takes/{job_id}/retry", status_code=202)
    def retry_feedback(job_id: str, background: BackgroundTasks):
        if not jobs.status(job_id)["can_retry"]:
            raise HTTPException(409, "This take cannot retry feedback")
        ready = jobs.readiness()
        if not ready["ready"]:
            raise HTTPException(503, ready["message"])
        jobs.reserve(job_id)
        jobs.save(job_id, status="processing", stage="feedback", analysis_step="labels", message="Retrying spoken feedback")
        background.add_task(jobs.process, job_id, retry=True)
        return {"id": job_id}

    @router.get("/api/studio/takes/{job_id}/audio")
    def take_audio(job_id: str):
        if jobs.status(job_id)["status"] != "complete":
            raise HTTPException(404, "Feedback audio is not ready")
        return FileResponse(jobs.directory(job_id) / "feedback" / "feedback.mp3",
                            media_type="audio/mpeg", filename="align-feedback.mp3")

    return router
