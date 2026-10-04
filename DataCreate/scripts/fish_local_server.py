"""Loopback-only Fish Speech 1.5 TTS service for ALIGN.

Uses the upstream inference engine unchanged, without its ASR/training/UI stack.
Run with the isolated .local/fish/venv interpreter, not the ALIGN environment.
"""
from contextlib import asynccontextmanager
import io
import os
from pathlib import Path
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / ".local" / "fish"
SOURCE = RUNTIME / "fish-speech-1.5.1"
CHECKPOINTS = RUNTIME / "checkpoints" / "fish-speech-1.5"
sys.path.insert(0, str(SOURCE))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model
from fish_speech.utils.schema import ServeTTSRequest


@asynccontextmanager
async def lifespan(app):
    if not torch.cuda.is_available():
        raise RuntimeError("This installation requires the NVIDIA CUDA GPU.")
    torch.set_num_threads(4)
    os.chdir(SOURCE)  # upstream reference_id resolves references/ under cwd
    decoder = load_model("firefly_gan_vq", str(CHECKPOINTS / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"), "cuda")
    queue = launch_thread_safe_queue(checkpoint_path=CHECKPOINTS, device="cuda",
                                     precision=torch.bfloat16, compile=False)
    app.state.engine = TTSInferenceEngine(queue, decoder, torch.bfloat16, False)
    app.state.lock = Lock()
    yield


app = FastAPI(title="ALIGN Local Fish Speech", lifespan=lifespan)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    format: str = "mp3"
    reference_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    seed: int = 42


@app.get("/v1/health")
def health():
    return {"status": "ok", "backend": "fish-speech-1.5", "pid": os.getpid(), "device": torch.cuda.get_device_name(),
            "local": True, "weights_license": "CC-BY-NC-SA-4.0"}


@app.post("/v1/tts")
def tts(body: SpeechRequest):
    if body.format != "mp3" or not body.text.strip():
        raise HTTPException(422, "Provide nonempty text and format=mp3.")
    if body.reference_id:
        folder = SOURCE / "references" / body.reference_id
        audios = [p for p in folder.glob("*") if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
        if not audios or any(not p.with_suffix(".lab").is_file() for p in audios):
            raise HTTPException(422, "Local voice requires paired audio and .lab transcript files under references/<id>.")
    if not app.state.lock.acquire(blocking=False):
        raise HTTPException(409, "Local GPU is already synthesizing another request.")
    try:
        audio = None
        request = ServeTTSRequest(text=body.text, reference_id=body.reference_id,
                                  format="mp3", streaming=False, seed=body.seed,
                                  # Keep brief feedback together instead of placing a
                                  # codec boundary just after a bar number.
                                  chunk_length=300, max_new_tokens=1024)
        for result in app.state.engine.inference(request):
            if result.code == "error":
                raise RuntimeError("Fish inference failed") from result.error
            if result.code == "final":
                audio = result.audio
        if audio is None or not len(audio[1]) or not np.isfinite(audio[1]).all():
            raise RuntimeError("No valid audio generated")
        stream = io.BytesIO()
        sf.write(stream, audio[1], audio[0], format="MP3")
        return Response(stream.getvalue(), media_type="audio/mpeg")
    finally:
        app.state.lock.release()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081, workers=1)
