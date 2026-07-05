from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datacreate.config import PipelineConfig
from datacreate.models import LabelsDocument
from datacreate.utils import read_json, write_json
from datacreate.validation import validate_labels_file


WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"


class LabelsPayload(BaseModel):
    labels: list[dict[str, Any]]
    self_reported: list[dict[str, Any]] = []
    annotator_id: str | None = None


class ReviewPayload(BaseModel):
    annotator_b_labels: LabelsPayload


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    config = config or PipelineConfig.load()
    samples_root = config.path("samples_root") or Path("samples")
    app = FastAPI(title="MusicEval Annotator", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (TEMPLATES_DIR / "annotate.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/samples")
    def list_samples() -> list[str]:
        if not samples_root.exists():
            return []
        return sorted(
            d.name for d in samples_root.iterdir() if d.is_dir() and (d / "performance_audio.wav").exists()
        )

    @app.get("/api/samples/{sample_id}")
    def get_sample(sample_id: str) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        candidates = read_json(sample_dir / "candidates.json") if (sample_dir / "candidates.json").exists() else {"labels": []}
        labels = read_json(sample_dir / "labels.json") if (sample_dir / "labels.json").exists() else {"labels": [], "self_reported": []}
        return {
            "sample_id": sample_id,
            "taxonomy": config.taxonomy,
            "schema_version": config.schema_version,
            "candidates": candidates.get("labels", []),
            "labels": labels.get("labels", []),
            "self_reported": labels.get("self_reported", []),
            "annotator_id": labels.get("annotator_id"),
            "audio_url": f"/api/samples/{sample_id}/audio",
            "score_url": f"/api/samples/{sample_id}/score",
        }

    @app.get("/api/samples/{sample_id}/audio")
    def get_audio(sample_id: str) -> FileResponse:
        path = samples_root / sample_id / "performance_audio.wav"
        if not path.exists():
            raise HTTPException(404, "Audio not found")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/samples/{sample_id}/score")
    def get_score(sample_id: str) -> FileResponse:
        path = samples_root / sample_id / "verified_score.musicxml"
        if not path.exists():
            raise HTTPException(404, "Score not found")
        return FileResponse(path, media_type="application/xml")

    @app.put("/api/samples/{sample_id}/labels")
    def save_labels(sample_id: str, payload: LabelsPayload) -> dict[str, str]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        doc = LabelsDocument(
            schema_version=config.schema_version,
            annotator_id=payload.annotator_id,
            labels=payload.labels,  # type: ignore[arg-type]
            self_reported=payload.self_reported,  # type: ignore[arg-type]
        )
        path = sample_dir / "labels.json"
        write_json(path, doc.model_dump())
        errors = validate_labels_file(path, config)
        if errors:
            raise HTTPException(400, "; ".join(errors))
        return {"status": "saved"}

    @app.post("/api/samples/{sample_id}/review")
    def compare_annotations(sample_id: str, payload: ReviewPayload) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        path_a = sample_dir / "labels.json"
        if not path_a.exists():
            raise HTTPException(404, "Primary labels not found")
        labels_a = read_json(path_a).get("labels", [])
        labels_b = payload.annotator_b_labels.labels
        return _diff_labels(labels_a, labels_b)

    return app


def _diff_labels(a: list[dict], b: list[dict]) -> dict[str, Any]:
    def iou(x: dict, y: dict) -> float:
        start = max(x["start_time"], y["start_time"])
        end = min(x["end_time"], y["end_time"])
        inter = max(0.0, end - start)
        union = max(x["end_time"], y["end_time"]) - min(x["start_time"], y["start_time"])
        return inter / union if union > 0 else 0.0

    matched, type_mismatch, only_a, only_b = [], [], list(a), list(b)
    for la in a:
        best = None
        best_iou = 0.0
        for lb in b:
            score = iou(la, lb)
            if score > best_iou:
                best_iou = score
                best = lb
        if best and best_iou >= 0.3:
            only_b.remove(best)
            if la.get("type") == best.get("type"):
                matched.append({"a": la, "b": best, "iou": best_iou})
            else:
                type_mismatch.append({"a": la, "b": best, "iou": best_iou})
        else:
            only_a.append(la)

    agreement = len(matched) / max(1, len(a) + len(b) - len(matched))
    return {
        "matched": matched,
        "type_mismatch": type_mismatch,
        "only_a": only_a,
        "only_b": only_b,
        "agreement_score": round(agreement, 4),
    }
