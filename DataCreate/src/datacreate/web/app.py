from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datacreate.batch_audio import list_available_audio_ids, run_batch_range
from datacreate.config import PipelineConfig
from datacreate.models import LabelsDocument
from datacreate.note_alignment import build_note_alignment
from datacreate.sample_prep import (
    apply_performance_trim,
    apply_score_segment,
    ensure_full_score,
    get_prep_state,
    reprocess_alignment,
)
from datacreate.score_segment import extract_measure_range
from datacreate.utils import read_json, setup_sample_logger, write_json
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


class ScoreSegmentPayload(BaseModel):
    start_measure: int
    end_measure: int
    start_beat: int = 1
    end_beat: int | None = None


class PerformanceTrimPayload(BaseModel):
    trim_start: float
    trim_end: float | None = None


class BatchRangePayload(BaseModel):
    id_from: int | str
    id_to: int | str
    score_path: str | None = None
    audio_dir: str | None = None
    id_width: int = 3
    skip_existing: bool = True


def _sample_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", name)
    return (int(match.group(1)) if match else 0, name)


def _resolve_score_path(config: PipelineConfig, override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        path = config.resolved_path("raw_data_score")
        if path is None:
            path = Path(__file__).resolve().parents[3] / "RawData" / "Score"
        if path.is_dir():
            candidates = sorted(path.glob("*.musicxml")) + sorted(path.glob("*.mxl"))
            if not candidates:
                raise FileNotFoundError(f"No MusicXML in {path}")
            return candidates[0]
    if not path.exists():
        raise FileNotFoundError(f"Score not found: {path}")
    return path


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    config = config or PipelineConfig.load()
    samples_root = config.resolved_path("samples_root") or Path("samples")
    app = FastAPI(title="MusicEval Annotator", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (TEMPLATES_DIR / "annotate.html").read_text(encoding="utf-8")
        js_v = int((STATIC_DIR / "annotate.js").stat().st_mtime)
        css_v = int((STATIC_DIR / "style.css").stat().st_mtime)
        html = html.replace("/static/style.css", f"/static/style.css?v={css_v}")
        html = html.replace("/static/annotate.js", f"/static/annotate.js?v={js_v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/api/samples")
    def list_samples() -> list[dict[str, Any]]:
        if not samples_root.exists():
            return []
        items = []
        for d in samples_root.iterdir():
            if not d.is_dir() or not (d / "performance_audio.wav").exists():
                continue
            label_count = 0
            labels_path = d / "labels.json"
            if labels_path.exists():
                label_count = len(read_json(labels_path).get("labels", []))
            items.append(
                {
                    "id": d.name,
                    "label_count": label_count,
                    "has_candidates": (d / "candidates.json").exists(),
                }
            )
        items.sort(key=lambda x: _sample_sort_key(x["id"]))
        return items

    @app.get("/api/batch/info")
    def batch_info() -> dict[str, Any]:
        audio_dir = config.resolved_path("raw_data_audio")
        score_path = config.resolved_path("raw_data_score")
        if score_path and score_path.is_dir():
            scores = sorted(score_path.glob("*.musicxml")) + sorted(score_path.glob("*.mxl"))
            score_file = str(scores[0]) if scores else None
        else:
            score_file = str(score_path) if score_path else None
        available = list_available_audio_ids(audio_dir) if audio_dir else []
        return {
            "audio_dir": str(audio_dir) if audio_dir else None,
            "score_path": score_file,
            "available_audio_ids": available,
        }

    @app.post("/api/batch/range")
    def batch_range(payload: BatchRangePayload) -> dict[str, Any]:
        try:
            score_path = _resolve_score_path(config, payload.score_path)
            audio_dir = Path(payload.audio_dir) if payload.audio_dir else config.resolved_path("raw_data_audio")
            if audio_dir is None or not audio_dir.is_dir():
                raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
            batch = run_batch_range(
                score_path=score_path,
                audio_dir=audio_dir,
                id_from=payload.id_from,
                id_to=payload.id_to,
                config=config,
                id_width=payload.id_width,
                skip_existing=payload.skip_existing,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc
        return {
            "succeeded": batch.succeeded,
            "skipped": batch.skipped,
            "failed": batch.failed,
            "results": [r.__dict__ for r in batch.results],
        }

    @app.get("/api/samples/{sample_id}")
    def get_sample(sample_id: str) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        candidates = read_json(sample_dir / "candidates.json") if (sample_dir / "candidates.json").exists() else {"labels": []}
        labels = read_json(sample_dir / "labels.json") if (sample_dir / "labels.json").exists() else {"labels": [], "self_reported": []}
        prep = get_prep_state(sample_dir, config)
        full_score = sample_dir / "full_score.musicxml"
        if not full_score.exists():
            try:
                ensure_full_score(sample_dir)
            except FileNotFoundError:
                pass
        perf_path = sample_dir / "performance_audio.wav"
        audio_mtime = int(perf_path.stat().st_mtime * 1000) if perf_path.exists() else 0
        align_path = sample_dir / "alignment.npz"
        candidate_labels = candidates.get("labels", [])
        return {
            "sample_id": sample_id,
            "taxonomy": config.taxonomy,
            "schema_version": config.schema_version,
            "candidates": candidate_labels,
            "candidate_count": len(candidate_labels),
            "has_alignment": align_path.exists(),
            "labels": labels.get("labels", []),
            "self_reported": labels.get("self_reported", []),
            "annotator_id": labels.get("annotator_id"),
            "audio_url": f"/api/samples/{sample_id}/audio",
            "audio_mtime": audio_mtime,
            "score_url": f"/api/samples/{sample_id}/score",
            "full_score_url": f"/api/samples/{sample_id}/full-score",
            "prep": prep,
        }

    @app.get("/api/samples/{sample_id}/audio")
    def get_audio(sample_id: str) -> FileResponse:
        path = samples_root / sample_id / "performance_audio.wav"
        if not path.exists():
            raise HTTPException(404, "Audio not found")
        return FileResponse(
            path,
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/samples/{sample_id}/score")
    def get_score(sample_id: str) -> FileResponse:
        path = samples_root / sample_id / "verified_score.musicxml"
        if not path.exists():
            raise HTTPException(404, "Score not found")
        return FileResponse(path, media_type="application/xml")

    @app.get("/api/samples/{sample_id}/full-score")
    def get_full_score(sample_id: str) -> FileResponse:
        sample_dir = samples_root / sample_id
        path = sample_dir / "full_score.musicxml"
        if not path.exists():
            try:
                path = ensure_full_score(sample_dir)
            except FileNotFoundError as exc:
                raise HTTPException(404, "Full score not found") from exc
        return FileResponse(path, media_type="application/xml")

    @app.get("/api/samples/{sample_id}/score-preview")
    def preview_score_segment(
        sample_id: str,
        start_measure: int = Query(..., ge=1),
        end_measure: int = Query(..., ge=1),
        start_beat: int = Query(1, ge=1),
        end_beat: int | None = Query(None, ge=1),
    ) -> Response:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        full_score = ensure_full_score(sample_dir)
        logger = setup_sample_logger(sample_dir, name="prep")
        fd, tmp_name = tempfile.mkstemp(suffix=".musicxml")
        tmp_path = Path(tmp_name)
        try:
            os.close(fd)
            extract_measure_range(
                full_score,
                tmp_path,
                start_measure,
                end_measure,
                logger,
                start_beat=start_beat,
                end_beat=end_beat,
            )
            xml = tmp_path.read_text(encoding="utf-8")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return Response(content=xml, media_type="application/xml")

    @app.post("/api/samples/{sample_id}/score-segment")
    def set_score_segment(sample_id: str, payload: ScoreSegmentPayload) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        logger = setup_sample_logger(sample_dir, name="prep")
        try:
            info = apply_score_segment(
                sample_dir,
                payload.start_measure,
                payload.end_measure,
                config,
                logger,
                start_beat=payload.start_beat,
                end_beat=payload.end_beat,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok", "score_segment": info}

    @app.get("/api/samples/{sample_id}/note-alignment")
    def get_note_alignment(sample_id: str) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        logger = setup_sample_logger(sample_dir, name="prep")
        try:
            return build_note_alignment(sample_dir, logger)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/api/samples/{sample_id}/re-align")
    def realign_sample(sample_id: str) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        logger = setup_sample_logger(sample_dir, name="prep")
        try:
            info = reprocess_alignment(sample_dir, config, logger)
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok", **info}

    @app.post("/api/samples/{sample_id}/trim-performance")
    def trim_performance(sample_id: str, payload: PerformanceTrimPayload) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        logger = setup_sample_logger(sample_dir, name="prep")
        try:
            info = apply_performance_trim(
                sample_dir,
                payload.trim_start,
                payload.trim_end,
                config,
                logger,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok", "performance_trim": info}

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
