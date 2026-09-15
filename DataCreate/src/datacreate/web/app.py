from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datacreate.batch_audio import list_available_audio_ids, run_batch_range
from datacreate.config import PipelineConfig
from datacreate.models import LabelsDocument
from datacreate.melody import match_note_wise_labels_detail
from datacreate.note_alignment import build_note_alignment, build_score_events
from datacreate.sample_prep import (
    apply_performance_trim,
    apply_score_segment,
    ensure_full_score,
    get_prep_state,
    reprocess_alignment,
)
from datacreate.score_segment import extract_measure_range
from datacreate.tools.musescore import renderer_status, warmup_synth_background
from datacreate.utils import read_json, setup_sample_logger, write_json
from datacreate.validation import validate_labels_file
from datacreate.web.compare_eval import default_eval_dir, load_summary, sample_payload


WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
LabelSource = Literal["human", "agent"]
LABEL_SOURCE_FILES: dict[LabelSource, str] = {
    "human": "labels.json",
    "agent": "labels_agent.json",
}


def _labels_path(sample_dir: Path, label_source: LabelSource) -> Path:
    return sample_dir / LABEL_SOURCE_FILES[label_source]


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

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        warmup_synth_background(config)
        yield

    app = FastAPI(title="MusicEval Annotator", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (TEMPLATES_DIR / "annotate.html").read_text(encoding="utf-8")
        js_v = int((STATIC_DIR / "annotate.js").stat().st_mtime)
        css_v = int((STATIC_DIR / "style.css").stat().st_mtime)
        html = html.replace("/static/style.css", f"/static/style.css?v={css_v}")
        html = html.replace("/static/annotate.js", f"/static/annotate.js?v={js_v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    def _inject_asset_versions(name: str) -> HTMLResponse:
        html = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
        js_name = "compare.js" if name == "compare.html" else "annotate.js"
        js_v = int((STATIC_DIR / js_name).stat().st_mtime)
        css_v = int((STATIC_DIR / "style.css").stat().st_mtime)
        html = html.replace("/static/style.css", f"/static/style.css?v={css_v}")
        html = html.replace(f"/static/{js_name}", f"/static/{js_name}?v={js_v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/compare", response_class=HTMLResponse)
    def compare_index() -> HTMLResponse:
        return _inject_asset_versions("compare.html")

    @app.get("/api/samples")
    def list_samples() -> list[dict[str, Any]]:
        if not samples_root.exists():
            return []
        items = []
        for d in samples_root.iterdir():
            if not d.is_dir() or not (d / "performance_audio.wav").exists():
                continue
            counts = {}
            for source, filename in LABEL_SOURCE_FILES.items():
                labels_path = d / filename
                counts[source] = (
                    len(read_json(labels_path).get("labels", []))
                    if labels_path.exists()
                    else 0
                )
            items.append(
                {
                    "id": d.name,
                    "label_count": counts["human"],
                    "agent_label_count": counts["agent"],
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

    @app.get("/api/renderer/status")
    def get_renderer_status() -> dict[str, Any]:
        return renderer_status()

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
    def get_sample(
        sample_id: str,
        label_source: LabelSource = Query("human"),
    ) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        labels_path = _labels_path(sample_dir, label_source)
        labels = (
            read_json(labels_path)
            if labels_path.exists()
            else {"labels": [], "self_reported": []}
        )
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
        return {
            "sample_id": sample_id,
            "label_source": label_source,
            "label_source_available": labels_path.exists(),
            "taxonomy": config.taxonomy,
            "schema_version": config.schema_version,
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

    @app.get("/api/samples/{sample_id}/score-events")
    def get_score_events(sample_id: str) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        logger = setup_sample_logger(sample_dir, name="prep")
        try:
            return build_score_events(sample_dir, logger)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.get("/api/samples/{sample_id}/note-alignment")
    def get_note_alignment(sample_id: str) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        logger = setup_sample_logger(sample_dir, name="prep")
        try:
            if not (sample_dir / "note_alignment_v2.json").exists():
                reprocess_alignment(sample_dir, config, logger)
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
    def save_labels(
        sample_id: str,
        payload: LabelsPayload,
        label_source: LabelSource = Query("human"),
    ) -> dict[str, str]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        doc = LabelsDocument(
            schema_version="1.2" if label_source == "agent" else config.schema_version,
            annotator_id=payload.annotator_id,
            labels=payload.labels,  # type: ignore[arg-type]
            self_reported=payload.self_reported,  # type: ignore[arg-type]
        )
        path = _labels_path(sample_dir, label_source)
        document = doc.model_dump()
        if label_source == "agent" and path.exists():
            metadata = read_json(path).get("agent_labeling")
            if metadata is not None:
                document["agent_labeling"] = {
                    **metadata,
                    "edited_in_gui": True,
                }
        write_json(path, document)
        errors = validate_labels_file(path, config)
        if errors:
            raise HTTPException(400, "; ".join(errors))
        return {"status": "saved", "label_source": label_source}

    @app.get("/api/compare/summary")
    def compare_summary() -> dict[str, Any]:
        eval_dir = default_eval_dir()
        try:
            summary = load_summary(eval_dir)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        samples = []
        for row in summary.get("samples") or []:
            sid = str(row.get("sample") or "")
            if not sid:
                continue
            hs = row.get("hard_type_sensitive") or {}
            hi = row.get("hard_type_insensitive") or {}
            samples.append(
                {
                    "id": sid,
                    "n_gold": row.get("n_gold"),
                    "n_pred": row.get("n_pred"),
                    "hard_type_sensitive_f1": hs.get("melody_f1"),
                    "hard_type_insensitive_f1": hi.get("melody_f1"),
                }
            )
        return {
            "eval_dir": str(eval_dir),
            "checkpoint": summary.get("checkpoint"),
            "n_samples": summary.get("n_samples"),
            "n_skipped": summary.get("n_skipped"),
            "criteria": summary.get("criteria"),
            "samples": samples,
        }

    @app.get("/api/compare/samples/{sample_id}")
    def compare_sample(sample_id: str) -> dict[str, Any]:
        sample_dir = samples_root / sample_id
        if not sample_dir.exists():
            raise HTTPException(404, "Sample not found")
        eval_dir = default_eval_dir()
        try:
            summary = load_summary(eval_dir)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return sample_payload(sample_dir, eval_dir, summary)

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

    detail = match_note_wise_labels_detail(a, b)
    if detail["status"] != "available":
        return {
            "official_note_wise": "unavailable",
            "reason": detail["reason"],
            "diagnostic_timestamp_iou": True,
            "matched": [],
            "type_mismatch": [],
            "only_a": list(a),
            "only_b": list(b),
            "agreement_score": None,
        }
    matched, type_mismatch = [], []
    paired_a, paired_b = set(), set()
    for pair in detail["pairs"]:
        b_index = int(pair["prediction_index"])
        a_index = int(pair["gold_index"])
        if float(pair["credit"]) <= 0.0:
            continue
        paired_a.add(a_index)
        paired_b.add(b_index)
        row = {
            "a": a[a_index],
            "b": b[b_index],
            "diagnostic_iou": iou(a[a_index], b[b_index]),
        }
        (matched if pair["type_match"] else type_mismatch).append(row)
    return {
        "official_note_wise": "available",
        "matched": matched,
        "type_mismatch": type_mismatch,
        "only_a": [value for index, value in enumerate(a) if index not in paired_a],
        "only_b": [value for index, value in enumerate(b) if index not in paired_b],
        "agreement_score": round(float(detail["f1"]), 4),
    }
