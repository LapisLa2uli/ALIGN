"""Re-align the DataCreate corpus; relocate unlabeled score excerpts from transcription."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datacreate.align_bridge import dump_transcription
from datacreate.config import PipelineConfig
from datacreate.sample_prep import apply_score_segment, reprocess_alignment
from datacreate.score_locate import (
    locate_score_span,
    locate_payload,
    notes_from_payload,
    should_apply_location,
)
from datacreate.score_segment import get_measure_count
from datacreate.melody import parse_sounding_notes
from datacreate.tools.musescore import warmup_synth
from datacreate.utils import read_json, setup_sample_logger


def sample_has_labels(sample_dir: Path) -> bool:
    path = sample_dir / "labels.json"
    if not path.exists():
        return False
    try:
        document = read_json(path)
    except Exception:  # noqa: BLE001
        return False
    return bool(document.get("labels"))


def list_sample_dirs(samples_root: Path) -> list[Path]:
    if not samples_root.exists():
        return []
    dirs = [
        path
        for path in samples_root.iterdir()
        if path.is_dir()
        and path.name != "synthetic"
        and (path / "performance_audio.wav").exists()
        and (
            (path / "verified_score.musicxml").exists()
            or (path / "full_score.musicxml").exists()
        )
    ]
    return sorted(dirs, key=lambda path: (len(path.name), path.name))


def _current_segment(sample_dir: Path) -> dict[str, Any] | None:
    meta_path = sample_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        metadata = read_json(meta_path)
    except Exception:  # noqa: BLE001
        return None
    segment = metadata.get("score_segment")
    return segment if isinstance(segment, dict) else None


def relocate_unlabeled_sample(
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    transcription_path = dump_transcription(sample_dir, config, logger)
    payload = json.loads(transcription_path.read_text(encoding="utf-8"))
    transcribed = notes_from_payload(
        payload.get("transcribed_notes") or payload.get("notes") or []
    )
    full_score = sample_dir / "full_score.musicxml"
    if not full_score.exists():
        full_score = sample_dir / "verified_score.musicxml"
    written = parse_sounding_notes(full_score)
    located = locate_score_span(transcribed, written, score_path=full_score)
    total = get_measure_count(full_score)
    current = _current_segment(sample_dir)
    info = {
        "relocated": False,
        "located": locate_payload(located),
        "previous": current,
        "total_measures": total,
    }
    if not should_apply_location(
        located, current=current, total_measures=total
    ):
        logger.info(
            "%s: keeping score segment %s (confidence=%s)",
            sample_dir.name,
            current,
            None if located is None else round(located.confidence, 3),
        )
        return info
    assert located is not None
    logger.info(
        "%s: relocating %s -> %s-%s beat %s (confidence=%.3f, mapped=%d)",
        sample_dir.name,
        current,
        located.start_measure,
        located.end_measure,
        located.start_beat,
        located.confidence,
        located.mapped_notes,
    )
    try:
        apply_score_segment(
            sample_dir,
            located.start_measure,
            located.end_measure,
            config,
            logger,
            start_beat=located.start_beat,
            end_beat=located.end_beat,
        )
    except ValueError:
        logger.warning(
            "%s: beat %s rejected, retrying from beat 1",
            sample_dir.name,
            located.start_beat,
        )
        apply_score_segment(
            sample_dir,
            located.start_measure,
            located.end_measure,
            config,
            logger,
            start_beat=1,
            end_beat=None,
        )
    info["relocated"] = True
    info["applied"] = located.as_segment()
    return info


def realign_sample(
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
    *,
    relocate: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample": sample_dir.name,
        "labeled": sample_has_labels(sample_dir),
        "relocated": False,
    }
    if relocate and not result["labeled"]:
        locate_info = relocate_unlabeled_sample(sample_dir, config, logger)
        result.update(locate_info)
    alignment = reprocess_alignment(sample_dir, config, logger)
    result["alignment_path"] = alignment.get("alignment_path")
    result["status"] = "ok"
    return result


def realign_corpus(
    samples_root: Path,
    config: PipelineConfig,
    logger: logging.Logger,
    *,
    relocate_unlabeled: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    warmup_synth(config, logger)
    rows = []
    dirs = list_sample_dirs(samples_root)
    if limit is not None:
        dirs = dirs[: max(0, int(limit))]
    report_path = samples_root / "realign_corpus_report.json"
    for index, sample_dir in enumerate(dirs, start=1):
        sample_log = setup_sample_logger(sample_dir, name="realign")
        logger.info("[%d/%d] %s", index, len(dirs), sample_dir.name)
        print(f"[{index}/{len(dirs)}] {sample_dir.name}", flush=True)
        try:
            row = realign_sample(
                sample_dir,
                config,
                sample_log,
                relocate=relocate_unlabeled,
            )
            rows.append(row)
            print(
                f"  ok relocated={row.get('relocated')} labeled={row.get('labeled')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed %s", sample_dir.name)
            row = {
                "sample": sample_dir.name,
                "status": "error",
                "error": str(exc),
                "labeled": sample_has_labels(sample_dir),
            }
            rows.append(row)
            print(f"  ERROR {exc}", flush=True)
        report_path.write_text(
            json.dumps({"completed": len(rows), "total": len(dirs), "rows": rows}, indent=2),
            encoding="utf-8",
        )
    return rows
