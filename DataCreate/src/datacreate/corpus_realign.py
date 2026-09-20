"""Re-align the DataCreate corpus; relocate unlabeled score excerpts from transcription."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from datacreate.align_bridge import dump_transcription
from datacreate.config import PipelineConfig
from datacreate.sample_prep import apply_score_segment, reprocess_alignment
from datacreate.score_locate import (
    list_score_candidates,
    locate_best_among_scores,
    locate_payload,
    notes_from_payload,
    should_apply_location,
)
from datacreate.score_segment import get_measure_count
from datacreate.tools.musescore import warmup_synth
from datacreate.utils import read_json, setup_sample_logger, write_json


def sample_has_labels(sample_dir: Path) -> bool:
    path = sample_dir / "labels.json"
    if not path.exists():
        return False
    try:
        document = read_json(path)
    except Exception:  # noqa: BLE001
        return False
    return bool(document.get("labels"))


def list_sample_dirs(
    samples_root: Path,
    *,
    unlabeled_only: bool = False,
) -> list[Path]:
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
    dirs = sorted(dirs, key=lambda path: (len(path.name), path.name))
    if unlabeled_only:
        dirs = [path for path in dirs if not sample_has_labels(path)]
    return dirs


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


def _current_full_score(sample_dir: Path) -> Path | None:
    full_score = sample_dir / "full_score.musicxml"
    if full_score.exists():
        return full_score
    verified = sample_dir / "verified_score.musicxml"
    return verified if verified.exists() else None


def _install_full_score(sample_dir: Path, source_score: Path, logger: logging.Logger) -> Path:
    destination = sample_dir / "full_score.musicxml"
    if destination.exists() and destination.resolve() == source_score.resolve():
        return destination
    if destination.exists() and destination.read_bytes() == source_score.read_bytes():
        return destination
    shutil.copy2(source_score, destination)
    logger.info(
        "%s: installed RawData score %s as full_score.musicxml",
        sample_dir.name,
        source_score.name,
    )
    meta_path = sample_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        try:
            metadata = read_json(meta_path)
        except Exception:  # noqa: BLE001
            metadata = {}
    metadata["source_score"] = source_score.name
    write_json(meta_path, metadata)
    return destination


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
    current_full = _current_full_score(sample_dir)
    raw_root = config.resolved_path("raw_data_score")
    candidates = list_score_candidates(
        raw_score_root=raw_root,
        sample_full_score=current_full,
    )
    if not candidates and current_full is not None:
        candidates = [current_full]
    best = locate_best_among_scores(transcribed, candidates)
    current = _current_segment(sample_dir)
    info: dict[str, Any] = {
        "relocated": False,
        "score_changed": False,
        "score_candidates": [path.name for path in candidates],
        "previous": current,
        "previous_score": None if current_full is None else current_full.name,
    }
    if best is None:
        info["located"] = locate_payload(None)
        info["total_measures"] = (
            get_measure_count(current_full) if current_full is not None else 0
        )
        logger.info("%s: no RawData score span matched transcription", sample_dir.name)
        return info

    located, source_score = best
    total = get_measure_count(source_score)
    info["located"] = locate_payload(located)
    info["total_measures"] = total
    info["chosen_score"] = source_score.name

    same_bytes = (
        current_full is not None
        and current_full.exists()
        and current_full.read_bytes() == source_score.read_bytes()
    )
    score_switch = not same_bytes
    apply = score_switch or should_apply_location(
        located, current=current, total_measures=total
    )
    if not apply:
        logger.info(
            "%s: keeping %s segment %s (best=%s %s-%s conf=%.3f)",
            sample_dir.name,
            None if current_full is None else current_full.name,
            current,
            source_score.name,
            located.start_measure,
            located.end_measure,
            located.confidence,
        )
        return info

    if score_switch:
        _install_full_score(sample_dir, source_score, logger)
        info["score_changed"] = True

    logger.info(
        "%s: relocating via %s %s -> %s-%s beat %s (confidence=%.3f, mapped=%d)",
        sample_dir.name,
        source_score.name,
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
    info["applied"]["source_score"] = source_score.name
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
    unlabeled_only: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    warmup_synth(config, logger)
    rows = []
    dirs = list_sample_dirs(samples_root, unlabeled_only=unlabeled_only)
    if limit is not None:
        dirs = dirs[: max(0, int(limit))]
    report_name = (
        "realign_unlabeled_corpus_report.json"
        if unlabeled_only
        else "realign_corpus_report.json"
    )
    report_path = samples_root / report_name
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
                f"  ok relocated={row.get('relocated')} "
                f"score_changed={row.get('score_changed')} "
                f"chosen={row.get('chosen_score')} labeled={row.get('labeled')}",
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
