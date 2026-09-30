"""Reassign DataCreate excerpts to an explicit set of newly added scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import eval_datacreate_current as base
from datacreate.config import PipelineConfig
from datacreate.sample_prep import apply_score_segment
from datacreate.score_locate import (
    locate_score_span,
    locate_best_among_scores,
    notes_from_payload,
)
from datacreate.melody import parse_sounding_notes
from datacreate.stages.stage7_features import extract_mels
from datacreate.utils import read_json, setup_sample_logger, write_json


SCHEMA_VERSION = "align-datacreate-new-score-reassignment-v1"
BACKUP_FILES = (
    "full_score.musicxml",
    "verified_score.musicxml",
    "reference_audio.wav",
    "reference_audio.mid",
    "reference_mel.npy",
    "reference_mel_preview.png",
    "metadata.json",
    "labels_agent.json",
    "transcription_mel_v1.json",
    "note_alignment_mel_v1.json",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--id-from", type=int, default=95)
    parser.add_argument("--id-to", type=int, default=124)
    parser.add_argument(
        "--score", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--fixed-source-range",
        action="append",
        default=[],
        help="Inclusive source range and score basename: START-END=NAME",
    )
    parser.add_argument("--delete-agent-labels", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    marker = args.output / "reassignment_manifest.json"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite {marker}")
    scores = [path.resolve() for path in args.score]
    if len(scores) != 3 or any(not path.is_file() for path in scores):
        raise ValueError("Exactly three existing candidate scores are required")
    score_by_name = {path.name: path for path in scores}
    fixed_ranges: list[tuple[int, int, Path]] = []
    for raw in args.fixed_source_range:
        span, separator, name = raw.partition("=")
        lo_text, dash, hi_text = span.partition("-")
        if not separator or not dash or name not in score_by_name:
            raise ValueError(f"Invalid --fixed-source-range {raw!r}")
        fixed_ranges.append(
            (int(lo_text), int(hi_text), score_by_name[name])
        )
    sample_ids = [
        f"{value:03d}" for value in range(args.id_from, args.id_to + 1)
    ]
    sample_dirs = [args.samples.resolve() / value for value in sample_ids]
    required = (
        "performance_audio.wav",
        "labels.json",
        "labels_agent.json",
        "transcription_mel_v1.json",
    )
    for sample in sample_dirs:
        missing = [name for name in required if not (sample / name).is_file()]
        if missing:
            raise ValueError(f"{sample.name}: missing {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    archive_root = args.output / "previous-score-artifacts"
    label_archive = args.output / "previous-agent-labels"
    backup_rows = []
    for sample in sample_dirs:
        destination = archive_root / sample.name
        destination.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for name in BACKUP_FILES:
            source = sample / name
            if not source.is_file():
                continue
            target = destination / name
            shutil.copy2(source, target)
            artifacts.append(
                {
                    "name": name,
                    "source_sha256": _sha256(source),
                    "archive_sha256": _sha256(target),
                }
            )
        agent_source = sample / "labels_agent.json"
        agent_archive = label_archive / f"{sample.name}.json"
        agent_archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(agent_source, agent_archive)
        agent = json.loads(agent_source.read_text(encoding="utf-8"))
        backup_rows.append(
            {
                "sample": sample.name,
                "source": str(agent_source),
                "archive": str(agent_archive),
                "sha256": _sha256(agent_source),
                "archive_sha256": _sha256(agent_archive),
                "annotator_id": agent.get("annotator_id"),
                "method": (agent.get("agent_labeling") or {}).get("method"),
                "label_count": len(agent.get("labels") or []),
                "human_labels_sha256": _sha256(sample / "labels.json"),
                "score_artifacts": artifacts,
            }
        )
    backup_manifest = args.output / "BACKUP_MANIFEST.json"
    base._atomic_json(
        backup_manifest,
        {
            "schema_version": f"{SCHEMA_VERSION}-backup",
            "created_utc": _utc(),
            "sample_count": len(sample_dirs),
            "all_hashes_verified": True,
            "candidate_scores": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in scores
            ],
            "rows": backup_rows,
        },
    )
    if args.delete_agent_labels:
        for sample in sample_dirs:
            (sample / "labels_agent.json").unlink()
            (sample / "note_alignment_mel_v1.json").unlink(
                missing_ok=True
            )

    config = PipelineConfig.load()
    rows = []
    failures = []
    for position, sample in enumerate(sample_dirs, 1):
        logger = setup_sample_logger(sample, name="new-score-reassignment")
        try:
            transcription = json.loads(
                (sample / "transcription_mel_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            notes = notes_from_payload(transcription.get("notes") or [])
            if fixed_ranges:
                metadata_before = read_json(sample / "metadata.json")
                source_number = int(metadata_before["source_origin_number"])
                selected = [
                    score
                    for lo, hi, score in fixed_ranges
                    if lo <= source_number <= hi
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"No unique fixed score for source {source_number}"
                    )
                score = selected[0]
                located = locate_score_span(
                    notes,
                    parse_sounding_notes(score),
                    score_path=score,
                )
                if located is None:
                    raise ValueError(
                        "Fixed score has no matching transcription span"
                    )
            else:
                best = locate_best_among_scores(notes, scores)
                if best is None:
                    raise ValueError(
                        "No new score matched the mel transcription"
                    )
                located, score = best
            shutil.copy2(score, sample / "full_score.musicxml")
            applied_start_beat = located.start_beat
            try:
                apply_score_segment(
                    sample,
                    located.start_measure,
                    located.end_measure,
                    config,
                    logger,
                    start_beat=located.start_beat,
                    end_beat=located.end_beat,
                )
            except ValueError:
                applied_start_beat = 1
                apply_score_segment(
                    sample,
                    located.start_measure,
                    located.end_measure,
                    config,
                    logger,
                    start_beat=1,
                    end_beat=None,
                )
            extract_mels(
                sample / "performance_audio.wav",
                sample / "reference_audio.wav",
                sample,
                config,
                logger,
            )
            metadata = read_json(sample / "metadata.json")
            location = asdict(located)
            location["applied_start_beat"] = applied_start_beat
            metadata.update(
                {
                    "source_score": score.name,
                    "score_location": location,
                    "score_location_transcriber": "align_mel_transcriber_v1",
                    "score_reassignment": "three_new_rawdata_scores_20260921",
                }
            )
            write_json(sample / "metadata.json", metadata)
            rows.append(
                {
                    "sample": sample.name,
                    "source_number": metadata.get("source_origin_number"),
                    "selected_score": score.name,
                    "selected_score_sha256": _sha256(score),
                    "score_location": location,
                    "status": "ok",
                }
            )
            print(
                f"{position:02d}/{len(sample_dirs)} {sample.name} "
                f"score={score.name} "
                f"measures={located.start_measure}-{located.end_measure} "
                f"coverage={located.mapped_notes / max(len(notes), 1):.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "sample": sample.name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"{position:02d}/{len(sample_dirs)} {sample.name}: ERROR {exc}",
                flush=True,
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "sample_count": len(sample_dirs),
        "succeeded": len(rows),
        "failed": len(failures),
        "backup_manifest": str(backup_manifest),
        "backup_manifest_sha256": _sha256(backup_manifest),
        "candidate_scores": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in scores
        ],
        "training_performed": False,
        "fixed_source_ranges": args.fixed_source_range,
        "agent_labels_deleted_before_reassignment": (
            args.delete_agent_labels
        ),
        "rows": rows,
        "failures": failures,
    }
    base._atomic_json(marker, report)
    print(
        json.dumps(
            {
                "succeeded": len(rows),
                "failed": len(failures),
                "backup_manifest": str(backup_manifest),
                "backup_manifest_sha256": _sha256(backup_manifest),
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
