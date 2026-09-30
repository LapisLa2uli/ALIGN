"""Import newly numbered RawData audio into DataCreate using mel score location."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import eval_datacreate_current as base
from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from datacreate.config import PipelineConfig
from datacreate.melody import parse_sounding_notes
from datacreate.sample_prep import apply_score_segment
from datacreate.score_locate import TranscribedNote, locate_score_span
from datacreate.stages.stage4_performance import ingest_performance
from datacreate.stages.stage7_features import extract_mels
from datacreate.stages.stage8_bundle import (
    write_labels_template,
    write_metadata,
)
from datacreate.utils import read_json, setup_sample_logger, write_json


SCHEMA_VERSION = "align-rawdata-mel-import-v1"
SOURCE_NUMBER = re.compile(r"古龙路\s*(\d+)", re.IGNORECASE)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_number(path: Path) -> int | None:
    match = SOURCE_NUMBER.search(path.stem)
    return int(match.group(1)) if match else None


def plan_import(
    audio_dir: Path,
    samples_root: Path,
) -> list[dict[str, object]]:
    used_source_numbers = set()
    numeric_ids = []
    for sample in samples_root.iterdir():
        if sample.is_dir() and sample.name.isdigit():
            numeric_ids.append(int(sample.name))
        metadata = sample / "metadata.json"
        if metadata.is_file():
            value = read_json(metadata)
            if value.get("source_origin") == "gulonglu":
                number = value.get("source_origin_number")
                if number is not None:
                    used_source_numbers.add(int(number))
    candidates = []
    for path in audio_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {
            ".mp3",
            ".m4a",
            ".wav",
            ".flac",
            ".aac",
            ".ogg",
        }:
            continue
        number = source_number(path)
        if number is not None and number not in used_source_numbers:
            candidates.append((number, path))
    candidates.sort(key=lambda value: (value[0], value[1].name))
    duplicates = [
        number
        for number in {number for number, _path in candidates}
        if sum(value == number for value, _path in candidates) != 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate new source numbers: {duplicates}")
    first_id = max(numeric_ids, default=0) + 1
    return [
        {
            "source_number": number,
            "sample_id": f"{first_id + index:03d}",
            "original_path": str(path.resolve()),
            "original_filename": path.name,
            "renamed_path": str(
                (audio_dir / f"{first_id + index:03d}{path.suffix.lower()}")
                .resolve()
            ),
        }
        for index, (number, path) in enumerate(candidates)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=root / "RawData" / "Audio",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=root / "DataCreate" / "samples",
    )
    parser.add_argument(
        "--score",
        type=Path,
        default=root / "RawData" / "Score" / "MozartClConcertoA.musicxml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-plan", type=Path)
    parser.add_argument(
        "--mel-checkpoint",
        type=Path,
        default=root
        / "align-model"
        / "runs"
        / "joint-outputraw-full-v1"
        / "mel-transcriber-v1"
        / "full-training-all4544-v2"
        / "candidate-epoch-018.pt",
    )
    parser.add_argument(
        "--expected-mel-sha256",
        default="3d8f93732a810c3f8470a88316debb9f92b4680b2333c2187866e6090a730a4e",
    )
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--reset-partial", action="store_true")
    parser.add_argument(
        "--minimum-location-confidence",
        type=float,
        default=0.28,
    )
    args = parser.parse_args(argv)

    args.output = args.output.resolve()
    marker = args.output / "import_manifest.json"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite {marker}")
    if _sha256(args.mel_checkpoint.resolve()) != args.expected_mel_sha256:
        raise ValueError("Frozen mel checkpoint mismatch")
    if not args.score.is_file():
        raise FileNotFoundError(args.score)
    resumed = args.resume_plan is not None
    if resumed:
        previous_plan = json.loads(
            args.resume_plan.read_text(encoding="utf-8")
        )
        plan = [dict(row) for row in previous_plan["rows"]]
        for row in plan:
            renamed = Path(str(row["renamed_path"]))
            if (
                not renamed.is_file()
                or _sha256(renamed) != row["source_sha256"]
            ):
                raise ValueError(f"Resumed renamed file mismatch: {renamed}")
    else:
        plan = plan_import(args.audio_dir.resolve(), args.samples.resolve())
        if not plan:
            raise ValueError("No new numbered RawData files found")
        if any((args.samples / str(row["sample_id"])).exists() for row in plan):
            raise ValueError("One or more planned DataCreate IDs already exist")
        if any(Path(str(row["renamed_path"])).exists() for row in plan):
            raise ValueError("One or more planned renamed audio paths already exist")
    if args.sample_id:
        requested = set(map(str, args.sample_id))
        plan = [
            row for row in plan if str(row["sample_id"]) in requested
        ]
        if {str(row["sample_id"]) for row in plan} != requested:
            raise ValueError("One or more requested retry sample IDs are absent")

    args.output.mkdir(parents=True, exist_ok=True)
    planned = (
        previous_plan
        if resumed
        else {
            "schema_version": f"{SCHEMA_VERSION}-plan",
            "created_utc": _utc(),
            "score": str(args.score.resolve()),
            "score_sha256": _sha256(args.score),
            "rows": [
                {
                    **row,
                    "source_sha256": _sha256(
                        Path(str(row["original_path"]))
                    ),
                }
                for row in plan
            ],
        }
    )
    base._atomic_json(args.output / "RENAME_PLAN.json", planned)

    # The plan is durable before source names change.
    if not resumed:
        for row in plan:
            source = Path(str(row["original_path"]))
            destination = Path(str(row["renamed_path"]))
            os.replace(source, destination)
            if _sha256(destination) != next(
                value["source_sha256"]
                for value in planned["rows"]
                if value["sample_id"] == row["sample_id"]
            ):
                raise RuntimeError(f"Rename hash mismatch: {destination}")

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model, frontend, decode, payload = load_mel_checkpoint(
        args.mel_checkpoint.resolve(), device
    )
    decode = replace(decode, min_confidence=args.min_confidence)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = PipelineConfig.load()
    written = parse_sounding_notes(args.score)
    if not written:
        raise ValueError("Source score has no sounding notes")
    rows = []
    failures = []
    for position, row in enumerate(plan, 1):
        sample_id = str(row["sample_id"])
        sample_dir = args.samples / sample_id
        try:
            if args.reset_partial and sample_dir.exists():
                agent = sample_dir / "labels_agent.json"
                labels = sample_dir / "labels.json"
                if agent.exists():
                    raise ValueError("Refusing to reset a labeled agent bundle")
                if labels.exists() and (
                    read_json(labels).get("labels") or []
                ):
                    raise ValueError("Refusing to reset human-labeled bundle")
                shutil.rmtree(sample_dir)
            if sample_dir.exists():
                unexpected = [
                    path
                    for path in sample_dir.iterdir()
                    if path.name
                    not in {"pipeline.log", "full_score.musicxml"}
                ]
                if unexpected:
                    raise ValueError(
                        f"Existing sample has bundle files: {unexpected}"
                    )
            else:
                sample_dir.mkdir(parents=True)
            existing_full = sample_dir / "full_score.musicxml"
            if (
                existing_full.exists()
                and _sha256(existing_full) != _sha256(args.score)
            ):
                raise ValueError("Existing partial full score hash mismatch")
            logger = setup_sample_logger(
                sample_dir, name="rawdata-mel-import"
            )
            shutil.copy2(args.score, sample_dir / "full_score.musicxml")
            audio_path = Path(str(row["renamed_path"]))
            performance = ingest_performance(
                audio_path, sample_dir, config, logger
            )
            write_labels_template(sample_dir, config)
            write_metadata(
                sample_dir,
                config,
                {
                    "sample_id": sample_id,
                    "piece": "mozart",
                    "source_filename": row["original_filename"],
                    "source_origin": "gulonglu",
                    "source_origin_number": row["source_number"],
                    "renamed_rawdata_filename": audio_path.name,
                    "import_method": "frozen_mel_transcriber_score_location",
                    "training_performed": False,
                },
                logger,
            )
            audio = load_audio_mono(performance, frontend.sample_rate)
            mel, normalization = extract_log_mel(
                audio, frontend, device=device
            )
            probabilities = infer_mel_probabilities(
                model,
                np.asarray(mel, np.float32),
                device,
                window_frames=2048,
                overlap_frames=512,
                batch_size=args.batch_size,
            )
            notes = decode_mel_notes(
                probabilities,
                midi_min=model.config.midi_min,
                hop_sec=frontend.hop_sec,
                config=decode,
            )
            located = locate_score_span(
                [
                    TranscribedNote(
                        note.pitch,
                        note.start,
                        note.end,
                        note.confidence,
                    )
                    for note in notes
                ],
                written,
                score_path=args.score,
            )
            if (
                located is None
                or located.confidence < args.minimum_location_confidence
                or located.mapped_notes < 6
            ):
                raise ValueError(
                    f"Could not locate score reliably: {located}"
                )
            applied_start_beat = located.start_beat
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
                    "%s: located beat %s rejected; retrying from beat 1",
                    sample_id,
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
                applied_start_beat = 1
            extract_mels(
                performance,
                sample_dir / "reference_audio.wav",
                sample_dir,
                config,
                logger,
            )
            metadata = read_json(sample_dir / "metadata.json")
            location_payload = asdict(located)
            location_payload["applied_start_beat"] = applied_start_beat
            metadata.update(
                {
                    "score_location": location_payload,
                    "score_location_transcriber": "align_mel_transcriber_v1",
                    "mel_checkpoint_sha256": args.expected_mel_sha256,
                }
            )
            write_json(sample_dir / "metadata.json", metadata)
            transcription_path = (
                args.output / "score-location-transcriptions" / f"{sample_id}.json"
            )
            base._atomic_json(
                transcription_path,
                {
                    "sample": sample_id,
                    "audio_sha256": _sha256(performance),
                    "normalization": normalization,
                    "notes": [note.to_dict() for note in notes],
                    "training_performed": False,
                },
            )
            rows.append(
                {
                    **row,
                    "status": "ok",
                    "sample_dir": str(sample_dir),
                    "renamed_sha256": _sha256(audio_path),
                    "performance_audio_sha256": _sha256(performance),
                    "score_location": location_payload,
                    "transcribed_notes": len(notes),
                    "transcription": str(transcription_path),
                    "transcription_sha256": _sha256(transcription_path),
                }
            )
            print(
                f"{position:02d}/{len(plan)} {sample_id} "
                f"source={row['source_number']} "
                f"measures={located.start_measure}-{located.end_measure} "
                f"confidence={located.confidence:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    **row,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"{position:02d}/{len(plan)} {sample_id}: ERROR {exc}",
                flush=True,
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "score": str(args.score.resolve()),
        "score_sha256": _sha256(args.score),
        "mel_checkpoint": str(args.mel_checkpoint.resolve()),
        "mel_checkpoint_sha256": args.expected_mel_sha256,
        "training_performed": False,
        "requested": len(plan),
        "succeeded": len(rows),
        "failed": len(failures),
        "rows": rows,
        "failures": failures,
    }
    base._atomic_json(marker, report)
    base._atomic_json(
        args.samples / f"import_batch_{plan[0]['sample_id']}_{plan[-1]['sample_id']}.json",
        report,
    )
    print(
        json.dumps(
            {
                "requested": len(plan),
                "succeeded": len(rows),
                "failed": len(failures),
                "first_id": plan[0]["sample_id"],
                "last_id": plan[-1]["sample_id"],
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
