"""Relabel every DataCreate take with the frozen ALIGN mel transcriber.

This performs inference only:

    performance WAV -> frozen mel transcriber -> frozen joint aligner
    -> frozen error heads -> labels_agent.json

All staged labels are validated before any sample is overwritten.  A required
pre-run backup manifest makes the commit reversible and proves that human
``labels.json`` files were not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import eval_datacreate_current as base
import label_datacreate_agent as agent_base
import train_error_heads_v5 as v5
from alignmodel.joint.candidates import add_score_repeat_hints
from alignmodel.joint.error_heads import (
    build_inference_rows,
    direct_operation_probabilities,
    direct_rhythm_probabilities,
    infer_error_heads,
    schema12_document_v3,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.transcription.mel_v1 import (
    MelNote,
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from train_error_heads_v3 import _config_from_json


SCHEMA_VERSION = "align-datacreate-agent-labels-mel-v1"
ANNOTATOR_ID = "cursor_agent_mel_transcriber_v1_error_heads_v5"
METHOD = "align_mel_transcriber_v1_joint_error_heads_v5"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "performance_audio.wav").is_file()
        and (path / "verified_score.musicxml").is_file()
    )


def mel_candidates(notes: Sequence[MelNote]) -> list[JointCandidate]:
    """Expose the same calibrated pitch alternatives used in Track B eval."""

    candidates: dict[tuple[int, int], JointCandidate] = {}
    for note in notes:
        pitches = note.pitch_candidates or (note.pitch,)
        probabilities = note.candidate_confidences or (note.confidence,)
        for rank, pitch in enumerate(pitches):
            probability = float(
                probabilities[min(rank, len(probabilities) - 1)]
            )
            confidence = (
                float(note.confidence)
                if rank == 0
                else float(note.confidence)
                * max(0.05, min(0.85, probability))
            )
            if rank > 0 and probability < 0.08:
                continue
            primary = float(probabilities[0])
            candidate = JointCandidate(
                pitch=int(pitch),
                start=float(note.start),
                end=float(note.end),
                confidence=confidence,
                score_hints=(),
                acoustic_features=(
                    float(note.onset_strength),
                    max(float(note.confidence), probability),
                    float(note.confidence),
                    min(0.0, probability - primary),
                    probability,
                ),
            )
            key = (int(round(float(note.start) * 1000.0)), int(pitch))
            previous = candidates.get(key)
            if previous is None or previous.confidence < candidate.confidence:
                candidates[key] = candidate
    return sorted(
        candidates.values(),
        key=lambda value: (value.start, value.pitch, value.end),
    )


def _agent_document(
    prediction: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    document = agent_base.agent_document_from_prediction(
        prediction, metadata=metadata
    )
    document["annotator_id"] = ANNOTATOR_ID
    labeling = document["agent_labeling"]
    labeling["method"] = METHOD
    labeling["transcriber"] = "frozen_align_mel_transcriber_v1"
    labeling["uses_project_alignment_or_error_models"] = True
    for label in document["labels"]:
        label["comment"] = (
            "Frozen ALIGN mel transcriber v1, joint aligner, and "
            "error-heads v5 prediction on the DataCreate take."
        )
    return document


def _alignment_document(
    *,
    sample: str,
    notes: Sequence[MelNote],
    candidates: Sequence[JointCandidate],
    path: Any,
    score_event_count: int,
) -> dict[str, Any]:
    events = path.joint_events(candidates)
    deleted = set(path.trailing_deletions)
    for step in path.steps:
        deleted.update(step.deleted_events)
    return {
        "schema_version": f"{SCHEMA_VERSION}-alignment",
        "sample": sample,
        "transcribed_note_count": len(notes),
        "candidate_count": len(candidates),
        "score_event_count": score_event_count,
        "path_score": float(path.score),
        "deleted_score_event_indices": sorted(deleted),
        "steps": [
            {
                "candidate_index": int(step.candidate_index),
                "score_span": (
                    list(step.score_span)
                    if step.score_span is not None
                    else None
                ),
                "operation": step.operation.value,
                "structural_operation": (
                    step.structural_operation.value
                    if step.structural_operation is not None
                    else None
                ),
                "resume_event": step.resume_event,
                "deleted_events": list(step.deleted_events),
            }
            for step in path.steps
        ],
        "events": [asdict(event) for event in events],
    }


def _verify_backup(
    samples: Sequence[Path],
    manifest_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if _sha256(manifest_path) != expected_sha256:
        raise ValueError("Agent-label rollback manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["sample"]): row for row in manifest["rows"]}
    if (
        int(manifest["sample_count"]) != len(samples)
        or set(rows) != {sample.name for sample in samples}
    ):
        raise ValueError("Rollback manifest population mismatch")
    for sample in samples:
        current = sample / "labels_agent.json"
        if not current.is_file() or _sha256(current) != rows[sample.name][
            "sha256"
        ]:
            raise ValueError(
                f"{sample.name}: current agent labels differ from rollback backup"
            )
        archive = Path(rows[sample.name]["archive"])
        if (
            not archive.is_file()
            or _sha256(archive) != rows[sample.name]["archive_sha256"]
        ):
            raise ValueError(f"{sample.name}: rollback archive mismatch")
    return manifest


def _label_one(
    *,
    sample: Path,
    output: Path,
    stack: Mapping[str, Any],
    selector: Mapping[str, Any],
    decode_config: Any,
    mel_model: Any,
    mel_frontend: Any,
    mel_decode: Any,
    device: torch.device,
    metadata: Mapping[str, Any],
    batch_size: int,
    reuse_transcription: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    audio_path = sample / "performance_audio.wav"
    if reuse_transcription:
        existing_transcription = json.loads(
            (sample / "transcription_mel_v1.json").read_text(
                encoding="utf-8"
            )
        )
        notes = [
            MelNote(
                pitch=int(value["pitch"]),
                start=float(value["start"]),
                end=float(value["end"]),
                confidence=float(value["confidence"]),
                pitch_candidates=tuple(
                    map(int, value.get("pitch_candidates") or ())
                ),
                candidate_confidences=tuple(
                    map(float, value.get("candidate_confidences") or ())
                ),
                onset_strength=float(value["onset_strength"]),
                boundary_strength=float(value["boundary_strength"]),
            )
            for value in existing_transcription["notes"]
        ]
        normalization = existing_transcription.get("normalization") or {}
    else:
        audio = load_audio_mono(audio_path, mel_frontend.sample_rate)
        mel, normalization = extract_log_mel(
            audio, mel_frontend, device=device
        )
        probabilities = infer_mel_probabilities(
            mel_model,
            np.asarray(mel, np.float32),
            device,
            window_frames=2048,
            overlap_frames=512,
            batch_size=batch_size,
        )
        notes = decode_mel_notes(
            probabilities,
            midi_min=mel_model.config.midi_min,
            hop_sec=mel_frontend.hop_sec,
            config=mel_decode,
        )
    score_path = sample / "verified_score.musicxml"
    score = tuple(ScoreEventIndex.from_musicxml(score_path).events)
    if not score:
        raise ValueError("verified score has no sounding events")
    candidates = mel_candidates(notes)
    admitted = tuple(add_score_repeat_hints(candidates, score))
    if not admitted:
        raise ValueError("mel transcriber produced no admitted candidates")
    path = stack["lattice"].decode(admitted, score)
    rows = build_inference_rows(
        stack["lattice"], admitted, score, path
    )
    events = tuple(path.joint_events(admitted))
    learned = infer_error_heads(
        stack["heads_model"],
        rows,
        stack["heads_payload"]["thresholds"],
        device="cpu",
    )
    direct, _diagnostics = direct_operation_probabilities(
        rows, events, score
    )
    clip = {
        "rows": rows,
        "learned_prediction": learned,
        "direct_probabilities": direct,
        "direct_rhythm_probabilities": direct_rhythm_probabilities(rows),
    }
    prediction = v5._blend(
        clip,
        weight=float(metadata["direct_weight"]),
        direct_source=str(metadata["direct_source"]),
        selector=selector,
    )
    prediction_document = schema12_document_v3(
        sample.name, rows, prediction, score, decode_config
    )
    agent_document = _agent_document(
        prediction_document, metadata=metadata
    )
    transcription_path = (
        output / "transcriptions" / f"{sample.name}.json"
    )
    alignment_path = output / "alignments" / f"{sample.name}.json"
    prediction_path = output / "predictions" / f"{sample.name}.json"
    base._atomic_json(
        transcription_path,
        {
            "schema_version": f"{SCHEMA_VERSION}-transcription",
            "sample": sample.name,
            "audio_sha256": _sha256(audio_path),
            "normalization": normalization,
            "notes": [note.to_dict() for note in notes],
            "training_performed": False,
            "reused_existing_transcription": reuse_transcription,
        },
    )
    base._atomic_json(
        alignment_path,
        _alignment_document(
            sample=sample.name,
            notes=notes,
            candidates=admitted,
            path=path,
            score_event_count=len(score),
        ),
    )
    base._atomic_json(prediction_path, prediction_document)
    return {
        "sample": sample.name,
        "document": agent_document,
        "runtime_seconds": time.perf_counter() - started,
        "transcribed_notes": len(notes),
        "candidates": len(admitted),
        "mapped_events": len(events),
        "label_count": len(agent_document["labels"]),
        "counts_by_type": dict(
            Counter(label["type"] for label in agent_document["labels"])
        ),
        "transcription": str(transcription_path),
        "transcription_sha256": _sha256(transcription_path),
        "alignment": str(alignment_path),
        "alignment_sha256": _sha256(alignment_path),
        "prediction": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    align = root / "align-model"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=Path, default=root / "DataCreate" / "samples"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup-manifest", type=Path)
    parser.add_argument("--expected-backup-sha256")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--min-id", type=int)
    parser.add_argument("--max-id", type=int)
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help=(
            "Write predictions under --output only. Do not read labels.json "
            "or rewrite sample-directory agent labels."
        ),
    )
    parser.add_argument(
        "--mel-checkpoint",
        type=Path,
        default=align
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
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--reuse-transcription", action="store_true")
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-audit-v2"
        / "end-to-end-v2"
        / "weak-note-continuation-optimized"
        / "joint_decoder.pt",
    )
    parser.add_argument(
        "--error-heads-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "error-heads-v2"
        / "predicted"
        / "best.pt",
    )
    parser.add_argument(
        "--active-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "training-v1-optimized"
        / "last_checkpoint.pt",
    )
    parser.add_argument(
        "--v5-selector",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "error-heads-v5"
        / "selector.json",
    )
    parser.add_argument(
        "--v5-decode-config",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "error-heads-v5"
        / "decode_config.json",
    )
    args = parser.parse_args(argv)

    output = args.output.resolve()
    marker = output / "label_manifest.json"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite {marker}")
    all_samples = _sample_dirs(args.samples.resolve())
    if args.sample_id:
        requested = set(map(str, args.sample_id))
        samples = [
            sample for sample in all_samples if sample.name in requested
        ]
        if {sample.name for sample in samples} != requested:
            raise ValueError("One or more requested sample IDs do not exist")
    else:
        samples = all_samples
        if not samples:
            raise ValueError("No DataCreate samples found")
    if args.min_id is not None or args.max_id is not None:
        minimum = args.min_id if args.min_id is not None else 0
        maximum = args.max_id if args.max_id is not None else 10**9
        samples = [
            sample
            for sample in samples
            if sample.name.isdigit()
            and minimum <= int(sample.name) <= maximum
        ]
        if not samples:
            raise ValueError("No samples in the requested ID range")
    backup = None
    if args.freeze_only:
        if args.backup_manifest is not None:
            raise ValueError("--freeze-only cannot be combined with --backup-manifest")
    elif args.backup_manifest is not None:
        if args.expected_backup_sha256 is None:
            raise ValueError(
                "--expected-backup-sha256 is required with --backup-manifest"
            )
        backup = _verify_backup(
            samples,
            args.backup_manifest.resolve(),
            args.expected_backup_sha256,
        )
    elif any((sample / "labels_agent.json").exists() for sample in samples):
        raise ValueError(
            "Existing agent labels require a verified --backup-manifest"
        )
    human_hashes: dict[str, str] = {}
    if not args.freeze_only:
        human_hashes = {
            sample.name: _sha256(sample / "labels.json")
            for sample in samples
        }
    if _sha256(args.mel_checkpoint.resolve()) != args.expected_mel_sha256:
        raise ValueError("Frozen mel-transcriber checkpoint mismatch")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be in [0, 1]")

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    mel_model, mel_frontend, checkpoint_decode, mel_payload = (
        load_mel_checkpoint(args.mel_checkpoint.resolve(), device)
    )
    mel_decode = replace(
        checkpoint_decode, min_confidence=float(args.min_confidence)
    )
    if mel_payload["data"].get("locked_test_materialized"):
        raise ValueError("Refusing a mel checkpoint trained on locked test data")
    for parameter in mel_model.parameters():
        parameter.requires_grad_(False)
    mel_model.eval()

    stack = base._load_completed_stack(args)
    selector = base._json(args.v5_selector.resolve())
    decode_payload = base._json(args.v5_decode_config.resolve())
    policy = decode_payload[agent_base.POLICY_NAME]
    decode_config = _config_from_json(policy["schema_config"])
    metadata = {
        "direct_source": policy["direct_source"],
        "direct_weight": policy["direct_weight"],
        "policy": agent_base.POLICY_NAME,
        "mel_checkpoint": str(args.mel_checkpoint.resolve()),
        "mel_checkpoint_sha256": args.expected_mel_sha256,
        "mel_checkpoint_schema": mel_payload.get("schema_version"),
        "mel_frontend": mel_frontend.to_dict(),
        "mel_decode_config": mel_decode.to_dict(),
        "training_performed": False,
        "score_used_only_downstream": True,
        "joint_checkpoint": str(args.joint_checkpoint.resolve()),
        "joint_checkpoint_sha256": stack["selection"][
            "joint_aligner_decoder"
        ]["checkpoint_sha256"],
        "error_heads_checkpoint": str(
            args.error_heads_checkpoint.resolve()
        ),
        "error_heads_checkpoint_sha256": stack["selection"][
            "error_heads_v2"
        ]["checkpoint_sha256"],
        "selector": str(args.v5_selector.resolve()),
        "selector_sha256": base._sha256(args.v5_selector.resolve()),
        "decode_config": str(args.v5_decode_config.resolve()),
        "decode_config_sha256": base._sha256(
            args.v5_decode_config.resolve()
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    staged_root = output / "staged-labels"
    sys.path.insert(0, str(root / "DataCreate" / "src"))
    from datacreate.validation import validate_labels_file

    rows = []
    failures = []
    totals: Counter[str] = Counter()
    for position, sample in enumerate(samples, 1):
        try:
            result = _label_one(
                sample=sample,
                output=output,
                stack=stack,
                selector=selector,
                decode_config=decode_config,
                mel_model=mel_model,
                mel_frontend=mel_frontend,
                mel_decode=mel_decode,
                device=device,
                metadata=metadata,
                batch_size=args.batch_size,
                reuse_transcription=args.reuse_transcription,
            )
            staged = staged_root / f"{sample.name}.json"
            base._atomic_json(staged, result.pop("document"))
            errors = validate_labels_file(staged)
            if errors:
                raise ValueError("; ".join(errors))
            result["staged_labels"] = str(staged)
            result["staged_labels_sha256"] = _sha256(staged)
            rows.append(result)
            totals.update(result["counts_by_type"])
            print(
                f"{position:02d}/{len(samples)} {sample.name}: "
                f"{result['transcribed_notes']} notes, "
                f"{result['label_count']} labels",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - finish diagnostic manifest
            failures.append(
                {
                    "sample": sample.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"{position:02d}/{len(samples)} {sample.name}: ERROR {exc}",
                flush=True,
            )

    if failures:
        base._atomic_json(
            output / "FAILED.json",
            {
                "schema_version": f"{SCHEMA_VERSION}-failed",
                "created_utc": _utc(),
                "committed": False,
                "failures": failures,
                "succeeded": len(rows),
                "human_labels_modified": False,
                "agent_labels_modified": False,
            },
        )
        raise SystemExit(1)

    if args.freeze_only:
        human_unchanged = True
    else:
        # Recheck rollback sources immediately before the all-sample commit.
        if backup is not None:
            _verify_backup(
                samples,
                args.backup_manifest.resolve(),
                args.expected_backup_sha256,
            )
        elif any((sample / "labels_agent.json").exists() for sample in samples):
            raise ValueError("New subset acquired agent labels before commit")
        row_by_sample = {row["sample"]: row for row in rows}
        for sample in samples:
            result_row = row_by_sample[sample.name]
            staged = Path(result_row["staged_labels"])
            document = json.loads(staged.read_text(encoding="utf-8"))
            base._atomic_json(sample / "labels_agent.json", document)
            base._atomic_json(
                sample / "transcription_mel_v1.json",
                json.loads(
                    Path(result_row["transcription"]).read_text(
                        encoding="utf-8"
                    )
                ),
            )
            base._atomic_json(
                sample / "note_alignment_mel_v1.json",
                json.loads(
                    Path(result_row["alignment"]).read_text(
                        encoding="utf-8"
                    )
                ),
            )

        human_unchanged = True
        for sample in samples:
            human = sample / "labels.json"
            if (
                not human.is_file()
                or _sha256(human) != human_hashes[sample.name]
            ):
                human_unchanged = False
                break
        if not human_unchanged:
            raise RuntimeError("Human labels changed during mel relabeling")

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "annotator_id": ANNOTATOR_ID,
        "method": METHOD,
        "sample_count": len(samples),
        "succeeded": len(rows),
        "failed": 0,
        "committed": not args.freeze_only,
        "freeze_only": bool(args.freeze_only),
        "output_filename": (
            "staged-labels" if args.freeze_only else "labels_agent.json"
        ),
        "kept_labels_by_type": dict(sorted(totals.items())),
        "rollback_manifest": (
            str(args.backup_manifest.resolve())
            if args.backup_manifest is not None
            else None
        ),
        "rollback_manifest_sha256": args.expected_backup_sha256,
        "human_labels_modified": False,
        "training_performed": False,
        "transcription_reused": args.reuse_transcription,
        "device": str(device),
        "gold_access": {
            "labels_opened": False if args.freeze_only else True,
            "phase": "freeze" if args.freeze_only else "commit",
            "opened_sample_inputs": (
                ["performance_audio.wav", "verified_score.musicxml"]
                + (
                    ["transcription_mel_v1.json"]
                    if args.reuse_transcription
                    else []
                )
            ),
        },
        "model_selection": {
            **stack["selection"],
            "transcriber": {
                "implementation": "ALIGN mel transcriber v1",
                "checkpoint": str(args.mel_checkpoint.resolve()),
                "checkpoint_sha256": args.expected_mel_sha256,
                "decode_config": mel_decode.to_dict(),
                "training_performed": False,
            },
        },
        "metadata": metadata,
        "samples": rows,
    }
    base._atomic_json(marker, report)
    if args.freeze_only:
        base._atomic_json(
            output / "freeze_manifest.json",
            {
                "schema_version": f"{SCHEMA_VERSION}-freeze",
                "created_utc": report["created_utc"],
                "annotator_id": ANNOTATOR_ID,
                "method": METHOD,
                "output": str(output),
                "samples_root": str(args.samples.resolve()),
                "sample_count": len(samples),
                "ids": [sample.name for sample in samples],
                "gold_access": report["gold_access"],
                "label_manifest": str(marker),
                "prediction_dir": str(output / "staged-labels"),
            },
        )
    else:
        report_copy = args.samples.resolve() / "agent_labeling_report.json"
        base._atomic_json(
            report_copy,
            {
                "sample_count": len(samples),
                "succeeded": len(rows),
                "failed": 0,
                "output_filename": "labels_agent.json",
                "annotator_id": ANNOTATOR_ID,
                "method": METHOD,
                "kept_labels_by_type": report["kept_labels_by_type"],
                "rollback_manifest": report["rollback_manifest"],
                "human_labels_modified": False,
                "training_performed": False,
            },
        )
    print(
        json.dumps(
            {
                "sample_count": len(samples),
                "labels": sum(row["label_count"] for row in rows),
                "kept_labels_by_type": report["kept_labels_by_type"],
                "training_performed": False,
                "freeze_only": bool(args.freeze_only),
                "rollback_manifest": report["rollback_manifest"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(marker, flush=True)


if __name__ == "__main__":
    main()
