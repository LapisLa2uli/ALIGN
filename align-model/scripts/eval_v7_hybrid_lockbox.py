"""One-shot ORN v7 evaluation of the frozen mel/Basic Pitch candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import calibrate_v2_acoustic_crf as acoustic
import calibrate_v2_template_rescue as rescue
from alignmodel.joint.identity_crf_v1 import IdentityCandidate
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.basic_pitch import (
    FROZEN_DECODE_CONFIG,
    decode_frozen_basic_pitch,
    extract_basic_pitch_features,
    sanitize_basic_pitch_notes,
)
from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from eval_v2_template_rescue_lockbox import (
    PITCH_METADATA,
    _decrypt_targets,
    _score,
)


SCHEMA_VERSION = "align-orn-v7-hybrid-lockbox-v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _overlap(left: Any, right: Any) -> float:
    return min(float(left.end), float(right.end)) - max(
        float(left.start), float(right.start)
    )


def _gap_fill(
    mel_notes: Sequence[Any],
    basic_pitch_notes: Sequence[Any],
    maximum_overlap_sec: float | None,
) -> list[Any]:
    if maximum_overlap_sec is None:
        return list(mel_notes)
    added = [
        note
        for note in basic_pitch_notes
        if not any(
            _overlap(note, other) > maximum_overlap_sec for other in mel_notes
        )
    ]
    return sorted(
        [*mel_notes, *added],
        key=lambda note: (float(note.start), int(note.pitch)),
    )


def _candidates(
    notes: Sequence[Any], features: Any
) -> tuple[IdentityCandidate, ...]:
    output: list[IdentityCandidate] = []
    for note in notes:
        confidences = getattr(note, "candidate_confidences", ())
        pitches = getattr(note, "pitch_candidates", ())
        if confidences and pitches:
            output.append(
                IdentityCandidate(
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                    confidence=float(note.confidence),
                    alternatives=tuple(map(int, pitches)),
                    alternative_confidences=tuple(map(float, confidences)),
                )
            )
        else:
            output.extend(acoustic._identity_candidates((note,), features))
    return tuple(output)


def _validate_artifact(raw: Mapping[str, Any]) -> Path:
    path = Path(raw["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(raw["bytes"])
        or sha256_file(path) != raw["sha256"]
    ):
        raise ValueError(f"Public lockbox input changed: {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--expected-targets-sha256", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--public-smoke", action="store_true")
    args = parser.parse_args(argv)

    expected = (
        (args.manifest, args.expected_manifest_sha256),
        (args.targets, args.expected_targets_sha256),
        (args.freeze, args.expected_freeze_sha256),
        (args.candidate, args.expected_candidate_sha256),
        (args.protocol, args.expected_protocol_sha256),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact mismatch: {path}")
    manifest = _load(args.manifest)
    freeze = _load(args.freeze)
    candidate = _load(args.candidate)
    protocol = _load(args.protocol)
    if int(manifest["rows"]) != 137:
        raise ValueError("Unexpected ORN v7 population")
    if freeze.get("lockbox_opened"):
        raise ValueError("ORN v7 freeze already opened")
    if (
        candidate["holdout"]["freeze_sha256"]
        != args.expected_freeze_sha256
        or candidate["holdout"]["targets_read"]
        or candidate["holdout"]["key_read"]
    ):
        raise ValueError("Candidate was not frozen against sealed v7")
    if protocol["evaluator_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Evaluator changed after protocol freeze")
    if (
        protocol["candidate_sha256"] != args.expected_candidate_sha256
        or protocol["freeze_sha256"] != args.expected_freeze_sha256
        or protocol["manifest_sha256"] != args.expected_manifest_sha256
        or protocol["targets_sha256"] != args.expected_targets_sha256
    ):
        raise ValueError("Protocol/frozen inputs mismatch")

    method = candidate["method"]
    mel_checkpoint = (
        Path(__file__).resolve().parents[1] / method["checkpoint"]
    )
    identity_checkpoint = (
        Path(__file__).resolve().parents[1]
        / method["identity_crf_checkpoint"]
    )
    if (
        sha256_file(mel_checkpoint) != method["checkpoint_sha256"]
        or sha256_file(identity_checkpoint)
        != method["identity_crf_checkpoint_sha256"]
    ):
        raise ValueError("Candidate checkpoint mismatch")
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    mel_model, frontend, decode, _payload = load_mel_checkpoint(
        mel_checkpoint, device
    )
    overrides = dict(method.get("decode_overrides") or {})
    decode = replace(
        decode,
        min_confidence=float(method["minimum_confidence"]),
        **overrides,
    )
    identity_model = acoustic._load_model(identity_checkpoint)[0]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    maximum_overlap = method.get("maximum_overlap_sec", None)
    if maximum_overlap is not None:
        maximum_overlap = float(maximum_overlap)

    def predict(row: Mapping[str, Any]) -> dict[str, Any]:
        row_id = str(row["row_id"])
        audio = _validate_artifact(row["audio"])
        score_path = _validate_artifact(row["score"])
        audio_value = load_audio_mono(audio, frontend.sample_rate)
        mel, _normalization = extract_log_mel(
            audio_value, frontend, device=device
        )
        probabilities = infer_mel_probabilities(
            mel_model,
            np.asarray(mel, np.float32),
            device,
            window_frames=2048,
            overlap_frames=512,
            batch_size=args.batch_size,
        )
        mel_notes = decode_mel_notes(
            probabilities,
            midi_min=mel_model.config.midi_min,
            hop_sec=frontend.hop_sec,
            config=decode,
        )
        features = None
        if maximum_overlap is not None:
            features = extract_basic_pitch_features(
                audio,
                source_metadata=PITCH_METADATA,
                cache_path=args.cache_dir / f"{row_id}.npz",
            )
            basic_pitch_notes = sanitize_basic_pitch_notes(
                decode_frozen_basic_pitch(features),
                features,
                FROZEN_DECODE_CONFIG,
            )
            notes = _gap_fill(
                mel_notes, basic_pitch_notes, maximum_overlap
            )
        else:
            notes = list(mel_notes)
        index = ScoreEventIndex.from_musicxml(score_path)
        metric_sample, diagnostics = rescue._map(
            identity_model,
            _candidates(notes, features),
            index,
            score_path,
        )
        return {
            "ordinal": int(row["ordinal"]),
            "row_id": row_id,
            "leakage_group": str(row["leakage_group"]),
            "score_event_count": len(index.events),
            "transcribed_note_count": len(notes),
            "predicted": [
                asdict(event) for event in metric_sample.predicted
            ],
            "predicted_deletions": sorted(
                metric_sample.predicted_deletions
            ),
            "diagnostics": {
                "copies": diagnostics.get("copies"),
                "source_span": diagnostics.get("source_span"),
            },
        }

    if args.public_smoke:
        row = predict(manifest["inputs"][0])
        print(
            json.dumps(
                {
                    "row_id": row["row_id"],
                    "transcribed_note_count": row[
                        "transcribed_note_count"
                    ],
                    "predicted_count": len(row["predicted"]),
                    "target_or_key_read": False,
                },
                indent=2,
            )
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "FROZEN_PREDICTIONS.json"
    sentinel_path = args.output_dir / "LOCKBOX_OPENED.json"
    repository_sentinel = args.manifest.parent / "LOCKBOX_OPENED.json"
    report_path = args.output_dir / "LOCKBOX_REPORT.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    if sentinel_path.exists() or repository_sentinel.exists():
        if not sentinel_path.exists() or not repository_sentinel.exists():
            raise ValueError("Inconsistent opening sentinels")
        sentinel = _load(sentinel_path)
        if (
            not prediction_path.exists()
            or sha256_file(prediction_path)
            != sentinel["frozen_predictions_sha256"]
        ):
            raise ValueError("Frozen predictions changed")
        predictions = _load(prediction_path)["rows"]
    else:
        predictions = []
        for position, row in enumerate(manifest["inputs"], 1):
            predictions.append(predict(row))
            if position == 1 or position % 10 == 0 or position == 137:
                print(f"freeze={position}/137", flush=True)
        acoustic._atomic_json(
            prediction_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-predictions",
                "created_utc": _utc(),
                "target_or_key_read": False,
                "rows": predictions,
            },
        )
        opening = {
            "schema_version": f"{SCHEMA_VERSION}-opened",
            "opened_utc": _utc(),
            "opening_number": 1,
            "protocol_sha256": args.expected_protocol_sha256,
            "candidate_sha256": args.expected_candidate_sha256,
            "freeze_sha256": args.expected_freeze_sha256,
            "frozen_predictions_sha256": sha256_file(prediction_path),
            "predictions_frozen_before_target_or_key_read": True,
            "rows": len(predictions),
        }
        acoustic._atomic_json(sentinel_path, opening)
        acoustic._atomic_json(repository_sentinel, opening)

    targets = _decrypt_targets(
        key_path=args.key,
        encrypted_path=args.targets,
        manifest=manifest,
    )
    result = _score(
        predictions=predictions,
        targets=targets,
        manifest=manifest,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    threshold = float(protocol["success_threshold_f1"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "protocol_sha256": args.expected_protocol_sha256,
        "candidate_sha256": args.expected_candidate_sha256,
        "freeze_sha256": args.expected_freeze_sha256,
        "frozen_predictions_sha256": sha256_file(prediction_path),
        "population": {"split": "orn_v7_lockbox", "rows": len(targets)},
        "official_note_wise": result,
        "success_threshold_f1": threshold,
        "success": float(result["f1"]) >= threshold,
        "post_lockbox_tuning_allowed": False,
    }
    acoustic._atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "bootstrap_95": result.get("bootstrap_95"),
                "success": report["success"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(report_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
