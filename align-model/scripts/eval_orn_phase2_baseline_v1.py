"""Evaluate the frozen Track-B/mapper-v6 baseline on ORN open validation.

Inference and scoring are separate commands.  ``infer`` reads only the frozen
release manifest's open-validation audio paths and the frozen acoustic
checkpoint.  ``score`` first verifies the prediction freeze, then opens only
the development target archive; the encrypted lockbox target is never read.
"""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from alignmodel.joint.baseline import current_note_aligner_baseline
from alignmodel.joint.grammar_mapper_v2 import (
    GrammarCosts,
    decode_grammar_mapper,
)
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    evaluate_joint_events,
    pair_exact_pitch_onset,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


SCHEMA_VERSION = "align-orn-phase2-baseline-v1"
EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "3d8f93732a810c3f8470a88316debb9f92b4680b2333c2187866e6090a730a4e"
)
EXPECTED_MAPPER_CANDIDATE_SHA256 = (
    "39554480001221fb25f5fd4b6eb044d3dc89f1b21364e55d5a3a7af549371d6e"
)
DEFAULT_RELEASE = Path(
    "runs/joint-outputraw-full-v1/orn-generalization-v1/release_manifest.json"
)
DEFAULT_CHECKPOINT = Path(
    "runs/joint-outputraw-full-v1/mel-transcriber-v1/"
    "full-training-all4544-v2/candidate-epoch-018.pt"
)
DEFAULT_MAPPER_CANDIDATE = Path(
    "runs/joint-outputraw-full-v1/mel-mapper-v6/VALIDATION_CANDIDATE.json"
)
DEFAULT_OUTPUT = Path(
    "runs/joint-outputraw-full-v1/orn-generalization-v1/development/"
    "baseline-trackb18-mapperv6-v1"
)
DEFAULT_RESOURCE_STATUS = Path("runs/TRAINING_RESOURCE_STATUS.json")
DEFAULT_SEED = 20260918


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False)
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_release(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release manifest SHA-256 mismatch")
    release = _load_json(path)
    if (
        release.get("schema_version")
        != "align-orn-generalization-release-v1"
        or release.get("phase") != "frozen_before_model_iteration"
    ):
        raise ValueError("Unsupported or unfrozen ORN release")
    if release.get("lockbox", {}).get("opened"):
        raise ValueError("Replacement lockbox is not sealed")
    if len(release["splits"]["development"]["open_validation"]) != 66:
        raise ValueError("Frozen open-validation population is not 66 rows")
    return release


def _assert_lockbox_sealed(release: Mapping[str, Any]) -> None:
    release_root = Path(
        str(release["artifacts"]["development_targets"]["path"])
    ).parent
    if (release_root / "lockbox" / "LOCKBOX_OPENED.json").exists():
        raise ValueError("Replacement lockbox opening sentinel exists")


def _cache_key(
    row: Mapping[str, Any],
    checkpoint_sha256: str,
    frontend: Mapping[str, Any],
    decode: Mapping[str, Any],
    *,
    window_frames: int,
    overlap_frames: int,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "schema": f"{SCHEMA_VERSION}-track-b-cache",
                "audio_sha256": row["source_hashes"][
                    "performance_audio.wav"
                ],
                "checkpoint_sha256": checkpoint_sha256,
                "frontend": frontend,
                "decode": decode,
                "window_frames": window_frames,
                "overlap_frames": overlap_frames,
            }
        )
    ).hexdigest()


def run_infer(args: argparse.Namespace) -> None:
    release = _verify_release(args.release_manifest)
    _assert_lockbox_sealed(release)
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen Track B epoch-18 checkpoint mismatch")
    if (
        sha256_file(args.mapper_candidate)
        != EXPECTED_MAPPER_CANDIDATE_SHA256
    ):
        raise ValueError("Frozen mapper-v6 candidate mismatch")
    mapper_candidate = _load_json(args.mapper_candidate)
    grammar_path = (
        args.repo / "src" / "alignmodel" / "joint" / "grammar_mapper_v2.py"
    )
    if (
        sha256_file(grammar_path)
        != mapper_candidate["grammar_mapper_sha256"]
    ):
        raise ValueError("Mapper-v6 grammar implementation hash mismatch")
    output = args.output_dir
    prediction_path = output / "predictions.jsonl"
    freeze_path = output / "prediction_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Baseline inference already frozen: {freeze_path}")
    rows = release["splits"]["development"]["open_validation"]
    device = torch.device(args.device)
    lease = resource_lease(
        args.resource_status,
        "gpu" if device.type == "cuda" else "cpu_validation",
        track="orn-phase2-baseline-trackb18-infer",
        command=[sys.executable, *sys.argv],
        metadata={
            "rows": len(rows),
            "split": "open_validation",
            "locked_test": False,
        },
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        model, frontend, decode, _payload = load_mel_checkpoint(
            args.checkpoint, device
        )
        decode = replace(decode, min_confidence=0.8)
        cache_root = output / "track-b-cache"
        predictions = []
        cache_hits = 0
        for position, row in enumerate(rows, 1):
            key = _cache_key(
                row,
                EXPECTED_CHECKPOINT_SHA256,
                frontend.to_dict(),
                decode.to_dict(),
                window_frames=args.window_frames,
                overlap_frames=args.overlap_frames,
            )
            cache_path = cache_root / key[:2] / f"{key}.json"
            if cache_path.is_file():
                cached = _load_json(cache_path)
                if cached.get("cache_key") != key:
                    raise ValueError(f"Cache identity mismatch: {cache_path}")
                notes = cached["notes"]
                cache_hits += 1
            else:
                audio_path = Path(row["sample_dir"]) / "performance_audio.wav"
                if (
                    sha256_file(audio_path)
                    != row["source_hashes"]["performance_audio.wav"]
                ):
                    raise ValueError(f"Audio hash mismatch: {row['sample']}")
                audio = load_audio_mono(audio_path, frontend.sample_rate)
                mel, normalization = extract_log_mel(
                    audio, frontend, device=device
                )
                probabilities = infer_mel_probabilities(
                    model,
                    mel,
                    device,
                    window_frames=args.window_frames,
                    overlap_frames=args.overlap_frames,
                    batch_size=args.batch_size,
                )
                notes = [
                    value.to_dict()
                    for value in decode_mel_notes(
                        probabilities,
                        midi_min=model.config.midi_min,
                        hop_sec=frontend.hop_sec,
                        config=decode,
                    )
                ]
                _atomic_json(
                    cache_path,
                    {
                        "schema_version": f"{SCHEMA_VERSION}-track-b-cache",
                        "cache_key": key,
                        "audio_sha256": row["source_hashes"][
                            "performance_audio.wav"
                        ],
                        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                        "frontend": frontend.to_dict(),
                        "decode": decode.to_dict(),
                        "normalization": normalization,
                        "notes": notes,
                    },
                )
            predictions.append(
                {
                    "sample": row["sample"],
                    "leakage_group": row["leakage_group"],
                    "audio_sha256": row["source_hashes"][
                        "performance_audio.wav"
                    ],
                    "cache_key": key,
                    "notes": notes,
                }
            )
            if position == 1 or position % 10 == 0 or position == len(rows):
                print(
                    f"infer={position}/{len(rows)} cache_hits={cache_hits}",
                    flush=True,
                )
        _atomic_jsonl(prediction_path, predictions)
        freeze = {
            "schema_version": f"{SCHEMA_VERSION}-prediction-freeze",
            "created_utc": _utc(),
            "command": [sys.executable, *sys.argv],
            "split": "open_validation",
            "rows": len(predictions),
            "release_manifest": str(args.release_manifest),
            "release_manifest_sha256": sha256_file(args.release_manifest),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "mapper_candidate": str(args.mapper_candidate),
            "mapper_candidate_sha256": sha256_file(args.mapper_candidate),
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": sha256_file(prediction_path),
            "frontend": frontend.to_dict(),
            "decode": decode.to_dict(),
            "score_input_to_acoustic_model": False,
            "target_archive_read": False,
            "lockbox_inputs_read": False,
            "lockbox_targets_read": False,
            "original_mapper_v6_test_read": False,
            "production_weights_mutated": False,
        }
        _atomic_json(freeze_path, freeze)
    finally:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def _load_targets(
    release: Mapping[str, Any],
    split: str,
) -> dict[str, dict[str, Any]]:
    artifact = release["artifacts"]["development_targets"]
    path = Path(str(artifact["path"]))
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("Frozen development target archive mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {
            str(row["sample"]): row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] == split
        }


def _candidates(
    notes: Sequence[Mapping[str, Any]]
) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=int(row["pitch"]),
            start=float(row["start"]),
            end=float(row["end"]),
            confidence=float(row.get("confidence", 1.0)),
        )
        for row in notes
    )


def _oracle_candidates(
    target: Sequence[JointEvent],
) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=event.pitch,
            start=event.start,
            end=event.end,
            confidence=1.0,
        )
        for event in target
    )


def _candidate_events(
    candidates: Sequence[JointCandidate],
) -> tuple[JointEvent, ...]:
    return tuple(
        JointEvent(
            pitch=event.pitch,
            start=event.start,
            end=event.end,
            score_span=None,
            relationship="extra",
            rendered_index=index,
            confidence=event.confidence,
        )
        for index, event in enumerate(candidates)
    )


def _transcription_prediction(event: JointEvent, index: int) -> JointEvent:
    if event.score_span is None:
        return replace(
            event,
            relationship="extra",
            rendered_index=index,
            copy_pass=0,
        )
    return replace(
        event,
        relationship="match",
        origin_relationship="match",
        copy_pass=0,
    )


def _transcription_target(event: JointEvent) -> JointEvent:
    if event.score_span is None:
        return event
    return replace(
        event,
        relationship="match",
        origin_relationship="match",
        copy_pass=0,
    )


def _counts(sample: JointMetricSample) -> tuple[float, int, int]:
    report = evaluate_joint_events(
        sample.predicted,
        sample.target,
        predicted_deletions=sample.predicted_deletions,
        target_deletions=sample.target_deletions,
        score_event_count=sample.score_event_count,
        tolerances_sec=(),
    )["official_note_wise"]
    return (
        float(report["credit"]),
        int(report["predicted"]),
        int(report["gold"]),
    )


def _fractional_prf(
    credit: float, predicted: int, gold: int
) -> dict[str, Any]:
    precision = credit / predicted if predicted else (1.0 if not gold else 0.0)
    recall = credit / gold if gold else (1.0 if not predicted else 0.0)
    return {
        "credit": credit,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def _aggregate(samples: Sequence[JointMetricSample]) -> dict[str, Any]:
    return evaluate_joint_dataset(samples, tolerances_sec=())["aggregate"][
        "official_note_wise"
    ]


def _bootstrap(
    samples: Sequence[JointMetricSample],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    counts = [_counts(sample) for sample in samples]
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = generator.integers(0, len(counts), len(counts))
        values.append(
            _fractional_prf(
                sum(counts[index][0] for index in chosen),
                sum(counts[index][1] for index in chosen),
                sum(counts[index][2] for index in chosen),
            )["f1"]
        )
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _type_samples(
    samples: Sequence[JointMetricSample], kind: str
) -> list[JointMetricSample]:
    output = []
    for sample in samples:
        if kind == "missed_note":
            output.append(
                JointMetricSample(
                    predicted=(),
                    target=(),
                    source=sample.source,
                    predicted_deletions=sample.predicted_deletions,
                    target_deletions=sample.target_deletions,
                    score_event_count=sample.score_event_count,
                )
            )
            continue
        output.append(
            JointMetricSample(
                predicted=tuple(
                    event
                    for event in sample.predicted
                    if ("copy" if event.is_copy else event.relationship)
                    == kind
                ),
                target=tuple(
                    event
                    for event in sample.target
                    if ("copy" if event.is_copy else event.relationship)
                    == kind
                ),
                source=sample.source,
                score_event_count=sample.score_event_count,
            )
        )
    return output


def _full_report(
    samples: Sequence[JointMetricSample],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    grouped: defaultdict[str, list[JointMetricSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source].append(sample)
    source_metrics = {
        source: _aggregate(values) for source, values in grouped.items()
    }
    source_f1 = [float(value["f1"]) for value in source_metrics.values()]
    return {
        **_aggregate(samples),
        "bootstrap_95": _bootstrap(
            samples, seed=seed, replicates=replicates
        ),
        "per_type": {
            kind: _aggregate(_type_samples(samples, kind))
            for kind in (
                "match",
                "copy",
                "substitute",
                "extra",
                "missed_note",
            )
        },
        "source_macro": {
            "sources": len(source_metrics),
            "macro_f1": float(np.mean(source_f1)) if source_f1 else None,
            "median_f1": float(np.median(source_f1)) if source_f1 else None,
            "minimum_f1": min(source_f1) if source_f1 else None,
            "maximum_f1": max(source_f1) if source_f1 else None,
        },
        "per_source": source_metrics,
    }


def _lcs(left: Sequence[int], right: Sequence[int]) -> int:
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            row[column] = (
                previous + 1
                if item == other
                else max(row[column], row[column - 1])
            )
            previous = saved
    return row[-1]


def _stratified_metric(
    samples: Sequence[JointMetricSample],
    strata: Mapping[str, str],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[JointMetricSample]] = defaultdict(list)
    for sample in samples:
        grouped[strata[sample.source]].append(sample)
    return {
        name: {"rows": len(values), **_aggregate(values)}
        for name, values in sorted(grouped.items())
    }


def run_score(args: argparse.Namespace) -> None:
    release = _verify_release(args.release_manifest)
    _assert_lockbox_sealed(release)
    output = args.output_dir
    freeze_path = output / "prediction_freeze.json"
    freeze = _load_json(freeze_path)
    if freeze["release_manifest_sha256"] != EXPECTED_RELEASE_SHA256:
        raise ValueError("Prediction freeze uses another release")
    prediction_path = Path(str(freeze["predictions"]))
    if sha256_file(prediction_path) != freeze["predictions_sha256"]:
        raise ValueError("Frozen Track B predictions changed")
    predictions = {
        str(row["sample"]): row for row in _read_jsonl(prediction_path)
    }
    manifest_rows = {
        str(row["sample"]): row
        for row in release["splits"]["development"]["open_validation"]
    }
    if set(predictions) != set(manifest_rows):
        raise ValueError("Prediction and open-validation populations differ")
    targets = _load_targets(release, "open_validation")
    if set(targets) != set(manifest_rows):
        raise ValueError("Target and open-validation populations differ")
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-phase2-baseline-trackb18-score",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(targets), "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        combined_samples = []
        oracle_samples = []
        transcription_samples = []
        sequence_counts = [0.0, 0, 0]
        sequence_per_source = []
        duration_groups: dict[str, list[int]] = {
            "lt_80ms": [0, 0],
            "80_to_120ms": [0, 0],
            "120_to_180ms": [0, 0],
            "ge_180ms": [0, 0],
        }
        row_strata: dict[str, dict[str, str]] = {
            "provenance": {},
            "ornament_realization": {},
            "polyphony": {},
            "unmapped_rendered": {},
            "repeat": {},
        }
        per_row = []
        predicted_same_pitch = target_same_pitch = 0
        predicted_total = target_total = 0
        costs = GrammarCosts()
        for position, sample in enumerate(sorted(manifest_rows), 1):
            row = manifest_rows[sample]
            target_row = targets[sample]
            lineage = target_row["lineage"]
            if (
                hashlib.sha256(_canonical_bytes(lineage)).hexdigest()
                != row["target_lineage_sha256"]
            ):
                raise ValueError(f"Target lineage hash mismatch: {sample}")
            score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
            if (
                sha256_file(score_path)
                != row["source_hashes"]["verified_score.musicxml"]
            ):
                raise ValueError(f"Score hash mismatch: {sample}")
            index = ScoreEventIndex.from_musicxml(score_path, lineage)
            target = index.rendered_events
            candidate = _candidates(predictions[sample]["notes"])
            combined, grammar = decode_grammar_mapper(
                candidate, index.events, costs=costs
            )
            oracle, oracle_grammar = decode_grammar_mapper(
                _oracle_candidates(target), index.events, costs=costs
            )
            baseline, _baseline_deletions = current_note_aligner_baseline(
                candidate, index.events
            )
            transcription_predicted = tuple(
                _transcription_prediction(event, event_index)
                for event_index, event in enumerate(baseline)
            )
            transcription_target = tuple(
                _transcription_target(event) for event in target
            )
            source = str(row["leakage_group"])
            combined_sample = JointMetricSample(
                predicted=combined,
                target=target,
                source=source,
                target_deletions=index.deleted_event_indices,
                score_event_count=len(index.events),
            )
            oracle_sample = JointMetricSample(
                predicted=oracle,
                target=target,
                source=source,
                target_deletions=index.deleted_event_indices,
                score_event_count=len(index.events),
            )
            transcription_sample = JointMetricSample(
                predicted=transcription_predicted,
                target=transcription_target,
                source=source,
                target_deletions=index.deleted_event_indices,
                score_event_count=len(index.events),
            )
            combined_samples.append(combined_sample)
            oracle_samples.append(oracle_sample)
            transcription_samples.append(transcription_sample)
            predicted_pitch = [event.pitch for event in candidate]
            target_pitch = [event.pitch for event in target]
            correct = _lcs(predicted_pitch, target_pitch)
            sequence_counts[0] += correct
            sequence_counts[1] += len(predicted_pitch)
            sequence_counts[2] += len(target_pitch)
            sequence_per_source.append(
                (float(correct), len(predicted_pitch), len(target_pitch))
            )
            pairs = pair_exact_pitch_onset(
                _candidate_events(candidate),
                target,
                tolerance_sec=0.050,
            )
            paired_target = {right for _left, right in pairs}
            for target_index, event in enumerate(target):
                duration = event.end - event.start
                name = (
                    "lt_80ms"
                    if duration < 0.080
                    else "80_to_120ms"
                    if duration < 0.120
                    else "120_to_180ms"
                    if duration < 0.180
                    else "ge_180ms"
                )
                duration_groups[name][1] += 1
                duration_groups[name][0] += int(
                    target_index in paired_target
                )
            target_stats = row["target_stats"]
            has_ornament = (
                int(target_stats["rendered_extra_events"]) > 0
                or int(target_stats["max_rendered_polyphony"]) > 1
            )
            row_strata["provenance"][source] = str(row["provenance"])
            row_strata["ornament_realization"][source] = (
                "ornament_realized" if has_ornament else "principal_only"
            )
            row_strata["polyphony"][source] = (
                "overlap_polyphonic"
                if int(target_stats["max_rendered_polyphony"]) > 1
                else "monophonic"
            )
            row_strata["unmapped_rendered"][source] = (
                "has_rendered_extra"
                if int(target_stats["rendered_extra_events"]) > 0
                else "fully_performed_mapped"
            )
            row_strata["repeat"][source] = (
                "has_copy"
                if any(event.is_copy for event in target)
                else "no_copy"
            )
            predicted_same_pitch += sum(
                left.pitch == right.pitch
                for left, right in zip(candidate, candidate[1:])
            )
            target_same_pitch += sum(
                left.pitch == right.pitch
                for left, right in zip(target, target[1:])
            )
            predicted_total += len(candidate)
            target_total += len(target)
            per_row.append(
                {
                    "sample": sample,
                    "source": source,
                    "provenance": row["provenance"],
                    "target_stats": target_stats,
                    "predicted_notes": len(candidate),
                    "target_notes": len(target),
                    "score_events": len(index.events),
                    "sequence_lcs": correct,
                    "grammar": grammar,
                    "oracle_grammar": oracle_grammar,
                    "transcriber_canonical": _fractional_prf(
                        *_counts(transcription_sample)
                    ),
                    "oracle_mapper": _fractional_prf(
                        *_counts(oracle_sample)
                    ),
                    "combined": _fractional_prf(
                        *_counts(combined_sample)
                    ),
                }
            )
            if position == 1 or position % 10 == 0 or position == len(targets):
                print(f"score={position}/{len(targets)}", flush=True)
        combined_report = _full_report(
            combined_samples,
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        )
        transcriber_report = _full_report(
            transcription_samples,
            seed=args.seed + 1,
            replicates=args.bootstrap_replicates,
        )
        oracle_report = _full_report(
            oracle_samples,
            seed=args.seed + 2,
            replicates=args.bootstrap_replicates,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "phase": "unchanged_frozen_baseline",
            "created_utc": _utc(),
            "command": [sys.executable, *sys.argv],
            "population": {
                "split": "open_validation",
                "rows": len(targets),
                "source_groups": len(
                    {row["leakage_group"] for row in manifest_rows.values()}
                ),
                "provenance_support": dict(
                    sorted(
                        Counter(
                            row["provenance"]
                            for row in manifest_rows.values()
                        ).items()
                    )
                ),
                "raw_support_note": (
                    "Open validation contains no raw rows because all admissible "
                    "raw source scores were related to exposed ORN-500 sources."
                ),
            },
            "metric": {
                "schema_version": "align-note-wise-score-event-metric-v1",
                "unit": "canonical score-note/event identity",
                "matching": (
                    "exclusive one-to-one; exact location+type=1.0; exact "
                    "location+wrong type=0.5; wrong location=0"
                ),
                "rendered_extras": "exclusive rendered_index identity",
                "timestamp_metrics": "diagnostic_only",
            },
            "models": {
                "transcriber": {
                    "name": "Track B epoch 18 confidence 0.8",
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": sha256_file(args.checkpoint),
                    "score_input": False,
                },
                "mapper": {
                    "name": "mapper-v6 generator grammar Viterbi",
                    "candidate": str(args.mapper_candidate),
                    "candidate_sha256": sha256_file(
                        args.mapper_candidate
                    ),
                    "costs": costs.__dict__,
                },
            },
            "score_agnostic_transcriber_diagnostic": {
                "pitch_sequence_lcs": _fractional_prf(*sequence_counts),
                "pitch_sequence_source_bootstrap_95": {
                    **_bootstrap_counts(
                        sequence_per_source,
                        seed=args.seed + 10,
                        replicates=args.bootstrap_replicates,
                    )
                },
                "count_ratio": predicted_total / max(target_total, 1),
                "short_note_recall_at_50ms_onset": {
                    name: {
                        "matched": value[0],
                        "support": value[1],
                        "recall": value[0] / max(value[1], 1),
                    }
                    for name, value in duration_groups.items()
                },
                "same_pitch_rearticulation": {
                    "predicted_adjacent_same_pitch": predicted_same_pitch,
                    "predicted_rate": predicted_same_pitch
                    / max(predicted_total, 1),
                    "target_adjacent_same_pitch": target_same_pitch,
                    "target_rate": target_same_pitch
                    / max(target_total, 1),
                },
                "timestamp_status": "diagnostic_only",
            },
            "canonical_transcriber_fixed_projection": transcriber_report,
            "oracle_note_mapper": oracle_report,
            "combined": {
                **combined_report,
                "stratified": {
                    name: _stratified_metric(
                        combined_samples, strata
                    )
                    for name, strata in row_strata.items()
                },
            },
            "per_row": per_row,
            "isolation": {
                "release_manifest_sha256": sha256_file(
                    args.release_manifest
                ),
                "prediction_freeze_sha256": sha256_file(freeze_path),
                "lockbox_opened": False,
                "lockbox_inputs_read": False,
                "lockbox_targets_read": False,
                "ORN500_predictions_or_outcomes_read": False,
                "original_mapper_v6_test_read": False,
                "production_weights_mutated": False,
                "paper_updated": False,
            },
        }
        report_path = output / "report.json"
        _atomic_json(report_path, report)
        status = {
            "schema_version": f"{SCHEMA_VERSION}-status",
            "status": (
                "open_validation_gate_passed"
                if float(combined_report["f1"]) >= 0.85
                else "baseline_below_gate"
            ),
            "completed_utc": _utc(),
            "scores": {
                "score_agnostic_sequence_f1": report[
                    "score_agnostic_transcriber_diagnostic"
                ]["pitch_sequence_lcs"]["f1"],
                "canonical_transcriber_f1": transcriber_report["f1"],
                "oracle_note_mapper_f1": oracle_report["f1"],
                "combined_f1": combined_report["f1"],
                "combined_precision": combined_report["precision"],
                "combined_recall": combined_report["recall"],
                "combined_bootstrap_lower_95": combined_report[
                    "bootstrap_95"
                ]["lower_95"],
            },
            "gate": {
                "metric": "combined canonical note-wise F1",
                "threshold": 0.85,
                "passed": float(combined_report["f1"]) >= 0.85,
                "lockbox_action": (
                    "eligible_for_single_open"
                    if float(combined_report["f1"]) >= 0.85
                    else "remain_sealed"
                ),
            },
            "artifacts": {
                "prediction_freeze": {
                    "path": str(freeze_path),
                    "sha256": sha256_file(freeze_path),
                },
                "predictions": {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                },
                "report": {
                    "path": str(report_path),
                    "sha256": sha256_file(report_path),
                },
            },
            "lockbox_opened": False,
            "production_weights_mutated": False,
            "paper_updated": False,
        }
        _atomic_json(output / "STATUS.json", status)
        print(json.dumps(status["scores"], indent=2), flush=True)
    finally:
        _assert_lockbox_sealed(release)
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def _bootstrap_counts(
    counts: Sequence[tuple[float, int, int]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = generator.integers(0, len(counts), len(counts))
        values.append(
            _fractional_prf(
                sum(counts[index][0] for index in chosen),
                sum(counts[index][1] for index in chosen),
                sum(counts[index][2] for index in chosen),
            )["f1"]
        )
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _resolve(repo: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo / value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "score"))
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--release-manifest", type=Path, default=DEFAULT_RELEASE
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--mapper-candidate", type=Path, default=DEFAULT_MAPPER_CANDIDATE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resource-status", type=Path, default=DEFAULT_RESOURCE_STATUS
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=2048)
    parser.add_argument("--overlap-frames", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.repo = args.repo.resolve()
    for name in (
        "release_manifest",
        "checkpoint",
        "mapper_candidate",
        "output_dir",
        "resource_status",
    ):
        setattr(args, name, _resolve(args.repo, getattr(args, name)))
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    if args.phase == "infer":
        run_infer(args)
    else:
        run_score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
