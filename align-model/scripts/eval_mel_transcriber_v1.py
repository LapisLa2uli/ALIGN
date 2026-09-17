"""Canonical note-wise evaluation of frozen Track B predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.joint.baseline import current_note_aligner_baseline
from alignmodel.joint.index import JointEvent
from alignmodel.joint.lattice import JointCandidate, LatticeConfig, SparseJointLattice
from alignmodel.joint.metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    evaluate_joint_events,
    pair_exact_pitch_onset,
)
from alignmodel.joint.outputraw_full import infer_full_pipeline, load_checkpoint
from alignmodel.joint.packed_data import PackedJointDataset, sha256_file
from alignmodel.melody import match_note_wise_labels_detail


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _event(value: Mapping[str, Any]) -> JointEvent:
    return JointEvent(
        pitch=int(value["pitch"]),
        start=float(value.get("start", value.get("start_sec"))),
        end=float(value.get("end", value.get("end_sec"))),
        score_span=(
            tuple(int(item) for item in value["score_span"])
            if value.get("score_span") is not None else None
        ),
        relationship=str(value.get("relationship") or "match"),
        copy_pass=int(value.get("copy_pass") or 0),
        origin_relationship=value.get("origin_relationship"),
        rendered_index=value.get("rendered_index"),
        source_indices=tuple(int(item) for item in value.get("source_indices") or ()),
        confidence=float(value.get("confidence", 1.0)),
    )


def _prediction_candidates(notes: Sequence[Mapping[str, Any]]) -> list[JointCandidate]:
    """Expose calibrated alternatives without consulting score content."""

    candidates: dict[tuple[int, int], JointCandidate] = {}
    for note in notes:
        start = float(note["start"])
        end = float(note["end"])
        confidence = float(note["confidence"])
        pitches = [int(value) for value in note.get("pitch_candidates") or [note["pitch"]]]
        probabilities = [
            float(value) for value in note.get("candidate_confidences") or [confidence]
        ]
        for rank, pitch in enumerate(pitches):
            pitch_probability = probabilities[min(rank, len(probabilities) - 1)]
            candidate_confidence = (
                confidence if rank == 0
                else confidence * max(0.05, min(0.85, pitch_probability))
            )
            if rank > 0 and pitch_probability < 0.08:
                continue
            margin = pitch_probability - probabilities[0]
            candidate = JointCandidate(
                pitch=pitch,
                start=start,
                end=end,
                confidence=candidate_confidence,
                score_hints=(),
                acoustic_features=(
                    float(note.get("onset_strength", confidence)),
                    max(confidence, pitch_probability),
                    confidence,
                    min(0.0, margin),
                    pitch_probability,
                ),
            )
            key = (int(round(start * 1000)), pitch)
            if key not in candidates or candidates[key].confidence < candidate.confidence:
                candidates[key] = candidate
    return sorted(candidates.values(), key=lambda item: (item.start, item.pitch, item.end))


def _primary_events(notes: Sequence[Mapping[str, Any]]) -> list[JointEvent]:
    return [
        JointEvent(
            pitch=int(note["pitch"]),
            start=float(note["start"]),
            end=float(note["end"]),
            score_span=None,
            relationship="extra",
            confidence=float(note["confidence"]),
        )
        for note in notes
    ]


def _primary_candidates(
    notes: Sequence[Mapping[str, Any]],
) -> list[JointCandidate]:
    return [
        JointCandidate(
            pitch=int(note["pitch"]),
            start=float(note["start"]),
            end=float(note["end"]),
            confidence=float(note["confidence"]),
            score_hints=(),
            acoustic_features=(
                float(note.get("onset_strength", note["confidence"])),
                float(
                    (note.get("candidate_confidences") or [note["confidence"]])[0]
                ),
                float(note["confidence"]),
                0.0,
                float(
                    (note.get("candidate_confidences") or [note["confidence"]])[0]
                ),
            ),
        )
        for note in notes
    ]


def _transcription_event(event: JointEvent, rendered_index: int) -> JointEvent:
    """Normalize type while retaining canonical score/rendered identity."""

    if event.score_span is None:
        return replace(
            event,
            relationship="extra",
            copy_pass=0,
            rendered_index=rendered_index,
        )
    return replace(
        event,
        relationship="match",
        origin_relationship="match",
        copy_pass=0,
    )


def _target_transcription_event(event: JointEvent) -> JointEvent:
    if event.score_span is None:
        return event
    return replace(
        event,
        relationship="match",
        origin_relationship="match",
        copy_pass=0,
    )


def _event_label(event: JointEvent, kind: str | None = None) -> dict[str, Any]:
    label: dict[str, Any] = {"type": kind or event.relationship}
    if event.score_span is not None:
        label["score_event_indices"] = list(range(*event.score_span))
        label["copy_pass"] = event.copy_pass
    elif event.rendered_index is not None:
        label["rendered_index"] = int(event.rendered_index)
    return label


def _fractional_prf(credit: float, predicted: int, gold: int) -> dict[str, Any]:
    precision = credit / predicted if predicted else (1.0 if not gold else 0.0)
    recall = credit / gold if gold else (1.0 if not predicted else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
        "credit": credit,
        "predicted": predicted,
        "support": gold,
    }


def _bootstrap(
    counts: Sequence[tuple[float, int, int]], seed: int, replicates: int
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = generator.integers(0, len(counts), len(counts))
        credit = sum(counts[index][0] for index in selected)
        predicted = sum(counts[index][1] for index in selected)
        gold = sum(counts[index][2] for index in selected)
        values.append(_fractional_prf(credit, predicted, gold)["f1"])
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _lcs(left: Sequence[int], right: Sequence[int]) -> int:
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            row[column] = previous + 1 if item == other else max(
                row[column], row[column - 1]
            )
            previous = saved
    return row[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--aligner-checkpoint", type=Path, required=True)
    parser.add_argument("--aligner-config", type=Path, required=True)
    parser.add_argument("--basic-pitch-report", type=Path, required=True)
    parser.add_argument("--track-a-report", type=Path)
    parser.add_argument(
        "--reuse-fixed-report",
        type=Path,
        help="Reuse a hash-stable fixed-aligner result for identical predictions",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()

    ready = json.loads(args.ready_marker.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if sha256_file(args.predictions) != freeze["predictions_sha256"]:
        raise ValueError("Frozen prediction checksum mismatch")
    if freeze["release_pack_id"] != ready["hashes"]["pack_id"]:
        raise ValueError("Frozen predictions belong to another release")
    predictions = {
        row["sample"]: row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if len(predictions) != 358 or int(freeze["rows"]) != 358:
        raise ValueError("Canonical validation requires exactly 358 frozen rows")
    manifest_path = Path(ready["paths"]["manifest"])
    if sha256_file(manifest_path) != ready["hashes"]["manifest_sha256"]:
        raise ValueError("Audited manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_rows = {str(row["sample"]): row for row in manifest["val"]}
    if set(predictions) != set(val_rows):
        raise ValueError("Frozen prediction IDs do not equal audited validation IDs")

    reused_report = (
        json.loads(args.reuse_fixed_report.read_text(encoding="utf-8"))
        if args.reuse_fixed_report is not None
        else None
    )
    if reused_report is not None:
        reused_freeze = reused_report.get("inference_isolation") or {}
        if reused_freeze.get("predictions_sha256") != freeze["predictions_sha256"]:
            raise ValueError("Reused fixed report belongs to other predictions")
        reused_aligner = reused_report.get("fixed_downstream_aligner") or {}
        if (
            reused_aligner.get("checkpoint_sha256")
            != sha256_file(args.aligner_checkpoint)
            or reused_aligner.get("config_sha256")
            != sha256_file(args.aligner_config)
        ):
            raise ValueError("Reused fixed report used another aligner")
        aligner = lattice = None
    else:
        aligner_config = json.loads(
            args.aligner_config.read_text(encoding="utf-8")
        )
        aligner, _aligner_payload = load_checkpoint(
            args.aligner_checkpoint,
            device="cpu",
            expected_data_fingerprint=ready["hashes"]["pack_id"],
        )
        lattice = SparseJointLattice(
            aligner, LatticeConfig(**aligner_config["lattice"])
        )
    packed = PackedJointDataset(
        Path(ready["paths"]["packed_root"]),
        manifest_sha256=ready["hashes"]["manifest_sha256"],
        verify_records=False,
        load_feature_arrays=False,
    )
    packed_val = {
        item.sample: item for item in (
            packed[index] for index in packed.ordinals("val")
        )
    }

    metric_samples = []
    transcription_samples = []
    per_clip_counts = []
    transcription_clip_counts = []
    per_type = defaultdict(lambda: [0.0, 0, 0])
    sequence_counts = [0, 0, 0]
    duration_groups = {
        "lt_80ms": [0, 0],
        "lt_120ms": [0, 0],
        "lt_180ms": [0, 0],
        "ge_180ms": [0, 0],
    }
    grouped = defaultdict(lambda: [0, 0, 0])
    same_pitch_splits = 0
    target_same_pitch_rearticulations = 0
    aligned_candidate_count = 0
    for position, sample in enumerate(sorted(predictions), 1):
        prediction_row = predictions[sample]
        packed_sample = packed_val[sample]
        target = tuple(_event(value) for value in packed_sample.target["target_events"])
        target_deletions = frozenset(
            int(value) for value in packed_sample.target["target_deletions"]
        )
        row = val_rows[sample]
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        if sha256_file(score_path) != row["source_hashes"]["verified_score.musicxml"]:
            raise ValueError(f"Verified score hash mismatch: {sample}")
        score = packed_sample.training_example().score
        if reused_report is None:
            assert aligner is not None and lattice is not None
            candidates = _prediction_candidates(prediction_row["notes"])
            aligned_candidate_count += len(candidates)
            aligned = infer_full_pipeline(aligner, lattice, candidates, score)
            metric_sample = JointMetricSample(
                predicted=aligned.events,
                target=target,
                source=str(row["source"]),
                predicted_deletions=frozenset(aligned.missed_score_events),
                target_deletions=target_deletions,
                score_event_count=len(score),
            )
            metric_samples.append(metric_sample)
            clip_report = evaluate_joint_events(
                aligned.events,
                target,
                predicted_deletions=aligned.missed_score_events,
                target_deletions=target_deletions,
                score_event_count=len(score),
            )["official_note_wise"]
            per_clip_counts.append((
                float(clip_report["credit"]),
                int(clip_report["predicted"]),
                int(clip_report["gold"]),
            ))

        transcription_events, _transcription_deletions = (
            current_note_aligner_baseline(
                _primary_candidates(prediction_row["notes"]),
                score,
            )
        )
        transcription_sample = JointMetricSample(
            predicted=tuple(
                _transcription_event(event, index)
                for index, event in enumerate(transcription_events)
            ),
            target=tuple(
                _target_transcription_event(event) for event in target
            ),
            source=str(row["source"]),
            predicted_deletions=frozenset(),
            target_deletions=frozenset(),
            score_event_count=len(score),
        )
        transcription_samples.append(transcription_sample)
        transcription_clip = evaluate_joint_events(
            transcription_sample.predicted,
            transcription_sample.target,
            score_event_count=len(score),
        )["official_note_wise"]
        transcription_clip_counts.append((
            float(transcription_clip["credit"]),
            int(transcription_clip["predicted"]),
            int(transcription_clip["gold"]),
        ))

        if reused_report is None:
            for kind in ("match", "substitute", "extra", "copy"):
                pred_labels = [
                    _event_label(value, kind)
                    for value in aligned.events
                    if value.relationship == kind
                ]
                gold_labels = [
                    _event_label(value, kind)
                    for value in target
                    if value.relationship == kind
                ]
                detail = match_note_wise_labels_detail(
                    gold_labels, pred_labels, score_event_count=len(score)
                )
                values = per_type[kind]
                values[0] += float(detail["credit"])
                values[1] += len(pred_labels)
                values[2] += len(gold_labels)
            missed = per_type["missed_note"]
            missed[0] += len(
                set(aligned.missed_score_events) & target_deletions
            )
            missed[1] += len(aligned.missed_score_events)
            missed[2] += len(target_deletions)

        primary = _primary_events(prediction_row["notes"])
        pred_pitch = [value.pitch for value in primary]
        gold_pitch = [value.pitch for value in target]
        correct = _lcs(pred_pitch, gold_pitch)
        sequence_counts[0] += correct
        sequence_counts[1] += len(primary)
        sequence_counts[2] += len(target)
        pairs = pair_exact_pitch_onset(primary, target, tolerance_sec=0.050)
        paired_gold = {gold for _, gold in pairs}
        for label, predicate in (
            ("lt_80ms", lambda value: value < 0.080),
            ("lt_120ms", lambda value: value < 0.120),
            ("lt_180ms", lambda value: value < 0.180),
            ("ge_180ms", lambda value: value >= 0.180),
        ):
            selected = {
                index for index, event in enumerate(target)
                if predicate(event.end - event.start)
            }
            duration_groups[label][0] += len(selected & paired_gold)
            duration_groups[label][1] += len(selected)
        same_pitch_splits += sum(
            left.pitch == right.pitch and right.start - left.end <= 0.100
            for left, right in zip(primary, primary[1:])
        )
        target_same_pitch_rearticulations += sum(
            left.pitch == right.pitch
            for left, right in zip(target, target[1:])
        )
        timbre_key = f"timbre:{row['audio_render']}"
        source_key = f"source:{row['source']}"
        for key in (timbre_key, source_key):
            grouped[key][0] += len(pairs)
            grouped[key][1] += len(primary)
            grouped[key][2] += len(target)
        if position == 1 or position % 25 == 0 or position == 358:
            print(f"canonical={position}/358 sample={sample}", flush=True)
    packed.close()

    if reused_report is None:
        canonical = evaluate_joint_dataset(metric_samples)["aggregate"][
            "official_note_wise"
        ]
        fixed_downstream = {
            **canonical,
            "per_type": {
                kind: _fractional_prf(*values)
                for kind, values in sorted(per_type.items())
            },
            "bootstrap": _bootstrap(
                per_clip_counts,
                seed=20260916,
                replicates=args.bootstrap_replicates,
            ),
        }
    else:
        fixed_downstream = dict(
            reused_report.get("fixed_downstream_note_wise")
            or reused_report["canonical_note_wise"]
        )
        canonical = fixed_downstream
        aligned_candidate_count = int(
            reused_report.get("fixed_downstream_aligner", {}).get(
                "candidate_count", 0
            )
        )
    transcription_evaluation = evaluate_joint_dataset(transcription_samples)
    transcription = transcription_evaluation["aggregate"]["official_note_wise"]
    transcription["bootstrap"] = _bootstrap(
        transcription_clip_counts,
        seed=20260916,
        replicates=args.bootstrap_replicates,
    )
    sequence = _fractional_prf(*sequence_counts)
    basic = json.loads(args.basic_pitch_report.read_text(encoding="utf-8"))
    basic_headline = (
        basic.get("headline", {}).get("transcription_note_wise_f1")
        or basic.get("full_schema", {}).get("transcription", {}).get("f1")
        or basic.get("validation", {}).get("transcription", {}).get("f1")
    )
    track_a_path = args.track_a_report or args.basic_pitch_report
    track_a = json.loads(track_a_path.read_text(encoding="utf-8"))
    track_a_headline = (
        track_a.get("canonical_note_wise_transcription", {})
        .get("aggregate", {})
        .get("official_note_wise", {})
        .get("f1")
    )
    report = {
        "schema_version": "align-mel-transcriber-evaluation-v1",
        "metric": {
            "schema_version": "align-note-wise-score-event-metric-v1",
            "unit": "canonical score-event identity",
            "matching": "exclusive one-to-one; exact type=1.0, different type=0.5",
            "timestamp_tolerances": "diagnostic_only",
        },
        "data": {
            "release": ready["release"],
            "validation_rows": 358,
            "manifest_sha256": ready["hashes"]["manifest_sha256"],
            "pack_id": ready["hashes"]["pack_id"],
            "locked_test_touched": False,
        },
        "inference_isolation": freeze,
        "fixed_downstream_aligner": {
            "checkpoint": str(args.aligner_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.aligner_checkpoint),
            "config": str(args.aligner_config.resolve()),
            "config_sha256": sha256_file(args.aligner_config),
            "score_used_only_downstream": True,
            "candidate_alternatives_used": True,
            "candidate_count": aligned_candidate_count,
            "reused_report": (
                str(args.reuse_fixed_report.resolve())
                if args.reuse_fixed_report is not None
                else None
            ),
            "reused_report_sha256": (
                sha256_file(args.reuse_fixed_report)
                if args.reuse_fixed_report is not None
                else None
            ),
        },
        "canonical_note_wise_transcription": {
            **transcription,
            "per_source": {
                key: value["official_note_wise"]
                for key, value in sorted(
                    transcription_evaluation.get("per_source", {}).items()
                )
            },
        },
        "fixed_downstream_note_wise": fixed_downstream,
        "canonical_note_wise": transcription,
        "score_agnostic_diagnostics": {
            "pitch_sequence_lcs": sequence,
            "count_ratio": sequence_counts[1] / max(sequence_counts[2], 1),
            "same_pitch_split_count": same_pitch_splits,
            "same_pitch_split_rate": same_pitch_splits / max(sequence_counts[1], 1),
            "target_same_pitch_rearticulation_count": (
                target_same_pitch_rearticulations
            ),
            "target_same_pitch_rearticulation_rate": (
                target_same_pitch_rearticulations / max(sequence_counts[2], 1)
            ),
            "short_note_recall": {
                key: {
                    "matched": value[0],
                    "support": value[1],
                    "recall": value[0] / max(value[1], 1),
                }
                for key, value in duration_groups.items()
            },
            "per_timbre_source": {
                key: _fractional_prf(*value)
                for key, value in sorted(grouped.items())
            },
            "onset_tolerance_sec": 0.050,
            "timestamp_status": "diagnostic_only",
        },
        "basic_pitch_same_validation_comparison": {
            "report": str(args.basic_pitch_report.resolve()),
            "report_sha256": sha256_file(args.basic_pitch_report),
            "basic_pitch_canonical_note_wise_f1": basic_headline,
            "mel_transcriber_canonical_note_wise_f1": transcription["f1"],
            "absolute_delta": (
                float(transcription["f1"]) - float(basic_headline)
                if basic_headline is not None else None
            ),
        },
        "track_a_same_validation_comparison": {
            "report": str(track_a_path.resolve()),
            "report_sha256": sha256_file(track_a_path),
            "track_a_canonical_note_wise_f1": track_a_headline,
            "mel_transcriber_canonical_note_wise_f1": transcription["f1"],
            "absolute_delta": (
                float(transcription["f1"]) - float(track_a_headline)
                if track_a_headline is not None else None
            ),
        },
        "promotion": {
            "target_f1": 0.95,
            "validation_gate_met": float(transcription["f1"]) > 0.95,
            "promoted": False,
            "lockbox_opened": False,
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps({
        "canonical_note_wise_transcription": (
            report["canonical_note_wise_transcription"]
        ),
        "fixed_downstream_note_wise": report["fixed_downstream_note_wise"],
        "comparison": report["basic_pitch_same_validation_comparison"],
        "promotion": report["promotion"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
