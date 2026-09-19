"""Calibrate score-template-supported weak Basic Pitch ornament rescue."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import calibrate_v2_acoustic_crf as acoustic
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.grammar_mapper_v2 import (
    GrammarHypothesis,
    grammar_hypotheses,
)
from alignmodel.joint.identity_crf_fast_v2 import fast_decode_identity_crf
from alignmodel.joint.identity_crf_v1 import (
    IdentityCandidate,
    build_identity_lattice,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.ornament_mapper_v1 import (
    expand_ornament_hypothesis,
    score_ornament_patterns,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    basic_pitch_cache_path,
    decode_basic_pitch_features,
    extract_sample_basic_pitch_features,
)


SCHEMA_VERSION = "align-orn-v2-template-rescue-calibration-v1"
BASE_DECODE = BasicPitchDecodeConfig()
WEAK_DECODE = BasicPitchDecodeConfig(
    onset_threshold=0.40,
    frame_threshold=0.30,
    minimum_note_length_ms=30.0,
)
CONFIDENCE_THRESHOLDS = (0.35, 0.50)
WINDOWS_SEC = (0.15, 0.30, 0.60)
DIRECT_EVIDENCE_THRESHOLDS = (0.15, 0.25, 0.35)
ALTERNATIVE_CONFIDENCE_THRESHOLDS = (0.20, 0.30)


def _selected_hypothesis(
    score: Sequence[Any],
    count: int,
    diagnostics: Mapping[str, Any],
) -> GrammarHypothesis:
    candidates = [
        value
        for value in grammar_hypotheses(
            score, count, edit_slack=100000
        )
        if value.copies == int(diagnostics["copies"])
        and value.source_span
        == (
            tuple(diagnostics["source_span"])
            if diagnostics["source_span"] is not None
            else None
        )
    ]
    if not candidates:
        return GrammarHypothesis(
            None,
            0,
            tuple((index, 0) for index in range(len(score))),
            tuple(float(event.ql_start) for event in score),
        )
    return candidates[0]


def _rescue(
    base: Sequence[Any],
    weak: Sequence[Any],
    score: Sequence[Any],
    score_path: Path,
    diagnostics: Mapping[str, Any],
    features: Any,
    *,
    confidence_threshold: float,
    window_sec: float,
) -> tuple[Any, ...]:
    patterns = score_ornament_patterns(score_path, tuple(score))
    hypothesis = _selected_hypothesis(score, len(base), diagnostics)
    template = expand_ornament_hypothesis(
        score, patterns, hypothesis
    )
    ornaments = [value for value in template if value.is_ornament]
    if not ornaments:
        return acoustic._identity_candidates(base, features)
    audio_start = min((float(value.start) for value in base), default=0.0)
    audio_end = max((float(value.end) for value in base), default=audio_start + 1.0)
    audio_extent = max(audio_end - audio_start, 1e-6)
    template_start = min(value.time for value in template)
    template_end = max(value.time for value in template)
    template_extent = max(template_end - template_start, 1e-6)
    accepted = list(base)
    for candidate in weak:
        if float(candidate.confidence) < confidence_threshold:
            continue
        if any(
            int(value.pitch) == int(candidate.pitch)
            and abs(float(value.start) - float(candidate.start)) <= 0.050
            for value in accepted
        ):
            continue
        position = (
            float(candidate.start) - audio_start
        ) / audio_extent
        supported = any(
            int(unit.pitch) == int(candidate.pitch)
            and abs(
                position
                - (float(unit.time) - template_start) / template_extent
            )
            * audio_extent
            <= window_sec
            for unit in ornaments
        )
        if supported:
            accepted.append(candidate)
    accepted.sort(key=lambda value: (value.start, value.pitch, value.end))
    return acoustic._identity_candidates(accepted, features)


def _direct_rescue(
    base: Sequence[Any],
    score: Sequence[Any],
    score_path: Path,
    diagnostics: Mapping[str, Any],
    features: Any,
    *,
    evidence_threshold: float,
    window_sec: float,
) -> tuple[IdentityCandidate, ...]:
    accepted = list(acoustic._identity_candidates(base, features))
    patterns = score_ornament_patterns(score_path, tuple(score))
    hypothesis = _selected_hypothesis(score, len(base), diagnostics)
    template = expand_ornament_hypothesis(score, patterns, hypothesis)
    ornaments = [value for value in template if value.is_ornament]
    if not ornaments:
        return tuple(accepted)
    audio_start = min((value.start for value in accepted), default=0.0)
    audio_end = max((value.end for value in accepted), default=audio_start + 1.0)
    audio_extent = max(audio_end - audio_start, 1e-6)
    template_start = min(value.time for value in template)
    template_end = max(value.time for value in template)
    template_extent = max(template_end - template_start, 1e-6)
    times = np.asarray(features.frame_times, np.float64)
    for unit in ornaments:
        axis = int(unit.pitch) - 21
        if not 0 <= axis < features.note.shape[1]:
            continue
        expected = audio_start + (
            (float(unit.time) - template_start) / template_extent
        ) * audio_extent
        frame_indices = np.flatnonzero(np.abs(times - expected) <= window_sec)
        if not len(frame_indices):
            continue
        evidence = (
            0.55 * np.asarray(features.onset[frame_indices, axis], np.float32)
            + 0.45 * np.asarray(features.note[frame_indices, axis], np.float32)
        )
        local = int(np.argmax(evidence))
        frame = int(frame_indices[local])
        confidence = float(evidence[local])
        if confidence < evidence_threshold:
            continue
        start = float(times[frame])
        if any(
            value.pitch == int(unit.pitch)
            and abs(value.start - start) <= 0.050
            for value in accepted
        ):
            continue
        support_threshold = max(0.10, evidence_threshold * 0.60)
        left = frame
        right = frame + 1
        while (
            left > 0
            and frame - left < 12
            and float(features.note[left - 1, axis]) >= support_threshold
        ):
            left -= 1
        while (
            right < len(times)
            and right - frame < 12
            and float(features.note[right, axis]) >= support_threshold
        ):
            right += 1
        frame_hop = (
            float(np.median(np.diff(times))) if len(times) > 1 else 0.01161
        )
        end = max(
            float(times[min(right, len(times) - 1)]) + frame_hop,
            start + 0.020,
        )
        pitch_probabilities = np.asarray(features.note[frame], np.float32)
        top = np.argsort(pitch_probabilities)[-4:][::-1]
        accepted.append(
            IdentityCandidate(
                pitch=int(unit.pitch),
                start=start,
                end=end,
                confidence=confidence,
                alternatives=tuple(int(value) + 21 for value in top),
                alternative_confidences=tuple(
                    float(pitch_probabilities[value]) for value in top
                ),
            )
        )
    return tuple(sorted(accepted, key=lambda value: (value.start, value.pitch, value.end)))


def _alternative_correct(
    base: Sequence[Any],
    score: Sequence[Any],
    score_path: Path,
    diagnostics: Mapping[str, Any],
    features: Any,
    *,
    confidence_threshold: float,
    window_sec: float,
    margin: float = -0.15,
) -> tuple[tuple[IdentityCandidate, ...], int]:
    candidates = list(acoustic._identity_candidates(base, features))
    patterns = score_ornament_patterns(score_path, tuple(score))
    hypothesis = _selected_hypothesis(score, len(base), diagnostics)
    template = expand_ornament_hypothesis(score, patterns, hypothesis)
    audio_start = min((value.start for value in candidates), default=0.0)
    audio_end = max((value.end for value in candidates), default=audio_start + 1.0)
    audio_extent = max(audio_end - audio_start, 1e-6)
    template_start = min((value.time for value in template), default=0.0)
    template_end = max((value.time for value in template), default=template_start + 1.0)
    template_extent = max(template_end - template_start, 1e-6)
    corrected = []
    changes = 0
    for candidate in candidates:
        position = (candidate.start - audio_start) / audio_extent
        supported = {
            int(unit.pitch)
            for unit in template
            if abs(
                position - (float(unit.time) - template_start) / template_extent
            )
            * audio_extent
            <= window_sec
        }
        probability_by_pitch = dict(
            zip(candidate.alternatives, candidate.alternative_confidences)
        )
        primary_confidence = float(
            probability_by_pitch.get(candidate.pitch, 0.0)
        )
        choices = [
            (float(confidence), int(pitch))
            for pitch, confidence in zip(
                candidate.alternatives,
                candidate.alternative_confidences,
            )
            if int(pitch) in supported
            and float(confidence) >= confidence_threshold
            and float(confidence) >= primary_confidence + margin
        ]
        selected_pitch = (
            max(choices)[1] if choices else candidate.pitch
        )
        changes += int(selected_pitch != candidate.pitch)
        corrected.append(
            IdentityCandidate(
                pitch=selected_pitch,
                start=candidate.start,
                end=candidate.end,
                confidence=candidate.confidence,
                alternatives=candidate.alternatives,
                alternative_confidences=candidate.alternative_confidences,
            )
        )
    return (
        tuple(sorted(corrected, key=lambda value: (value.start, value.pitch, value.end))),
        changes,
    )


def _map(
    model: Any,
    candidates: Sequence[Any],
    index: ScoreEventIndex,
    score_path: Path,
) -> tuple[JointMetricSample, dict[str, Any]]:
    lattice = build_identity_lattice(
        candidates,
        index.events,
        score_path,
        max_inference_hypotheses=16,
    )
    predicted, deletions, diagnostics = fast_decode_identity_crf(
        model, lattice
    )
    return (
        JointMetricSample(
            predicted=predicted,
            target=index.rendered_events,
            predicted_deletions=deletions,
            target_deletions=index.deleted_event_indices,
            score_event_count=len(index.events),
        ),
        diagnostics,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen ORN v2 release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("ORN v2 lockbox opening sentinel exists")
    model, payload = acoustic._load_model(args.checkpoint)
    if payload["data"]["release_manifest_sha256"] != release_sha:
        raise ValueError("CRF checkpoint release mismatch")
    rows = {
        row["sample"]: row
        for row in release["splits"]["development"]["calibration"]
    }
    targets = acoustic._targets(release, "calibration")
    contexts = {}
    for position, sample in enumerate(sorted(rows), 1):
        row = rows[sample]
        feature = extract_sample_basic_pitch_features(
            row["sample_dir"],
            cache_path=basic_pitch_cache_path(
                args.feature_cache, row["sample_dir"], "orn-v2-calibration"
            ),
        )
        base = decode_basic_pitch_features(feature, BASE_DECODE)
        weak = decode_basic_pitch_features(feature, WEAK_DECODE)
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        index = ScoreEventIndex.from_musicxml(
            score_path, targets[sample]["lineage"]
        )
        base_candidates = acoustic._identity_candidates(base, feature)
        baseline_sample, diagnostics = _map(
            model, base_candidates, index, score_path
        )
        contexts[sample] = {
            "row": row,
            "features": feature,
            "base": base,
            "weak": weak,
            "score_path": score_path,
            "index": index,
            "baseline": baseline_sample,
            "diagnostics": diagnostics,
        }
        if position == 1 or position % 10 == 0 or position == len(rows):
            print(f"context={position}/{len(rows)}", flush=True)
    variants = []
    strategies = [
        {
            "mode": "baseline",
            "confidence": None,
            "window": None,
            "evidence": None,
        }
    ]
    strategies.extend(
        {
            "mode": "decoded_weak",
            "confidence": confidence,
            "window": window,
            "evidence": None,
        }
        for confidence in CONFIDENCE_THRESHOLDS
        for window in WINDOWS_SEC
    )
    strategies.extend(
        {
            "mode": "direct_template",
            "confidence": None,
            "window": window,
            "evidence": evidence,
        }
        for evidence in DIRECT_EVIDENCE_THRESHOLDS
        for window in WINDOWS_SEC[:2]
    )
    strategies.extend(
        {
            "mode": "alternative_pitch",
            "confidence": confidence,
            "window": window,
            "evidence": None,
        }
        for confidence in ALTERNATIVE_CONFIDENCE_THRESHOLDS
        for window in WINDOWS_SEC[:2]
    )
    for variant_index, strategy in enumerate(strategies):
        confidence = strategy["confidence"]
        window = strategy["window"]
        samples = []
        rescue_counts = []
        correction_counts = []
        for position, sample in enumerate(sorted(contexts), 1):
            context = contexts[sample]
            if strategy["mode"] == "baseline":
                metric_sample = context["baseline"]
                candidates = acoustic._identity_candidates(
                    context["base"], context["features"]
                )
                corrections = 0
            elif strategy["mode"] == "decoded_weak":
                candidates = _rescue(
                    context["base"],
                    context["weak"],
                    context["index"].events,
                    context["score_path"],
                    context["diagnostics"],
                    context["features"],
                    confidence_threshold=float(confidence),
                    window_sec=float(window),
                )
                metric_sample, _diagnostics = _map(
                    model,
                    candidates,
                    context["index"],
                    context["score_path"],
                )
                corrections = 0
            elif strategy["mode"] == "direct_template":
                candidates = _direct_rescue(
                    context["base"],
                    context["index"].events,
                    context["score_path"],
                    context["diagnostics"],
                    context["features"],
                    evidence_threshold=float(strategy["evidence"]),
                    window_sec=float(window),
                )
                metric_sample, _diagnostics = _map(
                    model,
                    candidates,
                    context["index"],
                    context["score_path"],
                )
                corrections = 0
            else:
                candidates, corrections = _alternative_correct(
                    context["base"],
                    context["index"].events,
                    context["score_path"],
                    context["diagnostics"],
                    context["features"],
                    confidence_threshold=float(confidence),
                    window_sec=float(window),
                )
                metric_sample, _diagnostics = _map(
                    model,
                    candidates,
                    context["index"],
                    context["score_path"],
                )
            metric_sample = JointMetricSample(
                predicted=metric_sample.predicted,
                target=metric_sample.target,
                source=context["row"]["leakage_group"],
                predicted_deletions=metric_sample.predicted_deletions,
                target_deletions=metric_sample.target_deletions,
                score_event_count=metric_sample.score_event_count,
            )
            samples.append(metric_sample)
            rescue_counts.append(
                len(candidates) - len(context["base"])
            )
            correction_counts.append(corrections)
            if position == 1 or position % 16 == 0 or position == len(contexts):
                print(
                    f"variant={variant_index} rows={position}/{len(contexts)}",
                    flush=True,
                )
        variants.append(
            {
                "variant": variant_index,
                "mode": strategy["mode"],
                "confidence_threshold": confidence,
                "evidence_threshold": strategy["evidence"],
                "window_sec": window,
                "rescued_candidates": sum(rescue_counts),
                "rows_with_rescue": sum(value > 0 for value in rescue_counts),
                "pitch_corrections": sum(correction_counts),
                "rows_with_pitch_corrections": sum(
                    value > 0 for value in correction_counts
                ),
                "combined": baseline._full_report(
                    samples,
                    seed=args.seed + variant_index,
                    replicates=args.bootstrap_replicates,
                ),
            }
        )
    selected = max(
        variants,
        key=lambda value: (
            float(value["combined"]["f1"]),
            -int(value["variant"]),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "calibration_report.json"
    acoustic._atomic_json(
        report_path,
        {
            "schema_version": SCHEMA_VERSION,
            "release_manifest_sha256": release_sha,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "population": {"split": "calibration", "rows": len(rows)},
            "base_decode": asdict(BASE_DECODE),
            "weak_decode": asdict(WEAK_DECODE),
            "predeclared_confidence_thresholds": list(
                CONFIDENCE_THRESHOLDS
            ),
            "predeclared_windows_sec": list(WINDOWS_SEC),
            "predeclared_direct_evidence_thresholds": list(
                DIRECT_EVIDENCE_THRESHOLDS
            ),
            "predeclared_alternative_confidence_thresholds": list(
                ALTERNATIVE_CONFIDENCE_THRESHOLDS
            ),
            "variants": variants,
            "selection": selected,
            "selection_metric": "official combined canonical note-wise F1",
            "timestamps_used_for_selection": False,
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    candidate_path = args.output_dir / "RESCUE_CANDIDATE.json"
    acoustic._atomic_json(
        candidate_path,
        {
            "schema_version": f"{SCHEMA_VERSION}-candidate",
            "frozen": True,
            "release_manifest_sha256": release_sha,
            "identity_crf_checkpoint": str(args.checkpoint.resolve()),
            "identity_crf_checkpoint_sha256": sha256_file(args.checkpoint),
            "base_decode": asdict(BASE_DECODE),
            "weak_decode": asdict(WEAK_DECODE),
            "confidence_threshold": selected["confidence_threshold"],
            "mode": selected["mode"],
            "evidence_threshold": selected["evidence_threshold"],
            "window_sec": selected["window_sec"],
            "calibration_report": str(report_path.resolve()),
            "calibration_report_sha256": sha256_file(report_path),
            "calibration_combined": selected["combined"],
            "calibration_gate": 0.85,
            "calibration_gate_passed": float(selected["combined"]["f1"])
            >= 0.85,
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate_path.resolve()),
                "candidate_sha256": sha256_file(candidate_path),
                "selected_variant": selected["variant"],
                "rescued_candidates": selected["rescued_candidates"],
                "combined_f1": selected["combined"]["f1"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import argparse

    raise SystemExit(main())
