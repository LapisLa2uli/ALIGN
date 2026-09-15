"""Diagnose real-audio note coverage without modifying sample artifacts.

Transcription is generated only from a hash-validated Basic Pitch cache. The
verified score and independently recomputed DTW path are loaded afterward and
are used only for diagnostic matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
import wave
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np

from alignmodel.joint.candidates import (
    CANDIDATE_GENERATION_VERSION,
    HIGH_RECALL_DECODE_CONFIGS,
    LEGACY_CANDIDATE_GENERATION_VERSION,
    LEGACY_HIGH_RECALL_DECODE_CONFIGS,
    basic_pitch_candidate_union,
)
from alignmodel.transcription.basic_pitch import (
    MIDI_OFFSET,
    BasicPitchDecodeConfig,
    BasicPitchFeatures,
    FROZEN_DECODE_CONFIG,
    decode_basic_pitch_features,
    load_audio_metadata,
    load_basic_pitch_cache,
    sanitize_basic_pitch_notes,
    save_basic_pitch_cache,
    sha256_file,
)
from alignmodel.transcription.evaluate import evaluate_note_lists, match_notes


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _note_row(note: Any) -> dict[str, Any]:
    return {
        "pitch": int(note.pitch),
        "start": float(note.start),
        "end": float(note.end),
        "confidence": float(note.confidence),
    }


def _cache_features(sample: Path, output: Path) -> tuple[BasicPitchFeatures, dict]:
    wav = sample / "performance_audio.wav"
    source = sample / "basic_pitch_cache.npz"
    metadata = load_audio_metadata(sample)
    features = load_basic_pitch_cache(source, wav, metadata)
    if features is None:
        raise ValueError(f"Missing or stale hash/pitch-policy cache: {source}")
    copied = output / "basic_pitch_cache.hash-validated.npz"
    save_basic_pitch_cache(copied, features)
    return features, {
        "source": str(source),
        "diagnostic_copy": str(copied),
        "wav_sha256": sha256_file(wav),
        "cache_sha256": sha256_file(source),
        "metadata": dict(features.metadata),
    }


def _decode_audio_only(
    features: BasicPitchFeatures,
) -> tuple[dict[str, list[Any]], list[Any]]:
    canonical = sanitize_basic_pitch_notes(
        decode_basic_pitch_features(features, FROZEN_DECODE_CONFIG),
        features,
        FROZEN_DECODE_CONFIG,
    )
    v1 = basic_pitch_candidate_union(
        features,
        configs=LEGACY_HIGH_RECALL_DECODE_CONFIGS,
        minimum_confidence=0.0,
    )
    v2 = basic_pitch_candidate_union(
        features,
        configs=HIGH_RECALL_DECODE_CONFIGS,
        minimum_confidence=0.0,
    )
    streams = {
        "canonical_frozen": canonical,
        "joint_v1_all": v1,
        "joint_v1_confidence_065": [
            value for value in v1 if value.confidence >= 0.65
        ],
        "joint_v2_all": v2,
        "joint_v2_confidence_065": [
            value for value in v2 if value.confidence >= 0.65
        ],
    }
    low_base = decode_basic_pitch_features(
        features, HIGH_RECALL_DECODE_CONFIGS[0]
    )
    return streams, low_base


def _independent_reference(
    sample: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Import after all audio-only candidate streams have been generated.
    from datacreate.audio_utils import load_audio
    from datacreate.config import PipelineConfig
    from datacreate.note_alignment import align_score_events
    from datacreate.stages.stage5_alignment import run_alignment

    config = PipelineConfig.load()
    logger = logging.getLogger("real_audio_diagnostic")
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        shutil.copy2(
            sample / "verified_score.musicxml",
            workspace / "verified_score.musicxml",
        )
        run_alignment(
            sample / "performance_audio.wav",
            sample / "reference_audio.wav",
            workspace,
            config,
            logger,
            detect_candidates=False,
        )
        with np.load(workspace / "alignment.npz", allow_pickle=False) as saved:
            wp = np.asarray(saved["warping_path"])
            residuals = np.asarray(saved["frame_residuals"])
            ref_features = np.asarray(saved["ref_features"])
            perf_features = np.asarray(saved["perf_features"])
            hop = int(saved["hop_length"])
            sample_rate = int(saved["sample_rate"])

    audio, _ = load_audio(
        sample / "performance_audio.wav", sample_rate, mono=True
    )
    events = align_score_events(
        sample / "verified_score.musicxml",
        wp,
        int(ref_features.shape[1]),
        hop / sample_rate,
        residuals=residuals,
        perf_audio=audio,
        sample_rate=sample_rate,
        onset_refine=True,
        onset_lookback_sec=float(
            config.alignment.get("onset_lookback_sec", 0.15)
        ),
        onset_max_shift_sec=float(
            config.alignment.get("onset_max_shift_sec", 0.6)
        ),
        onset_rise_db=float(config.alignment.get("onset_rise_db", 8.0)),
        phrase_min_rest_ql=float(
            config.alignment.get("phrase_min_rest_ql", 0.25)
        ),
    )
    np.savez_compressed(
        output / "independent_dtw.npz",
        warping_path=wp,
        frame_residuals=residuals,
        ref_shape=np.asarray(ref_features.shape),
        perf_shape=np.asarray(perf_features.shape),
        hop_length=np.asarray(hop),
        sample_rate=np.asarray(sample_rate),
        engine=np.asarray("independent-chroma-midi-dtw-diagnostic"),
    )
    sounding = [event for event in events if not event["is_rest"]]
    return sounding, {
        "engine": "independent-chroma-midi-dtw-diagnostic",
        "score_event_count": len(events),
        "score_note_count": len(sounding),
        "score_rest_count": len(events) - len(sounding),
        "measure_min": min(int(event["measure"]) for event in sounding),
        "measure_max": max(int(event["measure"]) for event in sounding),
        "reference_note_span_sec": [
            min(float(event["ref_start"]) for event in sounding),
            max(float(event["ref_end"]) for event in sounding),
        ],
        "performance_note_span_sec": [
            min(float(event["perf_start"]) for event in sounding),
            max(float(event["perf_end"]) for event in sounding),
        ],
        "mean_frame_residual": float(np.mean(residuals)),
        "p95_frame_residual": float(np.quantile(residuals, 0.95)),
        "artifact": str(output / "independent_dtw.npz"),
    }


def _targets(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pitch": int(event["midi"]),
            "start": float(event["perf_start"]),
            "end": float(event["perf_end"]),
            "confidence": 1.0,
        }
        for event in events
    ]


def _bucket_recall(
    predicted: Sequence[Any],
    target: Sequence[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, dict[str, float | int]]:
    pairs = match_notes(
        predicted, target, onset_tolerance_sec=float(tolerance)
    )
    matched = {target_index for _pred, target_index in pairs}
    result: dict[str, dict[str, float | int]] = {}
    for milliseconds in (80, 120, 180, 250):
        selected = [
            index
            for index, note in enumerate(target)
            if 1000.0 * (float(note["end"]) - float(note["start"]))
            < milliseconds
        ]
        hits = sum(index in matched for index in selected)
        result[f"lt_{milliseconds}ms"] = {
            "target": len(selected),
            "matched": hits,
            "recall": hits / max(len(selected), 1),
        }
    selected = [
        index
        for index, note in enumerate(target)
        if float(note["end"]) - float(note["start"]) >= 0.250
    ]
    hits = sum(index in matched for index in selected)
    result["ge_250ms"] = {
        "target": len(selected),
        "matched": hits,
        "recall": hits / max(len(selected), 1),
    }
    return result


def _lcs_pairs(
    predicted: Sequence[Any],
    score_events: Sequence[dict[str, Any]],
) -> list[tuple[int, int]]:
    ordered = sorted(
        enumerate(predicted),
        key=lambda item: (float(item[1].start), int(item[1].pitch)),
    )
    n_pred = len(ordered)
    n_score = len(score_events)
    counts = [[0] * (n_score + 1) for _ in range(n_pred + 1)]
    for pred_i in range(n_pred - 1, -1, -1):
        pitch = int(ordered[pred_i][1].pitch)
        for score_i in range(n_score - 1, -1, -1):
            if pitch == int(score_events[score_i]["midi"]):
                counts[pred_i][score_i] = (
                    1 + counts[pred_i + 1][score_i + 1]
                )
            else:
                counts[pred_i][score_i] = max(
                    counts[pred_i + 1][score_i],
                    counts[pred_i][score_i + 1],
                )
    pairs = []
    pred_i = score_i = 0
    while pred_i < n_pred and score_i < n_score:
        original_i, note = ordered[pred_i]
        if (
            int(note.pitch) == int(score_events[score_i]["midi"])
            and counts[pred_i][score_i]
            == 1 + counts[pred_i + 1][score_i + 1]
        ):
            pairs.append((original_i, score_i))
            pred_i += 1
            score_i += 1
        elif counts[pred_i + 1][score_i] >= counts[pred_i][score_i + 1]:
            pred_i += 1
        else:
            score_i += 1
    return pairs


def _sequence_coverage(
    predicted: Sequence[Any],
    score_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    pairs = _lcs_pairs(predicted, score_events)
    matched = {score_i for _pred_i, score_i in pairs}
    by_measure = {}
    for measure in sorted(
        {int(event["measure"]) for event in score_events}
    ):
        selected = [
            index
            for index, event in enumerate(score_events)
            if int(event["measure"]) == measure
        ]
        by_measure[str(measure)] = {
            "matched": sum(index in matched for index in selected),
            "target": len(selected),
        }
    precision = len(pairs) / max(len(predicted), 1)
    recall = len(pairs) / max(len(score_events), 1)
    return {
        "warning": (
            "Timing-free pitch-sequence coverage is an optimistic upper bound "
            "in repeated motifs, not event F1."
        ),
        "matched": len(pairs),
        "predicted": len(predicted),
        "target": len(score_events),
        "precision_proxy": precision,
        "recall_upper_bound": recall,
        "f1_proxy": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "by_measure": by_measure,
    }


def _confidence_gate_sweep(
    predicted: Sequence[Any],
    score_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in (
        0.0,
        0.40,
        0.45,
        0.50,
        0.55,
        0.575,
        0.60,
        0.625,
        0.65,
        0.675,
        0.70,
        0.725,
        0.75,
    ):
        selected = [
            note
            for note in predicted
            if float(note.confidence) >= threshold
        ]
        coverage = _sequence_coverage(selected, score_events)
        rows.append(
            {
                "minimum_confidence": threshold,
                **{
                    key: value
                    for key, value in coverage.items()
                    if key not in {"warning", "by_measure"}
                },
                "extra_sequence_candidates": len(selected)
                - int(coverage["matched"]),
            }
        )
    return rows


def _timing_reference_quality(
    predicted: Sequence[Any],
    target: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    used: set[int] = set()
    errors = []
    for note in sorted(predicted, key=lambda value: float(value.start)):
        options = [
            (
                abs(float(note.start) - float(other["start"])),
                index,
                float(note.start) - float(other["start"]),
            )
            for index, other in enumerate(target)
            if index not in used
            and int(note.pitch) == int(other["pitch"])
            and abs(float(note.start) - float(other["start"])) <= 0.300
        ]
        if options:
            _absolute, target_index, signed = min(options)
            used.add(target_index)
            errors.append(signed)
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "broad_pitch_matched_pairs": len(errors),
        "median_signed_error_ms": (
            1000.0 * float(np.median(errors)) if errors else None
        ),
        "median_absolute_error_ms": (
            1000.0 * float(np.median(absolute)) if errors else None
        ),
        "p75_absolute_error_ms": (
            1000.0 * float(np.quantile(absolute, 0.75)) if errors else None
        ),
        "p90_absolute_error_ms": (
            1000.0 * float(np.quantile(absolute, 0.90)) if errors else None
        ),
        "fitness_for_20_50_100ms_scoring": "insufficient",
    }


def _split_rate(predicted: Sequence[Any], features: BasicPitchFeatures) -> dict:
    ordered = sorted(predicted, key=lambda value: (value.start, value.pitch))
    pairs = []
    for previous, current in zip(ordered, ordered[1:]):
        gap = float(current.start) - float(previous.end)
        if int(previous.pitch) != int(current.pitch) or not -0.02 <= gap <= 0.10:
            continue
        axis = int(current.pitch) - MIDI_OFFSET
        frame = int(
            np.argmin(np.abs(features.frame_times - float(current.start)))
        )
        onset = (
            float(features.onset[frame, axis])
            if 0 <= axis < features.onset.shape[1]
            else 0.0
        )
        pairs.append({"gap_sec": gap, "second_onset": onset})
    weak = sum(float(row["second_onset"]) < 0.60 for row in pairs)
    return {
        "adjacent_same_pitch_boundaries": len(pairs),
        "weak_second_onset_boundaries": weak,
        "weak_split_rate_per_prediction": weak / max(len(predicted), 1),
    }


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p25": None, "median": None, "p75": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        key: float(np.quantile(array, quantile))
        for key, quantile in (
            ("p10", 0.10),
            ("p25", 0.25),
            ("median", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
        )
    }


def _frame_audio_features(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, np.ndarray | float]:
    hop = 256
    rms = librosa.feature.rms(
        y=audio, frame_length=1024, hop_length=hop, center=True
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate, n_fft=1024, hop_length=hop
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=1024, hop_length=hop
    )[0]
    noise_floor = float(np.quantile(rms, 0.10))
    return {
        "rms": rms,
        "centroid": centroid,
        "flatness": flatness,
        "noise_floor": noise_floor,
        "hop_sec": hop / sample_rate,
    }


def _evidence(
    event: dict[str, Any],
    features: BasicPitchFeatures,
    audio_features: dict[str, np.ndarray | float],
) -> dict[str, float]:
    start = float(event["perf_start"])
    end = float(event["perf_end"])
    pitch = int(event["midi"])
    axis = pitch - MIDI_OFFSET
    first = int(np.searchsorted(features.frame_times, start, side="left"))
    last = int(np.searchsorted(features.frame_times, end, side="right"))
    first = max(0, min(first, len(features.frame_times) - 1))
    last = max(first + 1, min(last, len(features.frame_times)))
    onset_lo = max(0, first - 2)
    onset_hi = min(len(features.frame_times), first + 3)
    contour_first = axis * 3
    note_values = features.note[first:last, axis]
    contour_values = features.contour[
        first:last, contour_first : contour_first + 3
    ]
    hop_sec = float(audio_features["hop_sec"])
    audio_first = max(0, int(start / hop_sec))
    audio_last = max(
        audio_first + 1,
        min(len(audio_features["rms"]), int(np.ceil(end / hop_sec))),
    )
    rms = float(np.mean(audio_features["rms"][audio_first:audio_last]))
    noise = max(float(audio_features["noise_floor"]), 1e-8)
    return {
        "onset_peak": float(
            np.max(features.onset[onset_lo:onset_hi, axis])
        ),
        "frame_peak": float(np.max(note_values)),
        "frame_mean": float(np.mean(note_values)),
        "contour_peak": float(np.max(contour_values)),
        "contour_mean": float(np.mean(contour_values)),
        "rms_dbfs": float(20.0 * np.log10(max(rms, 1e-8))),
        "snr_proxy_db": float(20.0 * np.log10(max(rms, 1e-8) / noise)),
        "spectral_centroid_hz": float(
            np.mean(audio_features["centroid"][audio_first:audio_last])
        ),
        "spectral_flatness": float(
            np.mean(audio_features["flatness"][audio_first:audio_last])
        ),
    }


def _correlates(
    events: Sequence[dict[str, Any]],
    matched: set[int],
    evidence_rows: Sequence[dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    labels = np.asarray(
        [float(index in matched) for index in range(len(events))],
        dtype=np.float64,
    )
    output = {}
    for key in evidence_rows[0]:
        values = np.asarray([row[key] for row in evidence_rows])
        detected = values[labels > 0.5]
        missed = values[labels < 0.5]
        correlation = (
            float(np.corrcoef(values, labels)[0, 1])
            if np.std(values) > 1e-10 and np.std(labels) > 1e-10
            else None
        )
        output[key] = {
            "matched_median": (
                float(np.median(detected)) if len(detected) else None
            ),
            "missed_median": float(np.median(missed)) if len(missed) else None,
            "point_biserial_correlation": correlation,
        }
    return output


def _pitch_offset(
    predicted: Sequence[Any],
    target: Sequence[dict[str, Any]],
    tolerance: float = 0.100,
) -> dict[str, Any]:
    used: set[int] = set()
    deltas = []
    for note in sorted(predicted, key=lambda value: value.start):
        options = [
            (abs(float(note.start) - float(other["start"])), index)
            for index, other in enumerate(target)
            if index not in used
            and abs(float(note.start) - float(other["start"])) <= tolerance
        ]
        if not options:
            continue
        _distance, target_index = min(options)
        used.add(target_index)
        deltas.append(int(note.pitch) - int(target[target_index]["pitch"]))
    counts = {
        str(delta): deltas.count(delta) for delta in sorted(set(deltas))
    }
    return {
        "onset_paired": len(deltas),
        "written_minus_score_median_semitones": (
            float(np.median(deltas)) if deltas else None
        ),
        "written_minus_score_mode_semitones": (
            int(max(counts, key=lambda key: counts[key])) if counts else None
        ),
        "delta_histogram": counts,
        "implied_score_minus_sounding_mode": (
            2 - int(max(counts, key=lambda key: counts[key]))
            if counts
            else None
        ),
    }


def _stream_report(
    predicted: Sequence[Any],
    target: Sequence[dict[str, Any]],
    features: BasicPitchFeatures,
    score_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        f"{milliseconds}ms": evaluate_note_lists(
            predicted,
            target,
            onset_tolerance_sec=milliseconds / 1000.0,
        )
        for milliseconds in (20, 50, 100)
    }
    return {
        "count": len(predicted),
        "metrics": metrics,
        "duration_bucket_recall_50ms": _bucket_recall(
            predicted, target, tolerance=0.050
        ),
        "same_pitch_splits": _split_rate(predicted, features),
        "confidence": _quantiles(
            [float(value.confidence) for value in predicted]
        ),
        "pitch_offset": _pitch_offset(predicted, target),
        "timing_free_sequence_coverage": _sequence_coverage(
            predicted, score_events
        ),
    }


def _sweep(
    low_base: Sequence[Any],
    features: BasicPitchFeatures,
    target: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    profiles = (
        ("conservative", 0.28, 0.18, 0.20, 0.31),
        ("balanced", 0.24, 0.15, 0.17, 0.27),
        ("sensitive", 0.20, 0.12, 0.14, 0.24),
    )
    rows = []
    for (
        name,
        rescue_onset,
        rescue_frame,
        rescue_contour,
        rescue_combined,
    ) in profiles:
        for merge_onset in (0.50, 0.60, 0.70, 0.80):
            for merge_gap in (0.04, 0.07, 0.10):
                for rescue_max_ms in (110.0, 160.0):
                    config = replace(
                        HIGH_RECALL_DECODE_CONFIGS[0],
                        adaptive_short_note_rescue=True,
                        rescue_onset_threshold=rescue_onset,
                        rescue_frame_threshold=rescue_frame,
                        rescue_contour_threshold=rescue_contour,
                        rescue_combined_threshold=rescue_combined,
                        rescue_max_note_length_ms=rescue_max_ms,
                        merge_onset_threshold=merge_onset,
                        merge_same_pitch_gap_sec=merge_gap,
                    )
                    notes = sanitize_basic_pitch_notes(
                        list(low_base), features, config
                    )
                    metrics = evaluate_note_lists(
                        notes, target, onset_tolerance_sec=0.100
                    )
                    rows.append(
                        {
                            "profile": name,
                            "config": asdict(config),
                            "count": len(notes),
                            "matched_100ms": int(metrics["n_matched"]),
                            "precision_100ms": float(metrics["precision"]),
                            "recall_100ms": float(metrics["recall"]),
                            "f1_100ms": float(metrics["f1"]),
                            "extras_proxy": len(notes)
                            - int(metrics["n_matched"]),
                            "short_recall_50ms": _bucket_recall(
                                notes, target, tolerance=0.050
                            ),
                        }
                    )
    rows.sort(
        key=lambda row: (
            -float(row["f1_100ms"]),
            int(row["extras_proxy"]),
            -int(row["matched_100ms"]),
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample = args.sample.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    features, cache = _cache_features(sample, output)
    streams, low_base = _decode_audio_only(features)
    _write_json(
        output / "audio_only_candidates.json",
        {
            "candidate_versions": {
                "v1": LEGACY_CANDIDATE_GENERATION_VERSION,
                "v2": CANDIDATE_GENERATION_VERSION,
            },
            "streams": {
                key: [_note_row(note) for note in notes]
                for key, notes in streams.items()
            },
        },
    )

    score_events, reference = _independent_reference(sample, output)
    target = _targets(score_events)
    _write_json(output / "independent_score_events.json", score_events)

    sample_metadata = json.loads(
        (sample / "metadata.json").read_text(encoding="utf-8")
    )
    existing = json.loads(
        (sample / "note_alignment_v2.json").read_text(encoding="utf-8")
    )
    with wave.open(str(sample / "performance_audio.wav"), "rb") as handle:
        audio_duration = handle.getnframes() / handle.getframerate()
        sample_rate = handle.getframerate()
    audio, _ = librosa.load(
        sample / "performance_audio.wav", sr=sample_rate, mono=True
    )
    audio_features = _frame_audio_features(audio, sample_rate)

    stream_reports = {
        name: _stream_report(notes, target, features, score_events)
        for name, notes in streams.items()
    }
    canonical_pairs = match_notes(
        streams["canonical_frozen"],
        target,
        onset_tolerance_sec=0.100,
    )
    v2_pairs = match_notes(
        streams["joint_v2_all"],
        target,
        onset_tolerance_sec=0.100,
    )
    canonical_matched = {target_index for _pred, target_index in canonical_pairs}
    v2_matched = {target_index for _pred, target_index in v2_pairs}
    evidence_rows = [
        _evidence(event, features, audio_features) for event in score_events
    ]
    missed_short = []
    for index, (event, evidence) in enumerate(
        zip(score_events, evidence_rows)
    ):
        duration_ms = 1000.0 * (
            float(event["perf_end"]) - float(event["perf_start"])
        )
        if duration_ms >= 250.0 or index in canonical_matched:
            continue
        missed_short.append(
            {
                "score_event_index": index,
                "id": event["id"],
                "measure": event["measure"],
                "pitch": event["pitch"],
                "midi_written": event["midi"],
                "ref_start": event["ref_start"],
                "perf_start": event["perf_start"],
                "perf_end": event["perf_end"],
                "duration_ms": duration_ms,
                "canonical_matched_100ms": False,
                "v2_matched_100ms": index in v2_matched,
                "acoustic_evidence": evidence,
            }
        )

    sequence_pairs = _lcs_pairs(
        streams["canonical_frozen"], score_events
    )
    sequence_matched = {
        score_index for _pred_index, score_index in sequence_pairs
    }
    sequence_unmatched_short = [
        {
            "score_event_index": index,
            "id": event["id"],
            "measure": event["measure"],
            "pitch": event["pitch"],
            "midi_written": event["midi"],
            "perf_start_dtw_proxy": event["perf_start"],
            "perf_end_dtw_proxy": event["perf_end"],
            "duration_ms_dtw_proxy": 1000.0
            * (float(event["perf_end"]) - float(event["perf_start"])),
            "acoustic_evidence_at_dtw_proxy": evidence_rows[index],
        }
        for index, event in enumerate(score_events)
        if index not in sequence_matched
        and float(event["perf_end"]) - float(event["perf_start"]) < 0.250
    ]
    _write_json(
        output / "sequence_unmatched_short_events.upper_bound.json",
        {
            "warning": (
                "Timing-free LCS is optimistic for repeated motifs; DTW times "
                "and evidence are approximate because note timing is unverified."
            ),
            "events": sequence_unmatched_short,
        },
    )

    sweep = _sweep(low_base, features, target)
    _write_json(
        output / "rescue_merge_sweep.non_generalizable.json",
        {
            "warning": (
                "Single-sample diagnostic sensitivity only. Do not calibrate "
                "or promote these settings without validation-split evidence."
            ),
            "rows": sweep,
        },
    )
    _write_json(output / "missed_short_events.json", missed_short)
    confidence_sweep = _confidence_gate_sweep(
        streams["joint_v2_all"], score_events
    )
    _write_json(
        output / "confidence_gate_sweep.non_generalizable.json",
        {
            "warning": (
                "Single-sample, score-conditioned, timing-free upper-bound "
                "sensitivity only. Do not promote."
            ),
            "rows": confidence_sweep,
        },
    )

    existing_summary = dict(existing.get("summary") or {})
    current_alignment = sample / "alignment.npz"
    current_alignment_engine = None
    with np.load(current_alignment, allow_pickle=False) as saved:
        if "engine" in saved:
            current_alignment_engine = str(saved["engine"].item())
    report = {
        "schema_version": "align-real-audio-diagnostic-v1",
        "sample": str(sample),
        "locked_synthetic_test_touched": False,
        "audio": {
            "duration_sec": audio_duration,
            "sample_rate": sample_rate,
            "sha256": hashlib.sha256(
                (sample / "performance_audio.wav").read_bytes()
            ).hexdigest(),
            "trim": sample_metadata.get("performance_trim"),
        },
        "score": {
            "segment": sample_metadata.get("score_segment"),
            **reference,
            "timing_reference_quality": _timing_reference_quality(
                streams["canonical_frozen"], target
            ),
        },
        "cache": cache,
        "existing_artifacts": {
            "alignment_engine": current_alignment_engine,
            "joint_summary": existing_summary,
            "canonical_transcription_file_count": len(
                json.loads(
                    (sample / "transcription_notes.json").read_text(
                        encoding="utf-8"
                    )
                ).get("transcribed_notes")
                or []
            ),
            "labels_count": len(
                json.loads(
                    (sample / "labels.json").read_text(encoding="utf-8")
                ).get("labels")
                or []
            ),
        },
        "candidate_identity": {
            "joint_v1_count": len(streams["joint_v1_all"]),
            "joint_v2_count": len(streams["joint_v2_all"]),
            "v1_and_v2_are_identical": {
                (
                    int(value.pitch),
                    round(float(value.start), 6),
                    round(float(value.end), 6),
                )
                for value in streams["joint_v1_all"]
            }
            == {
                (
                    int(value.pitch),
                    round(float(value.start), 6),
                    round(float(value.end), 6),
                )
                for value in streams["joint_v2_all"]
            },
        },
        "streams": stream_reports,
        "canonical_detection_correlates_100ms": _correlates(
            score_events, canonical_matched, evidence_rows
        ),
        "missed_short_count_100ms": len(missed_short),
        "sequence_unmatched_short_count_upper_bound": len(
            sequence_unmatched_short
        ),
        "sweep": {
            "non_generalizable": True,
            "evaluated_settings": len(sweep),
            "best_five": sweep[:5],
            "confidence_gate": confidence_sweep,
        },
        "artifacts": {
            "audio_only_candidates": str(
                output / "audio_only_candidates.json"
            ),
            "independent_score_events": str(
                output / "independent_score_events.json"
            ),
            "missed_short_events": str(output / "missed_short_events.json"),
            "sequence_unmatched_short_events": str(
                output
                / "sequence_unmatched_short_events.upper_bound.json"
            ),
            "sweep": str(
                output / "rescue_merge_sweep.non_generalizable.json"
            ),
            "confidence_gate_sweep": str(
                output / "confidence_gate_sweep.non_generalizable.json"
            ),
        },
    }
    _write_json(output / "report.json", report)
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
