"""Cache-only candidate confidence evaluation on an explicit non-test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from alignmodel.joint.candidates import (
    CANDIDATE_GENERATION_VERSION,
    basic_pitch_candidate_union,
)
from alignmodel.joint.data import _validated_basic_pitch
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.validated_targets import target_note_map


TOLERANCES = (0.020, 0.050, 0.100)
DURATION_BUCKETS = (
    ("lt_80ms", 0.0, 0.080),
    ("lt_120ms", 0.0, 0.120),
    ("lt_180ms", 0.0, 0.180),
    ("lt_250ms", 0.0, 0.250),
    ("ge_250ms", 0.250, math.inf),
)
RELATIONSHIPS = ("match", "copy", "extra", "substitute")
RELIABILITY_BINS = (
    ("0.45_0.50", 0.45, 0.50),
    ("0.50_0.55", 0.50, 0.55),
    ("0.55_0.60", 0.55, 0.60),
    ("0.60_0.65", 0.60, 0.65),
)


def _event(candidate: Any) -> JointEvent:
    return JointEvent(
        pitch=int(candidate.pitch),
        start=float(candidate.start),
        end=float(candidate.end),
        score_span=None,
        relationship="extra",
        confidence=float(candidate.confidence),
    )


def _prf(correct: int, predicted: int, target: int) -> dict[str, float]:
    precision = correct / predicted if predicted else float(target == 0)
    recall = correct / target if target else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _sample_counts(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
    tolerance: float,
) -> dict[str, Any]:
    pairs = pair_exact_pitch_onset(
        predicted, target, tolerance_sec=tolerance
    )
    matched_target = {target_index for _candidate, target_index in pairs}
    linked_target = {
        index for index, event in enumerate(target) if not event.is_extra
    }
    extra_target = {
        index for index, event in enumerate(target) if event.is_extra
    }
    relationship_target = Counter(
        str(event.relationship) for event in target
    )
    relationship_matched = Counter(
        str(target[target_index].relationship)
        for _candidate, target_index in pairs
    )
    duration_target: Counter[str] = Counter()
    duration_matched: Counter[str] = Counter()
    for index, event in enumerate(target):
        duration = float(event.end) - float(event.start)
        for name, minimum, maximum in DURATION_BUCKETS:
            if minimum <= duration < maximum:
                duration_target[name] += 1
                if index in matched_target:
                    duration_matched[name] += 1
    return {
        "predicted": len(predicted),
        "target": len(target),
        "matched": len(pairs),
        "unmatched_predictions": len(predicted) - len(pairs),
        "linked_target": len(linked_target),
        "linked_matched": len(matched_target & linked_target),
        "extra_target": len(extra_target),
        "extra_matched": len(matched_target & extra_target),
        "relationship_target": dict(relationship_target),
        "relationship_matched": dict(relationship_matched),
        "duration_target": dict(duration_target),
        "duration_matched": dict(duration_matched),
    }


def _sum_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "predicted",
        "target",
        "matched",
        "unmatched_predictions",
        "linked_target",
        "linked_matched",
        "extra_target",
        "extra_matched",
    )
    rows = list(rows)
    result: dict[str, Any] = {
        key: sum(int(row[key]) for row in rows) for key in scalar_names
    }
    for field in (
        "relationship_target",
        "relationship_matched",
        "duration_target",
        "duration_matched",
    ):
        total: Counter[str] = Counter()
        for row in rows:
            total.update(
                {str(key): int(value) for key, value in row[field].items()}
            )
        result[field] = dict(total)
    return result


def _report_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    matched = int(counts["matched"])
    predicted = int(counts["predicted"])
    target = int(counts["target"])
    relationship_target = counts["relationship_target"]
    relationship_matched = counts["relationship_matched"]
    duration_target = counts["duration_target"]
    duration_matched = counts["duration_matched"]
    return {
        "note": {
            **_prf(matched, predicted, target),
            "matched": matched,
            "predicted": predicted,
            "target": target,
            "predicted_target_ratio": predicted / max(target, 1),
        },
        "score_linked_event_recall": int(counts["linked_matched"])
        / max(int(counts["linked_target"]), 1),
        "extra_note_false_positive_rate": int(
            counts["unmatched_predictions"]
        )
        / max(predicted, 1),
        "relationship_recall": {
            relationship: {
                "matched": int(relationship_matched.get(relationship, 0)),
                "target": int(relationship_target.get(relationship, 0)),
                "recall": int(relationship_matched.get(relationship, 0))
                / max(int(relationship_target.get(relationship, 0)), 1),
            }
            for relationship in RELATIONSHIPS
        },
        "duration_bucket_recall": {
            name: {
                "matched": int(duration_matched.get(name, 0)),
                "target": int(duration_target.get(name, 0)),
                "recall": int(duration_matched.get(name, 0))
                / max(int(duration_target.get(name, 0)), 1),
            }
            for name, _minimum, _maximum in DURATION_BUCKETS
        },
        "oracle_mapping_ceiling": {
            "paired_event_count": matched,
            "joint_f1_with_oracle_mapping": _prf(
                matched, predicted, target
            )["f1"],
            "score_linked_mapping_recall": int(counts["linked_matched"])
            / max(int(counts["linked_target"]), 1),
            "conditional_mapping_accuracy": 1.0 if matched else None,
            "copy_coverage": int(
                relationship_matched.get("copy", 0)
            )
            / max(int(relationship_target.get("copy", 0)), 1),
        },
    }


def _bootstrap(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(sample_rows), len(sample_rows))
        counts = _sum_counts(sample_rows[index] for index in indices)
        values.append(
            _prf(
                int(counts["matched"]),
                int(counts["predicted"]),
                int(counts["target"]),
            )["f1"]
        )
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.500)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _reliability(
    candidates: Sequence[JointEvent],
    target: Sequence[JointEvent],
) -> list[dict[str, float | int | str | None]]:
    pairs = pair_exact_pitch_onset(
        candidates, target, tolerance_sec=0.050
    )
    matched = {candidate_index for candidate_index, _target in pairs}
    rows = []
    for name, minimum, maximum in RELIABILITY_BINS:
        selected = [
            index
            for index, candidate in enumerate(candidates)
            if minimum <= float(candidate.confidence) < maximum
        ]
        positives = sum(index in matched for index in selected)
        mean_confidence = (
            float(np.mean([candidates[index].confidence for index in selected]))
            if selected
            else None
        )
        empirical_precision = positives / len(selected) if selected else None
        rows.append(
            {
                "bin": name,
                "minimum": minimum,
                "maximum_exclusive": maximum,
                "count": len(selected),
                "matched_50ms": positives,
                "mean_confidence": mean_confidence,
                "empirical_precision": empirical_precision,
                "calibration_gap": (
                    mean_confidence - empirical_precision
                    if mean_confidence is not None
                    and empirical_precision is not None
                    else None
                ),
            }
        )
    return rows


def _aggregate_reliability(
    samples: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, float | int | str | None]]:
    output = []
    for index, (name, minimum, maximum) in enumerate(RELIABILITY_BINS):
        count = sum(int(rows[index]["count"]) for rows in samples)
        matched = sum(int(rows[index]["matched_50ms"]) for rows in samples)
        weighted_confidence = sum(
            int(rows[index]["count"])
            * float(rows[index]["mean_confidence"] or 0.0)
            for rows in samples
        )
        mean_confidence = weighted_confidence / count if count else None
        empirical_precision = matched / count if count else None
        output.append(
            {
                "bin": name,
                "minimum": minimum,
                "maximum_exclusive": maximum,
                "count": count,
                "matched_50ms": matched,
                "mean_confidence": mean_confidence,
                "empirical_precision": empirical_precision,
                "calibration_gap": (
                    mean_confidence - empirical_precision
                    if mean_confidence is not None
                    and empirical_precision is not None
                    else None
                ),
            }
        )
    return output


def _breakdown(
    sample_results: Sequence[Mapping[str, Any]],
    threshold: float,
    tolerance: float,
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_results:
        groups[str(row[field])].append(
            row["thresholds"][str(threshold)][f"{round(tolerance * 1000)}ms"]
        )
    return {
        group: {
            "samples": len(
                [row for row in sample_results if str(row[field]) == group]
            ),
            **_report_counts(_sum_counts(rows)),
        }
        for group, rows in sorted(groups.items())
    }


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.55, 0.60, 0.65),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args()

    split_name = str(args.split)
    if any(term in split_name.lower() for term in ("test", "sealed")):
        raise ValueError("This diagnostic is validation-only")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_rows = manifest.get(split_name)
    if not isinstance(all_rows, list) or not all_rows:
        raise ValueError(f"Missing non-empty split {split_name!r}")
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    sample_results = []

    for position, row in enumerate(rows, 1):
        features = _validated_basic_pitch(row, args.cache_root)
        candidates = tuple(
            _event(candidate)
            for candidate in basic_pitch_candidate_union(
                features, minimum_confidence=0.0
            )
        )

        # Gold timing and lineage are intentionally loaded after audio-only decode.
        target_index = ScoreEventIndex.from_musicxml(
            Path(str(row["sample_dir"])) / "verified_score.musicxml",
            target_note_map(row),
        )
        target = target_index.rendered_events
        threshold_rows = {}
        for threshold in thresholds:
            selected = tuple(
                event
                for event in candidates
                if float(event.confidence) >= threshold
            )
            threshold_rows[str(threshold)] = {
                f"{round(tolerance * 1000)}ms": _sample_counts(
                    selected, target, tolerance
                )
                for tolerance in TOLERANCES
            }
        sample_results.append(
            {
                "sample": str(row.get("sample")),
                "corpus": str(row.get("corpus") or "unknown"),
                "source": str(row.get("source") or "unknown"),
                "source_group": str(row.get("source_group") or "unknown"),
                "audio_render": str(row.get("audio_render") or "unknown"),
                "repeated": bool(row.get("repeated")),
                "candidate_count_all": len(candidates),
                "target_count": len(target),
                "reliability": _reliability(candidates, target),
                "thresholds": threshold_rows,
            }
        )
        if position == 1 or position % 10 == 0 or position == len(rows):
            print(f"evaluated {position}/{len(rows)}", flush=True)

    aggregate = {}
    for threshold in thresholds:
        threshold_key = str(threshold)
        by_tolerance = {}
        for tolerance in TOLERANCES:
            tolerance_key = f"{round(tolerance * 1000)}ms"
            selected = [
                row["thresholds"][threshold_key][tolerance_key]
                for row in sample_results
            ]
            by_tolerance[tolerance_key] = {
                **_report_counts(_sum_counts(selected)),
                "bootstrap_note_f1": _bootstrap(
                    selected,
                    seed=args.seed + int(threshold * 1000) + round(tolerance * 1000),
                    replicates=args.bootstrap_replicates,
                ),
            }
        aggregate[threshold_key] = {
            "tolerances": by_tolerance,
            "per_corpus_50ms": _breakdown(
                sample_results, threshold, 0.050, "corpus"
            ),
            "per_render_50ms": _breakdown(
                sample_results, threshold, 0.050, "audio_render"
            ),
            "per_source_50ms": _breakdown(
                sample_results, threshold, 0.050, "source"
            ),
        }

    report = {
        "schema_version": "align-candidate-confidence-calibration-v2",
        "diagnostic_only": True,
        "production_defaults_changed": False,
        "locked_test_inspected": False,
        "candidate_generation": CANDIDATE_GENERATION_VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _manifest_sha256(args.manifest),
        "split": split_name,
        "selection": {
            "method": "deterministic manifest prefix; audit round-robin ordering",
            "selected": len(rows),
            "available": len(all_rows),
            "corpus": dict(Counter(row["corpus"] for row in sample_results)),
            "audio_render": dict(
                Counter(row["audio_render"] for row in sample_results)
            ),
            "repeated": sum(bool(row["repeated"]) for row in sample_results),
        },
        "thresholds": list(thresholds),
        "reference": {
            "threshold": 0.65,
            "validation_100_candidate_note_f1_50ms": 0.847196,
            "sample_007_sequence_coverage": {
                "0.50": 0.8739495798319328,
                "0.55": 0.8067226890756303,
                "0.60": 0.7142857142857143,
                "0.65": 0.5630252100840336,
            },
        },
        "aggregate": aggregate,
        "confidence_reliability_45_65_at_50ms": _aggregate_reliability(
            [row["reliability"] for row in sample_results]
        ),
        "samples": sample_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
