from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from alignmodel.joint.candidate_rescorer import (
    load_candidate_rescorer,
    rescore_candidates,
)
from alignmodel.joint.index import JointEvent
from alignmodel.joint.lattice import (
    JointCandidate,
    JointOperation,
    LatticeConfig,
    SparseJointLattice,
)
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.outputraw_full import (
    FullPipelinePrediction,
    infer_full_pipeline,
    load_checkpoint,
    verify_data_ready,
)
from alignmodel.joint.outputraw_metrics import (
    FullPipelineMetricSample,
    evaluate_full_pipeline,
)
from alignmodel.joint.outputraw_train import (
    _target_layer2,
    _target_resume_events,
)
from alignmodel.joint.packed_data import PackedJointDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _resume_events(prediction: FullPipelinePrediction) -> tuple[int, ...]:
    return tuple(
        int(step.resume_event)
        for step in prediction.path.steps
        if step.structural_operation == JointOperation.REPEAT_ENTER
        and step.resume_event is not None
    )


def _rhythm_target(packed: Any, count: int) -> tuple[bool, ...]:
    target = [False] * count
    for row in packed.target.get("layer3_rhythm") or []:
        index = row.get("rendered_event")
        if index is not None and 0 <= int(index) < count:
            target[int(index)] = bool(row.get("rhythm_error"))
    return tuple(target)


def _sample(
    prediction: FullPipelinePrediction,
    target: Sequence[JointEvent],
    *,
    target_resume: Sequence[int],
    target_rhythm: Sequence[bool],
    target_deletions: frozenset[int],
    score_event_count: int,
    source: str,
) -> FullPipelineMetricSample:
    return FullPipelineMetricSample(
        predicted=prediction.events,
        target=target,
        predicted_layer2=prediction.layer2_types,
        target_layer2=_target_layer2(target),
        predicted_rhythm=tuple(
            probability >= 0.5
            for probability in prediction.rhythm_probabilities
        ),
        target_rhythm=target_rhythm,
        predicted_duration_sec=prediction.corrected_durations_sec,
        predicted_deletions=frozenset(prediction.missed_score_events),
        target_deletions=target_deletions,
        predicted_resume_events=_resume_events(prediction),
        target_resume_events=target_resume,
        score_event_count=score_event_count,
        source=source,
    )


def _nearest_acoustics(
    target: JointEvent,
    candidates: Sequence[JointCandidate],
) -> tuple[float, tuple[float, ...]]:
    same_pitch = [candidate for candidate in candidates if candidate.pitch == target.pitch]
    pool = same_pitch or list(candidates)
    if not pool:
        return 1.0, (0.0,) * 5
    nearest = min(pool, key=lambda candidate: abs(candidate.start - target.start))
    return nearest.confidence, nearest.acoustic_features


def _oracle_note_candidates(
    target: Sequence[JointEvent],
    candidates: Sequence[JointCandidate],
) -> tuple[JointCandidate, ...]:
    rows = []
    for event in target:
        confidence, acoustics = _nearest_acoustics(event, candidates)
        rows.append(
            JointCandidate(
                pitch=event.pitch,
                start=event.start,
                end=event.end,
                confidence=confidence,
                acoustic_features=acoustics,
            )
        )
    return tuple(rows)


def _project_oracle_structure(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
) -> tuple[JointEvent, ...]:
    pairs = pair_exact_pitch_onset(predicted, target, tolerance_sec=0.050)
    by_prediction = {left: right for left, right in pairs}
    return tuple(
        replace(
            event,
            score_span=target[by_prediction[index]].score_span,
            relationship=target[by_prediction[index]].relationship,
            copy_pass=target[by_prediction[index]].copy_pass,
            origin_relationship=target[by_prediction[index]].origin_relationship,
        )
        if index in by_prediction
        else replace(event, score_span=None, relationship="extra", copy_pass=0)
        for index, event in enumerate(predicted)
    )


def _project_oracle_copy_states(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
) -> tuple[JointEvent, ...]:
    pairs = pair_exact_pitch_onset(predicted, target, tolerance_sec=0.050)
    by_prediction = {left: right for left, right in pairs}
    rows = []
    for index, event in enumerate(predicted):
        target_index = by_prediction.get(index)
        if target_index is None:
            rows.append(event)
            continue
        gold = target[target_index]
        if gold.is_copy and event.score_span is not None:
            rows.append(
                replace(
                    event,
                    relationship="copy",
                    copy_pass=gold.copy_pass,
                    origin_relationship=gold.origin_relationship,
                )
            )
        elif not gold.is_copy and event.is_copy:
            rows.append(replace(event, relationship="match", copy_pass=0))
        else:
            rows.append(event)
    return tuple(rows)


def _with_events(
    sample: FullPipelineMetricSample,
    events: Sequence[JointEvent],
    *,
    oracle_resume: bool = False,
) -> FullPipelineMetricSample:
    target_layer2 = _target_layer2(events)
    return replace(
        sample,
        predicted=tuple(events),
        predicted_layer2=target_layer2,
        predicted_rhythm=tuple(False for _ in events),
        predicted_duration_sec=tuple(event.end - event.start for event in events),
        predicted_resume_events=(
            sample.target_resume_events
            if oracle_resume
            else sample.predicted_resume_events
        ),
    )


def _first_pass_counts(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
) -> tuple[int, int, int]:
    pairs = pair_exact_pitch_onset(predicted, target, tolerance_sec=0.050)
    target_first = {
        index
        for index, event in enumerate(target)
        if not event.is_copy and not event.is_extra
    }
    predicted_first = {
        index
        for index, event in enumerate(predicted)
        if not event.is_copy and not event.is_extra
    }
    correct = sum(
        left in predicted_first
        and right in target_first
        and predicted[left].score_span == target[right].score_span
        for left, right in pairs
    )
    return correct, len(predicted_first), len(target_first)


def _prf(counts: Sequence[int]) -> dict[str, float | int]:
    correct, predicted, target = counts
    precision = correct / max(predicted, 1)
    recall = correct / max(target, 1)
    return {
        "correct": correct,
        "predicted": predicted,
        "target": target,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-rescorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ready = verify_data_ready(args.ready_marker)
    fingerprint = str(ready["hashes"]["pack_id"])
    checkpoint_hash = _sha256(args.checkpoint)
    model, _payload = load_checkpoint(
        args.checkpoint,
        device="cpu",
        expected_data_fingerprint=fingerprint,
    )
    if _sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Full checkpoint changed while loading")
    model.config = replace(model.config, residual_scale=args.residual_scale)
    model.eval()
    rescorer_hash = _sha256(args.candidate_rescorer)
    rescorer, threshold, rescorer_payload = load_candidate_rescorer(
        args.candidate_rescorer,
        device="cpu",
    )
    if _sha256(args.candidate_rescorer) != rescorer_hash:
        raise RuntimeError("Candidate checkpoint changed while loading")
    if float(rescorer_payload["validation"]["f1"]) < 0.85:
        raise ValueError("Candidate rescorer failed validation gate")

    lattice_config = LatticeConfig(
        max_options_per_candidate=12,
        max_states=48,
        max_delete_events=24,
        noise_inference_bias=-6.0,
        continuation_feature_enabled=True,
        continuation_score_weight=0.35,
        continuation_hard_negative_copies=1,
        repeat_fragment_penalty=0.25,
    )
    lattice = SparseJointLattice(model, lattice_config)
    baseline = []
    oracle_notes = []
    oracle_structure = []
    oracle_copy = []
    first_pass = [0, 0, 0]
    started = time.perf_counter()
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        ordinals = dataset.ordinals("val")
        for position, ordinal in enumerate(ordinals, 1):
            packed = dataset[ordinal]
            example = packed.training_example()
            target_resume = _target_resume_events(example.target_events, lattice)
            target_rhythm = _rhythm_target(packed, len(example.target_events))
            gated_candidates = rescore_candidates(
                rescorer,
                example.candidates,
                threshold=threshold,
                score=example.score,
            )
            prediction = infer_full_pipeline(
                model,
                lattice,
                gated_candidates,
                example.score,
            )
            baseline_sample = _sample(
                prediction,
                example.target_events,
                target_resume=target_resume,
                target_rhythm=target_rhythm,
                target_deletions=example.target_deletions,
                score_event_count=len(example.score),
                source=packed.source,
            )
            baseline.append(baseline_sample)

            oracle_prediction = infer_full_pipeline(
                model,
                lattice,
                _oracle_note_candidates(
                    example.target_events,
                    example.candidates,
                ),
                example.score,
            )
            oracle_notes.append(
                _sample(
                    oracle_prediction,
                    example.target_events,
                    target_resume=target_resume,
                    target_rhythm=target_rhythm,
                    target_deletions=example.target_deletions,
                    score_event_count=len(example.score),
                    source=packed.source,
                )
            )
            oracle_structure.append(
                _with_events(
                    baseline_sample,
                    _project_oracle_structure(
                        prediction.events,
                        example.target_events,
                    ),
                    oracle_resume=True,
                )
            )
            oracle_copy.append(
                _with_events(
                    baseline_sample,
                    _project_oracle_copy_states(
                        prediction.events,
                        example.target_events,
                    ),
                )
            )
            counts = _first_pass_counts(
                prediction.events,
                example.target_events,
            )
            first_pass = [
                left + right for left, right in zip(first_pass, counts)
            ]
            if position == 1 or position % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = position / max(elapsed, 1e-9)
                print(
                    f"oracle={position}/{len(ordinals)} "
                    f"rows_per_sec={rate:.3f} "
                    f"eta_sec={(len(ordinals) - position) / max(rate, 1e-9):.1f}",
                    flush=True,
                )

    def metrics(samples: Sequence[FullPipelineMetricSample]) -> dict[str, Any]:
        return evaluate_full_pipeline(
            samples,
            bootstrap_replicates=args.bootstrap_replicates,
        )

    report = {
        "schema_version": "align-outputraw-structural-oracles-v1",
        "metric_status": "official_note_wise",
        "data_fingerprint": fingerprint,
        "validation_rows": len(baseline),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "residual_scale": args.residual_scale,
        },
        "candidate_rescorer": {
            "path": str(args.candidate_rescorer.resolve()),
            "sha256": rescorer_hash,
            "threshold": threshold,
            "frozen": True,
        },
        "baseline": metrics(baseline),
        "oracle_notes_predicted_path": metrics(oracle_notes),
        "predicted_notes_oracle_source_repeat_resume": metrics(oracle_structure),
        "oracle_copy_states_only": metrics(oracle_copy),
        "first_pass_only_mapping": _prf(first_pass),
        "elapsed_seconds": time.perf_counter() - started,
        "locked_test_touched": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
