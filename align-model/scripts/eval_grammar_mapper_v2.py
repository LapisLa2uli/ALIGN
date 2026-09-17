from __future__ import annotations

import argparse
import atexit
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from dataclasses import replace

import torch
import numpy as np
import joblib

from alignmodel.joint.grammar_mapper_v2 import (
    GrammarCosts,
    decode_grammar_mapper,
    operation_features,
    emission_features,
    grammar_hypotheses,
    plan_features,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.joint.sequence_mapper import (
    FixedScoreSequenceMapper,
    predict_mapper_scores,
    predict_operation_types,
)
from alignmodel.training_resources import resource_lease
from alignmodel.joint.operation_tagger import MelOperationTagger
from alignmodel.joint.sequence_mapper import sequence_tensors


def _candidates(events: tuple) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=event.pitch,
            start=event.start,
            end=event.end,
            confidence=1.0,
        )
        for event in events
    )


def _metric(samples: list[JointMetricSample]) -> dict:
    return evaluate_joint_dataset(samples)["aggregate"]["official_note_wise"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--substitution-cost", type=float, default=1.5)
    parser.add_argument("--extra-cost", type=float, default=1.0)
    parser.add_argument("--deletion-cost", type=float, default=1.0)
    parser.add_argument("--repeat-cost", type=float, default=0.75)
    parser.add_argument("--operation-checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--resource-status", type=Path)
    parser.add_argument("--substitutes-as-extra", action="store_true")
    parser.add_argument("--operation-core", type=Path)
    parser.add_argument("--operation-override-threshold", type=float, default=0.5)
    parser.add_argument("--operation-in-viterbi", action="store_true")
    parser.add_argument("--operation-weight", type=float, default=0.4)
    parser.add_argument("--sequence-emission-checkpoint", type=Path)
    parser.add_argument("--location-weight", type=float, default=0.2)
    parser.add_argument("--emission-core", type=Path)
    parser.add_argument("--emission-weight", type=float, default=0.2)
    parser.add_argument("--operation-tagger", type=Path)
    parser.add_argument("--tagger-threshold", type=float, default=0.9)
    parser.add_argument("--plan-core", type=Path)
    parser.add_argument("--plan-weight", type=float, default=0.0)
    args = parser.parse_args()
    lease = None
    if args.resource_status is not None:
        lease = resource_lease(
            args.resource_status,
            "cpu_validation",
            track="mel-mapper-v2-full-note-wise",
            command=[str(value) for value in __import__("sys").argv],
            metadata={"split": args.split, "locked_test": False},
        )
        lease.__enter__()
        atexit.register(lease.__exit__, None, None, None)
    ready = verify_data_ready(args.ready_marker)
    costs = GrammarCosts(
        substitution=args.substitution_cost,
        extra=args.extra_cost,
        deletion=args.deletion_cost,
        repeat=args.repeat_cost,
    )
    manifest = json.loads(
        Path(str(ready["paths"]["manifest"])).read_text(encoding="utf-8")
    )
    source_root = Path(str(manifest["source_root"]))
    operation_model = None
    if args.operation_checkpoint is not None:
        payload = torch.load(
            args.operation_checkpoint, map_location="cpu", weights_only=False
        )
        operation_model = FixedScoreSequenceMapper(**payload["model_config"])
        operation_model.load_state_dict(payload["model_state_dict"])
        operation_model.eval()
    predictions = {}
    if args.predictions is not None:
        with args.predictions.open("r", encoding="utf-8") as stream:
            predictions = {
                row["sample"]: row["notes"]
                for row in (json.loads(line) for line in stream if line.strip())
            }
    operation_core = (
        joblib.load(args.operation_core)
        if args.operation_core is not None
        else None
    )
    sequence_model = None
    if args.sequence_emission_checkpoint is not None:
        payload = torch.load(
            args.sequence_emission_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        sequence_model = FixedScoreSequenceMapper(**payload["model_config"])
        sequence_model.load_state_dict(payload["model_state_dict"])
        sequence_model.eval()
    emission_core = (
        joblib.load(args.emission_core)
        if args.emission_core is not None
        else None
    )
    operation_tagger = None
    if args.operation_tagger is not None:
        payload = torch.load(
            args.operation_tagger, map_location="cpu", weights_only=False
        )
        operation_tagger = MelOperationTagger(**payload["model_config"])
        operation_tagger.load_state_dict(payload["model_state_dict"])
        operation_tagger.eval()
    plan_core = (
        joblib.load(args.plan_core) if args.plan_core is not None else None
    )
    samples = []
    by_copy_count = defaultdict(list)
    by_type = defaultdict(list)
    first_pass = []
    source_correct = resume_correct = repeat_rows = 0
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        ordinals = sorted(
            dataset.ordinals(args.split),
            key=lambda value: hashlib.sha256(
                f"{args.seed}:{value}".encode()
            ).digest(),
        )[: args.limit]
        for position, ordinal in enumerate(ordinals, 1):
            packed = dataset[ordinal]
            example = packed.training_example()
            score = ScoreEventIndex.from_musicxml(
                source_root / packed.sample / "verified_score.musicxml"
            ).events
            if len(score) != len(example.score):
                raise ValueError("Part-aware score index differs from packed index")
            candidates = (
                tuple(
                    JointCandidate(
                        pitch=int(row["pitch"]),
                        start=float(row["start"]),
                        end=float(row["end"]),
                        confidence=float(row.get("confidence", 1.0)),
                    )
                    for row in predictions[packed.sample]
                )
                if predictions
                else _candidates(example.target_events)
            )
            location_scores = (
                predict_mapper_scores(sequence_model, candidates, score)[0]
                if sequence_model is not None
                else None
            )
            forced_hypothesis = None
            plan_log_probabilities = None
            if plan_core is not None:
                hypotheses = grammar_hypotheses(score, len(candidates))
                features = np.stack(
                    [
                        plan_features(candidates, score, hypothesis)
                        for hypothesis in hypotheses
                    ]
                )
                probability = plan_core.predict_proba(features)[:, 1]
                if args.plan_weight > 0:
                    plan_log_probabilities = np.log(
                        probability.clip(1e-6, 1.0)
                    )
                else:
                    forced_hypothesis = hypotheses[int(np.argmax(probability))]
            emission_scores = None
            extra_scores = None
            if emission_core is not None:
                feature_rows = [
                    emission_features(
                        candidates, score, event_index, score_index, copy_pass
                    )
                    for event_index in range(len(candidates))
                    for score_index in range(len(score))
                    for copy_pass in (0, 1, 2)
                ]
                probability = emission_core.predict_proba(
                    np.stack(feature_rows)
                )[:, 1]
                emission_scores = np.log(
                    probability.clip(1e-6, 1.0)
                ).reshape(len(candidates), len(score), 3)
                extra_probability = emission_core.predict_proba(
                    np.stack(
                        [
                            emission_features(
                                candidates, score, event_index, -1, 0
                            )
                            for event_index in range(len(candidates))
                        ]
                    )
                )[:, 1]
                extra_scores = np.log(extra_probability.clip(1e-6, 1.0))
            mapped, grammar = decode_grammar_mapper(
                candidates,
                score,
                costs=costs,
                location_log_probabilities=location_scores,
                location_weight=args.location_weight,
                emission_log_probabilities=emission_scores,
                extra_log_probabilities=extra_scores,
                emission_weight=args.emission_weight,
                forced_hypothesis=forced_hypothesis,
                plan_log_probabilities=plan_log_probabilities,
                plan_weight=args.plan_weight,
            )
            if args.substitutes_as_extra:
                mapped = tuple(
                    replace(
                        event,
                        score_span=None,
                        relationship="extra",
                        rendered_index=index,
                    )
                    if event.relationship == "substitute"
                    else event
                    for index, event in enumerate(mapped)
                )
            if operation_core is not None:
                probabilities = operation_core.predict_proba(
                    np.stack(
                        [
                            operation_features(
                                candidates, mapped, score, index, grammar
                            )
                            for index in range(len(candidates))
                        ]
                    )
                )
                if args.operation_in_viterbi:
                    mapped, grammar = decode_grammar_mapper(
                        candidates,
                        score,
                        costs=costs,
                        operation_probabilities=probabilities,
                        operation_weight=args.operation_weight,
                    )
                    probabilities = None
                adjusted = []
                for index, event in enumerate(mapped):
                    if probabilities is None:
                        adjusted.append(event)
                        continue
                    probability = probabilities[index]
                    predicted_type = int(np.argmax(probability))
                    kind = ("match", "copy", "substitute", "extra")[predicted_type]
                    decisive = (
                        float(probability[predicted_type])
                        >= args.operation_override_threshold
                    )
                    if (
                        not decisive
                        or event.is_copy
                        or event.is_extra
                        or kind == "copy"
                    ):
                        adjusted.append(event)
                    elif kind == "extra":
                        adjusted.append(
                            replace(
                                event,
                                score_span=None,
                                relationship="extra",
                                copy_pass=0,
                                rendered_index=index,
                            )
                        )
                    else:
                        adjusted.append(
                            replace(event, relationship=kind, copy_pass=0)
                        )
                mapped = tuple(adjusted)
            if operation_tagger is not None:
                pitch, continuous = sequence_tensors(candidates)
                with torch.inference_mode():
                    probabilities = operation_tagger(
                        pitch[None], continuous[None]
                    )[0].softmax(-1).numpy()
                adjusted = []
                for index, (event, probability) in enumerate(
                    zip(mapped, probabilities)
                ):
                    if event.is_copy or event.is_extra:
                        adjusted.append(event)
                    elif (
                        probability[3] >= args.tagger_threshold
                    ):
                        adjusted.append(
                            replace(
                                event,
                                score_span=None,
                                relationship="extra",
                                copy_pass=0,
                                rendered_index=index,
                            )
                        )
                    elif probability[2] >= args.tagger_threshold:
                        adjusted.append(
                            replace(
                                event,
                                relationship="substitute",
                                copy_pass=0,
                            )
                        )
                    else:
                        adjusted.append(event)
                mapped = tuple(adjusted)
            if operation_model is not None:
                predicted_types = predict_operation_types(
                    operation_model, candidates, score
                )
                adjusted = []
                for index, (event, predicted_type) in enumerate(
                    zip(mapped, predicted_types)
                ):
                    if event.is_copy:
                        adjusted.append(event)
                    elif event.is_extra or predicted_type == "extra":
                        adjusted.append(
                            replace(
                                event,
                                score_span=None,
                                relationship="extra",
                                copy_pass=0,
                                rendered_index=index,
                            )
                        )
                    else:
                        adjusted.append(
                            replace(
                                event,
                                relationship=(
                                    "substitute"
                                    if predicted_type == "substitute"
                                    else "match"
                                ),
                            )
                        )
                mapped = tuple(adjusted)
            sample = JointMetricSample(
                predicted=mapped,
                target=example.target_events,
                source=packed.source,
                score_event_count=len(score),
            )
            samples.append(sample)
            copy_count = max(
                (event.copy_pass for event in example.target_events), default=0
            )
            by_copy_count[str(copy_count)].append(sample)
            predicted_first = tuple(event for event in mapped if not event.is_copy)
            target_first = tuple(
                event for event in example.target_events if not event.is_copy
            )
            first_pass.append(
                JointMetricSample(
                    predicted=predicted_first,
                    target=target_first,
                    source=packed.source,
                    score_event_count=len(score),
                )
            )
            for kind in ("match", "copy", "substitute", "extra"):
                predicted = tuple(
                    event
                    for event in mapped
                    if ("copy" if event.is_copy else event.relationship) == kind
                )
                target = tuple(
                    event
                    for event in example.target_events
                    if ("copy" if event.is_copy else event.relationship) == kind
                )
                by_type[kind].append(
                    JointMetricSample(
                        predicted=predicted,
                        target=target,
                        source=packed.source,
                        score_event_count=len(score),
                    )
                )
            repeats = [
                row
                for row in (packed.target.get("layer1_repeats") or [])
                if row.get("source_span") is not None
            ]
            if repeats:
                repeat_rows += 1
                expected_source = tuple(repeats[0]["source_span"])
                source_correct += int(grammar["source_span"] == expected_source)
                expected_resume = int(repeats[-1]["resume_event"])
                predicted_resume = (
                    grammar["source_span"][1]
                    if grammar["source_span"] is not None
                    else -1
                )
                resume_correct += int(predicted_resume == expected_resume)
            if position == 1 or position % 25 == 0:
                print(f"grammar={position}/{len(ordinals)}", flush=True)
    per_clip = []
    for sample in samples:
        metric = _metric([sample])
        per_clip.append(
            (
                float(metric["credit"]),
                int(metric["predicted"]),
                int(metric["gold"]),
            )
        )
    generator = np.random.default_rng(args.seed)
    bootstrap = []
    for _ in range(args.bootstrap_replicates):
        selected = generator.integers(0, len(per_clip), len(per_clip))
        credit = sum(per_clip[index][0] for index in selected)
        predicted = sum(per_clip[index][1] for index in selected)
        gold = sum(per_clip[index][2] for index in selected)
        precision = credit / max(predicted, 1)
        recall = credit / max(gold, 1)
        bootstrap.append(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    report = {
        "schema_version": "align-grammar-mapper-v2-calibration-v1",
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "split": args.split,
        "selection": "deterministic hash-ordered group-disjoint train calibration",
        "rows": len(samples),
        "input": "frozen_mel" if predictions else "oracle_notes",
        "costs": costs.__dict__,
        "operation_checkpoint": (
            str(args.operation_checkpoint.resolve())
            if args.operation_checkpoint is not None
            else None
        ),
        "substitutes_as_extra": args.substitutes_as_extra,
        "operation_core": (
            str(args.operation_core.resolve())
            if args.operation_core is not None
            else None
        ),
        "operation_override_threshold": args.operation_override_threshold,
        "operation_in_viterbi": args.operation_in_viterbi,
        "operation_weight": args.operation_weight,
        "sequence_emission_checkpoint": (
            str(args.sequence_emission_checkpoint.resolve())
            if args.sequence_emission_checkpoint is not None
            else None
        ),
        "location_weight": args.location_weight,
        "emission_core": (
            str(args.emission_core.resolve())
            if args.emission_core is not None
            else None
        ),
        "emission_weight": args.emission_weight,
        "operation_tagger": (
            str(args.operation_tagger.resolve())
            if args.operation_tagger is not None
            else None
        ),
        "tagger_threshold": args.tagger_threshold,
        "plan_core": (
            str(args.plan_core.resolve())
            if args.plan_core is not None
            else None
        ),
        "plan_weight": args.plan_weight,
        "overall": _metric(samples),
        "bootstrap_95": {
            "replicates": args.bootstrap_replicates,
            "lower": float(np.quantile(bootstrap, 0.025)),
            "median": float(np.quantile(bootstrap, 0.5)),
            "upper": float(np.quantile(bootstrap, 0.975)),
        },
        "first_pass_only": _metric(first_pass),
        "by_copy_count": {
            name: {"rows": len(rows), "metric": _metric(rows)}
            for name, rows in by_copy_count.items()
        },
        "by_type": {
            name: {"support_rows": len(rows), "metric": _metric(rows)}
            for name, rows in by_type.items()
        },
        "repeat_structure": {
            "rows": repeat_rows,
            "source_accuracy": source_correct / max(repeat_rows, 1),
            "resume_accuracy": resume_correct / max(repeat_rows, 1),
        },
        "timestamp_metrics_used": False,
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if lease is not None:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


if __name__ == "__main__":
    main()
