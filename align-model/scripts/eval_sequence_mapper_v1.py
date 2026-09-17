from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.mel_mapper import ContextualRepeatMapper
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.joint.sequence_mapper import (
    FixedScoreSequenceMapper,
    decode_sequence_mapper,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidates(rows: list[dict]) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=int(row["pitch"]),
            start=float(row["start"]),
            end=float(row["end"]),
            confidence=float(row.get("confidence", 1.0)),
        )
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--edge-checkpoint", type=Path)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FixedScoreSequenceMapper(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    edge_model = None
    if args.edge_checkpoint is not None:
        edge_payload = torch.load(
            args.edge_checkpoint, map_location="cpu", weights_only=False
        )
        edge_model = ContextualRepeatMapper(
            **edge_payload["model_config"]
        )
        edge_model.load_state_dict(edge_payload["model_state_dict"])
        edge_model.eval()
    predictions = {}
    if args.predictions is not None:
        with args.predictions.open("r", encoding="utf-8") as stream:
            predictions = {
                row["sample"]: row["notes"]
                for row in (json.loads(line) for line in stream if line.strip())
            }
    samples = []
    decomposition = {
        "events": 0,
        "start_correct": 0,
        "span_correct": 0,
        "type_correct": 0,
        "identity_and_type_correct": 0,
    }
    by_type = {}
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        for ordinal in dataset.ordinals(args.split)[: args.limit]:
            packed = dataset[ordinal]
            example = packed.training_example()
            candidates = (
                _candidates(predictions[packed.sample])
                if predictions
                else _candidates(
                    [
                        {
                            "pitch": event.pitch,
                            "start": event.start,
                            "end": event.end,
                        }
                        for event in example.target_events
                    ]
                )
            )
            mapped = decode_sequence_mapper(
                model,
                candidates,
                example.score,
                edge_model=edge_model,
            )
            for predicted, target in zip(mapped, example.target_events):
                decomposition["events"] += 1
                predicted_start = (
                    predicted.score_span[0]
                    if predicted.score_span is not None
                    else None
                )
                target_start = (
                    target.score_span[0]
                    if target.score_span is not None
                    else None
                )
                decomposition["start_correct"] += int(
                    predicted_start == target_start
                )
                decomposition["span_correct"] += int(
                    predicted.score_span == target.score_span
                )
                predicted_type = "copy" if predicted.is_copy else predicted.relationship
                target_type = "copy" if target.is_copy else target.relationship
                type_row = by_type.setdefault(
                    target_type,
                    {"events": 0, "start_correct": 0, "span_correct": 0, "type_correct": 0},
                )
                type_row["events"] += 1
                type_row["start_correct"] += int(predicted_start == target_start)
                type_row["span_correct"] += int(
                    predicted.score_span == target.score_span
                )
                type_row["type_correct"] += int(predicted_type == target_type)
                decomposition["type_correct"] += int(
                    predicted_type == target_type
                )
                decomposition["identity_and_type_correct"] += int(
                    predicted.score_span == target.score_span
                    and predicted_type == target_type
                )
            samples.append(
                JointMetricSample(
                    predicted=mapped,
                    target=example.target_events,
                    source=packed.source,
                    score_event_count=len(example.score),
                )
            )
    report = {
        "schema_version": "align-sequence-mapper-eval-v1",
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "edge_checkpoint": (
            {
                "path": str(args.edge_checkpoint.resolve()),
                "sha256": _sha256(args.edge_checkpoint),
            }
            if args.edge_checkpoint is not None
            else None
        ),
        "input": "mel_predictions" if predictions else "oracle_notes",
        "rows": len(samples),
        "split": args.split,
        "metrics": evaluate_joint_dataset(samples),
        "decomposition": {
            **decomposition,
            **{
                f"{name}_accuracy": value / max(decomposition["events"], 1)
                for name, value in decomposition.items()
                if name != "events"
            },
        },
        "decomposition_by_target_type": {
            name: {
                **values,
                "start_accuracy": values["start_correct"] / max(values["events"], 1),
                "span_accuracy": values["span_correct"] / max(values["events"], 1),
                "type_accuracy": values["type_correct"] / max(values["events"], 1),
            }
            for name, values in by_type.items()
        },
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
