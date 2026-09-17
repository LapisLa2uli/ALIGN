from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.mel_mapper import (
    decode_anchor_mapper,
    decode_segmental_mapper,
)
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset


def _candidates(rows: list[dict]) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=int(row["pitch"]),
            start=float(row["start"]),
            end=float(row["end"]),
            confidence=float(row["confidence"]),
        )
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument(
        "--decoder", choices=("anchor", "segmental"), default="anchor"
    )
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    predictions = {}
    if args.predictions is not None:
        with args.predictions.open("r", encoding="utf-8") as stream:
            predictions = {
                row["sample"]: row["notes"]
                for row in (json.loads(line) for line in stream if line.strip())
            }
    samples = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        ordinals = dataset.ordinals("val")[: args.limit]
        for ordinal in ordinals:
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
                            "confidence": 1.0,
                        }
                        for event in example.target_events
                    ]
                )
            )
            mapped = (
                decode_segmental_mapper(candidates, example.score)
                if args.decoder == "segmental"
                else decode_anchor_mapper(candidates, example.score)
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
        "schema_version": "align-mel-anchor-mapper-eval-v1",
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "input": "mel_predictions" if predictions else "oracle_notes",
        "decoder": args.decoder,
        "rows": len(samples),
        "metrics": evaluate_joint_dataset(samples),
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
