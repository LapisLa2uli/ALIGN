from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.joint.baseline import current_note_aligner_baseline
from alignmodel.joint.data import build_inference_example
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.validated_targets import target_note_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only canonical baseline for the current note aligner."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-candidate-confidence", type=float, default=0.65)
    args = parser.parse_args()
    if "test" in args.split.lower() or "sealed" in args.split.lower():
        raise ValueError("This iterative baseline is validation-only")
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = list(document.get(args.split) or [])
    if args.limit is not None:
        rows = rows[: args.limit]
    samples = []
    for position, row in enumerate(rows, 1):
        inference = build_inference_example(
            row,
            args.cache_root,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
        )
        predicted, deletions = current_note_aligner_baseline(
            inference.candidates, inference.score
        )
        target = ScoreEventIndex.from_musicxml(
            Path(str(row["sample_dir"])) / "verified_score.musicxml",
            target_note_map(row),
        )
        samples.append(
            JointMetricSample(
                predicted=predicted,
                target=target.rendered_events,
                source=inference.source,
                predicted_deletions=deletions,
                target_deletions=target.deleted_event_indices,
                score_event_count=len(inference.score),
            )
        )
        if position == 1 or position % 25 == 0:
            print(f"evaluated {position}/{len(rows)}", flush=True)
    report = {
        "schema_version": "align-joint-baseline-v1",
        "split": args.split,
        "n_samples": len(samples),
        "minimum_candidate_confidence": args.minimum_candidate_confidence,
        "metrics": evaluate_joint_dataset(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
