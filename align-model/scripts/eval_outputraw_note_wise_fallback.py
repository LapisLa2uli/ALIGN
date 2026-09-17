from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from alignmodel.joint.baseline import current_note_aligner_baseline
from alignmodel.joint.candidate_rescorer import (
    load_candidate_rescorer,
    rescore_candidates,
)
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _transcription_event(event):
    if event.score_span is None:
        return event
    return replace(
        event,
        relationship="match",
        origin_relationship="match",
        copy_pass=0,
    )


def _bootstrap(samples, seed: int = 365, replicates: int = 1000):
    counts = []
    for sample in samples:
        metric = evaluate_joint_dataset([sample])["aggregate"][
            "official_note_wise"
        ]
        counts.append(
            (
                float(metric["credit"]),
                int(metric["predicted"]),
                int(metric["gold"]),
            )
        )
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = generator.integers(0, len(counts), len(counts))
        credit = sum(counts[index][0] for index in selected)
        predicted = sum(counts[index][1] for index in selected)
        target = sum(counts[index][2] for index in selected)
        precision = credit / max(predicted, 1)
        recall = credit / max(target, 1)
        values.append(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-rescorer", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    ready = verify_data_ready(args.ready_marker)
    rescorer = None
    threshold = 0.0
    rescorer_hash = None
    if args.candidate_rescorer is not None:
        rescorer_hash = _sha256(args.candidate_rescorer)
        rescorer, threshold, _payload = load_candidate_rescorer(
            args.candidate_rescorer, device="cpu"
        )
        if _sha256(args.candidate_rescorer) != rescorer_hash:
            raise RuntimeError("Candidate rescorer changed while loading")

    samples = []
    transcription_samples = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        ordinals = dataset.ordinals("val")
        if args.limit is not None:
            ordinals = ordinals[: args.limit]
        for position, ordinal in enumerate(ordinals, 1):
            packed = dataset[ordinal]
            example = packed.training_example()
            candidates = example.candidates
            if rescorer is not None:
                candidates = rescore_candidates(
                    rescorer,
                    candidates,
                    threshold=threshold,
                    score=example.score,
                )
            events, deletions = current_note_aligner_baseline(
                candidates, example.score
            )
            samples.append(
                JointMetricSample(
                    predicted=events,
                    target=example.target_events,
                    source=packed.source,
                    predicted_deletions=deletions,
                    target_deletions=example.target_deletions,
                    score_event_count=len(example.score),
                )
            )
            transcription_samples.append(
                JointMetricSample(
                    predicted=tuple(_transcription_event(value) for value in events),
                    target=tuple(
                        _transcription_event(value)
                        for value in example.target_events
                    ),
                    source=packed.source,
                    predicted_deletions=frozenset(),
                    target_deletions=frozenset(),
                    score_event_count=len(example.score),
                )
            )
            if position == 1 or position % 25 == 0:
                print(f"fallback={position}/{len(ordinals)}", flush=True)

    transcription = evaluate_joint_dataset(transcription_samples)
    transcription["aggregate"]["official_note_wise"]["bootstrap"] = _bootstrap(
        transcription_samples
    )
    report = {
        "schema_version": "align-outputraw-note-wise-fallback-v1",
        "metric_status": "official_note_wise",
        "variant": (
            "candidate_rescorer_plus_legacy_note_aligner"
            if rescorer is not None
            else "legacy_note_aligner"
        ),
        "rows": len(samples),
        "data_fingerprint": ready["hashes"]["pack_id"],
        "candidate_rescorer": (
            {
                "path": str(args.candidate_rescorer.resolve()),
                "sha256": rescorer_hash,
                "threshold": threshold,
            }
            if args.candidate_rescorer is not None
            else None
        ),
        "canonical_note_wise_transcription": transcription,
        "downstream_fixed_aligner": evaluate_joint_dataset(samples),
        "metrics": evaluate_joint_dataset(samples),
        "locked_test_touched": False,
    }
    _atomic_json(args.output, report)


if __name__ == "__main__":
    main()
