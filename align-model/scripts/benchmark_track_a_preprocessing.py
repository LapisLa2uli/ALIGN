"""Bounded group-disjoint train-fold benchmark for predeclared Track A views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from alignmodel.joint.baseline import current_note_aligner_baseline
from alignmodel.joint.candidate_rescorer import (
    load_candidate_rescorer,
    rescore_candidates,
)
from alignmodel.joint.candidates import basic_pitch_candidate_union
from alignmodel.joint.index import JointEvent
from alignmodel.joint.lattice import candidates_from_events
from alignmodel.joint.metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    pair_exact_pitch_onset,
)
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.transcription.basic_pitch import (
    decode_frozen_basic_pitch,
    sanitize_basic_pitch_notes,
)
from alignmodel.transcription.track_a import (
    TrackAPreprocessConfig,
    extract_preprocessed_view,
    preprocessing_identity,
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _normalized(event):
    return event if event.score_span is None else replace(
        event, relationship="match", origin_relationship="match", copy_pass=0
    )


def _merge_candidates(*streams):
    output = []
    for candidate in sorted(
        (value for stream in streams for value in stream),
        key=lambda value: (value.start, value.pitch, value.end),
    ):
        match = next(
            (
                index for index, prior in enumerate(output)
                if prior.pitch == candidate.pitch
                and abs(prior.start - candidate.start) <= 0.035
            ),
            None,
        )
        if match is None:
            output.append(candidate)
        elif candidate.confidence > output[match].confidence:
            output[match] = candidate
    return tuple(sorted(output, key=lambda value: (value.start, value.pitch)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--candidate-rescorer", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    manifest = json.loads(Path(ready["paths"]["manifest"]).read_text())
    rows = {row["sample"]: row for row in manifest["train"]}
    heldout_groups = sorted(
        group for group in {row["leakage_group"] for row in rows.values()}
        if hashlib.sha256(f"{args.seed}:{group}".encode()).digest()[0] < 26
    )
    chosen_groups = set(heldout_groups[: args.clips])
    model, threshold, _payload = load_candidate_rescorer(
        args.candidate_rescorer, "cpu"
    )
    config = TrackAPreprocessConfig()
    samples = {
        name: [] for name in (
            "original-canonical",
            "preprocessing-only",
            "postprocessing-only",
            "postprocessing+rescorer",
            "combined",
            "combined+rescorer",
        )
    }
    diagnostics = {
        name: {
            "predicted": 0,
            "target": 0,
            "matched": 0,
            "split": 0,
            "rearticulation": 0,
            "short": {80: [0, 0], 120: [0, 0], 180: [0, 0]},
        }
        for name in samples
    }
    selected_samples = []
    with PackedJointDataset(
        Path(ready["paths"]["packed_root"]),
        manifest_sha256=ready["hashes"]["manifest_sha256"],
        verify_records=False,
        load_feature_arrays=True,
    ) as dataset:
        ordinals = [
            ordinal for ordinal in dataset.ordinals("train")
            if rows[dataset[ordinal].sample]["leakage_group"] in chosen_groups
        ][: args.clips]
        for position, ordinal in enumerate(ordinals, 1):
            packed = dataset[ordinal]
            row = rows[packed.sample]
            example = packed.training_example()
            identity = preprocessing_identity(
                Path(row["sample_dir"]) / "performance_audio.wav",
                config,
                {"effective_audio_transpose": row["effective_audio_transpose"]},
            )
            cache = (
                args.cache_root / identity["config_sha256"] / f"{packed.sample}.npz"
            )
            preprocessed = extract_preprocessed_view(
                Path(row["sample_dir"]) / "performance_audio.wav",
                source_metadata={
                    "effective_audio_transpose": row["effective_audio_transpose"]
                },
                cache_path=cache,
                config=config,
            )
            original_canonical = tuple(
                candidates_from_events(
                    sanitize_basic_pitch_notes(
                        decode_frozen_basic_pitch(packed.features),
                        packed.features,
                    )
                )
            )
            preprocessing_only = tuple(
                candidates_from_events(
                    sanitize_basic_pitch_notes(
                        decode_frozen_basic_pitch(preprocessed), preprocessed
                    )
                )
            )
            postprocessing = tuple(
                value for value in example.candidates if value.confidence >= 0.65
            )
            combined = _merge_candidates(
                postprocessing,
                basic_pitch_candidate_union(
                    preprocessed, minimum_confidence=0.65
                ),
            )
            variants = {
                "original-canonical": original_canonical,
                "preprocessing-only": preprocessing_only,
                "postprocessing-only": postprocessing,
                "postprocessing+rescorer": rescore_candidates(
                    model,
                    postprocessing,
                    threshold=threshold,
                    score=example.score,
                ),
                "combined": combined,
                "combined+rescorer": rescore_candidates(
                    model, combined, threshold=threshold, score=example.score
                ),
            }
            for name, candidates in variants.items():
                candidate_events = tuple(
                    JointEvent(
                        value.pitch,
                        value.start,
                        value.end,
                        None,
                        "extra",
                        confidence=value.confidence,
                    )
                    for value in candidates
                )
                pairs = pair_exact_pitch_onset(
                    candidate_events, example.target_events, tolerance_sec=0.050
                )
                matched_candidates = {left for left, _right in pairs}
                matched_targets = {right for _left, right in pairs}
                diagnostic = diagnostics[name]
                diagnostic["predicted"] += len(candidates)
                diagnostic["target"] += len(example.target_events)
                diagnostic["matched"] += len(pairs)
                for maximum, counts in diagnostic["short"].items():
                    target_indices = {
                        index
                        for index, value in enumerate(example.target_events)
                        if value.end - value.start < maximum / 1000.0
                    }
                    counts[0] += len(matched_targets & target_indices)
                    counts[1] += len(target_indices)
                for index, (first, second) in enumerate(
                    zip(candidates, candidates[1:])
                ):
                    if (
                        first.pitch == second.pitch
                        and second.start - first.end <= 0.100
                    ):
                        if index in matched_candidates and index + 1 in matched_candidates:
                            diagnostic["rearticulation"] += 1
                        else:
                            diagnostic["split"] += 1
                events, _deletions = current_note_aligner_baseline(
                    candidates, example.score
                )
                samples[name].append(
                    JointMetricSample(
                        predicted=tuple(_normalized(value) for value in events),
                        target=tuple(
                            _normalized(value) for value in example.target_events
                        ),
                        source=str(row.get("audio_render") or "unknown"),
                        score_event_count=len(example.score),
                    )
                )
            selected_samples.append(packed.sample)
            print(f"preprocess_train_fold={position}/{len(ordinals)}", flush=True)
    report = {
        "schema_version": "align-track-a-preprocessing-train-fold-v1",
        "split": "train_group_disjoint_bounded_benchmark",
        "selection": {
            "seed": args.seed,
            "clips": len(selected_samples),
            "samples": selected_samples,
            "leakage_groups": len(chosen_groups),
        },
        "preprocessing": {
            **preprocessing_identity(
                Path(rows[selected_samples[0]]["sample_dir"])
                / "performance_audio.wav",
                config,
                {
                    "effective_audio_transpose":
                    rows[selected_samples[0]]["effective_audio_transpose"]
                },
            ),
            "cache_root": str(args.cache_root.resolve()),
        },
        "results": {
            name: {
                "canonical_note_wise": evaluate_joint_dataset(values),
                "diagnostics": {
                    **diagnostics[name],
                    "count_ratio": diagnostics[name]["predicted"]
                    / max(diagnostics[name]["target"], 1),
                    "split_rate": diagnostics[name]["split"]
                    / max(diagnostics[name]["predicted"], 1),
                    "short": {
                        f"lt_{maximum}ms": {
                            "matched": counts[0],
                            "target": counts[1],
                            "recall": counts[0] / max(counts[1], 1),
                        }
                        for maximum, counts in diagnostics[name]["short"].items()
                    },
                },
            }
            for name, values in samples.items()
        },
        "validation_rows_opened": 0,
        "lockbox_touched": False,
    }
    _atomic_json(args.output, report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
