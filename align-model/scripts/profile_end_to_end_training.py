from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import psutil
import torch

import alignmodel.joint.data as joint_data
from alignmodel.joint.end_to_end import (
    EndToEndTrainConfig,
    _initialize_model,
    _manifest_rows,
    _vectorized_group_nll,
)
from alignmodel.joint.lattice import LatticeConfig, SparseJointLattice


@contextmanager
def _timed_wrappers(timings: dict[str, float]):
    originals: list[tuple[str, Callable[..., Any]]] = []

    def install(name: str, bucket: str) -> None:
        original = getattr(joint_data, name)
        originals.append((name, original))

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings[bucket] = (
                    timings.get(bucket, 0.0)
                    + time.perf_counter()
                    - started
                )

        setattr(joint_data, name, wrapped)

    install("_validated_basic_pitch", "npz_decompression_sec")
    install("target_note_map", "sqlite_target_sec")
    install("basic_pitch_candidate_union", "candidate_decode_sec")
    install("add_score_repeat_hints", "repeat_hint_sec")
    install("pair_exact_pitch_onset", "gold_pairing_sec")
    try:
        yield
    finally:
        for name, original in originals:
            setattr(joint_data, name, original)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--path-rows", type=int, default=100)
    parser.add_argument("--decode-rows", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": "align-end-to-end-profile-v1",
        "rows": args.rows,
        "path_rows": min(args.path_rows, args.rows),
        "decode_rows": min(args.decode_rows, args.rows),
        "timings": {},
        "locked_test_touched": False,
    }
    started = time.perf_counter()
    rows = _manifest_rows(args.manifest, "train")[: args.rows]
    report["timings"]["manifest_load_sec"] = time.perf_counter() - started
    config = EndToEndTrainConfig(
        manifest=args.manifest,
        cache_root=args.cache_root,
        output_dir=args.output.parent,
        initialize_checkpoint=args.checkpoint,
        max_train_samples=args.rows,
        path_train_samples=min(args.path_rows, args.rows),
        max_val_samples=min(args.decode_rows, args.rows),
        local_device=args.device,
        path_device="cpu",
        lattice=LatticeConfig(
            max_options_per_candidate=12,
            max_states=48,
            max_delete_events=24,
            noise_inference_bias=-6.0,
            continuation_feature_enabled=True,
            continuation_score_weight=0.35,
            continuation_hard_negative_copies=1,
            repeat_fragment_penalty=0.25,
        ),
    )
    model, initialization = _initialize_model(
        config, torch.device(args.device)
    )
    report["initialization"] = initialization
    phase_detail: dict[str, float] = {}
    started = time.perf_counter()
    with _timed_wrappers(phase_detail):
        examples = [
            joint_data.build_training_example(
                row,
                args.cache_root,
                pairing_tolerance_sec=config.pairing_tolerance_sec,
                minimum_candidate_confidence=(
                    config.minimum_candidate_confidence
                ),
            )
            for row in rows
        ]
    construction = time.perf_counter() - started
    phase_detail["other_example_construction_sec"] = max(
        0.0, construction - sum(phase_detail.values())
    )
    report["timings"]["first_example_construction_sec"] = construction
    report["example_construction_detail"] = phase_detail
    report["first_construction_rows_per_sec"] = args.rows / construction
    report["rss_after_construction_mb"] = _rss_mb()

    started = time.perf_counter()
    repeated = [
        joint_data.build_training_example(
            row,
            args.cache_root,
            pairing_tolerance_sec=config.pairing_tolerance_sec,
            minimum_candidate_confidence=config.minimum_candidate_confidence,
        )
        for row in rows
    ]
    repeated_sec = time.perf_counter() - started
    report["timings"]["repeated_example_construction_sec"] = repeated_sec
    report["repeated_construction_rows_per_sec"] = args.rows / repeated_sec
    del repeated

    lattice = SparseJointLattice(model, config.lattice)
    edge_rows: list[list[float]] = []
    groups: list[tuple[int, int, int]] = []
    started = time.perf_counter()
    for example in examples:
        rows_for_example, groups_for_example = lattice.local_warmup_edges(
            example.candidates,
            example.score,
            example.gold_spans,
            example.gold_keep_unlinked,
        )
        offset = len(edge_rows)
        edge_rows.extend(rows_for_example)
        groups.extend(
            (start + offset, end + offset, gold)
            for start, end, gold in groups_for_example
        )
    report["timings"]["local_edge_construction_sec"] = (
        time.perf_counter() - started
    )
    report["local_edges"] = len(edge_rows)
    report["candidate_groups"] = len(groups)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    loss = _vectorized_group_nll(
        model,
        edge_rows,
        groups,
        torch.device(args.device),
        config.local_distillation_weight,
    )
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    report["timings"]["local_gpu_forward_backward_sec"] = (
        time.perf_counter() - started
    )
    report["local_loss"] = float(loss.detach().cpu())
    report["peak_vram_mb"] = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    if args.cpu_threads is not None:
        torch.set_num_threads(max(1, args.cpu_threads))
    report["cpu_threads"] = torch.get_num_threads()
    model.to("cpu")
    path_lattice = SparseJointLattice(model, config.lattice)
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    path_losses = []
    for example in examples[: args.path_rows]:
        value = path_lattice.nll(
            example.candidates,
            example.score,
            example.gold_spans,
            example.gold_keep_unlinked,
        ) / max(1, len(example.candidates))
        value.backward()
        path_losses.append(float(value.detach()))
    report["timings"]["structured_path_cpu_forward_backward_sec"] = (
        time.perf_counter() - started
    )
    report["structured_path_mean_loss"] = (
        sum(path_losses) / max(1, len(path_losses))
    )

    decode_lattice = SparseJointLattice(model, config.lattice)
    started = time.perf_counter()
    for example in examples[: args.decode_rows]:
        decode_lattice.decode(example.candidates, example.score)
    report["timings"]["validation_decode_sec"] = time.perf_counter() - started
    report["peak_rss_mb"] = _rss_mb()
    report["total_profile_sec"] = sum(report["timings"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
