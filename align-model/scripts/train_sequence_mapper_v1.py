from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.joint.sequence_mapper import (
    TYPE_NAMES,
    FixedScoreSequenceMapper,
    decode_sequence_mapper,
    sequence_tensors,
)
from alignmodel.training_resources import resource_lease


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _split(ordinals: tuple[int, ...], seed: int) -> tuple[list[int], list[int]]:
    train, heldout = [], []
    for ordinal in ordinals:
        value = hashlib.sha256(f"{seed}:{ordinal}".encode()).digest()[0]
        (heldout if value < 26 else train).append(ordinal)
    return train, heldout


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


def _batch(examples: list, device: torch.device) -> tuple:
    maximum = max(len(example.target_events) for example in examples)
    maximum_score = max(len(example.score) for example in examples)
    pitch = torch.zeros(len(examples), maximum, dtype=torch.long, device=device)
    continuous = torch.zeros(
        len(examples), maximum, 4, dtype=torch.float32, device=device
    )
    location = torch.full(
        (len(examples), maximum), -100, dtype=torch.long, device=device
    )
    operation = torch.full_like(location, -100)
    span_length = torch.full_like(location, -100)
    score_pitch = torch.zeros(
        len(examples), maximum_score, dtype=torch.long, device=device
    )
    score_position = torch.zeros(
        len(examples), maximum_score, dtype=torch.float32, device=device
    )
    score_mask = torch.zeros(
        len(examples), maximum_score, dtype=torch.bool, device=device
    )
    for row, example in enumerate(examples):
        candidates = _candidates(example.target_events)
        current_pitch, current_continuous = sequence_tensors(
            candidates, device=device
        )
        count = len(candidates)
        pitch[row, :count] = current_pitch
        continuous[row, :count] = current_continuous
        location[row, :count] = torch.tensor(
            [
                event.score_span[0]
                if event.score_span is not None
                else maximum_score
                for event in example.target_events
            ],
            device=device,
        )
        operation[row, :count] = torch.tensor(
            [
                TYPE_NAMES.index(
                    "copy" if event.is_copy else event.relationship
                )
                for event in example.target_events
            ],
            device=device,
        )
        span_length[row, :count] = torch.tensor(
            [
                (
                    min(128, event.score_span[1] - event.score_span[0])
                    if event.score_span is not None
                    else 0
                )
                for event in example.target_events
            ],
            device=device,
        )
        score_length = len(example.score)
        score_pitch[row, :score_length] = torch.tensor(
            [event.pitch for event in example.score], device=device
        )
        score_end = max(
            (event.ql_end for event in example.score), default=1.0
        )
        score_position[row, :score_length] = torch.tensor(
            [
                event.ql_start / max(score_end, 1e-6)
                for event in example.score
            ],
            dtype=torch.float32,
            device=device,
        )
        score_mask[row, :score_length] = True
    return (
        pitch,
        continuous,
        score_pitch,
        score_position,
        score_mask,
        location,
        operation,
        span_length,
    )


@torch.inference_mode()
def _validate(
    model: FixedScoreSequenceMapper,
    dataset: PackedJointDataset,
    ordinals: list[int],
) -> dict:
    samples = []
    for position, ordinal in enumerate(ordinals, 1):
        packed = dataset[ordinal]
        example = packed.training_example()
        mapped = decode_sequence_mapper(
            model, _candidates(example.target_events), example.score
        )
        samples.append(
            JointMetricSample(
                predicted=mapped,
                target=example.target_events,
                source=packed.source,
                score_event_count=len(example.score),
            )
        )
        if position == 1 or position % 16 == 0:
            print(f"oracle_validation={position}/{len(ordinals)}", flush=True)
    return evaluate_joint_dataset(samples)["aggregate"]["official_note_wise"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--oracle-val-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    history = []
    with resource_lease(
        args.resource_status,
        "gpu",
        track="sequence-mapper-v1-oracle",
        command=[str(value) for value in __import__("sys").argv],
        metadata={"hidden_dim": args.hidden_dim, "locked_test": False},
    ):
        with PackedJointDataset(
            Path(str(ready["paths"]["packed_root"])),
            manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
            verify_records=False,
            load_feature_arrays=False,
        ) as dataset:
            train, heldout = _split(dataset.ordinals("train"), args.seed)
            model = FixedScoreSequenceMapper(hidden_dim=args.hidden_dim).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=1e-4
            )
            first_epoch = 1
            if args.resume is not None:
                resume = torch.load(
                    args.resume, map_location=device, weights_only=False
                )
                if resume.get("data_fingerprint") != ready["hashes"]["pack_id"]:
                    raise ValueError("Resume checkpoint data fingerprint mismatch")
                model.load_state_dict(resume["model_state_dict"])
                optimizer.load_state_dict(resume["optimizer_state_dict"])
                history = list(resume.get("history") or [])
                first_epoch = int(resume["cursor"]["epoch"])
            for epoch in range(first_epoch, args.epochs + 1):
                random.Random(args.seed + epoch).shuffle(train)
                model.train()
                total = 0.0
                steps = 0
                started = time.perf_counter()
                for start in range(0, len(train), args.batch_size):
                    examples = [
                        dataset[ordinal].training_example()
                        for ordinal in train[start : start + args.batch_size]
                    ]
                    (
                        pitch,
                        continuous,
                        score_pitch,
                        score_position,
                        score_mask,
                        location,
                        operation,
                        span_length,
                    ) = _batch(
                        examples, device
                    )
                    location_logits, operation_logits, span_logits = model(
                        pitch,
                        continuous,
                        score_pitch,
                        score_position,
                        score_mask,
                    )
                    location_values = F.cross_entropy(
                        location_logits.transpose(1, 2),
                        location,
                        ignore_index=-100,
                        reduction="none",
                    )
                    location_weight = torch.where(
                        location == score_mask.shape[1],
                        5.0,
                        1.0,
                    )
                    valid_location = location != -100
                    location_loss = (
                        location_values[valid_location]
                        * location_weight[valid_location]
                    ).sum() / location_weight[valid_location].sum()
                    operation_loss = F.cross_entropy(
                        operation_logits.transpose(1, 2),
                        operation,
                        ignore_index=-100,
                        weight=torch.tensor(
                            [1.0, 1.5, 8.0, 10.0], device=device
                        ),
                    )
                    span_loss = F.cross_entropy(
                        span_logits.transpose(1, 2),
                        span_length,
                        ignore_index=-100,
                    )
                    loss = location_loss + 0.8 * operation_loss + 0.5 * span_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    total += float(loss)
                    steps += 1
                    position = min(start + args.batch_size, len(train))
                    if position == args.batch_size or position % 480 == 0:
                        rate = position / max(time.perf_counter() - started, 1e-9)
                        print(
                            f"epoch={epoch} rows={position}/{len(train)} "
                            f"loss={total/steps:.5f} rows_per_sec={rate:.2f}",
                            flush=True,
                        )
                    if position % 480 == 0:
                        _atomic_torch(
                            output / "mid_epoch_checkpoint.pt",
                            {
                                "schema_version": "align-sequence-mapper-mid-epoch-v1",
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": None,
                                "rng_state": {
                                    "python": random.getstate(),
                                    "numpy": np.random.get_state(),
                                    "torch": torch.get_rng_state(),
                                    "cuda": torch.cuda.get_rng_state_all(),
                                },
                                "cursor": {"epoch": epoch, "position": position},
                                "data_fingerprint": ready["hashes"]["pack_id"],
                            },
                        )
                model.eval()
                heldout_loss = []
                for start in range(0, len(heldout), args.batch_size):
                    examples = [
                        dataset[ordinal].training_example()
                        for ordinal in heldout[start : start + args.batch_size]
                    ]
                    (
                        pitch,
                        continuous,
                        score_pitch,
                        score_position,
                        score_mask,
                        location,
                        operation,
                        span_length,
                    ) = _batch(
                        examples, device
                    )
                    location_logits, operation_logits, span_logits = model(
                        pitch,
                        continuous,
                        score_pitch,
                        score_position,
                        score_mask,
                    )
                    location_values = F.cross_entropy(
                                location_logits.transpose(1, 2),
                                location,
                                ignore_index=-100,
                                reduction="none",
                            )
                    location_weight = torch.where(
                        location == score_mask.shape[1], 5.0, 1.0
                    )
                    valid_location = location != -100
                    location_loss = (
                        location_values[valid_location]
                        * location_weight[valid_location]
                    ).sum() / location_weight[valid_location].sum()
                    heldout_loss.append(
                        float(
                            location_loss
                            + 0.8
                            * F.cross_entropy(
                                operation_logits.transpose(1, 2),
                                operation,
                                ignore_index=-100,
                                weight=torch.tensor(
                                    [1.0, 1.5, 8.0, 10.0], device=device
                                ),
                            )
                            + 0.5
                            * F.cross_entropy(
                                span_logits.transpose(1, 2),
                                span_length,
                                ignore_index=-100,
                            )
                        )
                    )
                row = {
                    "epoch": epoch,
                    "train_loss": total / max(steps, 1),
                    "heldout_loss": float(np.mean(heldout_loss)),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                _atomic_torch(
                    output / "last_checkpoint.pt",
                    {
                        "schema_version": "align-sequence-mapper-v1",
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "hidden_dim": args.hidden_dim,
                        },
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": None,
                        "rng_state": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all(),
                        },
                        "cursor": {"epoch": epoch + 1, "position": 0},
                        "data_fingerprint": ready["hashes"]["pack_id"],
                        "history": history,
                    },
                )
            oracle = _validate(
                model, dataset, dataset.ordinals("val")[: args.oracle_val_rows]
            )
    checkpoint = output / "last_checkpoint.pt"
    (output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "align-sequence-mapper-report-v1",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "data_fingerprint": ready["hashes"]["pack_id"],
                "curriculum_stage": "exact_oracle_notes",
                "history": history,
                "oracle_validation_rows": args.oracle_val_rows,
                "oracle_validation": oracle,
                "selection_metric": "canonical_note_wise_only",
                "locked_test_touched": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
