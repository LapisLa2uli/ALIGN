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

from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.mel_mapper import (
    ContextualRepeatMapper,
    MapperState,
    decode_mapper,
    edge_features,
    option_indices,
    transition,
)
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import _segment_logsumexp, verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.training_resources import resource_lease


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _oracle_candidates(events: tuple) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=event.pitch,
            start=event.start,
            end=event.end,
            confidence=1.0,
        )
        for event in events
    )


def _training_rows(example: object) -> tuple[np.ndarray, list[tuple[int, int, int]], np.ndarray]:
    events = _oracle_candidates(example.target_events)
    rows = []
    groups = []
    weights = []
    state = MapperState()
    for event_index, target in enumerate(example.target_events):
        gold = target.score_span[0] if target.score_span is not None else -1
        options = option_indices(
            events, example.score, event_index, state, gold=gold
        )
        start = len(rows)
        for option in options:
            rows.append(edge_features(events, example.score, event_index, option, state))
        groups.append((start, len(rows), options.index(gold)))
        destination, operation = transition(state, gold)
        weights.append(
            4.0
            if operation in {4, 5, 6}
            else 3.0
            if target.relationship in {"extra", "substitute"}
            else 1.0
        )
        state = destination
    return np.stack(rows), groups, np.asarray(weights, np.float32)


def _split(ordinals: tuple[int, ...], seed: int) -> tuple[list[int], list[int]]:
    train, heldout = [], []
    for ordinal in ordinals:
        value = hashlib.sha256(f"{seed}:{ordinal}".encode()).digest()[0]
        (heldout if value < 26 else train).append(ordinal)
    return train, heldout


def _local_loss(
    model: ContextualRepeatMapper,
    features: np.ndarray,
    groups: list[tuple[int, int, int]],
    weights: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    scores = model(torch.from_numpy(features).to(device))
    partition, gold = _segment_logsumexp(scores, groups)
    losses = partition - scores[gold].float()
    weight = torch.from_numpy(weights).to(device)
    return (losses * weight).sum() / weight.sum().clamp_min(1)


@torch.inference_mode()
def _oracle_validation(
    model: ContextualRepeatMapper,
    dataset: PackedJointDataset,
    ordinals: list[int],
) -> dict:
    samples = []
    for position, ordinal in enumerate(ordinals, 1):
        packed = dataset[ordinal]
        example = packed.training_example()
        events = decode_mapper(
            model,
            _oracle_candidates(example.target_events),
            example.score,
        )
        samples.append(
            JointMetricSample(
                predicted=events,
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
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clips-per-step", type=int, default=8)
    parser.add_argument("--oracle-val-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ready = verify_data_ready(args.ready_marker)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ContextualRepeatMapper(hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history = []
    command = [str(value) for value in __import__("sys").argv]
    with resource_lease(
        args.resource_status,
        "gpu",
        track="mel-mapper-v1-oracle-curriculum",
        command=command,
        metadata={"hidden_dim": 32, "locked_test": False},
    ):
        with PackedJointDataset(
            Path(str(ready["paths"]["packed_root"])),
            manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
            verify_records=False,
            load_feature_arrays=False,
        ) as dataset:
            train, heldout = _split(dataset.ordinals("train"), args.seed)
            for epoch in range(1, args.epochs + 1):
                random.Random(args.seed + epoch).shuffle(train)
                model.train()
                total = 0.0
                steps = 0
                started = time.perf_counter()
                for start in range(0, len(train), args.clips_per_step):
                    clip_rows = [
                        _training_rows(dataset[value].training_example())
                        for value in train[start : start + args.clips_per_step]
                    ]
                    losses = [
                        _local_loss(model, features, groups, weights, device)
                        for features, groups, weights in clip_rows
                    ]
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    total += float(loss)
                    steps += 1
                    position = min(start + args.clips_per_step, len(train))
                    if position == args.clips_per_step or position % 256 == 0:
                        rate = position / max(time.perf_counter() - started, 1e-9)
                        print(
                            f"epoch={epoch} clips={position}/{len(train)} "
                            f"loss={total/steps:.5f} clips_per_sec={rate:.2f}",
                            flush=True,
                        )
                    if position % 256 == 0:
                        _atomic_torch(
                            output / "mid_epoch_checkpoint.pt",
                            {
                                "schema_version": "align-mel-mapper-mid-epoch-v1",
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
                heldout_losses = []
                for ordinal in heldout:
                    features, groups, weights = _training_rows(
                        dataset[ordinal].training_example()
                    )
                    heldout_losses.append(
                        float(_local_loss(model, features, groups, weights, device))
                    )
                row = {
                    "epoch": epoch,
                    "train_loss": total / max(steps, 1),
                    "train_rows": len(train),
                    "heldout_train_rows": len(heldout),
                    "heldout_local_nll": float(np.mean(heldout_losses)),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                _atomic_torch(
                    output / "last_checkpoint.pt",
                    {
                        "schema_version": "align-mel-mapper-v1",
                        "model_state_dict": model.state_dict(),
                        "model_config": {"hidden_dim": 32},
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
            model.eval()
            oracle = _oracle_validation(
                model, dataset, dataset.ordinals("val")[: args.oracle_val_rows]
            )
    checkpoint = output / "last_checkpoint.pt"
    report = {
        "schema_version": "align-mel-mapper-training-report-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "data_fingerprint": ready["hashes"]["pack_id"],
        "curriculum_stage": "exact_oracle_notes",
        "history": history,
        "oracle_validation_rows": args.oracle_val_rows,
        "oracle_validation": oracle,
        "selection_metric": "canonical_note_wise_only",
        "locked_test_touched": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
