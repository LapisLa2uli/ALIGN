from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import random
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.operation_tagger import TYPE_NAMES, MelOperationTagger
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.joint.sequence_mapper import sequence_tensors
from alignmodel.training_resources import resource_lease


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _batch(rows: list[tuple[tuple[JointCandidate, ...], list[int]]], device: torch.device):
    maximum = max(len(candidates) for candidates, _labels in rows)
    pitch = torch.zeros(len(rows), maximum, dtype=torch.long, device=device)
    continuous = torch.zeros(len(rows), maximum, 4, device=device)
    labels = torch.full(
        (len(rows), maximum), -100, dtype=torch.long, device=device
    )
    for index, (candidates, target) in enumerate(rows):
        current_pitch, current_continuous = sequence_tensors(
            candidates, device=device
        )
        pitch[index, : len(candidates)] = current_pitch
        continuous[index, : len(candidates)] = current_continuous
        labels[index, : len(target)] = torch.tensor(target, device=device)
    return pitch, continuous, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    lease = resource_lease(
        args.resource_status,
        "gpu",
        track="mel-mapper-v2-operation-tagger",
        command=[str(value) for value in __import__("sys").argv],
        metadata={"locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    ready = verify_data_ready(args.ready_marker)
    with args.predictions.open("r", encoding="utf-8") as stream:
        predictions = {
            row["sample"]: row["notes"]
            for row in (json.loads(line) for line in stream if line.strip())
        }
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = MelOperationTagger().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        training, heldout = [], []
        for position, ordinal in enumerate(dataset.ordinals("train")):
            packed = dataset[ordinal]
            example = packed.training_example()
            candidates = tuple(
                JointCandidate(
                    int(row["pitch"]),
                    float(row["start"]),
                    float(row["end"]),
                    float(row["confidence"]),
                )
                for row in predictions[packed.sample]
            )
            labels = [-100] * len(candidates)
            matcher = SequenceMatcher(
                None,
                [value.pitch for value in candidates],
                [value.pitch for value in example.target_events],
                autojunk=False,
            )
            for left, right, count in matcher.get_matching_blocks():
                for offset in range(count):
                    target = example.target_events[right + offset]
                    labels[left + offset] = TYPE_NAMES.index(
                        "copy" if target.is_copy else target.relationship
                    )
            (
                heldout if position % 5 == 0 else training
            ).append((candidates, labels))
        for epoch in range(1, args.epochs + 1):
            random.Random(args.seed + epoch).shuffle(training)
            model.train()
            total = steps = 0
            for start in range(0, len(training), args.batch_size):
                pitch, continuous, labels = _batch(
                    training[start : start + args.batch_size], device
                )
                logits = model(pitch, continuous)
                loss = F.cross_entropy(
                    logits.transpose(1, 2),
                    labels,
                    ignore_index=-100,
                    weight=torch.tensor(
                        [1.0, 1.5, 8.0, 5.0], device=device
                    ),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                total += float(loss)
                steps += 1
            model.eval()
            confusion = np.zeros((4, 4), dtype=np.int64)
            with torch.inference_mode():
                for start in range(0, len(heldout), args.batch_size):
                    pitch, continuous, labels = _batch(
                        heldout[start : start + args.batch_size], device
                    )
                    predicted = model(pitch, continuous).argmax(-1)
                    valid = labels >= 0
                    for gold, guess in zip(
                        labels[valid].cpu().tolist(),
                        predicted[valid].cpu().tolist(),
                    ):
                        confusion[gold, guess] += 1
            row = {
                "epoch": epoch,
                "loss": total / max(steps, 1),
                "heldout_accuracy": float(
                    np.trace(confusion) / max(confusion.sum(), 1)
                ),
                "confusion": confusion.tolist(),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            _atomic(
                output / "last_checkpoint.pt",
                {
                    "schema_version": "align-mel-operation-tagger-v2",
                    "model_state_dict": model.state_dict(),
                    "model_config": {"hidden_dim": 128},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None,
                    "rng_state": {
                        "python": random.getstate(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                    },
                    "cursor": {"epoch": epoch + 1, "position": 0},
                    "data_fingerprint": ready["hashes"]["pack_id"],
                    "history": history,
                },
            )
    checkpoint = output / "last_checkpoint.pt"
    (output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "align-mel-operation-tagger-report-v2",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "history": history,
                "data_fingerprint": ready["hashes"]["pack_id"],
                "prediction_cache_sha256": _sha256(args.predictions),
                "timestamp_metrics_used": False,
                "locked_test_touched": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    atexit.unregister(lease.__exit__)
    lease.__exit__(None, None, None)


if __name__ == "__main__":
    main()
