from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alignmodel.joint.candidate_rescorer import (
    CANONICAL_FEATURE_DIM,
    SCHEMA_VERSION,
    CandidateRescorer,
    _atomic_torch,
    candidate_features,
)
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.training_resources import resource_lease


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(ordinals: tuple[int, ...], seed: int) -> tuple[list[int], list[int]]:
    train, heldout = [], []
    for ordinal in ordinals:
        digest = hashlib.sha256(f"{seed}:{ordinal}".encode()).digest()
        (heldout if digest[0] < 26 else train).append(ordinal)
    return train, heldout


def _target_duration(example: object, index: int) -> float:
    candidate = example.candidates[index]
    span = example.gold_spans[index]
    matches = [
        value
        for value in example.target_events
        if value.pitch == candidate.pitch
        and (
            value.score_span == span
            if span is not None
            else value.is_extra
        )
    ]
    if not matches:
        return candidate.end - candidate.start
    target = min(matches, key=lambda value: abs(value.start - candidate.start))
    return target.end - target.start


def _rows(packed: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    example = packed.training_example()
    labels = np.asarray(
        [
            span is not None or keep
            for span, keep in zip(
                example.gold_spans, example.gold_keep_unlinked
            )
        ],
        dtype=np.float32,
    )
    weights = np.ones(len(labels), dtype=np.float32)
    for index, (candidate, label) in enumerate(
        zip(example.candidates, labels)
    ):
        # Weight the audited rendered event, not an accidentally fragmented
        # candidate. Candidate-duration weighting over-labelled split fragments
        # as "short positives" in the first Track A rescorer.
        duration = _target_duration(example, index) if label else (
            candidate.end - candidate.start
        )
        if label:
            weights[index] *= (
                4.0
                if duration < 0.080
                else 3.0
                if duration < 0.120
                else 2.0
                if duration < 0.180
                else 1.0
            )
        elif 0.45 <= candidate.confidence <= 0.65:
            weights[index] *= 2.0
    return (
        candidate_features(
            example.candidates,
            example.score,
            feature_dim=CANONICAL_FEATURE_DIM,
        ),
        labels,
        weights,
    )


@torch.inference_mode()
def _evaluate(
    model: CandidateRescorer,
    dataset: PackedJointDataset,
    ordinals: list[int],
    device: torch.device,
) -> tuple[float, dict[str, float | int]]:
    probabilities, labels = [], []
    for ordinal in ordinals:
        features, target, _weight = _rows(dataset[ordinal])
        probabilities.extend(
            model(torch.from_numpy(features).to(device)).sigmoid().cpu().tolist()
        )
        labels.extend(target.tolist())
    probability = np.asarray(probabilities)
    target = np.asarray(labels, dtype=bool)
    best = (-1.0, 0.5, 0, 0, int(target.sum()))
    for threshold in np.linspace(0.20, 0.90, 141):
        predicted = probability >= threshold
        correct = int(np.sum(predicted & target))
        predicted_count = int(predicted.sum())
        target_count = int(target.sum())
        precision = correct / max(predicted_count, 1)
        recall = correct / max(target_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best[0]:
            best = (f1, float(threshold), correct, predicted_count, target_count)
    return best[1], {
        "f1": best[0],
        "correct": best[2],
        "predicted": best[3],
        "target": best[4],
        "metric_schema": "canonical-candidate-identity-train-holdout-v1",
        "timestamp_matching_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every-rows", type=int, default=250)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()

    ready = verify_data_ready(args.ready_marker)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = CandidateRescorer(feature_dim=CANONICAL_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    resume_epoch = 1
    resume_position = 0
    if args.resume_checkpoint is not None:
        resume = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        if resume.get("data_fingerprint") != ready["hashes"]["pack_id"]:
            raise ValueError("Resume checkpoint belongs to another packed release")
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        cursor = resume["cursor"]
        resume_epoch = int(cursor["epoch"])
        resume_position = int(cursor["position"])
        rng = resume["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    command = [str(value) for value in __import__("sys").argv]
    history = []
    best_f1 = -1.0
    with resource_lease(
        args.resource_status.resolve(),
        "gpu",
        track="canonical-candidate-rescorer-v2",
        command=command,
        metadata={"data_fingerprint": ready["hashes"]["pack_id"]},
    ):
        with PackedJointDataset(
            Path(str(ready["paths"]["packed_root"])),
            manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
            verify_records=False,
            load_feature_arrays=False,
        ) as dataset:
            train, heldout = _split(dataset.ordinals("train"), args.seed)
            for epoch in range(resume_epoch, args.epochs + 1):
                random.Random(args.seed + epoch).shuffle(train)
                total_loss = 0.0
                seen = 0
                started = time.perf_counter()
                model.train()
                batch_features, batch_labels, batch_weights = [], [], []

                def flush() -> None:
                    nonlocal total_loss, seen
                    if not batch_features:
                        return
                    features = torch.from_numpy(
                        np.concatenate(batch_features)
                    ).to(device)
                    labels = torch.from_numpy(np.concatenate(batch_labels)).to(device)
                    weights = torch.from_numpy(np.concatenate(batch_weights)).to(device)
                    logits = model(features)
                    positives = labels.sum()
                    negatives = labels.numel() - positives
                    pos_weight = (negatives / positives.clamp_min(1)).clamp(1, 8)
                    losses = F.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=pos_weight, reduction="none"
                    )
                    loss = (losses * weights).sum() / weights.sum().clamp_min(1)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    total_loss += float(loss) * len(labels)
                    seen += len(labels)
                    batch_features.clear()
                    batch_labels.clear()
                    batch_weights.clear()

                first_position = resume_position if epoch == resume_epoch else 0
                for position, ordinal in enumerate(
                    train[first_position:], first_position + 1
                ):
                    features, labels, weights = _rows(dataset[ordinal])
                    batch_features.append(features)
                    batch_labels.append(labels)
                    batch_weights.append(weights)
                    if sum(len(value) for value in batch_labels) >= args.batch_size:
                        flush()
                    if position % args.checkpoint_every_rows == 0:
                        flush()
                        _atomic_torch(
                            output / "mid_epoch_checkpoint.pt",
                            {
                                "schema_version": "align-canonical-rescorer-mid-epoch-v1",
                                "state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "rng_state": {
                                    "python": random.getstate(),
                                    "numpy": np.random.get_state(),
                                    "torch": torch.get_rng_state(),
                                    "cuda": torch.cuda.get_rng_state_all(),
                                },
                                "cursor": {
                                    "epoch": epoch,
                                    "position": position,
                                    "ordinal": ordinal,
                                },
                                "data_fingerprint": ready["hashes"]["pack_id"],
                            },
                        )
                        rate = position / max(time.perf_counter() - started, 1e-9)
                        print(
                            f"epoch={epoch} rows={position}/{len(train)} "
                            f"rows_per_sec={rate:.2f}",
                            flush=True,
                        )
                flush()
                model.eval()
                threshold, validation = _evaluate(
                    model, dataset, heldout, device
                )
                row = {
                    "epoch": epoch,
                    "mean_loss": total_loss / max(seen, 1),
                    "threshold": threshold,
                    "validation": validation,
                    "train_rows": len(train),
                    "heldout_train_rows": len(heldout),
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                if float(validation["f1"]) > best_f1:
                    best_f1 = float(validation["f1"])
                    _atomic_torch(
                        output / "candidate_rescorer.pt",
                        {
                            "schema_version": SCHEMA_VERSION,
                            "state_dict": model.state_dict(),
                            "model": {
                                "hidden_dim": 64,
                                "dropout": 0.05,
                                "feature_dim": CANONICAL_FEATURE_DIM,
                            },
                            "threshold": threshold,
                            "validation": validation,
                            "data_fingerprint": ready["hashes"]["pack_id"],
                            "training_target": (
                                "packed canonical score-event or retained EXTRA identity"
                            ),
                            "timestamp_matching_used": False,
                        },
                    )
                resume_position = 0
    checkpoint = output / "candidate_rescorer.pt"
    (output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "align-canonical-candidate-rescorer-report-v1",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "history": history,
                "best_train_holdout_f1": best_f1,
                "data_fingerprint": ready["hashes"]["pack_id"],
                "official_full_validation_used": False,
                "timestamp_matching_used": False,
                "locked_test_touched": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
