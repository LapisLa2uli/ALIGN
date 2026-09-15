from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .candidates import (
    CANDIDATE_GENERATION_VERSION,
    HIGH_RECALL_DECODE_CONFIGS,
)
from .data import JointTrainingExample, build_training_example
from .lattice import JointEdgeScorer, LatticeConfig, SparseJointLattice
from .metrics import JointMetricSample, evaluate_joint_dataset

SPARSE_SCHEMA_VERSION = "align-sparse-joint-v2"


@dataclass(frozen=True)
class JointTrainConfig:
    manifest: Path
    cache_root: Path
    output_dir: Path
    initialize_checkpoint: Path | None = None
    seed: int = 365
    warmup_epochs: int = 1
    epochs: int = 4
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    dropout: float = 0.0
    gradient_accumulation: int = 8
    gradient_clip: float = 5.0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    device: str = "cuda"
    pairing_tolerance_sec: float = 0.050
    minimum_candidate_confidence: float = 0.65
    lattice: LatticeConfig = LatticeConfig()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows(path: Path, split: str) -> list[Mapping[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest has no non-empty {split!r} split")
    return rows


def _examples(
    rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    *,
    maximum: int | None,
    pairing_tolerance_sec: float,
    minimum_candidate_confidence: float,
) -> list[JointTrainingExample]:
    selected = rows[:maximum] if maximum is not None else rows
    output: list[JointTrainingExample] = []
    for index, row in enumerate(selected, 1):
        output.append(
            build_training_example(
                row,
                cache_root,
                pairing_tolerance_sec=pairing_tolerance_sec,
                minimum_candidate_confidence=minimum_candidate_confidence,
            )
        )
        if index == 1 or index % 100 == 0 or index == len(selected):
            print(
                f"loaded examples {index}/{len(selected)}",
                flush=True,
            )
    return output


def _predicted_deletions(path) -> frozenset[int]:
    values = set(path.trailing_deletions)
    for step in path.steps:
        values.update(step.deleted_events)
    return frozenset(values)


@torch.no_grad()
def evaluate_joint_model(
    model: JointEdgeScorer,
    examples: Sequence[JointTrainingExample],
    lattice_config: LatticeConfig,
) -> dict[str, object]:
    model.eval()
    lattice = SparseJointLattice(model, lattice_config)
    samples: list[JointMetricSample] = []
    for example in examples:
        path = lattice.decode(example.candidates, example.score)
        samples.append(
            JointMetricSample(
                predicted=path.joint_events(example.candidates),
                target=example.target_events,
                source=example.source,
                predicted_deletions=_predicted_deletions(path),
                target_deletions=example.target_deletions,
                score_event_count=len(example.score),
            )
        )
    return evaluate_joint_dataset(samples)


def bootstrap_joint_f1(
    model: JointEdgeScorer,
    examples: Sequence[JointTrainingExample],
    lattice_config: LatticeConfig,
    *,
    seed: int = 365,
    replicates: int = 1000,
) -> dict[str, float]:
    model.eval()
    lattice = SparseJointLattice(model, lattice_config)
    counts: list[tuple[float, int, int]] = []
    for example in examples:
        path = lattice.decode(example.candidates, example.score)
        report = evaluate_joint_dataset(
            [
                JointMetricSample(
                    predicted=path.joint_events(example.candidates),
                    target=example.target_events,
                    score_event_count=len(example.score),
                )
            ]
        )
        row = report["aggregate"]["official_note_wise"]
        counts.append(
            (
                float(row["credit"]),
                int(row["predicted"]),
                int(row["gold"]),
            )
        )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.integers(0, len(counts), size=len(counts))
        correct = sum(counts[index][0] for index in selected)
        predicted = sum(counts[index][1] for index in selected)
        target = sum(counts[index][2] for index in selected)
        precision = correct / max(predicted, 1)
        recall = correct / max(target, 1)
        values.append(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    return {
        "replicates": float(replicates),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def train_joint_model(config: JointTrainConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        config.device
        if config.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    train_rows = _manifest_rows(config.manifest, "train")
    val_rows = _manifest_rows(config.manifest, "val")
    train_examples = _examples(
        train_rows,
        config.cache_root,
        maximum=config.max_train_samples,
        pairing_tolerance_sec=config.pairing_tolerance_sec,
        minimum_candidate_confidence=config.minimum_candidate_confidence,
    )
    val_examples = _examples(
        val_rows,
        config.cache_root,
        maximum=config.max_val_samples,
        pairing_tolerance_sec=config.pairing_tolerance_sec,
        minimum_candidate_confidence=config.minimum_candidate_confidence,
    )
    model = JointEdgeScorer(config.hidden_dim, config.dropout).to(device)
    if config.initialize_checkpoint is not None:
        initial = torch.load(
            config.initialize_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if initial.get("schema_version") not in {
            "align-sparse-joint-v1",
            SPARSE_SCHEMA_VERSION,
        }:
            raise ValueError("Unsupported initialization checkpoint")
        model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lattice = SparseJointLattice(model, config.lattice)
    history: list[dict[str, Any]] = []
    best_f1 = -math.inf
    checkpoint = config.output_dir / "joint_decoder.pt"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for warmup_epoch in range(1, config.warmup_epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(config.seed + 1000 + warmup_epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        trained = 0
        for position, example_index in enumerate(order, 1):
            example = train_examples[example_index]
            if not example.candidates or not example.score:
                continue
            loss = lattice.local_warmup_nll(
                example.candidates,
                example.score,
                example.gold_spans,
                example.gold_keep_unlinked,
            )
            (loss / config.gradient_accumulation).backward()
            total_loss += float(loss.detach().cpu())
            trained += 1
            if trained % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if position == 1 or position % 250 == 0:
                print(
                    f"warmup={warmup_epoch} train={position}/{len(order)} "
                    f"loss={total_loss / max(trained, 1):.5f}",
                    flush=True,
                )
        if trained % config.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "stage": "local_warmup",
                "epoch": warmup_epoch,
                "mean_train_nll": total_loss / max(trained, 1),
                "trained_samples": trained,
            }
        )
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if config.epochs == 0:
        validation = evaluate_joint_model(
            model, val_examples, config.lattice
        )
        val_f1 = float(
            validation["aggregate"]["official_note_wise"]["f1"]
        )
        history[-1]["validation"] = validation
        _atomic_torch_save(
            {
                "schema_version": SPARSE_SCHEMA_VERSION,
                "state_dict": model.state_dict(),
                "model": {
                    "hidden_dim": config.hidden_dim,
                    "dropout": config.dropout,
                },
                "lattice": asdict(config.lattice),
                "training": {
                    **asdict(config),
                    "manifest": str(config.manifest),
                    "cache_root": str(config.cache_root),
                    "output_dir": str(config.output_dir),
                    "lattice": asdict(config.lattice),
                    "device": str(device),
                    "candidate_generation": CANDIDATE_GENERATION_VERSION,
                    "candidate_decode_configs": [
                        asdict(value)
                        for value in HIGH_RECALL_DECODE_CONFIGS
                    ],
                },
                "manifest_sha256": _sha256(config.manifest),
                "best_epoch": f"warmup-{config.warmup_epochs}",
                "best_validation_note_wise_f1": val_f1,
                "legacy_validation_joint_f1_50ms": validation["aggregate"][
                    "diagnostic_timestamp_tolerances"
                ]["50ms"]["joint"]["f1"],
                "history": history,
            },
            checkpoint,
        )
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(config.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        trained = 0
        skipped = 0
        for position, example_index in enumerate(order, 1):
            example = train_examples[example_index]
            if not example.candidates or not example.score:
                skipped += 1
                continue
            loss = lattice.nll(
                example.candidates,
                example.score,
                example.gold_spans,
                example.gold_keep_unlinked,
            ) / max(len(example.candidates), 1)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite joint loss for {example.sample}: {loss}"
                )
            (loss / config.gradient_accumulation).backward()
            total_loss += float(loss.detach().cpu())
            trained += 1
            if trained % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if position == 1 or position % 100 == 0:
                print(
                    f"epoch={epoch} train={position}/{len(order)} "
                    f"loss={total_loss / max(trained, 1):.5f}",
                    flush=True,
                )
        if trained % config.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        validation = evaluate_joint_model(
            model, val_examples, config.lattice
        )
        val_f1 = float(
            validation["aggregate"]["official_note_wise"]["f1"]
        )
        epoch_row = {
            "stage": "path_crf",
            "epoch": epoch,
            "mean_train_nll": total_loss / max(trained, 1),
            "trained_samples": trained,
            "skipped_samples": skipped,
            "validation": validation,
        }
        history.append(epoch_row)
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"epoch={epoch} val_note_wise_f1={val_f1:.6f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            _atomic_torch_save(
                {
                    "schema_version": SPARSE_SCHEMA_VERSION,
                    "state_dict": model.state_dict(),
                    "model": {
                        "hidden_dim": config.hidden_dim,
                        "dropout": config.dropout,
                    },
                    "lattice": asdict(config.lattice),
                    "training": {
                        **asdict(config),
                        "manifest": str(config.manifest),
                        "cache_root": str(config.cache_root),
                        "output_dir": str(config.output_dir),
                        "lattice": asdict(config.lattice),
                        "device": str(device),
                        "candidate_generation": CANDIDATE_GENERATION_VERSION,
                        "candidate_decode_configs": [
                            asdict(value)
                            for value in HIGH_RECALL_DECODE_CONFIGS
                        ],
                    },
                    "manifest_sha256": _sha256(config.manifest),
                    "best_epoch": epoch,
                    "best_validation_note_wise_f1": val_f1,
                    "legacy_validation_joint_f1_50ms": validation["aggregate"][
                        "diagnostic_timestamp_tolerances"
                    ]["50ms"]["joint"]["f1"],
                    "history": history,
                },
                checkpoint,
            )
    return checkpoint


def load_joint_model(
    checkpoint: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[JointEdgeScorer, LatticeConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema_version") not in {
        "align-sparse-joint-v1",
        SPARSE_SCHEMA_VERSION,
        "align-end-to-end-joint-v1",
        "align-end-to-end-joint-v2",
    }:
        raise ValueError("Unsupported joint checkpoint schema")
    model = JointEdgeScorer(**payload["model"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, LatticeConfig(**payload["lattice"]), payload
