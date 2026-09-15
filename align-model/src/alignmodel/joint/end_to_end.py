from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .candidates import (
    CANDIDATE_GENERATION_VERSION,
    HIGH_RECALL_DECODE_CONFIGS,
)
from .data import JointTrainingExample, build_training_example
from .example_cache import CachedExampleSequence, JointExampleCache
from .lattice import (
    FEATURE_DIM,
    LEGACY_FEATURE_DIM,
    JointEdgeScorer,
    LatticeConfig,
    SparseJointLattice,
)
from .index import JointEvent
from .metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    pair_exact_pitch_onset,
)


SCHEMA_VERSION = "align-end-to-end-joint-v2"
LEGACY_SCHEMA_VERSION = "align-end-to-end-joint-v1"


@dataclass(frozen=True)
class EndToEndTrainConfig:
    manifest: Path
    cache_root: Path
    output_dir: Path
    initialize_checkpoint: Path
    resume_checkpoint: Path | None = None
    seed: int = 365
    local_epochs: int = 1
    path_epochs: int = 1
    local_learning_rate: float = 2e-4
    local_distillation_weight: float = 0.5
    path_learning_rate: float = 8e-5
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    component_dim: int = 32
    dropout: float = 0.0
    residual_scale: float = 0.10
    local_batch_edges: int = 65536
    path_gradient_accumulation: int = 8
    gradient_clip: float = 5.0
    max_train_samples: int | None = None
    path_train_samples: int = 1000
    max_val_samples: int | None = 100
    local_device: str = "cuda"
    path_device: str = "cpu"
    example_cache_path: Path | None = None
    preprocessing_workers: int = 4
    preprocessing_prefetch: int = 8
    checkpoint_every_rows: int = 250
    structured_cpu_threads: int = 1
    pairing_tolerance_sec: float = 0.050
    minimum_candidate_confidence: float = 0.65
    freeze_legacy_during_local: bool = True
    lattice: LatticeConfig = LatticeConfig(
        max_options_per_candidate=12,
        max_delete_events=24,
        max_states=48,
        noise_inference_bias=-6.0,
        continuation_feature_enabled=True,
        continuation_score_weight=0.35,
        continuation_hard_negative_copies=1,
        repeat_fragment_penalty=0.25,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _config_payload(config: EndToEndTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["lattice"] = asdict(config.lattice)
    payload["manifest"] = str(config.manifest)
    payload["cache_root"] = str(config.cache_root)
    payload["output_dir"] = str(config.output_dir)
    payload["initialize_checkpoint"] = str(config.initialize_checkpoint)
    payload["resume_checkpoint"] = (
        str(config.resume_checkpoint) if config.resume_checkpoint else None
    )
    payload["example_cache_path"] = (
        str(config.example_cache_path) if config.example_cache_path else None
    )
    payload["frontend"] = {
        "name": "Basic Pitch 0.4.0 activation cache",
        "frozen": True,
        "differentiated": False,
        "feature_dim": 5,
        "candidate_generation": CANDIDATE_GENERATION_VERSION,
        "decode_configs": [
            asdict(value) for value in HIGH_RECALL_DECODE_CONFIGS
        ],
    }
    payload["trainable_components"] = [
        "legacy edge network during path stage",
        "acoustic_projection",
        "emission_head",
        "score_projection",
        "option_head",
        "structural_projection",
        "transition_head",
        "path_head",
    ]
    payload["structured_math_dtype"] = "float32"
    return payload


def _resume_fingerprint(config: EndToEndTrainConfig) -> str:
    payload = _config_payload(config)
    for key in ("resume_checkpoint", "output_dir"):
        payload.pop(key, None)
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_rows(path: Path, split: str) -> list[Mapping[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest has no non-empty {split!r} split")
    return rows


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for vectorized local training")
    return torch.device(requested)


def _optimizer_to(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )


def _new_model(config: EndToEndTrainConfig) -> JointEdgeScorer:
    return JointEdgeScorer(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        component_mode=True,
        component_dim=config.component_dim,
        residual_scale=config.residual_scale,
    )


def _model_payload(config: EndToEndTrainConfig) -> dict[str, Any]:
    return {
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "component_mode": True,
        "component_dim": config.component_dim,
        "residual_scale": config.residual_scale,
    }


def _initialize_model(
    config: EndToEndTrainConfig,
    device: torch.device,
) -> tuple[JointEdgeScorer, dict[str, Any]]:
    payload = torch.load(
        config.initialize_checkpoint,
        map_location=device,
        weights_only=False,
    )
    schema = payload.get("schema_version")
    if schema not in {
        "align-sparse-joint-v1",
        "align-sparse-joint-v2",
        LEGACY_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported initialization schema: {schema!r}")
    if schema in {"align-sparse-joint-v1", "align-sparse-joint-v2"}:
        best_f1 = float(
            payload.get(
                "best_validation_note_wise_f1",
                payload.get("best_validation_joint_f1_50ms", -1.0),
            )
        )
        if best_f1 < 0.0:
            raise ValueError("Refusing invalid negative-NLL alignment checkpoint")
        negative_nll = [
            float(row["mean_train_nll"])
            for row in payload.get("history", [])
            if row.get("mean_train_nll") is not None
            and float(row["mean_train_nll"]) < -1e-4
        ]
        if negative_nll:
            raise ValueError("Refusing checkpoint with negative path NLL history")
    model = _new_model(config).to(device)
    if schema in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        model.load_state_dict(payload["state_dict"])
    else:
        incompatible = model.load_state_dict(
            payload["state_dict"], strict=False
        )
        if incompatible.unexpected_keys:
            raise ValueError(
                "Unexpected legacy checkpoint keys: "
                f"{incompatible.unexpected_keys}"
            )
        if any(key.startswith("network.") for key in incompatible.missing_keys):
            raise ValueError("Legacy alignment network was not fully initialized")
    return model, {
        "path": str(config.initialize_checkpoint),
        "sha256": _sha256(config.initialize_checkpoint),
        "schema_version": schema,
        "validation_note_wise_f1": payload.get("best_validation_note_wise_f1"),
        "legacy_validation_joint_f1_50ms": payload.get(
            "best_validation_joint_f1_50ms"
        ),
        "legacy_timestamp_metric_initialization": (
            payload.get("best_validation_note_wise_f1") is None
        ),
    }


def _load_resume(
    config: EndToEndTrainConfig,
    device: torch.device,
) -> tuple[JointEdgeScorer, dict[str, Any]]:
    if config.resume_checkpoint is None:
        raise ValueError("No resume checkpoint configured")
    payload = torch.load(
        config.resume_checkpoint, map_location=device, weights_only=False
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported end-to-end resume checkpoint")
    if payload.get("manifest_sha256") != _sha256(config.manifest):
        raise ValueError("Resume manifest hash mismatch")
    if payload.get("config_fingerprint") != _resume_fingerprint(config):
        raise ValueError("Resume configuration mismatch")
    model = _new_model(config).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, payload


def _checkpoint_payload(
    *,
    config: EndToEndTrainConfig,
    model: JointEdgeScorer,
    optimizer: torch.optim.Optimizer,
    optimizer_stage: str,
    initialization: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    progress: Mapping[str, Any],
    best_f1: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state_dict": model.state_dict(),
        "model": _model_payload(config),
        "lattice": asdict(config.lattice),
        "training": _config_payload(config),
        "manifest_sha256": _sha256(config.manifest),
        "config_fingerprint": _resume_fingerprint(config),
        "initialization": dict(initialization),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": (
            dict(scaler_state) if scaler_state is not None else None
        ),
        "rng_state": _rng_state(),
        "optimizer_stage": optimizer_stage,
        "history": list(history),
        "progress": dict(progress),
        "best_validation_note_wise_f1": float(best_f1),
    }


def _example(
    row: Mapping[str, Any],
    config: EndToEndTrainConfig,
) -> JointTrainingExample:
    return build_training_example(
        row,
        config.cache_root,
        pairing_tolerance_sec=config.pairing_tolerance_sec,
        minimum_candidate_confidence=config.minimum_candidate_confidence,
    )


def _load_examples(
    rows: Sequence[Mapping[str, Any]],
    config: EndToEndTrainConfig,
    *,
    label: str,
) -> list[JointTrainingExample]:
    output = []
    for position, row in enumerate(rows, 1):
        output.append(_example(row, config))
        if position == 1 or position % 100 == 0 or position == len(rows):
            print(f"loaded {label} examples {position}/{len(rows)}", flush=True)
    return output


@torch.no_grad()
def _evaluate_model_once(
    model: JointEdgeScorer,
    examples: Sequence[JointTrainingExample],
    lattice_config: LatticeConfig,
) -> tuple[dict[str, object], list[tuple[float, int, int]]]:
    model.eval()
    lattice = SparseJointLattice(model, lattice_config)
    samples: list[JointMetricSample] = []
    counts: list[tuple[float, int, int]] = []
    for position, example in enumerate(examples, 1):
        path = lattice.decode(example.candidates, example.score)
        predicted_deletions = set(path.trailing_deletions)
        for step in path.steps:
            predicted_deletions.update(step.deleted_events)
        sample = JointMetricSample(
            predicted=path.joint_events(example.candidates),
            target=example.target_events,
            source=example.source,
            predicted_deletions=frozenset(predicted_deletions),
            target_deletions=example.target_deletions,
            score_event_count=len(example.score),
        )
        samples.append(sample)
        row = evaluate_joint_dataset([sample])["aggregate"][
            "official_note_wise"
        ]
        counts.append(
            (
                float(row["credit"]),
                int(row["predicted"]),
                int(row["gold"]),
            )
        )
        if (
            position == 1
            or position % 100 == 0
            or position == len(examples)
        ):
            print(
                f"validation_decode={position}/{len(examples)}",
                flush=True,
            )
    return evaluate_joint_dataset(samples), counts


def _bootstrap_from_counts(
    counts: Sequence[tuple[float, int, int]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = generator.integers(0, len(counts), size=len(counts))
        correct = sum(counts[index][0] for index in selected)
        predicted = sum(counts[index][1] for index in selected)
        target = sum(counts[index][2] for index in selected)
        precision = correct / max(predicted, 1)
        recall = correct / max(target, 1)
        values.append(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )
    return {
        "replicates": float(replicates),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _transcriber_metrics(
    examples: Sequence[JointTrainingExample],
    *,
    tolerance_sec: float = 0.050,
    short_note_sec: float = 0.120,
) -> dict[str, float | int]:
    predicted_total = 0
    target_total = 0
    paired_total = 0
    short_total = 0
    short_paired = 0
    same_pitch_splits = 0
    for example in examples:
        predicted = [
            JointEvent(
                pitch=value.pitch,
                start=value.start,
                end=value.end,
                score_span=None,
                relationship="extra",
                confidence=value.confidence,
            )
            for value in example.candidates
        ]
        target = list(example.target_events)
        pairs = pair_exact_pitch_onset(
            predicted, target, tolerance_sec=tolerance_sec
        )
        predicted_total += len(predicted)
        target_total += len(target)
        paired_total += len(pairs)
        short_indices = {
            index
            for index, value in enumerate(target)
            if value.end - value.start <= short_note_sec
        }
        short_total += len(short_indices)
        short_paired += sum(
            target_index in short_indices for _, target_index in pairs
        )
        same_pitch_splits += sum(
            current.pitch == previous.pitch
            and current.start - previous.end <= 0.100
            for previous, current in zip(predicted, predicted[1:])
        )
    precision = paired_total / max(predicted_total, 1)
    recall = paired_total / max(target_total, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tolerance_ms": int(round(tolerance_sec * 1000)),
        "predicted": predicted_total,
        "target": target_total,
        "paired": paired_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_ratio": predicted_total / max(target_total, 1),
        "short_note_threshold_ms": int(round(short_note_sec * 1000)),
        "short_note_target": short_total,
        "short_note_paired": short_paired,
        "short_note_recall": short_paired / max(short_total, 1),
        "same_pitch_split_count": same_pitch_splits,
        "same_pitch_split_rate": (
            same_pitch_splits / max(predicted_total, 1)
        ),
    }


def _vectorized_group_nll(
    model: JointEdgeScorer,
    rows: Sequence[Sequence[float]],
    groups: Sequence[tuple[int, int, int]],
    device: torch.device,
    distillation_weight: float = 0.0,
) -> torch.Tensor:
    features = torch.tensor(rows, dtype=torch.float32, device=device)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected local feature matrix [N, {FEATURE_DIM}]")
    scores = model(features)
    starts = torch.tensor(
        [group[0] for group in groups], dtype=torch.long, device=device
    )
    lengths = torch.tensor(
        [group[1] - group[0] for group in groups],
        dtype=torch.long,
        device=device,
    )
    gold = starts + torch.tensor(
        [group[2] for group in groups], dtype=torch.long, device=device
    )
    group_ids = torch.repeat_interleave(
        torch.arange(len(groups), device=device), lengths
    )
    maxima = torch.full(
        (len(groups),),
        -torch.inf,
        dtype=scores.dtype,
        device=device,
    )
    maxima.scatter_reduce_(0, group_ids, scores, reduce="amax", include_self=True)
    sums = torch.zeros_like(maxima)
    sums.scatter_add_(0, group_ids, torch.exp(scores - maxima[group_ids]))
    log_partitions = maxima + torch.log(sums.clamp_min(1e-20))
    local_nll = (log_partitions - scores[gold]).mean()
    if distillation_weight <= 0.0:
        return local_nll
    with torch.no_grad():
        teacher = model.network(
            features[..., :LEGACY_FEATURE_DIM]
        ).squeeze(-1)
    preservation = torch.mean(torch.square(scores - teacher))
    return local_nll + float(distillation_weight) * preservation


def _train_local_epoch(
    *,
    model: JointEdgeScorer,
    optimizer: torch.optim.Optimizer,
    lattice: SparseJointLattice,
    rows: Sequence[Mapping[str, Any]],
    cache: JointExampleCache,
    config: EndToEndTrainConfig,
    epoch: int,
    device: torch.device,
    start_cursor: int = 0,
    initial_total_loss: float = 0.0,
    initial_total_groups: int = 0,
    on_checkpoint: Callable[[int, float, int], None] | None = None,
) -> tuple[float, int]:
    order = list(range(len(rows)))
    random.Random(config.seed + 10_000 + epoch).shuffle(order)
    batch_rows: list[list[float]] = []
    batch_groups: list[tuple[int, int, int]] = []
    total_loss = float(initial_total_loss)
    total_groups = int(initial_total_groups)
    optimizer.zero_grad(set_to_none=True)

    def flush() -> bool:
        nonlocal batch_rows, batch_groups, total_loss, total_groups
        if not batch_groups:
            return False
        loss = _vectorized_group_nll(
            model,
            batch_rows,
            batch_groups,
            device,
            config.local_distillation_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite vectorized local loss: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.gradient_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        count = len(batch_groups)
        total_loss += float(loss.detach().cpu()) * count
        total_groups += count
        batch_rows = []
        batch_groups = []
        return True

    last_checkpoint_cursor = int(start_cursor)
    for position in range(start_cursor + 1, len(order) + 1):
        row_index = order[position - 1]
        example = cache.get(rows[row_index])
        if example.candidates and example.score:
            edge_rows, groups = lattice.local_warmup_edges(
                example.candidates,
                example.score,
                example.gold_spans,
                example.gold_keep_unlinked,
            )
            if (
                batch_rows
                and len(batch_rows) + len(edge_rows)
                > config.local_batch_edges
            ):
                flushed = flush()
                completed_cursor = position - 1
                if (
                    flushed
                    and on_checkpoint is not None
                    and completed_cursor - last_checkpoint_cursor
                    >= config.checkpoint_every_rows
                ):
                    on_checkpoint(
                        completed_cursor, total_loss, total_groups
                    )
                    last_checkpoint_cursor = completed_cursor
            offset = len(batch_rows)
            batch_rows.extend(edge_rows)
            batch_groups.extend(
                (start + offset, end + offset, gold_offset)
                for start, end, gold_offset in groups
            )
        if position == 1 or position % 250 == 0 or position == len(order):
            print(
                f"local_epoch={epoch} train={position}/{len(order)} "
                f"groups={total_groups + len(batch_groups)}",
                flush=True,
            )
    flush()
    if on_checkpoint is not None:
        on_checkpoint(len(order), total_loss, total_groups)
    return total_loss / max(total_groups, 1), total_groups


def _train_path_epoch(
    *,
    model: JointEdgeScorer,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[JointTrainingExample],
    config: EndToEndTrainConfig,
    epoch: int,
    start_cursor: int = 0,
    initial_total_loss: float = 0.0,
    initial_trained: int = 0,
    on_checkpoint: Callable[[int, float, int], None] | None = None,
) -> tuple[float, int]:
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Structured path training requires FP32 parameters")
    lattice = SparseJointLattice(model, config.lattice)
    order = list(range(len(examples)))
    random.Random(config.seed + 20_000 + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    total_loss = float(initial_total_loss)
    trained = int(initial_trained)
    last_checkpoint_cursor = int(start_cursor)
    for position in range(start_cursor + 1, len(order) + 1):
        example = examples[order[position - 1]]
        if not example.candidates or not example.score:
            continue
        loss = lattice.nll(
            example.candidates,
            example.score,
            example.gold_spans,
            example.gold_keep_unlinked,
        ) / max(len(example.candidates), 1)
        if not torch.isfinite(loss) or float(loss.detach()) < -1e-4:
            raise FloatingPointError(
                f"Invalid structured path NLL for {example.sample}: {loss}"
            )
        (loss / config.path_gradient_accumulation).backward()
        total_loss += float(loss.detach().cpu())
        trained += 1
        if trained % config.path_gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if (
                on_checkpoint is not None
                and position - last_checkpoint_cursor
                >= config.checkpoint_every_rows
            ):
                on_checkpoint(position, total_loss, trained)
                last_checkpoint_cursor = position
        if position == 1 or position % 100 == 0 or position == len(order):
            print(
                f"path_epoch={epoch} train={position}/{len(order)} "
                f"loss={total_loss / max(trained, 1):.6f}",
                flush=True,
            )
    if trained % config.path_gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if on_checkpoint is not None:
        on_checkpoint(len(order), total_loss, trained)
    return total_loss / max(trained, 1), trained


def _save_epoch(
    *,
    config: EndToEndTrainConfig,
    model: JointEdgeScorer,
    optimizer: torch.optim.Optimizer,
    stage: str,
    initialization: Mapping[str, Any],
    history: list[dict[str, Any]],
    progress: Mapping[str, Any],
    best_f1: float,
    validation: Mapping[str, Any],
    bootstrap_counts: Sequence[tuple[float, int, int]] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    val_f1 = float(validation["aggregate"]["official_note_wise"]["f1"])
    next_best = max(best_f1, val_f1)
    payload = _checkpoint_payload(
        config=config,
        model=model,
        optimizer=optimizer,
        optimizer_stage=stage,
        initialization=initialization,
        history=history,
        progress=progress,
        best_f1=next_best,
        scheduler=scheduler,
    )
    _atomic_torch(config.output_dir / "last_checkpoint.pt", payload)
    if val_f1 >= best_f1:
        payload["best_validation"] = dict(validation)
        payload["best_validation_bootstrap_counts"] = list(
            bootstrap_counts or ()
        )
        _atomic_torch(config.output_dir / "joint_decoder.pt", payload)
    _atomic_json(config.output_dir / "history.json", {"history": history})
    return next_best


def _save_progress(
    *,
    config: EndToEndTrainConfig,
    model: JointEdgeScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    stage: str,
    initialization: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    progress: Mapping[str, Any],
    best_f1: float,
) -> None:
    payload = _checkpoint_payload(
        config=config,
        model=model,
        optimizer=optimizer,
        optimizer_stage=stage,
        initialization=initialization,
        history=history,
        progress=progress,
        best_f1=best_f1,
        scheduler=scheduler,
    )
    _atomic_torch(config.output_dir / "last_checkpoint.pt", payload)
    _atomic_json(
        config.output_dir / "progress.json",
        {
            "stage": stage,
            "progress": dict(progress),
            "updated_at_unix": time.time(),
            "checkpoint": "last_checkpoint.pt",
        },
    )


def train_end_to_end_model(config: EndToEndTrainConfig) -> Path:
    if config.local_epochs < 0 or config.path_epochs < 0:
        raise ValueError("Epoch counts must be non-negative")
    if config.max_train_samples is None and config.path_train_samples > 14718:
        raise ValueError("path_train_samples exceeds audited training split")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(config.output_dir / "config.json", _config_payload(config))
    train_rows = _manifest_rows(config.manifest, "train")
    val_rows = _manifest_rows(config.manifest, "val")
    if config.max_train_samples is not None:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_val_samples is not None:
        val_rows = val_rows[: config.max_val_samples]

    cache_path = (
        config.example_cache_path
        or config.output_dir.parent / "example-cache-v2.sqlite"
    )
    example_cache = JointExampleCache(
        cache_path,
        pairing_tolerance_sec=config.pairing_tolerance_sec,
        minimum_candidate_confidence=config.minimum_candidate_confidence,
    )

    def cache_progress(completed: int, total: int, built: int) -> None:
        if completed == total or completed == 1 or completed % 100 == 0:
            print(
                f"example_cache={completed}/{total} newly_built={built}",
                flush=True,
            )

    cache_started = time.perf_counter()
    cache_report = example_cache.prepare(
        [*train_rows, *val_rows],
        config.cache_root,
        workers=max(1, config.preprocessing_workers),
        prefetch=max(1, config.preprocessing_prefetch),
        progress=cache_progress,
    )
    cache_report["wall_seconds"] = time.perf_counter() - cache_started
    cache_report["metadata"] = example_cache.export_metadata()
    _atomic_json(config.output_dir / "example_cache_report.json", cache_report)
    val_examples: Sequence[JointTrainingExample] = CachedExampleSequence(
        example_cache, val_rows
    )

    local_device = _device(config.local_device)
    initialization: dict[str, Any]
    resume_payload: dict[str, Any] | None = None
    if config.resume_checkpoint is not None:
        model, resume_payload = _load_resume(config, local_device)
        initialization = dict(resume_payload["initialization"])
        history = list(resume_payload.get("history", []))
        progress = dict(resume_payload.get("progress", {}))
        best_f1 = float(
            resume_payload.get("best_validation_note_wise_f1", -math.inf)
        )
    else:
        model, initialization = _initialize_model(config, local_device)
        history = []
        progress = {
            "local_epoch": 0,
            "local_in_epoch": 0,
            "local_cursor": 0,
            "local_total_loss": 0.0,
            "local_total_groups": 0,
            "path_epoch": 0,
            "path_in_epoch": 0,
            "path_cursor": 0,
            "path_total_loss": 0.0,
            "path_trained": 0,
        }
        best_f1 = -math.inf

    for parameter in model.network.parameters():
        parameter.requires_grad = not config.freeze_legacy_during_local
    local_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.local_learning_rate,
        weight_decay=config.weight_decay,
    )
    local_scheduler = torch.optim.lr_scheduler.LambdaLR(
        local_optimizer, lambda _step: 1.0
    )
    if (
        resume_payload is not None
        and resume_payload.get("optimizer_stage") == "vectorized_local"
    ):
        local_optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        _optimizer_to(local_optimizer, local_device)
        if resume_payload.get("scheduler_state_dict") is not None:
            local_scheduler.load_state_dict(
                resume_payload["scheduler_state_dict"]
            )
        _restore_rng_state(resume_payload.get("rng_state"))
    local_lattice = SparseJointLattice(model, config.lattice)
    if resume_payload is None:
        torch.set_num_threads(max(1, config.structured_cpu_threads))
        model.to("cpu")
        validation_started = time.perf_counter()
        initial_validation, initial_counts = _evaluate_model_once(
            model, val_examples, config.lattice
        )
        initial_validation_seconds = time.perf_counter() - validation_started
        model.to(local_device)
        history.append(
            {
                "stage": "initialized_alignment",
                "epoch": 0,
                "validation_wall_seconds": initial_validation_seconds,
                "validation": initial_validation,
            }
        )
        best_f1 = _save_epoch(
            config=config,
            model=model,
            optimizer=local_optimizer,
            stage="vectorized_local",
            initialization=initialization,
            history=history,
            progress=progress,
            best_f1=best_f1,
            validation=initial_validation,
            bootstrap_counts=initial_counts,
            scheduler=local_scheduler,
        )
    local_start = int(
        progress.get("local_in_epoch")
        or (int(progress.get("local_epoch", 0)) + 1)
    )
    for epoch in range(local_start, config.local_epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        start_cursor = (
            int(progress.get("local_cursor", 0))
            if int(progress.get("local_in_epoch", 0)) == epoch
            else 0
        )
        initial_total_loss = (
            float(progress.get("local_total_loss", 0.0))
            if start_cursor
            else 0.0
        )
        initial_total_groups = (
            int(progress.get("local_total_groups", 0))
            if start_cursor
            else 0
        )

        def save_local_progress(
            cursor: int, total_loss: float, total_groups: int
        ) -> None:
            progress.update(
                {
                    "local_in_epoch": epoch,
                    "local_cursor": cursor,
                    "local_total_loss": total_loss,
                    "local_total_groups": total_groups,
                }
            )
            _save_progress(
                config=config,
                model=model,
                optimizer=local_optimizer,
                scheduler=local_scheduler,
                stage="vectorized_local",
                initialization=initialization,
                history=history,
                progress=progress,
                best_f1=best_f1,
            )

        mean_loss, candidate_groups = _train_local_epoch(
            model=model,
            optimizer=local_optimizer,
            lattice=local_lattice,
            rows=train_rows,
            cache=example_cache,
            config=config,
            epoch=epoch,
            device=local_device,
            start_cursor=start_cursor,
            initial_total_loss=initial_total_loss,
            initial_total_groups=initial_total_groups,
            on_checkpoint=save_local_progress,
        )
        train_seconds = time.perf_counter() - epoch_started
        local_scheduler.step()
        torch.set_num_threads(max(1, config.structured_cpu_threads))
        model.to("cpu")
        validation_started = time.perf_counter()
        validation, validation_counts = _evaluate_model_once(
            model, val_examples, config.lattice
        )
        validation_seconds = time.perf_counter() - validation_started
        model.to(local_device)
        history.append(
            {
                "stage": "vectorized_local",
                "epoch": epoch,
                "mean_train_nll": mean_loss,
                "candidate_groups": candidate_groups,
                "trained_samples": len(train_rows),
                "train_wall_seconds": train_seconds,
                "train_rows_per_second": len(train_rows)
                / max(train_seconds, 1e-12),
                "validation_wall_seconds": validation_seconds,
                "validation": validation,
            }
        )
        progress.update(
            {
                "local_epoch": epoch,
                "local_in_epoch": 0,
                "local_cursor": 0,
                "local_total_loss": 0.0,
                "local_total_groups": 0,
            }
        )
        best_f1 = _save_epoch(
            config=config,
            model=model,
            optimizer=local_optimizer,
            stage="vectorized_local",
            initialization=initialization,
            history=history,
            progress=progress,
            best_f1=best_f1,
            validation=validation,
            bootstrap_counts=validation_counts,
            scheduler=local_scheduler,
        )
        print(
            f"local_epoch={epoch} val_note_wise_f1="
            f"{history[-1]['validation']['aggregate']['official_note_wise']['f1']:.6f}",
            flush=True,
        )

    for parameter in model.parameters():
        parameter.requires_grad = True
    torch.set_num_threads(max(1, config.structured_cpu_threads))
    path_device = _device(config.path_device)
    model.to(path_device)
    path_start = int(
        progress.get("path_in_epoch")
        or (int(progress.get("path_epoch", 0)) + 1)
    )
    local_order = list(range(len(train_rows)))
    random.Random(config.seed + 10_001).shuffle(local_order)
    path_rows = [
        train_rows[index]
        for index in local_order[: config.path_train_samples]
    ]
    path_examples: Sequence[JointTrainingExample] = CachedExampleSequence(
        example_cache, path_rows
    )
    path_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.path_learning_rate,
        weight_decay=config.weight_decay,
    )
    path_scheduler = torch.optim.lr_scheduler.LambdaLR(
        path_optimizer, lambda _step: 1.0
    )
    if (
        resume_payload is not None
        and resume_payload.get("optimizer_stage") == "structured_path"
    ):
        path_optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        _optimizer_to(path_optimizer, path_device)
        if resume_payload.get("scheduler_state_dict") is not None:
            path_scheduler.load_state_dict(
                resume_payload["scheduler_state_dict"]
            )
        _restore_rng_state(resume_payload.get("rng_state"))
    for epoch in range(path_start, config.path_epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        start_cursor = (
            int(progress.get("path_cursor", 0))
            if int(progress.get("path_in_epoch", 0)) == epoch
            else 0
        )
        initial_total_loss = (
            float(progress.get("path_total_loss", 0.0))
            if start_cursor
            else 0.0
        )
        initial_trained = (
            int(progress.get("path_trained", 0))
            if start_cursor
            else 0
        )

        def save_path_progress(
            cursor: int, total_loss: float, trained: int
        ) -> None:
            progress.update(
                {
                    "path_in_epoch": epoch,
                    "path_cursor": cursor,
                    "path_total_loss": total_loss,
                    "path_trained": trained,
                }
            )
            _save_progress(
                config=config,
                model=model,
                optimizer=path_optimizer,
                scheduler=path_scheduler,
                stage="structured_path",
                initialization=initialization,
                history=history,
                progress=progress,
                best_f1=best_f1,
            )

        mean_loss, trained = _train_path_epoch(
            model=model,
            optimizer=path_optimizer,
            examples=path_examples,
            config=config,
            epoch=epoch,
            start_cursor=start_cursor,
            initial_total_loss=initial_total_loss,
            initial_trained=initial_trained,
            on_checkpoint=save_path_progress,
        )
        train_seconds = time.perf_counter() - epoch_started
        path_scheduler.step()
        validation_started = time.perf_counter()
        validation, validation_counts = _evaluate_model_once(
            model, val_examples, config.lattice
        )
        validation_seconds = time.perf_counter() - validation_started
        history.append(
            {
                "stage": "structured_path",
                "epoch": epoch,
                "mean_train_nll": mean_loss,
                "trained_samples": trained,
                "train_wall_seconds": train_seconds,
                "train_rows_per_second": trained
                / max(train_seconds, 1e-12),
                "validation_wall_seconds": validation_seconds,
                "validation": validation,
            }
        )
        progress.update(
            {
                "path_epoch": epoch,
                "path_in_epoch": 0,
                "path_cursor": 0,
                "path_total_loss": 0.0,
                "path_trained": 0,
            }
        )
        best_f1 = _save_epoch(
            config=config,
            model=model,
            optimizer=path_optimizer,
            stage="structured_path",
            initialization=initialization,
            history=history,
            progress=progress,
            best_f1=best_f1,
            validation=validation,
            bootstrap_counts=validation_counts,
            scheduler=path_scheduler,
        )
        print(
            f"path_epoch={epoch} val_note_wise_f1="
            f"{history[-1]['validation']['aggregate']['official_note_wise']['f1']:.6f}",
            flush=True,
        )

    best_path = config.output_dir / "joint_decoder.pt"
    if not best_path.is_file():
        raise RuntimeError("No completed end-to-end epoch produced a checkpoint")
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = _new_model(config)
    best_model.load_state_dict(best_payload["state_dict"])
    best_validation = best_payload["best_validation"]
    bootstrap_counts = best_payload.get(
        "best_validation_bootstrap_counts"
    )
    if not bootstrap_counts:
        _validation, bootstrap_counts = _evaluate_model_once(
            best_model, val_examples, config.lattice
        )
    bootstrap = _bootstrap_from_counts(
        bootstrap_counts,
        seed=config.seed,
        replicates=1000,
    )
    selected_f1 = float(best_validation["aggregate"]["official_note_wise"]["f1"])
    transcriber_metrics = _transcriber_metrics(val_examples)
    full_validation = len(val_examples) == 1871
    report = {
        "schema_version": "align-end-to-end-experiment-report-v2",
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "initialization": initialization,
        "train_rows": len(train_rows),
        "validation_rows": len(val_examples),
        "path_train_rows": min(config.path_train_samples, len(train_rows)),
        "example_cache": cache_report,
        "structured_cpu_threads": config.structured_cpu_threads,
        "frozen_frontend": _config_payload(config)["frontend"],
        "transcriber": transcriber_metrics,
        "best_validation": best_validation,
        "full_end_to_end_note_wise_f1": selected_f1,
        "bootstrap_note_wise_f1": bootstrap,
        "comparisons": {
            "path_1000_v4_interval": 0.7430444282592862,
            "current_baseline": 0.6254916241806264,
            "selected": selected_f1,
            "delta_vs_path_1000_v4_interval": (
                selected_f1 - 0.7430444282592862
            ),
            "delta_vs_current_baseline": (
                selected_f1 - 0.6254916241806264
            ),
        },
        "promotion_gate": {
            "full_validation_required": True,
            "full_validation_completed": full_validation,
            "f1_threshold": 0.83,
            "bootstrap_lower_threshold": 0.80,
            "passed": (
                full_validation
                and selected_f1 >= 0.83
                and bootstrap["lower_95"] >= 0.80
            ),
        },
        "locked_test_touched": False,
    }
    _atomic_json(config.output_dir / "report.json", report)
    example_cache.close()
    return best_path


def load_end_to_end_model(
    checkpoint: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[JointEdgeScorer, LatticeConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema_version") not in {
        LEGACY_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        raise ValueError("Unsupported end-to-end checkpoint schema")
    model = JointEdgeScorer(**payload["model"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, LatticeConfig(**payload["lattice"]), payload
