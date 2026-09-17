from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import JointTrainingExample
from .example_cache import CachedExampleSequence, JointExampleCache
from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate
from .metrics import pair_exact_pitch_onset


FEATURE_DIM = 17
CANONICAL_FEATURE_DIM = 22
SCHEMA_VERSION = "align-candidate-rescorer-v1"
MID_EPOCH_SCHEMA_VERSION = "align-candidate-rescorer-mid-epoch-v1"
PACKED_ROWS_SCHEMA_VERSION = "align-candidate-rescorer-packed-rows-v1"


@dataclass(frozen=True)
class CandidateRescorerConfig:
    manifest: Path
    basic_cache_root: Path
    example_cache_path: Path
    output_dir: Path
    seed: int = 365
    epochs: int = 3
    batch_candidates: int = 65536
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    dropout: float = 0.05
    candidate_floor: float = 0.50
    global_candidate_gate: float = 0.65
    hard_negative_ratio: float = 1.0
    short_weight_lt_80ms: float = 4.0
    short_weight_lt_120ms: float = 3.0
    short_weight_lt_180ms: float = 2.0
    workers: int = 3
    prefetch: int = 6
    device: str = "cuda"
    resume_checkpoint: Path | None = None
    checkpoint_every_clips: int = 0
    packed_training_root: Path | None = None


class PackedCandidateRows:
    def __init__(self, root: Path, expected_fingerprint: str) -> None:
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("schema_version") != PACKED_ROWS_SCHEMA_VERSION:
            raise ValueError("Unsupported packed candidate rows")
        if self.metadata.get("fingerprint") != expected_fingerprint:
            raise ValueError("Packed candidate row configuration mismatch")
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        rows = int(self.metadata["candidate_rows"])
        self.features = np.memmap(
            self.root / "features.float32.bin",
            dtype=np.float32,
            mode="r",
            shape=(rows, FEATURE_DIM),
        )
        self.labels = np.memmap(
            self.root / "labels.float32.bin",
            dtype=np.float32,
            mode="r",
            shape=(rows,),
        )
        self.weights = np.memmap(
            self.root / "weights.float32.bin",
            dtype=np.float32,
            mode="r",
            shape=(rows,),
        )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return (
            np.asarray(self.features[start:stop]),
            np.asarray(self.labels[start:stop]),
            np.asarray(self.weights[start:stop]),
        )


class CandidateRescorer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.05,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        confidence = features[..., 0].clamp(1e-4, 1.0 - 1e-4)
        prior = torch.log(confidence) - torch.log1p(-confidence)
        return prior + self.network(features).squeeze(-1)


def candidate_features(
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent] | None = None,
    *,
    feature_dim: int = FEATURE_DIM,
) -> np.ndarray:
    rows = []
    for index, candidate in enumerate(candidates):
        previous = candidates[index - 1] if index else None
        following = candidates[index + 1] if index + 1 < len(candidates) else None
        duration = max(candidate.end - candidate.start, 1e-3)
        previous_gap = (
            candidate.start - previous.end if previous is not None else 1.0
        )
        next_gap = (
            following.start - candidate.end if following is not None else 1.0
        )
        acoustic = tuple(candidate.acoustic_features) or (0.0,) * 5
        row = (
                candidate.confidence,
                min(duration / 1.5, 2.0),
                max(-2.0, min(2.0, math.log(duration / 0.12))),
                *acoustic,
                max(-2.0, min(2.0, previous_gap / 0.25)),
                max(-2.0, min(2.0, next_gap / 0.25)),
                (
                    max(-2.0, min(2.0, (candidate.pitch - previous.pitch) / 12.0))
                    if previous is not None
                    else 0.0
                ),
                (
                    max(-2.0, min(2.0, (following.pitch - candidate.pitch) / 12.0))
                    if following is not None
                    else 0.0
                ),
                float(previous is not None and previous.pitch == candidate.pitch),
                float(following is not None and following.pitch == candidate.pitch),
                float(previous is not None and candidate.start < previous.end),
                float(following is not None and following.start < candidate.end),
                candidate.confidence * candidate.confidence,
            )
        if feature_dim == CANONICAL_FEATURE_DIM:
            if score is None:
                raise ValueError("Canonical candidate features require a score")
            duration = max(
                max((value.end for value in candidates), default=1.0),
                1e-3,
            )
            expected = (
                candidate.start / duration * max(len(score) - 1, 0)
            )
            exact = [
                value.index for value in score if value.pitch == candidate.pitch
            ]
            nearest_index_error = (
                min(abs(value - expected) for value in exact)
                / max(len(score), 1)
                if exact
                else 1.0
            )
            local_exact = sum(abs(value - expected) <= 8 for value in exact)
            pitch_distance = min(
                (abs(value.pitch - candidate.pitch) for value in score),
                default=12,
            )
            row = (
                *row,
                expected / max(len(score), 1),
                float(bool(exact)),
                nearest_index_error,
                min(local_exact / 8.0, 1.0),
                min(pitch_distance / 12.0, 1.0),
            )
        elif feature_dim != FEATURE_DIM:
            raise ValueError(f"Unsupported candidate feature dimension: {feature_dim}")
        rows.append(row)
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (len(candidates), feature_dim):
        raise RuntimeError(
            f"Candidate feature shape mismatch: {result.shape}"
        )
    return result


def _duration_weight(
    duration: float, config: CandidateRescorerConfig
) -> float:
    if duration < 0.080:
        return config.short_weight_lt_80ms
    if duration < 0.120:
        return config.short_weight_lt_120ms
    if duration < 0.180:
        return config.short_weight_lt_180ms
    return 1.0


def _training_rows(
    example: JointTrainingExample,
    config: CandidateRescorerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = candidate_features(example.candidates)
    labels = np.asarray(example.gold_keep_unlinked, dtype=np.float32)
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels < 0.5)
    requested = min(
        len(negative),
        max(1, int(round(len(positive) * config.hard_negative_ratio))),
    )
    hard = (
        negative[
            np.argsort(
                [
                    -float(example.candidates[index].confidence)
                    for index in negative
                ]
            )[:requested]
        ]
        if requested
        else np.empty(0, dtype=np.int64)
    )
    selected = np.sort(np.concatenate((positive, hard)))
    weights = np.asarray(
        [
            (
                _duration_weight(
                    example.candidates[index].end
                    - example.candidates[index].start,
                    config,
                )
                if labels[index] > 0.5
                else 1.0
            )
            for index in selected
        ],
        dtype=np.float32,
    )
    return features[selected], labels[selected], weights


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _packed_rows_fingerprint(
    config: CandidateRescorerConfig, manifest_sha256: str
) -> str:
    payload = {
        "schema_version": PACKED_ROWS_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "candidate_floor": config.candidate_floor,
        "hard_negative_ratio": config.hard_negative_ratio,
        "short_weight_lt_80ms": config.short_weight_lt_80ms,
        "short_weight_lt_120ms": config.short_weight_lt_120ms,
        "short_weight_lt_180ms": config.short_weight_lt_180ms,
        "feature_dim": FEATURE_DIM,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _prepare_packed_training_rows(
    examples: Sequence[JointTrainingExample],
    config: CandidateRescorerConfig,
    root: Path,
    *,
    manifest_sha256: str,
) -> PackedCandidateRows:
    fingerprint = _packed_rows_fingerprint(config, manifest_sha256)
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        return PackedCandidateRows(root, fingerprint)
    staging = root.with_name(f".{root.name}.{os.getpid()}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    offsets = [0]
    try:
        with (
            (staging / "features.float32.bin").open("wb") as feature_file,
            (staging / "labels.float32.bin").open("wb") as label_file,
            (staging / "weights.float32.bin").open("wb") as weight_file,
        ):
            for position, example in enumerate(examples, 1):
                features, labels, weights = _training_rows(example, config)
                feature_file.write(features.astype(np.float32, copy=False).tobytes())
                label_file.write(labels.astype(np.float32, copy=False).tobytes())
                weight_file.write(weights.astype(np.float32, copy=False).tobytes())
                offsets.append(offsets[-1] + len(labels))
                if position == 1 or position % 500 == 0 or position == len(examples):
                    print(
                        f"candidate_pack={position}/{len(examples)} "
                        f"rows={offsets[-1]}",
                        flush=True,
                    )
        np.save(staging / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        metadata = {
            "schema_version": PACKED_ROWS_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "manifest_sha256": manifest_sha256,
            "examples": len(examples),
            "candidate_rows": offsets[-1],
            "feature_dim": FEATURE_DIM,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PackedCandidateRows(root, fingerprint)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state.get("torch", state.get("torch_cpu"))
    if torch_state is None:
        raise ValueError("Candidate checkpoint has no CPU torch RNG state")
    torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda", state.get("torch_cuda"))
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)


def _prf(correct: int, predicted: int, target: int) -> dict[str, float]:
    precision = correct / max(predicted, 1)
    recall = correct / max(target, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


@torch.no_grad()
def evaluate_candidate_rescorer(
    model: CandidateRescorer,
    examples: Sequence[JointTrainingExample],
    *,
    device: torch.device,
    thresholds: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    all_probabilities = []
    for position, example in enumerate(examples, 1):
        features = torch.from_numpy(candidate_features(example.candidates)).to(
            device
        )
        probabilities = model(features).sigmoid().cpu().numpy()
        all_probabilities.append(probabilities)
        if position == 1 or position % 250 == 0 or position == len(examples):
            print(
                f"candidate_validation={position}/{len(examples)}",
                flush=True,
            )
    best_threshold = 0.5
    best_f1 = -1.0
    best_counts = None
    best_sample_counts = None
    for threshold in thresholds:
        total = [0, 0, 0]
        per_sample = []
        short = [0, 0]
        split_count = 0
        diagnostic_pairs = 0
        for example, probabilities in zip(examples, all_probabilities):
            selected_indices = {
                index
                for index, probability in enumerate(probabilities)
                if probability >= threshold
            }
            target_indices = {
                index
                for index, keep in enumerate(example.gold_keep_unlinked)
                if keep
            }
            selected = [
                candidate
                for candidate, probability in zip(
                    example.candidates, probabilities
                )
                if probability >= threshold
            ]
            predicted = [
                JointEvent(
                    value.pitch,
                    value.start,
                    value.end,
                    None,
                    "extra",
                    confidence=float(probability),
                )
                for value, probability in zip(
                    example.candidates, probabilities
                )
                if probability >= threshold
            ]
            pairs = pair_exact_pitch_onset(
                predicted, example.target_events, tolerance_sec=0.050
            )
            diagnostic_pairs += len(pairs)
            counts = (
                len(selected_indices & target_indices),
                len(selected_indices),
                len(target_indices),
            )
            per_sample.append(counts)
            total = [left + right for left, right in zip(total, counts)]
            matched_targets = {target for _candidate, target in pairs}
            short_targets = {
                index
                for index, value in enumerate(example.target_events)
                if value.end - value.start < 0.120
            }
            short[0] += len(matched_targets & short_targets)
            short[1] += len(short_targets)
            split_count += sum(
                current.pitch == previous.pitch
                and current.start - previous.end <= 0.100
                for previous, current in zip(selected, selected[1:])
            )
        metrics = _prf(*total)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_counts = total
            best_sample_counts = per_sample
            best_detail = {
                **metrics,
                "matched": total[0],
                "predicted": total[1],
                "target": total[2],
                "count_ratio": total[1] / max(total[2], 1),
                "metric": "official audited candidate-event identity F1",
                "diagnostic_timestamp_pairs_50ms": diagnostic_pairs,
                "short_note_recall_lt_120ms": short[0] / max(short[1], 1),
                "short_note_matched": short[0],
                "short_note_target": short[1],
                "same_pitch_split_count": split_count,
                "same_pitch_split_rate": split_count / max(total[1], 1),
            }
    assert best_counts is not None and best_sample_counts is not None
    return best_threshold, {
        **best_detail,
        "threshold": best_threshold,
        "sample_counts": best_sample_counts,
    }


def train_candidate_rescorer(config: CandidateRescorerConfig) -> Path:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(
        config.device
        if config.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    manifest_bytes = config.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    train_rows = manifest["train"]
    val_rows = manifest["val"]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with JointExampleCache(
        config.example_cache_path,
        pairing_tolerance_sec=0.050,
        minimum_candidate_confidence=config.candidate_floor,
    ) as cache:
        cache_report = cache.prepare(
            [*train_rows, *val_rows],
            config.basic_cache_root,
            workers=config.workers,
            prefetch=config.prefetch,
            progress=lambda done, total, built: (
                print(
                    f"candidate_cache={done}/{total} built={built}",
                    flush=True,
                )
                if done == total or done == 1 or done % 250 == 0
                else None
            ),
        )
        train_examples = CachedExampleSequence(cache, train_rows)
        val_examples = CachedExampleSequence(cache, val_rows)
        training_examples: Sequence[Any] = train_examples
        packed_fingerprint = None
        if config.packed_training_root is not None:
            config.packed_training_root.parent.mkdir(parents=True, exist_ok=True)
            training_examples = _prepare_packed_training_rows(
                train_examples,
                config,
                config.packed_training_root,
                manifest_sha256=manifest_sha256,
            )
            packed_fingerprint = training_examples.metadata["fingerprint"]
        model = CandidateRescorer(
            config.hidden_dim, config.dropout
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        history = []
        best_f1 = -1.0
        best_path = config.output_dir / "candidate_rescorer.pt"
        mid_epoch_path = config.output_dir / "mid_epoch_checkpoint.pt"
        start_epoch = 1
        resume_cursor = 0
        resume_total_loss = 0.0
        resume_total_candidates = 0
        if config.resume_checkpoint is not None:
            resume = torch.load(
                config.resume_checkpoint, map_location="cpu", weights_only=False
            )
            schema = resume.get("schema_version")
            if schema not in {
                "align-candidate-rescorer-pause-v1",
                MID_EPOCH_SCHEMA_VERSION,
            }:
                raise ValueError("Unsupported candidate resume checkpoint")
            sampler = resume.get("sampler") or {}
            if (
                schema == "align-candidate-rescorer-pause-v1"
                and not resume.get("recoverable_training_state", False)
                and int(sampler.get("cursor", -1)) != 0
            ):
                raise ValueError("Unrecoverable candidate checkpoint cursor")
            model.load_state_dict(resume["state_dict"])
            if resume.get("optimizer_state_dict"):
                optimizer.load_state_dict(resume["optimizer_state_dict"])
            start_epoch = int(sampler.get("epoch", 1))
            resume_cursor = int(sampler.get("cursor", 0))
            resume_total_loss = float(resume.get("total_loss", 0.0))
            resume_total_candidates = int(resume.get("total_candidates", 0))
            history = list(resume.get("history") or [])
            best_f1 = float(resume.get("best_f1", -1.0))
            _restore_rng_state(resume.get("rng_state"))

        stop_requested = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True

        previous_handlers = {}
        for event in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[event] = signal.signal(event, request_stop)
        for epoch in range(start_epoch, config.epochs + 1):
            model.train()
            order = np.random.default_rng(config.seed + epoch).permutation(
                len(training_examples)
            )
            batch_features = []
            batch_labels = []
            batch_weights = []
            total_loss = resume_total_loss if epoch == start_epoch else 0.0
            total_candidates = (
                resume_total_candidates if epoch == start_epoch else 0
            )
            started = time.perf_counter()

            def flush() -> None:
                nonlocal total_loss, total_candidates
                if not batch_features:
                    return
                features = torch.from_numpy(
                    np.concatenate(batch_features)
                )
                labels = torch.from_numpy(np.concatenate(batch_labels))
                weights = torch.from_numpy(np.concatenate(batch_weights))
                if device.type == "cuda":
                    features = features.pin_memory()
                    labels = labels.pin_memory()
                    weights = weights.pin_memory()
                features = features.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                weights = weights.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                losses = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss = (losses * weights).sum() / weights.sum().clamp_min(1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                total_candidates += len(labels)
                batch_features.clear()
                batch_labels.clear()
                batch_weights.clear()

            def checkpoint(position: int) -> None:
                _atomic_torch(
                    mid_epoch_path,
                    {
                        "schema_version": MID_EPOCH_SCHEMA_VERSION,
                        "recoverable_training_state": True,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": None,
                        "scaler_state_dict": None,
                        "rng_state": _rng_state(),
                        "sampler": {"epoch": epoch, "cursor": position},
                        "total_loss": total_loss,
                        "total_candidates": total_candidates,
                        "history": history,
                        "best_f1": best_f1,
                        "manifest_sha256": manifest_sha256,
                        "packed_training_fingerprint": packed_fingerprint,
                        "config": {
                            **asdict(config),
                            "manifest": str(config.manifest),
                            "basic_cache_root": str(config.basic_cache_root),
                            "example_cache_path": str(config.example_cache_path),
                            "output_dir": str(config.output_dir),
                            "resume_checkpoint": (
                                str(config.resume_checkpoint)
                                if config.resume_checkpoint is not None
                                else None
                            ),
                            "packed_training_root": (
                                str(config.packed_training_root)
                                if config.packed_training_root is not None
                                else None
                            ),
                        },
                    },
                )

            queued = 0
            first_position = resume_cursor if epoch == start_epoch else 0
            for position, index in enumerate(
                order[first_position:], first_position + 1
            ):
                features, labels, weights = _training_rows(
                    train_examples[int(index)], config
                ) if training_examples is train_examples else (
                    training_examples[int(index)]
                )
                if queued and queued + len(labels) > config.batch_candidates:
                    flush()
                    queued = 0
                batch_features.append(features)
                batch_labels.append(labels)
                batch_weights.append(weights)
                queued += len(labels)
                if (
                    config.checkpoint_every_clips > 0
                    and position % config.checkpoint_every_clips == 0
                ):
                    flush()
                    queued = 0
                    checkpoint(position)
                if position % 500 == 0 or position == len(order):
                    elapsed = time.perf_counter() - started
                    print(
                        f"candidate_epoch={epoch} clips={position}/{len(order)} "
                        f"clips_per_sec={(position-first_position)/max(elapsed,1e-9):.3f} "
                        f"eta_sec={(len(order)-position)/max((position-first_position)/max(elapsed,1e-9),1e-9):.1f}",
                        flush=True,
                    )
                if stop_requested:
                    flush()
                    checkpoint(position)
                    print(
                        f"candidate_paused epoch={epoch} cursor={position}",
                        flush=True,
                    )
                    return mid_epoch_path
            flush()
            resume_cursor = 0
            resume_total_loss = 0.0
            resume_total_candidates = 0
            threshold, validation = evaluate_candidate_rescorer(
                model,
                val_examples,
                device=device,
                thresholds=np.linspace(0.20, 0.90, 71),
            )
            row = {
                "epoch": epoch,
                "mean_loss": total_loss / max(total_candidates, 1),
                "trained_candidates": total_candidates,
                "train_wall_seconds": time.perf_counter() - started,
                "validation": {
                    key: value
                    for key, value in validation.items()
                    if key != "sample_counts"
                },
            }
            history.append(row)
            if validation["f1"] > best_f1:
                best_f1 = float(validation["f1"])
                _atomic_torch(
                    best_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "state_dict": model.state_dict(),
                        "model": {
                            "hidden_dim": config.hidden_dim,
                            "dropout": config.dropout,
                        },
                        "threshold": threshold,
                        "validation": validation,
                        "config": {
                            **asdict(config),
                            "manifest": str(config.manifest),
                            "basic_cache_root": str(config.basic_cache_root),
                            "example_cache_path": str(config.example_cache_path),
                            "output_dir": str(config.output_dir),
                        },
                        "manifest_sha256": manifest_sha256,
                    },
                )
            (config.output_dir / "history.json").write_text(
                json.dumps(history, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            print(
                f"candidate_epoch={epoch} val_f1={validation['f1']:.6f} "
                f"threshold={threshold:.3f}",
                flush=True,
            )
        for event, handler in previous_handlers.items():
            signal.signal(event, handler)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        sample_counts = best["validation"]["sample_counts"]
        generator = np.random.default_rng(config.seed)
        bootstrap_values = []
        for _ in range(2000):
            selected = generator.integers(
                0, len(sample_counts), len(sample_counts)
            )
            aggregate = [
                sum(sample_counts[index][column] for index in selected)
                for column in range(3)
            ]
            bootstrap_values.append(_prf(*aggregate)["f1"])
        report = {
            "schema_version": "align-candidate-rescorer-report-v1",
            "checkpoint": str(best_path),
            "candidate_generation_floor": config.candidate_floor,
            "global_candidate_gate": config.global_candidate_gate,
            "confidence_band_trained": [0.50, 0.65],
            "duration_positive_weights": {
                "lt_80ms": config.short_weight_lt_80ms,
                "lt_120ms": config.short_weight_lt_120ms,
                "lt_180ms": config.short_weight_lt_180ms,
            },
            "hard_negative_ratio": config.hard_negative_ratio,
            "cache": cache_report,
            "validation": {
                key: value
                for key, value in best["validation"].items()
                if key != "sample_counts"
            },
            "bootstrap_candidate_event_identity_f1": {
                "replicates": 2000,
                "lower_95": float(np.quantile(bootstrap_values, 0.025)),
                "median": float(np.quantile(bootstrap_values, 0.5)),
                "upper_95": float(np.quantile(bootstrap_values, 0.975)),
            },
            "locked_test_touched": False,
        }
        (config.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return best_path


def load_candidate_rescorer(
    path: Path | str,
    device: str | torch.device = "cpu",
) -> tuple[CandidateRescorer, float, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported candidate rescorer checkpoint")
    model = CandidateRescorer(**payload["model"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, float(payload["threshold"]), payload


@torch.inference_mode()
def rescore_candidates(
    model: CandidateRescorer,
    candidates: Sequence[JointCandidate],
    *,
    threshold: float,
    score: Sequence[ScoreEvent] | None = None,
) -> tuple[JointCandidate, ...]:
    """Apply a validated candidate gate without changing candidate content."""

    rescored, _indices = rescore_candidates_with_indices(
        model,
        candidates,
        threshold=threshold,
        score=score,
    )
    return rescored


@torch.inference_mode()
def rescore_candidates_with_indices(
    model: CandidateRescorer,
    candidates: Sequence[JointCandidate],
    *,
    threshold: float,
    score: Sequence[ScoreEvent] | None = None,
) -> tuple[tuple[JointCandidate, ...], tuple[int, ...]]:
    """Apply the gate and retain exact source indices for aligned targets."""

    if not candidates:
        return (), ()
    device = next(model.parameters()).device
    probabilities = (
        model(
            torch.from_numpy(
                candidate_features(
                    candidates,
                    score,
                    feature_dim=model.feature_dim,
                )
            ).to(device)
        )
        .sigmoid()
        .cpu()
        .tolist()
    )
    kept = [
        (
            JointCandidate(
                pitch=candidate.pitch,
                start=candidate.start,
                end=candidate.end,
                confidence=float(probability),
                score_hints=candidate.score_hints,
                acoustic_features=candidate.acoustic_features,
            ),
            index,
        )
        for index, (candidate, probability) in enumerate(
            zip(candidates, probabilities)
        )
        if probability >= threshold
    ]
    return (
        tuple(candidate for candidate, _index in kept),
        tuple(index for _candidate, index in kept),
    )
