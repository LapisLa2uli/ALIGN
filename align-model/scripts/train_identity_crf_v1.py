"""Train the globally normalized identity CRF on a frozen ORN release."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_v1 import (
    SCHEMA_VERSION,
    IdentityCandidate,
    IdentityLattice,
    OrnamentIdentityCRF,
    build_identity_lattice,
    decode_identity_crf,
    identity_crf_nll,
)
from alignmodel.joint.identity_crf_fast_v2 import (
    fast_decode_identity_crf,
    fast_identity_crf_nll,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in value["cuda"]]
        )


def _targets(release: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifact = release["artifacts"]["development_targets"]
    path = Path(artifact["path"])
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("Development target archive mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }


def _build_examples(
    release: Mapping[str, Any],
    split: str,
    targets: Mapping[str, Mapping[str, Any]],
    *,
    max_rows: int | None,
    max_negative_hypotheses: int,
) -> list[dict[str, Any]]:
    rows = list(release["splits"]["development"][split])
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    output = []
    for position, row in enumerate(rows, 1):
        target_row = targets[row["sample"]]
        if target_row["split"] != split:
            raise ValueError(f"Target split mismatch: {row['sample']}")
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        index = ScoreEventIndex.from_musicxml(
            score_path, target_row["lineage"]
        )
        candidates = tuple(
            IdentityCandidate(event.pitch, event.start, event.end, 1.0)
            for event in index.rendered_events
        )
        lattice = build_identity_lattice(
            candidates,
            index.events,
            score_path,
            targets=index.rendered_events,
            target_deletions=index.deleted_event_indices,
            max_negative_hypotheses=max_negative_hypotheses,
        )
        output.append(
            {
                "sample": row["sample"],
                "source": row["leakage_group"],
                "score_path": score_path,
                "index": index,
                "candidates": candidates,
                "target": index.rendered_events,
                "lattice": lattice,
            }
        )
        if position == 1 or position % 25 == 0 or position == len(rows):
            print(f"lattice={split}:{position}/{len(rows)}", flush=True)
    return output


def _checkpoint(
    *,
    model: OrnamentIdentityCRF,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    release_sha256: str,
    coverage_sha256: str,
    progress: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": {"hidden": model.hidden},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": {
            "release_manifest_sha256": release_sha256,
            "coverage_report_sha256": coverage_sha256,
            "train_rows": args.max_train_rows or 901,
            "calibration_rows": args.max_calibration_rows or 64,
            "lockbox_targets_read": False,
        },
        "progress": dict(progress),
        "history": list(history),
        "rng_state": _rng_state(),
    }


@torch.no_grad()
def _evaluate_oracle(
    model: OrnamentIdentityCRF,
    examples: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
    runtime: str,
) -> dict[str, Any]:
    samples = []
    per_row = []
    for position, example in enumerate(examples, 1):
        inference_lattice = build_identity_lattice(
            example["candidates"],
            example["index"].events,
            example["score_path"],
            max_inference_hypotheses=16,
        )
        predicted, deletions, diagnostics = (
            fast_decode_identity_crf(model, inference_lattice)
            if runtime == "fast_v2"
            else decode_identity_crf(model, inference_lattice)
        )
        sample = JointMetricSample(
            predicted=predicted,
            target=example["target"],
            source=example["source"],
            predicted_deletions=deletions,
            target_deletions=example["index"].deleted_event_indices,
            score_event_count=len(example["index"].events),
        )
        samples.append(sample)
        per_row.append(
            {
                "sample": example["sample"],
                "metric": baseline._fractional_prf(
                    *baseline._counts(sample)
                ),
                "decode": diagnostics,
            }
        )
        if position == 1 or position % 10 == 0 or position == len(examples):
            print(f"calibration={position}/{len(examples)}", flush=True)
    return {
        **baseline._full_report(
            samples,
            seed=seed,
            replicates=bootstrap_replicates,
        ),
        "per_row": per_row,
    }


def train(args: argparse.Namespace) -> Path:
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen release hash mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Lockbox opening sentinel exists")
    coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    if (
        coverage["release_manifest_sha256"] != release_sha
        or not coverage["all_gold_paths_covered"]
        or not coverage["all_exact_identity_round_trips"]
    ):
        raise ValueError("Mandatory all-row identity coverage gate failed")
    coverage_sha = sha256_file(args.coverage_report)
    targets = _targets(release)
    train_examples = _build_examples(
        release,
        "train",
        targets,
        max_rows=args.max_train_rows,
        max_negative_hypotheses=args.max_negative_hypotheses,
    )
    calibration_examples = _build_examples(
        release,
        "calibration",
        targets,
        max_rows=args.max_calibration_rows,
        max_negative_hypotheses=0,
    )
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = OrnamentIdentityCRF(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(train_examples) * args.epochs),
        eta_min=args.learning_rate * 0.05,
    )
    epoch_start = 1
    position_start = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (
            saved.get("schema_version") != SCHEMA_VERSION
            or saved["data"]["release_manifest_sha256"] != release_sha
            or saved["data"]["coverage_report_sha256"] != coverage_sha
        ):
            raise ValueError("Identity CRF resume fingerprint mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        progress = saved["progress"]
        epoch_start = int(progress["epoch"])
        position_start = int(progress["position"])
        global_step = int(progress["global_step"])
        history = list(saved.get("history") or [])
        best_f1 = max(
            (
                float(row.get("calibration_oracle", {}).get("f1", -1.0))
                for row in history
            ),
            default=-1.0,
        )
        _restore_rng(saved["rng_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    mid_path = args.output_dir / "mid_epoch_checkpoint.pt"
    for epoch in range(epoch_start, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
        order = torch.randperm(
            len(train_examples), generator=generator
        ).tolist()
        position = position_start if epoch == epoch_start else 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        started = time.perf_counter()
        for order_position in range(position, len(order)):
            example = train_examples[order[order_position]]
            loss, _parts = (
                fast_identity_crf_nll(
                    model, example["lattice"], normalize=True
                )
                if args.runtime == "fast_v2"
                else identity_crf_nll(
                    model, example["lattice"], normalize=True
                )
            )
            regularization = args.transition_regularization * (
                model.transitions.square().mean()
                + model.initial.square().mean()
            )
            total = loss + regularization
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            position = order_position + 1
            global_step += 1
            running += float(total.detach())
            if (
                global_step % args.checkpoint_every_steps == 0
                and position < len(order)
            ):
                _atomic_torch(
                    mid_path,
                    _checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        release_sha256=release_sha,
                        coverage_sha256=coverage_sha,
                        progress={
                            "epoch": epoch,
                            "position": position,
                            "global_step": global_step,
                            "optimizer_boundary": True,
                        },
                        history=history,
                    ),
                )
            if global_step == 1 or global_step % 10 == 0:
                rate = (order_position + 1) / max(
                    time.perf_counter() - started, 1e-9
                )
                print(
                    f"epoch={epoch} rows={position}/{len(order)} "
                    f"loss={float(total.detach()):.5f} "
                    f"rows_per_sec={rate:.3f}",
                    flush=True,
                )
        model.eval()
        calibration = _evaluate_oracle(
            model,
            calibration_examples,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed + epoch,
            runtime=args.runtime,
        )
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": running / max(len(order), 1),
                "calibration_oracle": calibration,
            }
        )
        saved = _checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            release_sha256=release_sha,
            coverage_sha256=coverage_sha,
            progress={
                "epoch": epoch + 1,
                "position": 0,
                "global_step": global_step,
                "optimizer_boundary": True,
            },
            history=history,
        )
        epoch_path = args.output_dir / f"candidate-epoch-{epoch:03d}.pt"
        _atomic_torch(epoch_path, saved)
        _atomic_torch(args.output_dir / "last.pt", saved)
        if float(calibration["f1"]) > best_f1:
            best_f1 = float(calibration["f1"])
            _atomic_torch(best_path, saved)
        _atomic_json(args.output_dir / "history.json", {"history": history})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "calibration_oracle_f1": calibration["f1"],
                    "best_f1": best_f1,
                }
            ),
            flush=True,
        )
        position_start = 0
    _atomic_json(
        args.output_dir / "STATUS.json",
        {
            "schema_version": f"{SCHEMA_VERSION}-status",
            "status": (
                "calibration_oracle_gate_passed"
                if best_f1 >= args.oracle_gate
                else "calibration_oracle_gate_failed"
            ),
            "best_checkpoint": str(best_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path),
            "best_calibration_oracle_f1": best_f1,
            "oracle_gate": args.oracle_gate,
            "oracle_gate_passed": best_f1 >= args.oracle_gate,
            "release_manifest_sha256": release_sha,
            "coverage_report_sha256": coverage_sha,
            "lockbox_opened": False,
        },
    )
    return best_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--runtime", choices=("fast_v2", "reference_v1"), default="fast_v2"
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--transition-regularization", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-hypotheses", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-calibration-rows", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--oracle-gate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    role = "gpu" if args.device.startswith("cuda") else "cpu_validation"
    with resource_lease(
        args.resource_status,
        role,
        track="ornament-identity-crf-v1-training",
        command=[sys.executable, *sys.argv],
        metadata={
            "train_rows": args.max_train_rows or 901,
            "calibration_rows": args.max_calibration_rows or 64,
            "locked_test": False,
        },
    ):
        best = train(args)
    print(
        json.dumps(
            {"best": str(best), "best_sha256": sha256_file(best)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
