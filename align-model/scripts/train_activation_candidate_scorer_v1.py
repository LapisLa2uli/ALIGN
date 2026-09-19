"""Train exact-target activation candidate scorer and calibrate with frozen CRF."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import tempfile
import time
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import calibrate_v2_acoustic_crf as acoustic
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_fast_v2 import fast_decode_identity_crf
from alignmodel.joint.identity_crf_v1 import (
    IdentityCandidate,
    OrnamentIdentityCRF,
    build_identity_lattice,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.activation_candidate_scorer_v1 import (
    SCHEMA_VERSION,
    ActivationCandidateScorer,
    candidate_features,
    focal_candidate_loss,
    score_candidates,
)


THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


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


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _identity(value: Mapping[str, Any], probability: float) -> IdentityCandidate:
    return IdentityCandidate(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        confidence=float(probability),
        alternatives=tuple(int(item) for item in value["alternatives"]),
        alternative_confidences=tuple(
            float(item) for item in value["alternative_confidences"]
        ),
    )


def _load_crf(path: Path) -> OrnamentIdentityCRF:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = OrnamentIdentityCRF(hidden=payload["model_config"]["hidden"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--sequence-supervision", type=Path, required=True)
    parser.add_argument("--expected-supervision-sha256", required=True)
    parser.add_argument("--crf-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--checkpoint-every-batches", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    if sha256_file(args.candidate_pool) != args.expected_pool_sha256:
        raise ValueError("Candidate pool mismatch")
    if (
        sha256_file(args.sequence_supervision)
        != args.expected_supervision_sha256
    ):
        raise ValueError("Sequence supervision mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Lockbox opening sentinel exists")
    pools = {
        row["sample"]: row for row in _read_jsonl_gz(args.candidate_pool)
    }
    supervision = {
        row["sample"]: row
        for row in _read_jsonl_gz(args.sequence_supervision)
    }
    targets = acoustic._targets(release, "calibration")
    train_rows = release["splits"]["development"]["train"]
    calibration_rows = release["splits"]["development"]["calibration"]
    features = []
    labels = []
    weights = []
    for position, row in enumerate(train_rows, 1):
        pool = pools[row["sample"]]
        truth = supervision[row["sample"]]
        positive = {
            int(value["selected_interval_candidate"])
            for value in truth["assignments"]
        }
        hard = {
            int(candidate)
            for assignment in truth["assignments"]
            for candidate in assignment["candidate_indices"]
        } - positive
        for index, candidate in enumerate(pool["candidates"]):
            features.append(candidate_features(candidate))
            labels.append(int(index in positive))
            weights.append(3.0 if index in hard else 1.0)
        if position == 1 or position % 50 == 0 or position == len(train_rows):
            print(f"pack={position}/{len(train_rows)}", flush=True)
    x = torch.from_numpy(np.asarray(features, np.float32))
    y = torch.from_numpy(np.asarray(labels, np.float32))
    weight = torch.from_numpy(np.asarray(weights, np.float32))
    positives = int(y.sum())
    positive_weight = min(30.0, max(1.0, (len(y) - positives) / max(positives, 1)))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = ActivationCandidateScorer(hidden=64).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    batches_per_epoch = (len(y) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=batches_per_epoch * args.epochs,
        eta_min=args.learning_rate * 0.05,
    )
    epoch_start = 1
    batch_start = 0
    global_step = 0
    history = []
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (
            saved["release_manifest_sha256"] != release_sha
            or saved["candidate_pool_sha256"] != args.expected_pool_sha256
        ):
            raise ValueError("Candidate scorer resume mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        progress = saved["progress"]
        epoch_start = int(progress["epoch"])
        batch_start = int(progress["batch"])
        global_step = int(progress["global_step"])
        history = list(saved.get("history") or [])
        random.setstate(saved["rng_state"]["python"])
        np.random.set_state(saved["rng_state"]["numpy"])
        torch.set_rng_state(saved["rng_state"]["torch"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with resource_lease(
        args.resource_status,
        "gpu",
        track="activation-candidate-scorer-v1-training",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"train_rows": 901, "calibration_rows": 64, "locked_test": False},
    ):
        for epoch in range(epoch_start, args.epochs + 1):
            generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
            order = torch.randperm(len(y), generator=generator)
            dataset = TensorDataset(x[order], y[order], weight[order])
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            model.train()
            running = 0.0
            started = time.perf_counter()
            for batch_index, (batch_x, batch_y, batch_weight) in enumerate(loader):
                if epoch == epoch_start and batch_index < batch_start:
                    continue
                logits = model(batch_x.to(device, non_blocking=True))
                loss = focal_candidate_loss(
                    logits,
                    batch_y.to(device, non_blocking=True),
                    batch_weight.to(device, non_blocking=True),
                    positive_weight=positive_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                running += float(loss.detach())
                global_step += 1
                if global_step % args.checkpoint_every_batches == 0:
                    _atomic_torch(
                        args.output_dir / "mid_epoch_checkpoint.pt",
                        {
                            "schema_version": SCHEMA_VERSION,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "progress": {
                                "epoch": epoch,
                                "batch": batch_index + 1,
                                "global_step": global_step,
                                "optimizer_boundary": True,
                            },
                            "rng_state": {
                                "python": random.getstate(),
                                "numpy": np.random.get_state(),
                                "torch": torch.get_rng_state(),
                            },
                            "release_manifest_sha256": release_sha,
                            "candidate_pool_sha256": args.expected_pool_sha256,
                            "history": history,
                        },
                    )
                if batch_index == 0 or (batch_index + 1) % 50 == 0:
                    print(
                        f"epoch={epoch} batch={batch_index+1}/{len(loader)} "
                        f"loss={float(loss):.5f} "
                        f"batches_per_sec={(batch_index+1)/max(time.perf_counter()-started,1e-9):.2f}",
                        flush=True,
                    )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running / max(len(loader), 1),
                }
            )
            _atomic_torch(
                args.output_dir / f"candidate-scorer-epoch-{epoch:03d}.pt",
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_state_dict": model.state_dict(),
                    "model_config": {"hidden": model.hidden},
                    "release_manifest_sha256": release_sha,
                    "candidate_pool_sha256": args.expected_pool_sha256,
                    "sequence_supervision_sha256": args.expected_supervision_sha256,
                    "positive_weight": positive_weight,
                    "history": history,
                },
            )
            batch_start = 0
    crf = _load_crf(args.crf_checkpoint)
    calibration_scored = {}
    for row in calibration_rows:
        pool = pools[row["sample"]]
        calibration_scored[row["sample"]] = score_candidates(
            model, pool["candidates"], device
        )
    variants = []
    for variant_index, threshold in enumerate(THRESHOLDS):
        samples = []
        selected_counts = []
        for position, row in enumerate(calibration_rows, 1):
            pool = pools[row["sample"]]
            truth = supervision[row["sample"]]
            probability = calibration_scored[row["sample"]]
            selected = []
            for group in truth["candidate_groups"]:
                best = max(
                    group["candidate_indices"],
                    key=lambda index: (
                        float(probability[index]),
                        float(pool["candidates"][index]["confidence"]),
                        -int(index),
                    ),
                )
                if float(probability[best]) >= threshold:
                    selected.append(best)
            selected.sort(
                key=lambda index: (
                    float(pool["candidates"][index]["start"]),
                    int(pool["candidates"][index]["pitch"]),
                    float(pool["candidates"][index]["end"]),
                )
            )
            candidates = tuple(
                _identity(pool["candidates"][index], float(probability[index]))
                for index in selected
            )
            target_row = targets[row["sample"]]
            score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
            index = ScoreEventIndex.from_musicxml(
                score_path, target_row["lineage"]
            )
            lattice = build_identity_lattice(
                candidates,
                index.events,
                score_path,
                max_inference_hypotheses=16,
            )
            predicted, deletions, _diagnostics = fast_decode_identity_crf(
                crf, lattice
            )
            samples.append(
                JointMetricSample(
                    predicted=predicted,
                    target=index.rendered_events,
                    source=row["leakage_group"],
                    predicted_deletions=deletions,
                    target_deletions=index.deleted_event_indices,
                    score_event_count=len(index.events),
                )
            )
            selected_counts.append(len(candidates))
            if position == 1 or position % 16 == 0 or position == len(calibration_rows):
                print(
                    f"threshold={threshold:.2f} rows={position}/{len(calibration_rows)}",
                    flush=True,
                )
        variants.append(
            {
                "variant": variant_index,
                "threshold": threshold,
                "selected_candidates": sum(selected_counts),
                "combined": baseline._full_report(
                    samples,
                    seed=args.seed + variant_index,
                    replicates=args.bootstrap_replicates,
                ),
            }
        )
    selected = max(
        variants,
        key=lambda value: (
            float(value["combined"]["f1"]),
            -abs(float(value["threshold"]) - 0.5),
        ),
    )
    model_path = args.output_dir / "best.pt"
    _atomic_torch(
        model_path,
        {
            "schema_version": SCHEMA_VERSION,
            "model_state_dict": model.state_dict(),
            "model_config": {"hidden": model.hidden},
            "release_manifest_sha256": release_sha,
            "candidate_pool_sha256": args.expected_pool_sha256,
            "sequence_supervision_sha256": args.expected_supervision_sha256,
            "threshold": selected["threshold"],
            "history": history,
        },
    )
    report_path = args.output_dir / "calibration_report.json"
    _atomic_json(
        report_path,
        {
            "schema_version": f"{SCHEMA_VERSION}-calibration",
            "release_manifest_sha256": release_sha,
            "candidate_pool_sha256": args.expected_pool_sha256,
            "sequence_supervision_sha256": args.expected_supervision_sha256,
            "crf_checkpoint_sha256": sha256_file(args.crf_checkpoint),
            "train_examples": len(y),
            "train_positive_intervals": positives,
            "positive_weight": positive_weight,
            "predeclared_thresholds": list(THRESHOLDS),
            "variants": variants,
            "selection": selected,
            "official_metric": "exclusive one-to-one canonical identity/type 1/0.5/0",
            "timestamps_used_for_selection": False,
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    _atomic_json(
        args.output_dir / "STATUS.json",
        {
            "schema_version": f"{SCHEMA_VERSION}-status",
            "status": (
                "combined_calibration_gate_passed"
                if float(selected["combined"]["f1"]) >= 0.85
                else "combined_calibration_gate_failed"
            ),
            "model": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "threshold": selected["threshold"],
            "best_combined_calibration_f1": selected["combined"]["f1"],
            "combined_gate": 0.85,
            "combined_gate_passed": float(selected["combined"]["f1"]) >= 0.85,
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "open_validation_read": False,
            "lockbox_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "model_sha256": sha256_file(model_path),
                "threshold": selected["threshold"],
                "combined_f1": selected["combined"]["f1"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
