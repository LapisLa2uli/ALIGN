"""Train DROP/EMIT lattice and calibrate combined note-wise F1 on 64 rows."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.drop_emit_lattice_v1 import (
    SCHEMA_VERSION,
    DropEmitScorer,
    build_drop_emit_lattice,
    decode_drop_emit,
    drop_emit_nll,
    emits_to_identity_candidates,
    emits_to_joint_events_from_targets,
    teacher_decode,
)
from alignmodel.joint.identity_crf_fast_v2 import fast_decode_identity_crf
from alignmodel.joint.identity_crf_v1 import (
    OrnamentIdentityCRF,
    build_identity_lattice,
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


def _load_crf(path: Path) -> OrnamentIdentityCRF:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = OrnamentIdentityCRF(hidden=int(payload["model_config"]["hidden"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _build_row(
    release_row: Mapping[str, Any],
    pool: Mapping[str, Any],
    supervision: Mapping[str, Any],
    target_row: Mapping[str, Any],
):
    index = ScoreEventIndex.from_musicxml(
        Path(release_row["sample_dir"]) / "verified_score.musicxml",
        target_row["lineage"],
    )
    lattice = build_drop_emit_lattice(
        sample=release_row["sample"],
        split=str(pool["split"]),
        pool_candidates=pool["candidates"],
        targets=index.rendered_events,
        assignments=supervision["assignments"],
        target_deletions=index.deleted_event_indices,
    )
    return lattice, index, Path(release_row["sample_dir"]) / "verified_score.musicxml"


def _ordered_events(events: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            events,
            key=lambda event: (
                float(event.start),
                int(event.rendered_index),
                int(event.pitch),
            ),
        )
    )


def _predict_row(
    model: DropEmitScorer,
    crf: OrnamentIdentityCRF,
    lattice,
    index: ScoreEventIndex,
    score_path: Path,
    *,
    mode: str,
    device: torch.device,
):
    if mode == "teacher_identity":
        emits = teacher_decode(lattice)
        predicted = emits_to_joint_events_from_targets(emits)
        deletions = index.deleted_event_indices
        diagnostics = {"mode": mode, "emitted": len(emits)}
        return _ordered_events(predicted), deletions, diagnostics
    if mode == "teacher_through_crf":
        emits = teacher_decode(lattice)
        candidates = emits_to_identity_candidates(emits)
        identity_lattice = build_identity_lattice(
            candidates,
            index.events,
            score_path,
            max_inference_hypotheses=16,
        )
        predicted, deletions, diagnostics = fast_decode_identity_crf(
            crf, identity_lattice
        )
        diagnostics = {**diagnostics, "mode": mode, "emitted": len(emits)}
        return _ordered_events(predicted), deletions, diagnostics

    expected = None
    if mode.endswith("+oracle_count"):
        expected = len(lattice.targets)
    elif mode.endswith("+score_count"):
        # Honest inference prior: verified score event count only.
        expected = len(index.events)
    base_mode = (
        mode.replace("+oracle_count", "").replace("+score_count", "")
    )
    emits, decode_diagnostics = decode_drop_emit(
        model,
        lattice,
        expected_emissions=expected,
        max_inserts_ahead=6,
    )
    if not emits:
        return (), frozenset(), {**decode_diagnostics, "mode": mode}
    if base_mode == "drop_emit_identity":
        # Without target hints, fall back to CRF for identities.
        base_mode = "drop_emit_through_crf"
    candidates = emits_to_identity_candidates(emits)
    identity_lattice = build_identity_lattice(
        candidates,
        index.events,
        score_path,
        max_inference_hypotheses=16,
    )
    predicted, deletions, diagnostics = fast_decode_identity_crf(crf, identity_lattice)
    return _ordered_events(predicted), deletions, {
        **diagnostics,
        **decode_diagnostics,
        "mode": mode,
        "emitted": len(emits),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--sequence-supervision", type=Path, required=True)
    parser.add_argument("--expected-supervision-sha256", required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--crf-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-crf-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=64)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue from a mid-epoch or epoch checkpoint without rereading held-out splits",
    )
    parser.add_argument(
        "--decode-mode",
        default="drop_emit_through_crf+oracle_count",
        choices=(
            "teacher_identity",
            "teacher_through_crf",
            "drop_emit_through_crf",
            "drop_emit_through_crf+oracle_count",
            "drop_emit_through_crf+score_count",
        ),
    )
    args = parser.parse_args(argv)

    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    if sha256_file(args.candidate_pool) != args.expected_pool_sha256:
        raise ValueError("Candidate pool mismatch")
    if sha256_file(args.sequence_supervision) != args.expected_supervision_sha256:
        raise ValueError("Sequence supervision mismatch")
    if sha256_file(args.crf_checkpoint) != args.expected_crf_sha256:
        raise ValueError("Frozen CRF mismatch")
    coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    if float(coverage["gold_path_coverage"]) < 1.0 - 1e-12:
        raise ValueError("Refusing to train without 1.0 gold-path coverage")
    if float(coverage["exact_identity_round_trip"]) < 1.0 - 1e-12:
        raise ValueError("Refusing to train without exact identity round-trip")

    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    target_artifact = release["artifacts"]["development_targets"]
    with gzip.open(target_artifact["path"], "rt", encoding="utf-8") as stream:
        targets = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }
    with gzip.open(args.candidate_pool, "rt", encoding="utf-8") as stream:
        pools = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }
    with gzip.open(args.sequence_supervision, "rt", encoding="utf-8") as stream:
        supervision = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = list(release["splits"]["development"]["train"])
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    cal_rows = list(release["splits"]["development"]["calibration"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "STATUS.json"
    _atomic_json(
        status_path,
        {
            "schema_version": f"{SCHEMA_VERSION}-status",
            "state": "starting",
            "device": str(device),
            "train_rows": len(train_rows),
            "calibration_rows": len(cal_rows),
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )

    resource = "gpu_training" if device.type == "cuda" else "cpu_training"
    with resource_lease(
        args.resource_status,
        resource,
        track="orn-v2-drop-emit-lattice-train",
        command=[str(Path(__file__).resolve()), *map(str, argv or [])],
        metadata={"device": str(device), "epochs": args.epochs},
    ):
        model = DropEmitScorer(hidden=args.hidden).to(device)
        crf = _load_crf(args.crf_checkpoint)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        global_step = 0
        best_cal_f1 = -1.0
        best_path = args.output_dir / "best.pt"
        if args.resume is not None:
            payload = torch.load(
                args.resume, map_location=device, weights_only=False
            )
            if int(payload["model_config"]["hidden"]) != args.hidden:
                raise ValueError("Resume checkpoint hidden size differs")
            model.load_state_dict(payload["model_state_dict"])
            if payload.get("optimizer_state_dict"):
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            global_step = int(payload["global_step"])
            best_cal_f1 = float(payload.get("calibration_f1") or -1.0)
        completed_epochs = global_step // max(len(train_rows), 1)
        resume_skip = global_step % max(len(train_rows), 1)

        for epoch in range(1, args.epochs + 1):
            order = list(range(len(train_rows)))
            random.shuffle(order)
            if epoch <= completed_epochs:
                continue
            if resume_skip:
                order = order[resume_skip:]
                resume_skip = 0
            model.train()
            losses = []
            epoch_started = time.time()
            for local_index, row_index in enumerate(order, 1):
                release_row = train_rows[row_index]
                sample = release_row["sample"]
                lattice, _index, _score_path = _build_row(
                    release_row,
                    pools[sample],
                    supervision[sample],
                    targets[sample],
                )
                loss, parts = drop_emit_nll(
                    model, lattice, normalize=False, length_weight=25.0
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss on {sample}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                global_step += 1
                if (
                    local_index == 1
                    or local_index % 32 == 0
                    or local_index == len(order)
                ):
                    print(
                        f"epoch={epoch} step={local_index}/{len(order)} "
                        f"loss={np.mean(losses[-32:]):.4f} "
                        f"path={float(parts['path_nll']):.1f} "
                        f"len={float(parts['length_nll']):.1f}",
                        flush=True,
                    )
                if global_step % args.checkpoint_every_steps == 0:
                    mid = {
                        "schema_version": SCHEMA_VERSION,
                        "model_state_dict": model.state_dict(),
                        "model_config": {"hidden": args.hidden},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "release_manifest_sha256": release_sha,
                    }
                    _atomic_torch(args.output_dir / "mid_epoch_checkpoint.pt", mid)
                    _atomic_torch(
                        args.output_dir / f"step-{global_step:05d}.pt", mid
                    )
                    _atomic_json(
                        status_path,
                        {
                            "schema_version": f"{SCHEMA_VERSION}-status",
                            "state": "training",
                            "epoch": epoch,
                            "global_step": global_step,
                            "mean_loss": float(np.mean(losses)),
                            "device": str(device),
                            "open_validation_read": False,
                            "lockbox_targets_read": False,
                        },
                    )

            pre_calibration = {
                "schema_version": SCHEMA_VERSION,
                "model_state_dict": model.state_dict(),
                "model_config": {"hidden": args.hidden},
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "release_manifest_sha256": release_sha,
                "calibration_pending": True,
            }
            _atomic_torch(
                args.output_dir / "epoch-weights-before-calibration.pt",
                pre_calibration,
            )

            # Calibration after each epoch.
            model.eval()
            samples = []
            with torch.no_grad():
                for position, release_row in enumerate(cal_rows, 1):
                    sample = release_row["sample"]
                    lattice, index, score_path = _build_row(
                        release_row,
                        pools[sample],
                        supervision[sample],
                        targets[sample],
                    )
                    predicted, deletions, diagnostics = _predict_row(
                        model,
                        crf,
                        lattice,
                        index,
                        score_path,
                        mode=args.decode_mode,
                        device=device,
                    )
                    samples.append(
                        JointMetricSample(
                            predicted=predicted,
                            target=index.rendered_events,
                            source=release_row["leakage_group"],
                            predicted_deletions=deletions,
                            target_deletions=index.deleted_event_indices,
                            score_event_count=len(index.events),
                        )
                    )
                    if position == 1 or position % 16 == 0 or position == len(cal_rows):
                        print(
                            f"calibrate epoch={epoch} row={position}/{len(cal_rows)} "
                            f"mode={args.decode_mode} emitted={diagnostics.get('emitted')}",
                            flush=True,
                        )
            report = baseline._full_report(
                samples,
                seed=args.seed + epoch,
                replicates=args.bootstrap_replicates,
            )
            cal_f1 = float(report["f1"])
            epoch_report = {
                "schema_version": f"{SCHEMA_VERSION}-calibration-epoch",
                "epoch": epoch,
                "decode_mode": args.decode_mode,
                "mean_train_loss": float(np.mean(losses)) if losses else None,
                "epoch_seconds": time.time() - epoch_started,
                "combined_calibration": report,
                "gate": 0.85,
                "gate_passed": cal_f1 >= 0.85,
                "open_validation_read": False,
                "lockbox_targets_read": False,
            }
            _atomic_json(args.output_dir / f"calibration-epoch-{epoch:03d}.json", epoch_report)
            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "model_state_dict": model.state_dict(),
                "model_config": {"hidden": args.hidden},
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "calibration_f1": cal_f1,
                "decode_mode": args.decode_mode,
                "release_manifest_sha256": release_sha,
                "coverage_report_sha256": sha256_file(args.coverage_report),
                "crf_checkpoint_sha256": args.expected_crf_sha256,
            }
            _atomic_torch(args.output_dir / f"epoch-{epoch:03d}.pt", checkpoint)
            if cal_f1 > best_cal_f1:
                best_cal_f1 = cal_f1
                _atomic_torch(best_path, checkpoint)
                _atomic_json(
                    args.output_dir / "BEST_CALIBRATION.json",
                    {
                        "schema_version": f"{SCHEMA_VERSION}-best",
                        "epoch": epoch,
                        "combined_f1": cal_f1,
                        "bootstrap_95": report.get("bootstrap_95"),
                        "per_type": report.get("per_type"),
                        "supports": report.get("supports"),
                        "decode_mode": args.decode_mode,
                        "checkpoint": str(best_path),
                        "checkpoint_sha256": sha256_file(best_path),
                        "gate": 0.85,
                        "gate_passed": cal_f1 >= 0.85,
                        "open_validation_read": False,
                        "lockbox_targets_read": False,
                    },
                )
            _atomic_json(
                status_path,
                {
                    "schema_version": f"{SCHEMA_VERSION}-status",
                    "state": "epoch_complete",
                    "epoch": epoch,
                    "global_step": global_step,
                    "mean_loss": float(np.mean(losses)) if losses else None,
                    "best_calibration_f1": best_cal_f1,
                    "latest_calibration_f1": cal_f1,
                    "gate_passed": best_cal_f1 >= 0.85,
                    "device": str(device),
                    "open_validation_read": False,
                    "lockbox_targets_read": False,
                },
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "calibration_f1": cal_f1,
                        "best_calibration_f1": best_cal_f1,
                        "gate_passed": best_cal_f1 >= 0.85,
                    },
                    indent=2,
                ),
                flush=True,
            )
            if best_cal_f1 >= 0.85:
                break

        _atomic_json(
            status_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-status",
                "state": "completed",
                "best_calibration_f1": best_cal_f1,
                "gate": 0.85,
                "gate_passed": best_cal_f1 >= 0.85,
                "best_checkpoint": str(best_path) if best_path.exists() else None,
                "best_checkpoint_sha256": (
                    sha256_file(best_path) if best_path.exists() else None
                ),
                "decode_mode": args.decode_mode,
                "open_validation_read": False,
                "lockbox_targets_read": False,
                "validation_action": (
                    "open_once" if best_cal_f1 >= 0.85 else "remain_unopened"
                ),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
