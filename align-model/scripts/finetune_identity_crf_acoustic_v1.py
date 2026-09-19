"""Fine-tune identity CRF on train-only Basic Pitch candidate sequences."""

from __future__ import annotations

import argparse
import atexit
import gzip
import json
import os
import random
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import calibrate_v2_acoustic_crf as acoustic
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_fast_v2 import (
    fast_decode_identity_crf,
    fast_identity_crf_nll,
)
from alignmodel.joint.identity_crf_v1 import (
    IdentityCandidate,
    IdentityTarget,
    OrnamentIdentityCRF,
    build_identity_lattice,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import JointMetricSample
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


SCHEMA_VERSION = "align-identity-crf-acoustic-finetune-v1"


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


def _lcs_pairs(left: Sequence[int], right: Sequence[int]) -> list[tuple[int, int]]:
    counts = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, value in enumerate(left, 1):
        for j, other in enumerate(right, 1):
            counts[i][j] = (
                counts[i - 1][j - 1] + 1
                if value == other
                else max(counts[i - 1][j], counts[i][j - 1])
            )
    pairs = []
    i, j = len(left), len(right)
    while i and j:
        if left[i - 1] == right[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif counts[i - 1][j] >= counts[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def _load_targets(release: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifact = release["artifacts"]["development_targets"]
    path = Path(artifact["path"])
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("Development target archive mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
            if row["split"] in {"train", "calibration"}
        }


def _candidate(value: Mapping[str, Any]) -> IdentityCandidate:
    return IdentityCandidate(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        confidence=float(value.get("confidence", 1.0)),
        alternatives=tuple(int(item) for item in value.get("alternatives") or ()),
        alternative_confidences=tuple(
            float(item)
            for item in value.get("alternative_confidences") or ()
        ),
    )


def _pseudo_targets(
    candidates: Sequence[IdentityCandidate],
    target_events: Sequence[Any],
    score_count: int,
    original_deletions: Sequence[int],
) -> tuple[tuple[IdentityTarget, ...], frozenset[int], dict[str, int]]:
    pairs = _lcs_pairs(
        [value.pitch for value in candidates],
        [value.pitch for value in target_events],
    )
    target_by_candidate = dict(pairs)
    pseudo = []
    linked_score = set()
    exact_extra_identity = 0
    for index, candidate in enumerate(candidates):
        target_index = target_by_candidate.get(index)
        if target_index is None:
            pseudo.append(
                IdentityTarget("extra", None, 0, index, candidate.pitch)
            )
            continue
        event = target_events[target_index]
        if event.score_span is not None:
            value = IdentityTarget.from_event(event)
            pseudo.append(value)
            if value.copy_pass == 0:
                linked_score.update(range(*value.score_span))
        elif event.rendered_index == index:
            pseudo.append(IdentityTarget.from_event(event))
            exact_extra_identity += 1
        else:
            pseudo.append(
                IdentityTarget("extra", None, 0, index, candidate.pitch)
            )
    deletions = frozenset(
        set(int(value) for value in original_deletions)
        | (set(range(score_count)) - linked_score)
    )
    return tuple(pseudo), deletions, {
        "pitch_lcs_pairs": len(pairs),
        "candidates": len(candidates),
        "target_events": len(target_events),
        "pseudo_candidate_extras": sum(
            value.relationship == "extra" and value.score_span is None
            for value in pseudo
        ),
        "exact_extra_identities": exact_extra_identity,
        "pseudo_deletions": len(deletions),
    }


def _build_train(
    release: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
    predictions_path: Path,
    *,
    max_negative_hypotheses: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    predictions = {
        row["sample"]: row
        for row in (
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    rows = release["splits"]["development"]["train"]
    if set(predictions) != {row["sample"] for row in rows}:
        raise ValueError("Train candidate population mismatch")
    output = []
    totals: dict[str, int] = {}
    for position, row in enumerate(rows, 1):
        target_row = targets[row["sample"]]
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        index = ScoreEventIndex.from_musicxml(score_path, target_row["lineage"])
        candidates = tuple(
            _candidate(value)
            for value in predictions[row["sample"]]["candidates"]
        )
        pseudo, deletions, stats = _pseudo_targets(
            candidates,
            index.rendered_events,
            len(index.events),
            index.deleted_event_indices,
        )
        synthetic_events = tuple(
            baseline.JointEvent(
                pitch=value.pitch,
                start=candidates[i].start,
                end=candidates[i].end,
                score_span=value.score_span,
                relationship=value.relationship,
                copy_pass=value.copy_pass,
                rendered_index=i,
            )
            for i, value in enumerate(pseudo)
        )
        lattice = build_identity_lattice(
            candidates,
            index.events,
            score_path,
            targets=synthetic_events,
            target_deletions=deletions,
            max_negative_hypotheses=max_negative_hypotheses,
        )
        output.append(
            {
                "sample": row["sample"],
                "lattice": lattice,
            }
        )
        for name, value in stats.items():
            totals[name] = totals.get(name, 0) + int(value)
        if position == 1 or position % 25 == 0 or position == len(rows):
            print(f"train-lattice={position}/{len(rows)}", flush=True)
    totals["rows"] = len(output)
    return output, totals


def _calibration_contexts(
    release: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
    calibration_report: Path,
) -> list[dict[str, Any]]:
    report = json.loads(calibration_report.read_text(encoding="utf-8"))
    variant = next(
        value
        for value in report["variants"]
        if value["variant"] == report["selection"]["variant"]
    )
    predictions = variant["predictions"]
    output = []
    for row in release["splits"]["development"]["calibration"]:
        score_path = Path(row["sample_dir"]) / "verified_score.musicxml"
        index = ScoreEventIndex.from_musicxml(
            score_path, targets[row["sample"]]["lineage"]
        )
        output.append(
            {
                "sample": row["sample"],
                "source": row["leakage_group"],
                "score_path": score_path,
                "index": index,
                "target": index.rendered_events,
                "candidates": tuple(
                    _candidate(value)
                    for value in predictions[row["sample"]]["notes"]
                ),
            }
        )
    return output


@torch.no_grad()
def _evaluate(
    model: OrnamentIdentityCRF,
    contexts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    samples = []
    for position, context in enumerate(contexts, 1):
        lattice = build_identity_lattice(
            context["candidates"],
            context["index"].events,
            context["score_path"],
            max_inference_hypotheses=16,
        )
        predicted, deletions, _diagnostics = fast_decode_identity_crf(
            model, lattice
        )
        samples.append(
            JointMetricSample(
                predicted=predicted,
                target=context["target"],
                source=context["source"],
                predicted_deletions=deletions,
                target_deletions=context["index"].deleted_event_indices,
                score_event_count=len(context["index"].events),
            )
        )
        if position == 1 or position % 16 == 0 or position == len(contexts):
            print(f"calibration={position}/{len(contexts)}", flush=True)
    return baseline._full_report(
        samples, seed=seed, replicates=replicates
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--train-candidate-freeze", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-negative-hypotheses", type=int, default=4)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    release_sha = sha256_file(args.release_manifest)
    if release_sha != args.expected_release_sha256:
        raise ValueError("Frozen release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Lockbox opening sentinel exists")
    freeze = json.loads(args.train_candidate_freeze.read_text(encoding="utf-8"))
    if (
        freeze["release_manifest_sha256"] != release_sha
        or freeze["predictions_sha256"] != sha256_file(args.train_candidates)
    ):
        raise ValueError("Train acoustic candidate freeze mismatch")
    held_lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="identity-crf-acoustic-finetune-v1",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={
            "train_rows": 901,
            "calibration_rows": 64,
            "locked_test": False,
        },
    )
    held_lease.__enter__()
    atexit.register(held_lease.__exit__, None, None, None)
    targets = _load_targets(release)
    train_examples, projection = _build_train(
        release,
        targets,
        args.train_candidates,
        max_negative_hypotheses=args.max_negative_hypotheses,
    )
    calibration = _calibration_contexts(
        release, targets, args.calibration_report
    )
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model = OrnamentIdentityCRF(hidden=base["model_config"]["hidden"])
    model.load_state_dict(base["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_examples) * args.epochs,
        eta_min=args.learning_rate * 0.1,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    epoch_start = 1
    position_start = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            saved.get("schema_version") != SCHEMA_VERSION
            or saved["release_manifest_sha256"] != release_sha
            or saved["train_candidates_sha256"]
            != sha256_file(args.train_candidates)
        ):
            raise ValueError("Acoustic fine-tune resume fingerprint mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        progress = saved["progress"]
        epoch_start = int(progress["epoch"])
        position_start = int(progress["position"])
        history = list(saved.get("history") or [])
        best_f1 = max(
            (float(row["calibration"]["f1"]) for row in history),
            default=-1.0,
        )
        random.setstate(saved["rng_state"]["python"])
        np.random.set_state(saved["rng_state"]["numpy"])
        torch.set_rng_state(saved["rng_state"]["torch"])
    best_path = args.output_dir / "best.pt"
    with nullcontext():
        for epoch in range(epoch_start, args.epochs + 1):
            generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
            order = torch.randperm(len(train_examples), generator=generator).tolist()
            start_position = position_start if epoch == epoch_start else 0
            started = time.perf_counter()
            running = 0.0
            for position_index in range(start_position, len(order)):
                position = position_index + 1
                index_value = order[position_index]
                loss, _parts = fast_identity_crf_nll(
                    model,
                    train_examples[index_value]["lattice"],
                    normalize=True,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                running += float(loss.detach())
                if position % args.checkpoint_every_steps == 0:
                    _atomic_torch(
                        args.output_dir / "mid_epoch_checkpoint.pt",
                        {
                            "schema_version": SCHEMA_VERSION,
                            "model_state_dict": model.state_dict(),
                            "model_config": base["model_config"],
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "progress": {
                                "epoch": epoch,
                                "position": position,
                                "global_step": (epoch - 1) * len(order) + position,
                                "optimizer_boundary": True,
                            },
                            "rng_state": {
                                "python": random.getstate(),
                                "numpy": np.random.get_state(),
                                "torch": torch.get_rng_state(),
                            },
                            "release_manifest_sha256": release_sha,
                            "train_candidates_sha256": sha256_file(args.train_candidates),
                            "history": history,
                        },
                    )
                if position == 1 or position % 50 == 0:
                    print(
                        f"epoch={epoch} rows={position}/{len(order)} "
                        f"loss={float(loss):.5f} "
                        f"rows_per_sec={position/max(time.perf_counter()-started,1e-9):.2f}",
                        flush=True,
                    )
            model.eval()
            report = _evaluate(
                model,
                calibration,
                seed=args.seed + epoch,
                replicates=args.bootstrap_replicates,
            )
            model.train()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running / len(order),
                    "calibration": report,
                }
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "model_state_dict": model.state_dict(),
                "model_config": base["model_config"],
                "release_manifest_sha256": release_sha,
                "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
                "train_candidates_sha256": sha256_file(args.train_candidates),
                "projection": projection,
                "history": history,
            }
            epoch_path = args.output_dir / f"candidate-epoch-{epoch:03d}.pt"
            _atomic_torch(epoch_path, payload)
            if float(report["f1"]) > best_f1:
                best_f1 = float(report["f1"])
                _atomic_torch(best_path, payload)
            _atomic_json(args.output_dir / "history.json", {"history": history})
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "calibration_combined_f1": report["f1"],
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
            "status": "complete",
            "best_checkpoint": str(best_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path),
            "best_calibration_combined_f1": best_f1,
            "base_calibration_combined_f1": 0.8013462485634543,
            "improved": best_f1 > 0.8013462485634543,
            "calibration_gate": 0.85,
            "calibration_gate_passed": best_f1 >= 0.85,
            "projection": projection,
            "lockbox_opened": False,
        },
    )
    atexit.unregister(held_lease.__exit__)
    held_lease.__exit__(None, None, None)
    print(json.dumps({"best_f1": best_f1}, indent=2))
    return 0


if __name__ == "__main__":
    import argparse

    raise SystemExit(main())
