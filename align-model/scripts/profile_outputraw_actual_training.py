from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from alignmodel.joint.lattice import LatticeConfig
from alignmodel.joint.outputraw_full import (
    FullJointPipelineModel,
    FullPipelineLossConfig,
)
from alignmodel.joint.outputraw_train import (
    collate_local_samples,
    train_local_batch,
)
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.joint.prepared_local import PreparedLocalDataset


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _amp_dtype(name: str) -> torch.dtype | None:
    return {
        "float32": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _batch(
    prepared: PreparedLocalDataset,
    order: list[int],
    max_edges: int,
) -> Any:
    rows = []
    edges = 0
    for ordinal in order:
        row = prepared[ordinal]
        if rows and edges + row.edge_count > max_edges:
            break
        rows.append(row)
        edges += row.edge_count
    return collate_local_samples(rows)


def _profile(
    template: FullJointPipelineModel,
    batch: Any,
    *,
    amp_name: str,
    fused: bool,
    iterations: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    dtype = _amp_dtype(amp_name)
    model = copy.deepcopy(template).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        fused=fused,
    )
    scaler = (
        torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    )
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    torch.cuda.reset_peak_memory_stats()
    totals = {
        "host_to_device": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    losses = []
    components = None
    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss, components, timing = train_local_batch(
            model,
            batch,
            device=device,
            loss_config=FullPipelineLossConfig(),
            amp_dtype=dtype,
        )
        scaled = loss if scaler is None else scaler.scale(loss)
        phase_started = time.perf_counter()
        scaled.backward()
        torch.cuda.synchronize()
        totals["backward"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        torch.cuda.synchronize()
        totals["optimizer"] += time.perf_counter() - phase_started
        totals["host_to_device"] += timing["host_to_device"]
        totals["forward"] += timing["forward"]
        losses.append(float(loss.detach().float().cpu()))
    elapsed = time.perf_counter() - started
    finite_gradients = all(
        parameter.grad is None or bool(torch.all(torch.isfinite(parameter.grad)))
        for parameter in model.parameters()
    )
    gradient_l2 = math.sqrt(
        sum(
            float(torch.sum(parameter.grad.detach().float().square()))
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    return {
        "ok": all(math.isfinite(value) for value in losses) and finite_gradients,
        "amp_dtype": amp_name,
        "fused_optimizer": fused,
        "iterations": iterations,
        "rows": len(batch.samples),
        "edges": batch.edge_count,
        "seconds": elapsed,
        "rows_per_sec": len(batch.samples) * iterations / elapsed,
        "edges_per_sec": batch.edge_count * iterations / elapsed,
        "losses": losses,
        "first_components": components,
        "gradient_l2": gradient_l2,
        "finite_gradients": finite_gradients,
        "phase_seconds": totals,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--prepared-local-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ready = json.loads(args.ready_marker.read_text(encoding="utf-8"))
    lattice_config = LatticeConfig(
        max_options_per_candidate=12,
        max_states=48,
        max_delete_events=24,
        noise_inference_bias=-6.0,
        continuation_feature_enabled=True,
        continuation_score_weight=0.35,
        continuation_hard_negative_copies=1,
        repeat_fragment_penalty=0.25,
    )
    template = FullJointPipelineModel()
    results = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset, PreparedLocalDataset(
        args.prepared_local_cache,
        pack_id=str(ready["hashes"]["pack_id"]),
        lattice_config=lattice_config,
        max_open_shards=64,
    ) as prepared:
        order = dataset.deterministic_order(
            "train",
            epoch=0,
            seed=20260915,
        )
        batches = {
            limit: _batch(prepared, order, limit)
            for limit in (32768, 65536)
        }
        for limit, batch in batches.items():
            for amp_name in ("float32", "float16", "bfloat16"):
                for fused in (False, True):
                    try:
                        result = _profile(
                            template,
                            batch,
                            amp_name=amp_name,
                            fused=fused,
                            iterations=max(1, args.iterations),
                        )
                    except (RuntimeError, TypeError) as error:
                        result = {
                            "ok": False,
                            "amp_dtype": amp_name,
                            "fused_optimizer": fused,
                            "error": str(error),
                        }
                    result["batch_edge_limit"] = limit
                    results.append(result)
    baselines = {
        int(row["batch_edge_limit"]): row
        for row in results
        if row.get("ok")
        and row["amp_dtype"] == "float32"
        and not row["fused_optimizer"]
    }
    for row in results:
        if not row.get("ok"):
            continue
        baseline = baselines[int(row["batch_edge_limit"])]
        row["first_loss_relative_error_vs_fp32"] = abs(
            row["losses"][0] - baseline["losses"][0]
        ) / max(abs(baseline["losses"][0]), 1e-12)
    eligible = [
        row
        for row in results
        if row.get("ok")
        and row["first_loss_relative_error_vs_fp32"] <= 0.01
    ]
    if not eligible:
        raise RuntimeError("No numerically stable actual training profile")
    selected = max(eligible, key=lambda row: float(row["edges_per_sec"]))
    report = {
        "schema_version": "align-outputraw-actual-training-profile-v1",
        "created_unix": time.time(),
        "pack_id": ready["hashes"]["pack_id"],
        "results": results,
        "selected": selected,
        "selection": {
            "maximum_first_loss_relative_error": 0.01,
            "metric": "actual full-loss edges/sec",
            "fp32_path_reduction": True,
            "channels_last": "not_applicable_to_2d_edge_mlp",
        },
        "protected_test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
