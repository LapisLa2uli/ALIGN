from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil

from alignmodel.joint.candidates import (
    add_score_repeat_hints,
    basic_pitch_candidate_union,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.outputraw_train import collate_local_samples
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


def _data_builder() -> Any:
    path = Path(__file__).with_name("prepare_joint_outputraw_data.py")
    spec = importlib.util.spec_from_file_location("outputraw_data_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _process_start() -> tuple[float, float, int, int]:
    process = psutil.Process()
    io = process.io_counters()
    return (
        time.perf_counter(),
        time.process_time(),
        int(io.read_bytes),
        int(process.memory_info().rss),
    )


def _process_finish(
    started: tuple[float, float, int, int],
    rows: int,
) -> dict[str, Any]:
    wall = time.perf_counter() - started[0]
    cpu = time.process_time() - started[1]
    process = psutil.Process()
    io = process.io_counters()
    memory = process.memory_info()
    return {
        "rows": rows,
        "seconds": wall,
        "rows_per_sec": rows / max(wall, 1e-9),
        "process_cpu_seconds": cpu,
        "process_cpu_one_core_percent": 100.0 * cpu / max(wall, 1e-9),
        "process_read_bytes": int(io.read_bytes) - started[2],
        "process_read_mib_per_sec": (
            (int(io.read_bytes) - started[2]) / 1024**2 / max(wall, 1e-9)
        ),
        "rss_start_mb": started[3] / 1024**2,
        "rss_end_mb": memory.rss / 1024**2,
    }


def _baseline(
    rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    checkpoints: set[int],
) -> dict[str, Any]:
    builder = _data_builder()
    phases = {
        "target_sqlite_zlib_json": 0.0,
        "musicxml_score_index": 0.0,
        "npz_metadata_validation_decompression": 0.0,
        "candidate_construction": 0.0,
        "target_pairing_reconstruction": 0.0,
    }
    reports = {}
    started = _process_start()
    for position, row in enumerate(rows, 1):
        phase_started = time.perf_counter()
        lineage = builder._load_target(row)
        phases["target_sqlite_zlib_json"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        index = ScoreEventIndex.from_musicxml(
            Path(str(row["sample_dir"])) / "verified_score.musicxml",
            lineage,
        )
        phases["musicxml_score_index"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        features = builder._load_features(row, cache_root)
        phases["npz_metadata_validation_decompression"] += (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        candidates = tuple(
            add_score_repeat_hints(
                basic_pitch_candidate_union(features, minimum_confidence=0.0),
                index.events,
            )
        )
        phases["candidate_construction"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        pair_exact_pitch_onset(
            tuple(builder._candidate_event(value) for value in candidates),
            index.rendered_events,
            tolerance_sec=0.050,
        )
        phases["target_pairing_reconstruction"] += (
            time.perf_counter() - phase_started
        )
        if position in checkpoints:
            report = _process_finish(started, position)
            report["phase_seconds"] = dict(phases)
            reports[str(position)] = report
            print(
                f"baseline_phases={position}/{len(rows)} "
                f"rows_per_sec={report['rows_per_sec']:.3f}",
                flush=True,
            )
    return reports


def _packed_and_prepared(
    *,
    pack_root: Path,
    prepared_root: Path,
    lattice_config: Any,
    counts: Sequence[int],
) -> dict[str, Any]:
    reports = {}
    with PackedJointDataset(
        pack_root,
        verify_records=False,
        load_feature_arrays=False,
    ) as packed, PreparedLocalDataset(
        prepared_root,
        pack_id=str(packed.metadata["pack_id"]),
        lattice_config=lattice_config,
    ) as prepared:
        ordinals = packed.ordinals("train")
        for count in counts:
            selected = ordinals[:count]
            phase = {
                "packed_sqlite_zlib_json_candidates": 0.0,
                "packed_target_reconstruction": 0.0,
                "prepared_mmap_tensor_load": 0.0,
                "collation": 0.0,
            }
            started = _process_start()
            for ordinal in selected:
                phase_started = time.perf_counter()
                sample = packed[ordinal]
                phase["packed_sqlite_zlib_json_candidates"] += (
                    time.perf_counter() - phase_started
                )
                phase_started = time.perf_counter()
                sample.training_example()
                phase["packed_target_reconstruction"] += (
                    time.perf_counter() - phase_started
                )
            prepared_rows = []
            for ordinal in selected:
                phase_started = time.perf_counter()
                prepared_rows.append(prepared[ordinal])
                phase["prepared_mmap_tensor_load"] += (
                    time.perf_counter() - phase_started
                )
            phase_started = time.perf_counter()
            batch = collate_local_samples(prepared_rows)
            phase["collation"] += time.perf_counter() - phase_started
            report = _process_finish(started, count)
            report.update(
                {
                    "edges": batch.edge_count,
                    "phase_seconds": phase,
                    "prepared_rows_per_sec": count
                    / max(
                        phase["prepared_mmap_tensor_load"]
                        + phase["collation"],
                        1e-9,
                    ),
                }
            )
            reports[str(count)] = report
            del prepared_rows, batch
            gc.collect()
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--prepared-local-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=(100, 1000))
    args = parser.parse_args()
    ready = json.loads(args.ready_marker.read_text(encoding="utf-8"))
    if (
        ready.get("verification", {}).get("test_features_materialized") is not False
        or ready.get("verification", {}).get("test_targets_materialized") is not False
    ):
        raise ValueError("Protected test split is not sealed")
    from alignmodel.joint.lattice import LatticeConfig

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
    counts = sorted(set(max(1, value) for value in args.counts))
    manifest_started = time.perf_counter()
    manifest = json.loads(
        Path(str(ready["paths"]["manifest"])).read_text(encoding="utf-8")
    )
    manifest_seconds = time.perf_counter() - manifest_started
    rows = list(manifest["train"][: counts[-1]])
    report = {
        "schema_version": "align-outputraw-data-phase-profile-v1",
        "created_unix": time.time(),
        "pack_id": ready["hashes"]["pack_id"],
        "manifest_seconds": manifest_seconds,
        "counts": counts,
        "baseline": _baseline(
            rows,
            Path("runs/joint-audit-v2/basic-pitch-cache").resolve(),
            set(counts),
        ),
        "optimized": _packed_and_prepared(
            pack_root=Path(str(ready["paths"]["packed_root"])),
            prepared_root=args.prepared_local_cache,
            lattice_config=lattice_config,
            counts=counts,
        ),
        "protected_test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
