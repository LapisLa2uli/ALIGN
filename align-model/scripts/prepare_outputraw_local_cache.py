from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from alignmodel.joint.lattice import LatticeConfig, SparseJointLattice
from alignmodel.joint.outputraw_full import FullJointPipelineModel
from alignmodel.joint.outputraw_train import prepare_local_sample
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.joint.prepared_local import (
    PreparedLocalDataset,
    PreparedLocalWriter,
    find_staging,
)


_WORKER_DATASET: PackedJointDataset | None = None
_WORKER_LATTICE: SparseJointLattice | None = None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _worker_init(pack_root: str, lattice_config: dict[str, Any]) -> None:
    global _WORKER_DATASET, _WORKER_LATTICE
    torch.set_num_threads(1)
    _WORKER_DATASET = PackedJointDataset(
        Path(pack_root),
        verify_records=False,
        load_feature_arrays=False,
    )
    _WORKER_LATTICE = SparseJointLattice(
        FullJointPipelineModel(),
        LatticeConfig(**lattice_config),
    )


def _worker_prepare(source_ordinal: int) -> tuple[int, str, Any]:
    if _WORKER_DATASET is None or _WORKER_LATTICE is None:
        raise RuntimeError("Prepared-cache worker is not initialized")
    packed = _WORKER_DATASET[source_ordinal]
    prepared = prepare_local_sample(packed, _WORKER_LATTICE)
    return source_ordinal, packed.sample, prepared


def _local_prepare(
    dataset: PackedJointDataset,
    lattice: SparseJointLattice,
    ordinals: Sequence[int],
) -> Iterable[tuple[int, str, Any]]:
    for source_ordinal in ordinals:
        packed = dataset[source_ordinal]
        yield (
            source_ordinal,
            packed.sample,
            prepare_local_sample(packed, lattice),
        )


def _parallel_prepare(
    pack_root: Path,
    lattice_config: LatticeConfig,
    ordinals: Sequence[int],
    *,
    workers: int,
    prefetch: int,
) -> Iterable[tuple[int, str, Any]]:
    """Yield in source order with a strict bound on serialized results."""

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(pack_root), asdict(lattice_config)),
    ) as executor:
        iterator = iter(ordinals)
        pending = deque()
        limit = max(workers, prefetch)
        for _ in range(limit):
            try:
                pending.append(executor.submit(_worker_prepare, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(_worker_prepare, next(iterator)))
            except StopIteration:
                pass


def _assert_equivalent(expected: Any, actual: Any) -> None:
    if expected.sample != actual.sample or expected.groups != actual.groups:
        raise ValueError(f"Prepared local identity mismatch: {expected.sample}")
    if expected.copy_count_target != actual.copy_count_target:
        raise ValueError(f"Prepared copy target mismatch: {expected.sample}")
    for name in (
        "edge_features",
        "keep",
        "boundary",
        "split",
        "emission",
        "structure",
        "layer2",
        "rhythm",
        "duration_target",
        "duration_weight",
        "rearticulation_weight",
    ):
        if not torch.equal(getattr(expected, name), getattr(actual, name)):
            raise ValueError(
                f"Prepared local tensor mismatch: {expected.sample} {name}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build resumable mmap local-lattice training shards."
    )
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--shard-rows", type=int, default=128)
    parser.add_argument("--checkpoint-rows", type=int, default=16)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-options", type=int, default=12)
    parser.add_argument("--max-states", type=int, default=48)
    parser.add_argument("--max-delete-events", type=int, default=24)
    args = parser.parse_args()

    lattice_config = LatticeConfig(
        max_options_per_candidate=args.max_options,
        max_states=args.max_states,
        max_delete_events=args.max_delete_events,
        noise_inference_bias=-6.0,
        continuation_feature_enabled=True,
        continuation_score_weight=0.35,
        continuation_hard_negative_copies=1,
        repeat_fragment_penalty=0.25,
    )
    started = time.perf_counter()
    with PackedJointDataset(
        args.pack_root,
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        pack_id = str(dataset.metadata["pack_id"])
        source = [
            (int(ordinal), str(sample))
            for ordinal, sample in dataset.connection.execute(
                "SELECT ordinal,sample FROM records "
                "WHERE split='train' ORDER BY ordinal"
            )
        ]
        if args.max_rows is not None:
            source = source[: max(0, args.max_rows)]
        staging = find_staging(args.destination)
        with PreparedLocalWriter(
            args.destination,
            pack_id=pack_id,
            lattice_config=lattice_config,
            shard_rows=max(1, args.shard_rows),
            checkpoint_rows=max(1, args.checkpoint_rows),
            staging=staging,
        ) as writer:
            writer.validate_prefix(source)
            initial = writer.committed_records
            remaining = [ordinal for ordinal, _sample in source[initial:]]
            if args.workers > 1 and remaining:
                prepared_rows = _parallel_prepare(
                    args.pack_root,
                    lattice_config,
                    remaining,
                    workers=args.workers,
                    prefetch=max(args.workers, args.prefetch),
                )
            else:
                lattice = SparseJointLattice(
                    FullJointPipelineModel(), lattice_config
                )
                prepared_rows = _local_prepare(dataset, lattice, remaining)
            for position, (source_ordinal, sample, prepared) in enumerate(
                prepared_rows,
                initial + 1,
            ):
                writer.add(
                    source_ordinal=source_ordinal,
                    sample=sample,
                    prepared=prepared,
                )
                if (
                    position == 1
                    or position % max(1, args.checkpoint_rows) == 0
                    or position == len(source)
                ):
                    elapsed = time.perf_counter() - started
                    rate = position / max(elapsed, 1e-9)
                    eta = (len(source) - position) / max(rate, 1e-9)
                    progress = {
                        "schema_version": "align-prepared-local-progress-v1",
                        "pack_id": pack_id,
                        "destination": str(args.destination.resolve()),
                        "committed_cursor": position,
                        "total_rows": len(source),
                        "rows_per_sec": rate,
                        "eta_seconds": eta,
                        "workers": max(0, args.workers),
                        "prefetch": max(0, args.prefetch),
                        "lattice": asdict(lattice_config),
                        "protected_test_accessed": False,
                    }
                    _atomic_json(
                        args.destination.parent
                        / f"{args.destination.name}.progress.json",
                        progress,
                    )
                    print(
                        f"prepared_local={position}/{len(source)} "
                        f"rows_per_sec={rate:.3f} eta_sec={eta:.1f}",
                        flush=True,
                    )
            metadata = writer.finalize()

        check_ordinals = [
            source[index][0]
            for index in sorted(
                {
                    0,
                    min(1, len(source) - 1),
                    len(source) // 2,
                    max(0, len(source) - 2),
                    max(0, len(source) - 1),
                }
            )
        ] if source else []
        lattice = SparseJointLattice(FullJointPipelineModel(), lattice_config)
        with PreparedLocalDataset(
            args.destination,
            pack_id=pack_id,
            lattice_config=lattice_config,
        ) as prepared_dataset:
            validation = prepared_dataset.validate(deep=True)
            for source_ordinal in check_ordinals:
                expected = prepare_local_sample(
                    dataset[source_ordinal], lattice
                )
                _assert_equivalent(
                    expected, prepared_dataset[source_ordinal]
                )

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "align-prepared-local-report-v1",
        "pack_id": pack_id,
        "destination": str(args.destination.resolve()),
        "records": metadata["record_count"],
        "shard_rows": metadata["shard_rows"],
        "bytes": sum(int(row["size"]) for row in metadata["files"]),
        "seconds": elapsed,
        "rows_per_sec": len(source) / max(elapsed, 1e-9),
        "workers": max(0, args.workers),
        "prefetch": max(0, args.prefetch),
        "lattice": asdict(lattice_config),
        "equivalence_source_ordinals": check_ordinals,
        "validation": validation,
        "protected_test_accessed": False,
    }
    _atomic_json(
        args.destination.parent / f"{args.destination.name}.report.json",
        report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
