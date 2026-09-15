from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch

from alignmodel.joint.candidate_rescorer import load_candidate_rescorer
from alignmodel.joint.lattice import FEATURE_DIM, LatticeConfig, SparseJointLattice
from alignmodel.joint.outputraw_full import (
    FullJointPipelineModel,
    FullPipelineModelConfig,
    load_checkpoint,
    verify_data_ready,
)
from alignmodel.joint.outputraw_train import (
    _target_resume_events,
    evaluate_packed_validation,
)
from alignmodel.joint.packed_data import PackedJointDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_load(
    path: Path,
    loader: Any,
) -> tuple[Any, str]:
    if not path.is_file() or path.suffix == ".tmp":
        raise ValueError(f"Checkpoint is not a complete file: {path}")
    before = _sha256(path)
    result = loader(path)
    after = _sha256(path)
    if before != after:
        raise RuntimeError(f"Checkpoint changed while being read: {path}")
    return result, before


class _OrdinalView:
    def __init__(
        self,
        dataset: PackedJointDataset,
        ordinals: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.selected = list(ordinals)
        self.metadata = dataset.metadata

    def ordinals(self, split: str) -> list[int]:
        if split != "val":
            raise ValueError("Ablation view exposes only validation rows")
        return list(self.selected)

    def __getitem__(self, ordinal: int) -> Any:
        return self.dataset[ordinal]


def _balanced_ordinals(
    dataset: PackedJointDataset,
    *,
    limit: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    repeat_rows = []
    ordinary_rows = []
    for ordinal in dataset.ordinals("val"):
        packed = dataset[ordinal]
        target = (
            repeat_rows
            if packed.target.get("layer1_repeats")
            else ordinary_rows
        )
        target.append(ordinal)
    rng = random.Random(seed)
    rng.shuffle(repeat_rows)
    rng.shuffle(ordinary_rows)
    ordinary_count = min(len(ordinary_rows), limit // 2)
    repeat_count = min(len(repeat_rows), limit - ordinary_count)
    selected = ordinary_rows[:ordinary_count] + repeat_rows[:repeat_count]
    if len(selected) < limit:
        remainder = ordinary_rows[ordinary_count:] + repeat_rows[repeat_count:]
        rng.shuffle(remainder)
        selected.extend(remainder[: limit - len(selected)])
    rng.shuffle(selected)
    return selected, {
        "selection": "deterministic balanced copy-lineage strata",
        "seed": seed,
        "rows": len(selected),
        "repeat_positive_rows": repeat_count,
        "repeat_negative_rows": ordinary_count,
        "full_validation_rows": len(dataset.ordinals("val")),
    }


def _oracle_summary(
    dataset: PackedJointDataset,
    ordinals: Sequence[int],
    lattice_config: LatticeConfig,
) -> dict[str, Any]:
    lattice = SparseJointLattice(
        FullJointPipelineModel(FullPipelineModelConfig()),
        lattice_config,
    )
    copy_events = 0
    resume_events = 0
    packed_resume_events = 0
    mismatch_rows = 0
    for ordinal in ordinals:
        packed = dataset[ordinal]
        events = packed.training_example().target_events
        copy_events += sum(event.is_copy for event in events)
        reconstructed = _target_resume_events(events, lattice)
        packed_resume = tuple(
            int(row["resume_event"])
            for row in packed.target.get("layer1_repeats") or []
        )
        resume_events += len(reconstructed)
        packed_resume_events += len(packed_resume)
        mismatch_rows += int(reconstructed != packed_resume)
    return {
        "oracle_repeat_f1": 1.0,
        "oracle_repeat_events": copy_events,
        "oracle_resume_accuracy": 1.0,
        "oracle_resume_events": resume_events,
        "legacy_packed_resume_events": packed_resume_events,
        "legacy_resume_definition_mismatch_rows": mismatch_rows,
    }


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-replicates", type=int, default=250)
    parser.add_argument("--candidate-rescorer-checkpoint", type=Path)
    parser.add_argument("--skip-initial-validation", action="store_true")
    parser.add_argument("--full-residual-scale", type=float)
    parser.add_argument("--structure-path-weight", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ready = verify_data_ready(args.ready_marker)
    fingerprint = str(ready["hashes"]["pack_id"])
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
    started = time.time()
    candidate_rescorer = None
    candidate_threshold = 0.0
    candidate_checkpoint_sha = None
    if args.candidate_rescorer_checkpoint is not None:
        (
            candidate_rescorer,
            candidate_threshold,
            _candidate_payload,
        ), candidate_checkpoint_sha = _stable_load(
            args.candidate_rescorer_checkpoint,
            lambda path: load_candidate_rescorer(path, device="cpu"),
        )
    with ExitStack() as stack:
        dataset = stack.enter_context(
            PackedJointDataset(
                Path(str(ready["paths"]["packed_root"])),
                manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
                verify_records=False,
                load_feature_arrays=False,
            )
        )
        ordinals, selection = _balanced_ordinals(
            dataset,
            limit=args.limit,
            seed=args.seed,
        )
        view = _OrdinalView(dataset, ordinals)

        torch.manual_seed(args.seed)
        initial = FullJointPipelineModel(
            FullPipelineModelConfig(path_component_dim=32)
        )
        initialization, source_sha = _stable_load(
            args.source_checkpoint,
            lambda path: initial.initialize_path(
                path,
                transfer_mode="exact",
                minimum_coverage=1.0,
            ),
        )
        initial.eval()
        initial_metrics = None
        if not args.skip_initial_validation:
            initial_metrics = evaluate_packed_validation(
                initial,
                view,
                lattice_config=lattice_config,
                bootstrap_replicates=args.bootstrap_replicates,
                candidate_rescorer=candidate_rescorer,
                candidate_threshold=candidate_threshold,
            )

        final_result = None
        if args.final_checkpoint is not None:
            (final, payload), final_sha = _stable_load(
                args.final_checkpoint,
                lambda path: load_checkpoint(
                    path,
                    device="cpu",
                    expected_data_fingerprint=fingerprint,
                ),
            )
            final.eval()
            if args.full_residual_scale is not None:
                final.config = replace(
                    final.config,
                    residual_scale=args.full_residual_scale,
                )
            if args.structure_path_weight is not None:
                final.config = replace(
                    final.config,
                    structure_path_weight=args.structure_path_weight,
                )
            final_metrics = evaluate_packed_validation(
                final,
                view,
                lattice_config=lattice_config,
                bootstrap_replicates=args.bootstrap_replicates,
                candidate_rescorer=candidate_rescorer,
                candidate_threshold=candidate_threshold,
            )
            final_result = {
                "path": str(args.final_checkpoint.resolve()),
                "sha256": final_sha,
                "progress": payload.get("progress"),
                "validation": final_metrics,
            }
        oracle = _oracle_summary(dataset, ordinals, lattice_config)

    document = {
        "schema_version": "align-outputraw-checkpoint-ablation-v1",
        "metric_status": "official_note_wise",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "locked_test_touched": False,
        "data_fingerprint": fingerprint,
        "selection": selection,
        "candidate_rescorer": (
            {
                "path": str(args.candidate_rescorer_checkpoint.resolve()),
                "sha256": candidate_checkpoint_sha,
                "threshold": candidate_threshold,
            }
            if args.candidate_rescorer_checkpoint is not None
            else None
        ),
        "checkpoints": {
            "initialization": {
                "path": str(args.source_checkpoint.resolve()),
                "sha256": source_sha,
                "transfer": initialization,
                "validation": initial_metrics,
            },
            "final_after_path": final_result,
        },
        "architecture_transfer_comparison": {
            "exact_component_dim_32": initialization,
            "selected": "exact_component_dim_32",
            "required_transfer_coverage": 1.0,
        },
        "unavailable_checkpoints": {
            "local_only": "not recoverable; run overwrote one atomic last_checkpoint.pt",
            "final_before_path": "not recoverable; run overwrote one atomic last_checkpoint.pt",
        },
        "oracles": oracle,
    }
    _atomic_json(args.output, document)
    print(json.dumps({"output": str(args.output.resolve()), **selection}, indent=2))


if __name__ == "__main__":
    main()
