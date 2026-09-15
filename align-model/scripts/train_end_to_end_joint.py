from __future__ import annotations

import argparse
from pathlib import Path

from alignmodel.joint.end_to_end import (
    EndToEndTrainConfig,
    train_end_to_end_model,
)
from alignmodel.joint.lattice import LatticeConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train repository-owned acoustic, option, transition, and path "
            "heads over the frozen audited Basic Pitch cache."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--path-epochs", type=int, default=1)
    parser.add_argument("--local-learning-rate", type=float, default=2e-4)
    parser.add_argument("--local-distillation-weight", type=float, default=0.5)
    parser.add_argument("--path-learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--component-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--local-batch-edges", type=int, default=65536)
    parser.add_argument("--path-gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--path-train-samples", type=int, default=1000)
    parser.add_argument("--max-val-samples", type=int, default=100)
    parser.add_argument("--local-device", default="cuda")
    parser.add_argument("--path-device", default="cpu")
    parser.add_argument("--example-cache-path", type=Path)
    parser.add_argument("--preprocessing-workers", type=int, default=4)
    parser.add_argument("--preprocessing-prefetch", type=int, default=8)
    parser.add_argument("--checkpoint-every-rows", type=int, default=250)
    parser.add_argument("--structured-cpu-threads", type=int, default=1)
    parser.add_argument("--pairing-tolerance-sec", type=float, default=0.050)
    parser.add_argument("--minimum-candidate-confidence", type=float, default=0.65)
    parser.add_argument(
        "--train-legacy-during-local",
        action="store_true",
        help="Also update the initialized legacy edge network during local training.",
    )
    parser.add_argument("--max-options", type=int, default=12)
    parser.add_argument("--max-states", type=int, default=48)
    parser.add_argument("--max-delete-events", type=int, default=24)
    parser.add_argument("--noise-inference-bias", type=float, default=-6.0)
    parser.add_argument(
        "--disable-continuation-feature", action="store_true"
    )
    parser.add_argument(
        "--continuation-score-weight", type=float, default=0.35
    )
    parser.add_argument(
        "--continuation-lookahead-notes", type=int, default=3
    )
    parser.add_argument(
        "--continuation-hard-negative-copies", type=int, default=1
    )
    parser.add_argument(
        "--repeat-fragment-penalty", type=float, default=0.25
    )
    args = parser.parse_args()

    checkpoint = train_end_to_end_model(
        EndToEndTrainConfig(
            manifest=args.manifest,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            initialize_checkpoint=args.initialize_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
            seed=args.seed,
            local_epochs=args.local_epochs,
            path_epochs=args.path_epochs,
            local_learning_rate=args.local_learning_rate,
            local_distillation_weight=args.local_distillation_weight,
            path_learning_rate=args.path_learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            component_dim=args.component_dim,
            dropout=args.dropout,
            residual_scale=args.residual_scale,
            local_batch_edges=args.local_batch_edges,
            path_gradient_accumulation=args.path_gradient_accumulation,
            gradient_clip=args.gradient_clip,
            max_train_samples=args.max_train_samples,
            path_train_samples=args.path_train_samples,
            max_val_samples=args.max_val_samples,
            local_device=args.local_device,
            path_device=args.path_device,
            example_cache_path=args.example_cache_path,
            preprocessing_workers=max(1, args.preprocessing_workers),
            preprocessing_prefetch=max(1, args.preprocessing_prefetch),
            checkpoint_every_rows=max(1, args.checkpoint_every_rows),
            structured_cpu_threads=max(1, args.structured_cpu_threads),
            pairing_tolerance_sec=args.pairing_tolerance_sec,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
            freeze_legacy_during_local=not args.train_legacy_during_local,
            lattice=LatticeConfig(
                max_options_per_candidate=args.max_options,
                max_states=args.max_states,
                max_delete_events=args.max_delete_events,
                noise_inference_bias=args.noise_inference_bias,
                continuation_feature_enabled=not args.disable_continuation_feature,
                continuation_lookahead_notes=max(
                    2, args.continuation_lookahead_notes
                ),
                continuation_score_weight=args.continuation_score_weight,
                continuation_hard_negative_copies=max(
                    0, args.continuation_hard_negative_copies
                ),
                repeat_fragment_penalty=max(
                    0.0, args.repeat_fragment_penalty
                ),
            ),
        )
    )
    print(checkpoint.resolve())


if __name__ == "__main__":
    main()
