from __future__ import annotations

import argparse
from pathlib import Path

from alignmodel.joint.lattice import LatticeConfig
from alignmodel.joint.train import JointTrainConfig, train_joint_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the sparse score-conditioned joint path CRF."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--minimum-candidate-confidence", type=float, default=0.65)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--max-states", type=int, default=256)
    parser.add_argument("--max-delete-events", type=int, default=16)
    parser.add_argument("--noise-inference-bias", type=float, default=-3.0)
    parser.add_argument(
        "--enable-continuation-feature", action="store_true"
    )
    parser.add_argument(
        "--continuation-score-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--continuation-hard-negative-copies", type=int, default=0
    )
    parser.add_argument(
        "--repeat-fragment-penalty", type=float, default=0.0
    )
    args = parser.parse_args()
    checkpoint = train_joint_model(
        JointTrainConfig(
            manifest=args.manifest,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            initialize_checkpoint=args.initialize_checkpoint,
            seed=args.seed,
            warmup_epochs=args.warmup_epochs,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
            hidden_dim=args.hidden_dim,
            gradient_accumulation=args.gradient_accumulation,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            device=args.device,
            lattice=LatticeConfig(
                max_options_per_candidate=args.max_options,
                max_states=args.max_states,
                max_delete_events=args.max_delete_events,
                noise_inference_bias=args.noise_inference_bias,
                continuation_feature_enabled=args.enable_continuation_feature,
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
