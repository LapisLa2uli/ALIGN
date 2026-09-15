from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alignmodel.joint.candidate_rescorer import (
    CandidateRescorerConfig,
    train_candidate_rescorer,
)
from alignmodel.training_resources import (
    DEFAULT_STATUS_PATH,
    claim_resource,
    release_resource,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train confidence-aware candidate scoring while retaining the "
            "global 0.65 candidate gate as the production reference."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--basic-cache-root", type=Path, required=True)
    parser.add_argument("--example-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-candidates", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--candidate-floor", type=float, default=0.50)
    parser.add_argument("--global-candidate-gate", type=float, default=0.65)
    parser.add_argument("--hard-negative-ratio", type=float, default=1.0)
    parser.add_argument("--short-weight-lt-80ms", type=float, default=4.0)
    parser.add_argument("--short-weight-lt-120ms", type=float, default=3.0)
    parser.add_argument("--short-weight-lt-180ms", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--prefetch", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-every-clips", type=int, default=0)
    parser.add_argument("--packed-training-root", type=Path)
    parser.add_argument("--resource-status", type=Path, default=DEFAULT_STATUS_PATH)
    args = parser.parse_args()
    lease_id = None
    try:
        if args.device == "cuda":
            lease_id = claim_resource(
                args.resource_status,
                "gpu",
                track="candidate-rescorer-training",
                command=sys.argv,
            )
        checkpoint = train_candidate_rescorer(
            CandidateRescorerConfig(
                manifest=args.manifest,
                basic_cache_root=args.basic_cache_root,
                example_cache_path=args.example_cache_path,
                output_dir=args.output_dir,
                seed=args.seed,
                epochs=max(1, args.epochs),
                batch_candidates=max(1024, args.batch_candidates),
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                candidate_floor=args.candidate_floor,
                global_candidate_gate=args.global_candidate_gate,
                hard_negative_ratio=args.hard_negative_ratio,
                short_weight_lt_80ms=args.short_weight_lt_80ms,
                short_weight_lt_120ms=args.short_weight_lt_120ms,
                short_weight_lt_180ms=args.short_weight_lt_180ms,
                workers=max(1, args.workers),
                prefetch=max(1, args.prefetch),
                device=args.device,
                resume_checkpoint=args.resume_checkpoint,
                checkpoint_every_clips=max(0, args.checkpoint_every_clips),
                packed_training_root=args.packed_training_root,
            )
        )
        print(checkpoint.resolve())
    finally:
        if lease_id is not None:
            release_resource(args.resource_status, "gpu", lease_id)


if __name__ == "__main__":
    main()
