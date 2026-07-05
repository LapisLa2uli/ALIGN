from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from datacreate.config import PipelineConfig
from datacreate.pipeline import DataCreatePipeline
from datacreate.stages.stage9_synthetic import generate_synthetic_samples
from datacreate.utils import ensure_dir
from datacreate.validation import validate_corpus, validate_labels_file


def _config_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="MusicEval data creation pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run full batch pipeline (stages 1-5, 7)")
    run.add_argument("--score", type=Path, required=True)
    run.add_argument("--performance", type=Path, required=True)
    run.add_argument("--sample-id", type=str)
    run.add_argument("--verified-score", type=Path)

    resume = sub.add_parser("resume", help="Resume an existing sample")
    resume.add_argument("--sample-id", type=str, required=True)

    sub.add_parser("serve", help="Launch annotation web UI")

    args = parser.parse_args()
    config = PipelineConfig.load(_config_path(args.config))
    pipeline = DataCreatePipeline(config)

    if args.command == "run":
        job = pipeline.run_batch(
            score=args.score,
            performance=args.performance,
            sample_id=args.sample_id,
            verified_score=args.verified_score,
        )
        print(f"Batch complete: {job.sample_dir}")
    elif args.command == "resume":
        job = pipeline.resume(args.sample_id)
        print(f"Resumed sample {job.sample_id}: state={job.state}")
    elif args.command == "serve":
        serve_main(config)


def batch_main() -> None:
    sys.argv = ["datacreate-batch", "run"] + sys.argv[1:]
    main()


def synthetic_main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic error samples")
    parser.add_argument("--config", type=str)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    config = PipelineConfig.load(_config_path(args.config))
    root = ensure_dir(
        config.path("synthetic_root")
        if config.synthetic.get("output_separate_root", True)
        else config.path("samples_root") or Path("samples")
    )
    from datacreate.utils import setup_sample_logger

    logger = setup_sample_logger(root, name="synthetic")
    dirs = generate_synthetic_samples(args.score, root, config, logger, args.count)
    print(f"Created {len(dirs)} synthetic samples under {root}")


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate labels.json corpus")
    parser.add_argument("--config", type=str)
    parser.add_argument("path", type=Path, help="labels.json file or corpus root")
    args = parser.parse_args()
    config = PipelineConfig.load(_config_path(args.config))
    if args.path.is_file():
        errors = validate_labels_file(args.path, config)
    else:
        errors = validate_corpus(args.path, config)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        sys.exit(1)
    print("Validation passed.")


def serve_main(config: PipelineConfig | None = None) -> None:
    from datacreate.web.app import create_app

    config = config or PipelineConfig.load()
    app = create_app(config)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
