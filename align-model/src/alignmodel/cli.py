from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.config import TrainConfig
from alignmodel.device import resolve_device
from alignmodel.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="ALIGN error detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the four-stage pipeline on one bundle")
    p_run.add_argument("--sample", type=Path, required=True)
    p_run.add_argument("--out", type=Path, default=None)
    p_run.add_argument(
        "--stages",
        default="1,2,3",
        help="Comma-separated stages to run (default 1,2,3; pass 4 or use --timbre)",
    )
    p_run.add_argument(
        "--timbre",
        action="store_true",
        help="Enable stage 4 timbre/squeak/bad_start heuristics",
    )
    p_run.add_argument("--include-state", action="store_true")
    p_run.add_argument("--device", default="cuda", help="cuda | cpu | auto")

    p_smoke = sub.add_parser("smoke", help="Run pipeline on one synth repetition bundle")
    p_smoke.add_argument(
        "--data",
        type=Path,
        default=Path("synth-pipeline/output"),
        help="Root of ALIGN synth bundles",
    )
    p_smoke.add_argument("--out", type=Path, default=None)
    p_smoke.add_argument("--timbre", action="store_true")
    p_smoke.add_argument("--device", default="cuda", help="cuda | cpu | auto")

    p_train = sub.add_parser("train", help="Train legacy RUMAA-lite on ALIGN synth bundles")
    p_train.add_argument(
        "--data",
        type=Path,
        default=Path("synth-pipeline/1000dataexport"),
        help="Root of sample folders (verified_score.musicxml + performance_mel.npy)",
    )
    p_train.add_argument("--out", type=Path, default=Path("align-model/runs/rumaa-lite"))
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--overfit", type=int, default=0)
    p_train.add_argument("--device", default="cuda")

    p_st = sub.add_parser("train-stages", help="Train pipeline stages 1-3 on ALIGN synth bundles")
    p_st.add_argument(
        "--data",
        type=Path,
        default=Path("synth-pipeline/output"),
        help="Root of synth sample folders",
    )
    p_st.add_argument("--out", type=Path, default=Path("align-model/runs/stages"))
    p_st.add_argument("--epochs", type=int, default=8)
    p_st.add_argument("--batch-size", type=int, default=32)
    p_st.add_argument("--lr", type=float, default=1e-3)
    p_st.add_argument("--device", default="cuda")
    p_st.add_argument("--stages", default="1,2,3")
    p_st.add_argument("--max-samples", type=int, default=0)

    p_run.add_argument(
        "--weights",
        type=Path,
        default=Path("align-model/runs/stages"),
        help="Directory with stage1.pt / stage2.pt / stage3.pt",
    )

    p_inf = sub.add_parser("infer", help="Run a trained RUMAA-lite checkpoint on one sample")
    p_inf.add_argument("--ckpt", type=Path, required=True)
    p_inf.add_argument("--sample", type=Path, required=True)
    p_inf.add_argument("--out", type=Path, default=None)
    p_inf.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.cmd == "run":
        from alignmodel.pipeline import run_pipeline, write_prediction

        stages = {int(x.strip()) for x in str(args.stages).split(",") if x.strip()}
        timbre = bool(args.timbre) or (4 in stages)
        stages.discard(4)
        state = run_pipeline(
            args.sample,
            stages=stages,
            timbre=timbre,
            device=args.device,
            weights_dir=args.weights,
        )
        out = args.out or (args.sample / "pipeline_pred.json")
        write_prediction(state, out, include_state=bool(args.include_state))
        print(f"Wrote {out}")
        print(f"device={state.device}")
        counts: dict[str, int] = {}
        for lab in state.labels:
            counts[lab.type] = counts.get(lab.type, 0) + 1
        print(
            f"stages={state.stages_run} segments={len(state.segments)} "
            f"pairs={len(state.pairs)} labels={counts}"
        )
        return

    if args.cmd == "smoke":
        from alignmodel.pipeline_smoke import smoke_pipeline

        report = smoke_pipeline(args.data, args.out, timbre=bool(args.timbre), device=args.device)
        print(json.dumps(report, indent=2))
        return

    if args.cmd == "train":
        cfg = TrainConfig(
            data_root=args.data.resolve(),
            output_dir=args.out.resolve(),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            overfit=args.overfit,
            device=args.device,
        )
        train(cfg)
        return

    if args.cmd == "train-stages":
        from alignmodel.stage_train import StageTrainConfig, train_stages

        stages = tuple(int(x.strip()) for x in str(args.stages).split(",") if x.strip())
        train_stages(
            StageTrainConfig(
                data_root=args.data.resolve(),
                output_dir=args.out.resolve(),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                stages=stages,
                max_samples=args.max_samples,
            )
        )
        return

    from alignmodel.infer import infer_sample, load_model, write_prediction

    device = resolve_device(args.device)
    print(f"device={device}")
    model = load_model(args.ckpt, device)
    result = infer_sample(model, args.sample, device)
    out = args.out or (args.sample / "rumaa_lite_pred.json")
    write_prediction(result, out)
    print(f"Wrote {out}")
    print(
        f"repeat_pred={result['repeat_pred']} "
        f"prob={result['repeat_prob']:.3f} label={result['repeat_label']}"
    )
    mismatches = [n for n in result["notes"] if n["pred"] != n["label"]]
    print(f"notes={len(result['notes'])} mismatches={len(mismatches)} extra_spans={len(result['extra_spans'])}")


if __name__ == "__main__":
    main()
