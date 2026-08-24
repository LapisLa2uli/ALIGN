from __future__ import annotations

import argparse
from pathlib import Path

from alignmodel.config import TrainConfig
from alignmodel.train import resolve_device, train


def main() -> None:
    parser = argparse.ArgumentParser(description="ALIGN RUMAA-lite error detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on ALIGN synth bundles")
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
    p_train.add_argument("--device", default="auto")

    p_inf = sub.add_parser("infer", help="Run a trained checkpoint on one sample")
    p_inf.add_argument("--ckpt", type=Path, required=True)
    p_inf.add_argument("--sample", type=Path, required=True)
    p_inf.add_argument("--out", type=Path, default=None)
    p_inf.add_argument("--device", default="auto")

    args = parser.parse_args()
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

    from alignmodel.infer import infer_sample, load_model, write_prediction

    device = resolve_device(args.device)
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
