"""Train + holdout-eval isolated even bakeoff versions. Reuses shared ES helpers."""

from __future__ import annotations

import importlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from alignmodel.bakeoff.even_common import infer_and_eval_holdout
from alignmodel.config import FRAME_HOP_SEC, ModelConfig
from alignmodel.dataset import list_sample_dirs
from alignmodel.device import device_label, resolve_device
from alignmodel.melody_model import class_index
from alignmodel.melody_train import (
    MelodyBundleDataset,
    MelodyTrainConfig,
    collate_melody,
    dump_train_history,
    step_ema_is_plateau,
    step_ema_update,
)
from datacreate.melody import WeakMelody, match_melodies_detail

FORBIDDEN = {
    "melody-random12k-set",
    "melody-raw2k-set",
    "melody-b-v1-es",
    "melody-b-v3-bio",
    "melody-b-v5-dice",
    "melody-b-v7-combo",
}

VARIANTS = {
    "v2": "alignmodel.bakeoff.v2",
    "v4": "alignmodel.bakeoff.v4",
    "v6": "alignmodel.bakeoff.v6",
}


def load_variant(name: str):
    if name not in VARIANTS:
        raise ValueError(f"unknown even variant {name}")
    return importlib.import_module(VARIANTS[name])


def _assert_safe_out(out: Path) -> None:
    if out.name in FORBIDDEN:
        raise RuntimeError(f"refusing to write sibling/legacy folder {out}")


@torch.no_grad()
def evaluate_variant(mod, model, loader, device, cfg: MelodyTrainConfig) -> dict:
    model.eval()
    totals = {"loss": 0.0, "type": 0.0, "err": 0.0, "copies": 0.0, "coverage": 0.0}
    n_correct = 0
    n_notes = 0
    n_err_correct = 0
    n_err = 0
    n_pred_err = 0
    copies_correct = 0
    n_clip = 0
    f1_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    n_pred_sum = 0
    n_gold_sum = 0
    n_set = 0
    match_i = class_index("match")
    for batch in loader:
        batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(
            batch_dev["mel"],
            batch_dev["mel_mask"],
            batch_dev["pitch"],
            batch_dev["onset"],
            batch_dev["duration"],
            batch_dev["note_mask"],
            FRAME_HOP_SEC,
        )
        _, parts = mod.compute_loss(out, batch_dev, cfg)
        for k, val in parts.items():
            if k in totals:
                totals[k] += val
        mask = batch_dev["note_mask"]
        bsz = int(batch_dev["score_y"].size(0))
        copies_correct += int((out["copies_logits"].argmax(-1) == batch_dev["copies_y"]).sum())
        n_clip += int(batch_dev["copies_y"].numel())
        for b in range(bsz):
            n = int(batch["n_notes"][b])
            types, pred_labs = mod.decode_outputs(out, batch_dev, b)
            pred_ids = torch.tensor(
                [class_index(t) if t in {"match", "miss", "wrong", "extra", "rhythm", "intonation", "repetition"} else match_i for t in types],
                device=device,
            )
            gold = batch_dev["score_y"][b, :n]
            note_m = mask[b, :n]
            n_correct += int(((pred_ids == gold) & note_m).sum())
            n_notes += int(note_m.sum())
            err_mask = note_m & (gold != match_i)
            n_err_correct += int(((pred_ids == gold) & err_mask).sum())
            n_err += int(err_mask.sum())
            n_pred_err += int((note_m & (pred_ids != match_i)).sum())
            pred_mels = [WeakMelody(pitches=lab["pitches"]) for lab in pred_labs]
            gold_mels = [WeakMelody(pitches=list(p)) for p in batch["gold_pitches"][b]]
            detail = match_melodies_detail(gold_mels, pred_mels)
            f1_sum += detail["f1"]
            prec_sum += detail["precision"]
            rec_sum += detail["recall"]
            n_pred_sum += len(pred_mels)
            n_gold_sum += len(gold_mels)
            n_set += 1
    n = max(len(loader), 1)
    n_set = max(n_set, 1)
    return {
        "loss": totals["loss"] / n,
        "note_acc": n_correct / max(n_notes, 1),
        "error_acc": n_err_correct / max(n_err, 1),
        "pred_err_frac": n_pred_err / max(n_notes, 1),
        "copies_acc": copies_correct / max(n_clip, 1),
        "set_f1": f1_sum / n_set,
        "set_precision": prec_sum / n_set,
        "set_recall": rec_sum / n_set,
        "mean_n_pred": n_pred_sum / n_set,
        "mean_n_gold": n_gold_sum / n_set,
        "n_notes": n_notes,
        "n_err": n_err,
    }


def train_variant(mod, cfg: MelodyTrainConfig) -> tuple[Path, dict]:
    _assert_safe_out(Path(cfg.output_dir))
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dirs = list_sample_dirs(cfg.data_root)
    if cfg.max_samples:
        dirs = dirs[: cfg.max_samples]
    if cfg.overfit:
        dirs = dirs[: cfg.overfit]
    if not dirs:
        raise FileNotFoundError(f"No ALIGN bundles under {cfg.data_root}")

    dataset = MelodyBundleDataset(dirs, cfg.model)
    n_val = max(1, int(len(dataset) * cfg.val_fraction)) if cfg.overfit == 0 else max(
        1, min(2, len(dataset) // 5)
    )
    n_train = len(dataset) - n_val
    if n_train < 1:
        n_train = len(dataset)
        n_val = 0
        train_set = dataset
        val_set = dataset
    else:
        train_set, val_set = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
        )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_melody,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_melody,
    )

    model = mod.build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"variant={getattr(mod, 'VARIANT', cfg.variant)} params={n_params / 1e6:.2f}M "
        f"device={device_label(device)} train={n_train} val={n_val} batch={cfg.batch_size}",
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "best.pt"
    last_path = cfg.output_dir / "last.pt"
    history: list[dict] = []
    best_f1 = -1.0
    patience_best = -1.0
    stale_epochs = 0
    ema: float | None = None
    ema_prev: float | None = None
    plateau_logged = 0
    global_step = 0
    stopped_reason = "max_epochs"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        stop_train = False
        for step, batch in enumerate(train_loader, start=1):
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(
                batch_dev["mel"],
                batch_dev["mel_mask"],
                batch_dev["pitch"],
                batch_dev["onset"],
                batch_dev["duration"],
                batch_dev["note_mask"],
                FRAME_HOP_SEC,
            )
            loss, parts = mod.compute_loss(out, batch_dev, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += parts["loss"]
            steps += 1
            global_step += 1
            ema = step_ema_update(ema, parts["loss"], cfg.es_ema_alpha)
            if step % cfg.log_every == 0:
                print(
                    f"epoch {epoch} step {step} loss={parts['loss']:.4f} "
                    f"type={parts['type']:.4f} err={parts.get('err', 0.0):.4f} "
                    f"copies={parts['copies']:.4f} coverage={parts.get('coverage', 0.0):.4f} "
                    f"ema={ema:.4f}",
                    flush=True,
                )
                if global_step >= cfg.es_min_steps and ema_prev is not None:
                    if step_ema_is_plateau(ema, ema_prev, cfg.es_rel_tol):
                        plateau_logged += 1
                    else:
                        plateau_logged = 0
                    if plateau_logged >= cfg.es_plateau_steps:
                        stopped_reason = "step_ema_plateau"
                        stop_train = True
                ema_prev = ema
            if stop_train:
                break
        val_metrics = evaluate_variant(mod, model, val_loader, device, cfg)
        row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": running / max(steps, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"set_f1={val_metrics['set_f1']:.3f} "
            f"set_p={val_metrics['set_precision']:.3f} "
            f"set_r={val_metrics['set_recall']:.3f} "
            f"n_pred={val_metrics['mean_n_pred']:.2f} "
            f"n_gold={val_metrics['mean_n_gold']:.2f} "
            f"note_acc={val_metrics['note_acc']:.3f} "
            f"error_acc={val_metrics['error_acc']:.3f} "
            f"pred_err={val_metrics['pred_err_frac']:.3f} "
            f"copies_acc={val_metrics['copies_acc']:.3f}",
            flush=True,
        )
        ckpt = {
            "model": model.state_dict(),
            "config": cfg.model.__dict__,
            "variant": getattr(mod, "VARIANT", cfg.variant),
            "epoch": epoch,
            "step": global_step,
            "metrics": val_metrics,
        }
        torch.save(ckpt, last_path)
        if val_metrics["set_f1"] >= best_f1:
            best_f1 = val_metrics["set_f1"]
            torch.save(ckpt, best_path)
        if val_metrics["set_f1"] >= patience_best + cfg.es_f1_delta:
            patience_best = val_metrics["set_f1"]
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.es_patience_epochs and stopped_reason == "max_epochs":
                stopped_reason = "val_f1_patience"
                stop_train = True
        if stop_train:
            history[-1]["stopped_reason"] = stopped_reason
        dump_train_history(
            cfg.output_dir / "history.json",
            history,
            stopped_reason if stop_train else None,
            epoch,
            global_step,
        )
        if stop_train:
            break

    if history and "stopped_reason" not in history[-1]:
        history[-1]["stopped_reason"] = stopped_reason
    dump_train_history(
        cfg.output_dir / "history.json",
        history,
        stopped_reason,
        history[-1]["epoch"] if history else 0,
        global_step,
    )
    last_epoch = history[-1]["epoch"] if history else 0
    print(
        f"Wrote {best_path} stopped_reason={stopped_reason} epoch={last_epoch} step={global_step}",
        flush=True,
    )
    info = {
        "best_pt": str(best_path),
        "stopped_reason": stopped_reason,
        "epoch": last_epoch,
        "step": global_step,
        "best_val_set_f1": best_f1,
        "history": history,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path, info


def load_trained(mod, ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["config"])
    model = mod.build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def run_one(
    variant: str,
    *,
    data_root: Path,
    runs_root: Path,
    epochs: int = 8,
    batch_size: int = 2,
    device: str = "cuda",
    seed: int = 365,
    n_eval: int = 100,
) -> dict:
    mod = load_variant(variant)
    run_id = getattr(mod, "RUN_ID")
    out = runs_root / "melody-bakeoff" / run_id
    _assert_safe_out(out)
    bakeoff_dir = out
    bakeoff_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = runs_root / "eval-dual" / f"{run_id}_on_random12k-holdout"
    oom = False
    used_batch = batch_size
    t0 = time.time()
    last_err = None
    train_info = None
    for batch in [batch_size, 1] if batch_size > 1 else [1]:
        cfg = MelodyTrainConfig(
            data_root=data_root,
            output_dir=out,
            epochs=epochs,
            batch_size=batch,
            lr=2e-4,
            device=device,
            seed=seed,
            variant=variant,
        )
        try:
            print(f"=== START train {run_id} batch={batch} ===", flush=True)
            _best, train_info = train_variant(mod, cfg)
            used_batch = batch
            last_err = None
            break
        except RuntimeError as exc:
            last_err = exc
            if "out of memory" not in str(exc).lower():
                raise
            oom = True
            print(f"{run_id} OOM at batch {batch}; emptying cache", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch == 1:
                raise
    if train_info is None:
        raise RuntimeError(f"train {run_id} failed: {last_err}")
    elapsed_train = (time.time() - t0) / 60.0
    print(f"=== START eval {run_id} holdout ===", flush=True)
    torch_device = resolve_device(device)
    model, blob = load_trained(mod, out / "best.pt", torch_device)
    holdout = infer_and_eval_holdout(
        infer_fn=mod.infer_sample,
        model=model,
        device=torch_device,
        data_root=data_root,
        pred_dir=pred_dir,
        n_eval=n_eval,
        seed=seed,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val_metrics = blob.get("metrics") or {}
    metrics = {
        "id": run_id,
        "variant": variant,
        "set_f1": holdout["set_f1"],
        "P": holdout["precision"],
        "R": holdout["recall"],
        "mean_n_pred": holdout["mean_n_pred"],
        "mean_n_gold": holdout["mean_n_gold"],
        "val": val_metrics,
        "epochs": train_info["epoch"],
        "steps": train_info["step"],
        "stopped_reason": train_info["stopped_reason"],
        "best.pt": str((out / "best.pt").resolve()),
        "batch_size": used_batch,
        "oom": oom,
        "train_elapsed_min": round(elapsed_train, 2),
        "n_holdout": holdout["n_samples"],
        "pred_dir": str(pred_dir),
    }
    (bakeoff_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"=== DONE {run_id} holdout f1={metrics['set_f1']:.3f} "
        f"p={metrics['P']:.3f} r={metrics['R']:.3f} n_pred={metrics['mean_n_pred']:.2f} "
        f"stop={metrics['stopped_reason']} ===",
        flush=True,
    )
    return metrics
