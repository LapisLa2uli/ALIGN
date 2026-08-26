from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

from alignmodel.device import device_label, resolve_device
from alignmodel.config import FRAME_HOP_SEC, SCORE_CLASSES, ModelConfig, TrainConfig
from alignmodel.dataset import AlignBundleDataset, collate_bundles, list_sample_dirs
from alignmodel.model import RumaLite


def downsample_extra(extra_y: Tensor, audio_mask: Tensor, stride: int) -> Tensor:
    """Max-pool extra labels onto the encoder's strided time grid."""
    b, t_full = extra_y.shape
    t = audio_mask.size(1)
    need = t * stride
    if extra_y.size(1) < need:
        pad = extra_y.new_zeros(b, need - extra_y.size(1))
        extra_y = torch.cat([extra_y, pad], dim=1)
    pooled = extra_y[:, :need].view(b, t, stride).amax(dim=-1)
    return pooled


def compute_loss(
    outputs: dict[str, Tensor],
    batch: dict,
    cfg: ModelConfig,
) -> tuple[Tensor, dict[str, float]]:
    score_logits = outputs["score_logits"]
    extra_logits = outputs["extra_logits"]
    repeat_logits = outputs["repeat_logits"]
    note_mask = batch["note_mask"]
    weights = torch.ones(cfg.num_score_classes, device=score_logits.device)
    match_idx = SCORE_CLASSES.index("match")
    weights[:] = cfg.error_loss_weight
    weights[match_idx] = 1.0
    score_ce = nn.functional.cross_entropy(
        score_logits.transpose(1, 2),
        batch["score_y"],
        weight=weights,
        reduction="none",
    )
    score_loss = (score_ce * note_mask).sum() / note_mask.sum().clamp_min(1)

    extra_tgt = downsample_extra(batch["extra_y"], outputs["audio_mask"], cfg.audio_stride)
    extra_bce = nn.functional.binary_cross_entropy_with_logits(
        extra_logits, extra_tgt, reduction="none"
    )
    extra_loss = (extra_bce * outputs["audio_mask"]).sum() / outputs["audio_mask"].sum().clamp_min(
        1
    )
    repeat_loss = nn.functional.binary_cross_entropy_with_logits(
        repeat_logits, batch["repeat_y"]
    )
    total = (
        score_loss
        + cfg.extra_loss_weight * extra_loss
        + cfg.repeat_loss_weight * repeat_loss
    )
    return total, {
        "loss": float(total.detach()),
        "score": float(score_loss.detach()),
        "extra": float(extra_loss.detach()),
        "repeat": float(repeat_loss.detach()),
    }


@torch.no_grad()
def evaluate(model: RumaLite, loader: DataLoader, device: torch.device, cfg: ModelConfig) -> dict:
    model.eval()
    totals = {"loss": 0.0, "score": 0.0, "extra": 0.0, "repeat": 0.0}
    n_correct = 0
    n_notes = 0
    n_err_correct = 0
    n_err = 0
    extra_tp = extra_fp = extra_fn = 0
    repeat_correct = 0
    n_clip = 0
    for batch in loader:
        batch_dev = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }
        out = model(
            batch_dev["mel"],
            batch_dev["mel_mask"],
            batch_dev["pitch"],
            batch_dev["onset"],
            batch_dev["duration"],
            batch_dev["note_mask"],
            FRAME_HOP_SEC,
        )
        _, parts = compute_loss(out, batch_dev, cfg)
        for k, val in parts.items():
            totals[k] += val
        pred = out["score_logits"].argmax(-1)
        mask = batch_dev["note_mask"]
        n_correct += int(((pred == batch_dev["score_y"]) & mask).sum())
        n_notes += int(mask.sum())
        err_mask = mask & (batch_dev["score_y"] != SCORE_CLASSES.index("match"))
        n_err_correct += int(((pred == batch_dev["score_y"]) & err_mask).sum())
        n_err += int(err_mask.sum())
        extra_tgt = downsample_extra(
            batch_dev["extra_y"], out["audio_mask"], cfg.audio_stride
        )
        extra_pred = (out["extra_logits"] > 0) & out["audio_mask"]
        extra_true = (extra_tgt > 0.5) & out["audio_mask"]
        extra_tp += int((extra_pred & extra_true).sum())
        extra_fp += int((extra_pred & ~extra_true).sum())
        extra_fn += int((~extra_pred & extra_true).sum())
        repeat_pred = (out["repeat_logits"] > 0).float()
        repeat_correct += int((repeat_pred == batch_dev["repeat_y"]).sum())
        n_clip += batch_dev["repeat_y"].numel()
    n = max(len(loader), 1)
    prec = extra_tp / max(extra_tp + extra_fp, 1)
    rec = extra_tp / max(extra_tp + extra_fn, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {
        "loss": totals["loss"] / n,
        "score_acc": n_correct / max(n_notes, 1),
        "error_acc": n_err_correct / max(n_err, 1),
        "extra_f1": f1,
        "repeat_acc": repeat_correct / max(n_clip, 1),
        "n_notes": n_notes,
        "n_err": n_err,
    }


def train(cfg: TrainConfig) -> Path:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dirs = list_sample_dirs(cfg.data_root)
    if cfg.overfit:
        dirs = dirs[: cfg.overfit]
    if not dirs:
        raise FileNotFoundError(f"No ALIGN bundles under {cfg.data_root}")

    dataset = AlignBundleDataset(dirs, cfg.model)
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
        collate_fn=collate_bundles,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_bundles,
    )

    model = RumaLite(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_params / 1e6:.2f}M device={device_label(device)} train={n_train} val={n_val}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "best.pt"
    last_path = cfg.output_dir / "last.pt"
    history: list[dict] = []
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for step, batch in enumerate(train_loader, start=1):
            batch_dev = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
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
            loss, parts = compute_loss(out, batch_dev, cfg.model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += parts["loss"]
            steps += 1
            if step % cfg.log_every == 0:
                print(
                    f"epoch {epoch} step {step} loss={parts['loss']:.4f} "
                    f"score={parts['score']:.4f} extra={parts['extra']:.4f} "
                    f"repeat={parts['repeat']:.4f}"
                )
        val_metrics = evaluate(model, val_loader, device, cfg.model)
        row = {
            "epoch": epoch,
            "train_loss": running / max(steps, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"score_acc={val_metrics['score_acc']:.3f} "
            f"error_acc={val_metrics['error_acc']:.3f} "
            f"extra_f1={val_metrics['extra_f1']:.3f} "
            f"repeat_acc={val_metrics['repeat_acc']:.3f}"
        )
        ckpt = {
            "model": model.state_dict(),
            "config": cfg.model.__dict__,
            "epoch": epoch,
            "metrics": val_metrics,
        }
        torch.save(ckpt, last_path)
        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, best_path)
        (cfg.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    print(f"Wrote {best_path}")
    return best_path
