"""Train the score-informed note-presence verifier on synthetic train splits.

Selection uses the val splits' examples only (AUC and the share of played
notes rejected at a 95% keep rate of true missed notes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from alignmodel.joint.presence_verifier_v1 import MARGIN, WINDOW, PresenceConfig, PresenceNet, pitch_template, save_verifier


def _load(paths):
    parts = [np.load(path) for path in paths]
    keys = ("windows", "pitch", "previous", "following", "duration", "uncertainty", "label")
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--val", nargs="+", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    train, val = _load(args.train), _load(args.val)
    table = torch.as_tensor(np.stack([pitch_template(p) for p in range(0, 128)]), device=device)

    def batch(data, index, train_mode):
        windows = torch.as_tensor(data["windows"][index].astype(np.float32), device=device)
        if train_mode:
            shift = int(rng.integers(0, 2 * MARGIN + 1))
            windows = windows[:, :, shift:shift + WINDOW]
            windows = windows + torch.randn_like(windows) * 0.05 + (torch.rand(len(index), 1, 1, device=device) - 0.5) * 0.6
        else:
            windows = windows[:, :, MARGIN:MARGIN + WINDOW]
        pitches = [np.clip(data[key][index].astype(np.int64), 0, 127) for key in ("pitch", "previous", "following")]
        templates = torch.stack([table[torch.as_tensor(p, device=device)] for p in pitches], dim=1)
        scalars = torch.as_tensor(np.stack([np.log1p(np.maximum(data["duration"][index], 0)) / 4.0,
                                            np.log1p(np.maximum(data["uncertainty"][index], 0)) / 4.0], 1),
                                  device=device, dtype=torch.float32)
        labels = torch.as_tensor(data["label"][index].astype(np.float32), device=device)
        return windows, templates, scalars, labels

    model = PresenceNet(PresenceConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    n = len(train["label"])
    steps = int(np.ceil(n / args.batch_size)) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=steps)
    positives = float((train["label"] == 1).sum())
    negatives = float((train["label"] == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(positives, 1.0), device=device))
    print(json.dumps({"train": n, "train_negatives": int(negatives), "val": len(val["label"]),
                      "val_negatives": int((val["label"] == 0).sum())}), flush=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, args.batch_size):
            index = order[start:start + args.batch_size]
            windows, templates, scalars, labels = batch(train, index, True)
            loss = loss_fn(model(windows, templates, scalars), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss) * len(index)
        model.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(val["label"]), 1024):
                index = np.arange(start, min(start + 1024, len(val["label"])))
                windows, templates, scalars, _labels = batch(val, index, False)
                scores.append(torch.sigmoid(model(windows, templates, scalars)).cpu().numpy())
        scores = np.concatenate(scores)
        labels = val["label"]
        auc = _auc(scores, labels)
        negative_scores = np.sort(scores[labels == 0])
        keep95 = float(negative_scores[int(0.95 * (len(negative_scores) - 1))]) if len(negative_scores) else 1.0
        rejected = float((scores[labels == 1] > keep95).mean())
        row = {"epoch": epoch, "train_loss": total / n, "val_auc": auc,
               "threshold_keeping_95pct_missed": keep95, "played_rejected_at_that_threshold": rejected}
        history.append(row)
        print(json.dumps(row), flush=True)
        if auc > best:
            best = auc
            save_verifier(args.output, model, {"epoch": epoch, "val": row, "history": history})
    print(json.dumps({"best_val_auc": best}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
