from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

from alignmodel.config import (
    ALIGN_TO_MELODY,
    FRAME_HOP_SEC,
    ModelConfig,
)
from alignmodel.dataset import list_sample_dirs
from alignmodel.device import device_label, resolve_device
from alignmodel.melody import gold_melodies_from_labels, load_bundle_notes, melody_span_from_label, official_label_metrics
from alignmodel.bakeoff.common import span_targets_from_score_y
from alignmodel.melody_model import MelodyFirst, class_index, decode_note_runs, types_from_logits
from alignmodel.stages.gold import extra_copies_of, load_first_pass_labels
from datacreate.melody import ScoreSoundingNote, WeakMelody, match_melodies_detail

# Specific faults beat repetition so one restart cannot paint the whole clip.
_PRIORITY = {
    class_index("wrong"): 6,
    class_index("extra"): 5,
    class_index("miss"): 4,
    class_index("rhythm"): 3,
    class_index("intonation"): 2,
    class_index("repetition"): 1,
    class_index("match"): 0,
}

_MAX_REP_CORE_NOTES = 16
_MAX_REP_CORE_FRAC = 0.40


@dataclass
class MelodyTrainConfig:
    data_root: Path = field(default_factory=lambda: Path("synth-pipeline/output"))
    output_dir: Path = field(default_factory=lambda: Path("align-model/runs/melody"))
    epochs: int = 8
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-2
    val_fraction: float = 0.1
    seed: int = 365
    num_workers: int = 0
    overfit: int = 0
    device: str = "cuda"
    grad_clip: float = 1.0
    log_every: int = 10
    max_samples: int = 0
    error_loss_weight: float = 6.0
    copies_loss_weight: float = 1.0
    coverage_loss_weight: float = 2.0
    error_aux_weight: float = 3.0
    model: ModelConfig = field(default_factory=ModelConfig)
    # Shared early stop is always on; versions cannot disable it.
    es_min_steps: int = 80
    es_plateau_steps: int = 150
    es_ema_alpha: float = 0.05
    es_rel_tol: float = 0.002
    es_f1_delta: float = 0.005
    es_patience_epochs: int = 2
    variant: str = "v1"


def step_ema_update(ema: float | None, loss: float, alpha: float = 0.05) -> float:
    loss_f = float(loss)
    if ema is None:
        return loss_f
    return alpha * loss_f + (1.0 - alpha) * float(ema)


def step_ema_is_plateau(ema: float, ema_prev: float, rel_tol: float = 0.002) -> bool:
    return abs(float(ema) - float(ema_prev)) / max(float(ema_prev), 1e-6) < rel_tol


def dump_train_history(
    path: Path,
    history: list[dict],
    stopped_reason: str | None,
    epoch: int,
    step: int,
) -> None:
    payload = {
        "stopped_reason": stopped_reason,
        "epoch": epoch,
        "step": step,
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _variant_name(cfg: MelodyTrainConfig) -> str:
    return (getattr(cfg, "variant", "v1") or "v1").lower()


def build_train_model(cfg: MelodyTrainConfig):
    v = _variant_name(cfg)
    if v in {"v1", "control", ""}:
        return MelodyFirst(cfg.model)
    from alignmodel.bakeoff import build_model

    return build_model(v, cfg.model)


def compute_train_loss(outputs, batch, cfg: MelodyTrainConfig):
    v = _variant_name(cfg)
    if v in {"v1", "control", ""}:
        return compute_loss(outputs, batch, cfg)
    from alignmodel.bakeoff import compute_loss as variant_loss

    return variant_loss(v, outputs, batch, cfg)


def predict_eval_labels(out, batch, b: int, cfg: MelodyTrainConfig) -> list[dict]:
    n = int(batch["n_notes"][b])
    notes = _notes_from_tensors(
        batch["pitch"][b],
        batch["onset"][b],
        batch["duration"][b],
        n,
    )
    v = _variant_name(cfg)
    if v in {"v1", "control", ""}:
        types = types_from_logits(out["type_logits"][b, :n])
        copies = int(out["copies_logits"][b].argmax(-1).cpu())
        return decode_note_runs(types, notes, copies)
    from alignmodel.bakeoff import decode_sample

    return decode_sample(v, out, b, n, notes)


class MelodyBundleDataset(Dataset):
    def __init__(self, sample_dirs: list[Path], cfg: ModelConfig):
        self.sample_dirs = sample_dirs
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.sample_dirs[idx]
        mel = np.load(sample_dir / "performance_mel.npy").astype(np.float32)
        if mel.ndim != 2:
            raise ValueError(f"Unexpected mel shape {mel.shape} in {sample_dir}")
        if mel.shape[0] > mel.shape[1] and mel.shape[0] != self.cfg.n_mels:
            mel = mel.T
        if mel.shape[0] != self.cfg.n_mels:
            if mel.shape[0] > self.cfg.n_mels:
                mel = mel[: self.cfg.n_mels]
            else:
                pad = np.zeros((self.cfg.n_mels - mel.shape[0], mel.shape[1]), dtype=np.float32)
                mel = np.concatenate([mel, pad], axis=0)

        t = min(mel.shape[1], self.cfg.max_audio_frames)
        mel = mel[:, :t]
        mel_mask = np.ones(t, dtype=np.bool_)

        notes = load_bundle_notes(sample_dir)[: self.cfg.max_score_notes]
        labels = load_first_pass_labels(sample_dir)
        n = len(notes)
        if n == 0:
            from datacreate.melody import ScoreSoundingNote

            notes = [
                ScoreSoundingNote(
                    index=0,
                    pitch=60,
                    start=0.0,
                    end=0.25,
                    ql_start=0.0,
                    ql_end=0.25,
                    measure=1,
                    note_id="note_0000",
                )
            ]
            n = 1

        pitch = np.zeros(self.cfg.max_score_notes, dtype=np.int64)
        onset = np.zeros(self.cfg.max_score_notes, dtype=np.float32)
        duration = np.zeros(self.cfg.max_score_notes, dtype=np.float32)
        note_mask = np.zeros(self.cfg.max_score_notes, dtype=np.bool_)
        score_y = np.zeros(self.cfg.max_score_notes, dtype=np.int64)
        pri = np.zeros(self.cfg.max_score_notes, dtype=np.int64)
        for i, note in enumerate(notes):
            pitch[i] = max(0, min(127, note.pitch))
            onset[i] = note.start
            duration[i] = max(0.04, note.end - note.start)
            note_mask[i] = True

        copies = 0
        for lab in labels:
            kind = lab.get("type")
            cls_name = ALIGN_TO_MELODY.get(kind)
            if cls_name is None:
                continue
            cls = class_index(cls_name)
            core = _train_core_span(lab, notes, n)
            if core is None:
                continue
            lo, hi = core
            for i in range(lo, hi + 1):
                if i >= n:
                    continue
                if _PRIORITY[cls] > pri[i]:
                    score_y[i] = cls
                    pri[i] = _PRIORITY[cls]
            if kind == "repetition":
                copies = max(copies, extra_copies_of(lab))

        bio_y, type_span_y, mask_y = span_targets_from_score_y(
            score_y, n, class_index("match")
        )
        golds = gold_melodies_from_labels(labels)
        return {
            "mel": torch.from_numpy(mel),
            "mel_mask": torch.from_numpy(mel_mask),
            "pitch": torch.from_numpy(pitch),
            "onset": torch.from_numpy(onset),
            "duration": torch.from_numpy(duration),
            "note_mask": torch.from_numpy(note_mask),
            "score_y": torch.from_numpy(score_y),
            "bio_y": torch.from_numpy(bio_y),
            "type_span_y": torch.from_numpy(type_span_y),
            "mask_y": torch.from_numpy(mask_y),
            "copies_y": torch.tensor(copies, dtype=torch.int64),
            "n_notes": n,
            "sample_id": sample_dir.name,
            "sample_dir": str(sample_dir),
            "gold_pitches": [list(g.pitches) for g in golds],
            "gold_labels": list(labels),
        }


def _train_core_span(
    lab: dict, notes: list, n: int
) -> tuple[int, int] | None:
    """Inclusive core [lo, hi]. Drop clip-wide repetition so it cannot swallow the score."""
    part = lab.get("score_part") if isinstance(lab.get("score_part"), dict) else None
    if part is not None and part.get("start_note_index") is not None:
        pad = max(0, int(part.get("pad_notes") or 2))
        i0 = max(0, min(int(part["start_note_index"]), max(n - 1, 0)))
        i1 = max(i0, min(int(part.get("end_note_index", i0)), max(n - 1, 0)))
        lo = i0 + pad
        hi = i1 - pad
        if lo > hi:
            mid = (i0 + i1) // 2
            lo = hi = mid
    else:
        span = melody_span_from_label(lab, notes, pad_notes=0)
        if span is None:
            return None
        lo, hi = span.start_note_index, span.end_note_index
    span_len = hi - lo + 1
    if lab.get("type") == "repetition" and (
        span_len > _MAX_REP_CORE_NOTES or span_len / max(n, 1) > _MAX_REP_CORE_FRAC
    ):
        return None
    return lo, hi


def collate_melody(batch: list[dict]) -> dict:
    max_t = max(item["mel"].shape[-1] for item in batch)
    n_mels = batch[0]["mel"].shape[0]
    b = len(batch)
    mel = torch.zeros(b, n_mels, max_t)
    mel_mask = torch.zeros(b, max_t, dtype=torch.bool)
    for i, item in enumerate(batch):
        t = item["mel"].shape[-1]
        mel[i, :, :t] = item["mel"]
        mel_mask[i, :t] = item["mel_mask"]
    return {
        "mel": mel,
        "mel_mask": mel_mask,
        "pitch": torch.stack([item["pitch"] for item in batch]),
        "onset": torch.stack([item["onset"] for item in batch]),
        "duration": torch.stack([item["duration"] for item in batch]),
        "note_mask": torch.stack([item["note_mask"] for item in batch]),
        "score_y": torch.stack([item["score_y"] for item in batch]),
        "bio_y": torch.stack([item["bio_y"] for item in batch]),
        "type_span_y": torch.stack([item["type_span_y"] for item in batch]),
        "mask_y": torch.stack([item["mask_y"] for item in batch]),
        "copies_y": torch.stack([item["copies_y"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "sample_dir": [item["sample_dir"] for item in batch],
        "n_notes": [item["n_notes"] for item in batch],
        "gold_pitches": [item["gold_pitches"] for item in batch],
        "gold_labels": [item["gold_labels"] for item in batch],
    }


def compute_loss(
    outputs: dict[str, Tensor],
    batch: dict,
    cfg: MelodyTrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    type_logits = outputs["type_logits"]
    note_mask = batch["note_mask"]
    y = batch["score_y"]
    match_i = class_index("match")
    rep_i = class_index("repetition")
    ce = nn.functional.cross_entropy(
        type_logits.transpose(1, 2),
        y,
        reduction="none",
    )
    match_mask = note_mask & (y == match_i)
    err_mask = note_mask & (y != match_i)
    match_loss = (ce * match_mask).sum() / match_mask.sum().clamp_min(1)
    err_loss = (ce * err_mask).sum() / err_mask.sum().clamp_min(1)
    type_loss = 0.25 * match_loss + cfg.error_aux_weight * err_loss
    copies_loss = nn.functional.cross_entropy(outputs["copies_logits"], batch["copies_y"])
    probs = nn.functional.softmax(type_logits, dim=-1)
    mask_f = note_mask.float()
    denom = mask_f.sum().clamp_min(1)
    pred_rep = (probs[..., rep_i] * mask_f).sum() / denom
    coverage = nn.functional.relu(pred_rep - 0.30)
    total = type_loss + cfg.copies_loss_weight * copies_loss + cfg.coverage_loss_weight * coverage
    return total, {
        "loss": float(total.detach()),
        "type": float(type_loss.detach()),
        "err": float(err_loss.detach()),
        "copies": float(copies_loss.detach()),
        "coverage": float(coverage.detach()),
    }


@torch.no_grad()
def evaluate(
    model: MelodyFirst, loader: DataLoader, device: torch.device, cfg: MelodyTrainConfig
) -> dict:
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
    note_credit = 0.0
    note_predicted = 0
    note_gold = 0
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
        _, parts = compute_train_loss(out, batch_dev, cfg)
        for k, val in parts.items():
            totals[k] = totals.get(k, 0.0) + val
        variant = _variant_name(cfg)
        mask = batch_dev["note_mask"]
        if variant in {"v3", "bio"} and "bio_logits" in out:
            pred = out["bio_logits"].argmax(-1)
            target = batch_dev["bio_y"]
            n_pred_err += int((mask & (pred != 0)).sum())
        elif variant in {"v5", "dice"} and "mask_logits" in out:
            pred = (out["mask_logits"].sigmoid() >= 0.5).long()
            target = (batch_dev["mask_y"] > 0.5).long()
            n_pred_err += int((mask & (pred > 0)).sum())
        else:
            pred = out["type_logits"].argmax(-1)
            target = batch_dev["score_y"]
            n_pred_err += int((mask & (pred != class_index("match"))).sum())
        n_correct += int(((pred == target) & mask).sum())
        n_notes += int(mask.sum())
        err_mask = mask & (batch_dev["score_y"] != class_index("match"))
        n_err_correct += int(((pred == target) & err_mask).sum())
        n_err += int(err_mask.sum())
        copies_correct += int((out["copies_logits"].argmax(-1) == batch_dev["copies_y"]).sum())
        n_clip += int(batch_dev["copies_y"].numel())
        bsz = int(out["copies_logits"].size(0))
        for b in range(bsz):
            pred_labs = predict_eval_labels(out, batch_dev, b, cfg)
            pred_mels = [WeakMelody(pitches=lab["pitches"]) for lab in pred_labs]
            gold_mels = [WeakMelody(pitches=list(p)) for p in batch["gold_pitches"][b]]
            detail = match_melodies_detail(gold_mels, pred_mels)
            f1_sum += detail["f1"]
            prec_sum += detail["precision"]
            rec_sum += detail["recall"]
            n_pred_sum += len(pred_mels)
            n_gold_sum += len(gold_mels)
            n_set += 1
            official = official_label_metrics(
                batch.get("gold_labels")[b] if batch.get("gold_labels") else [],
                pred_labs,
                score_event_count=int(batch["n_notes"][b]) or None,
            )
            if official["status"] == "available":
                note_credit += official["credit"]
                note_predicted += official["predicted"]
                note_gold += official["gold"]
    n = max(len(loader), 1)
    n_set = max(n_set, 1)
    if not note_predicted and not note_gold:
        note_precision = note_recall = note_f1 = 1.0 if n_set else 0.0
    else:
        note_precision = note_credit / note_predicted if note_predicted else 0.0
        note_recall = note_credit / note_gold if note_gold else 0.0
        note_f1 = (
            2.0 * note_precision * note_recall / (note_precision + note_recall)
            if note_precision + note_recall
            else 0.0
        )
    return {
        "loss": totals["loss"] / n,
        "note_acc": n_correct / max(n_notes, 1),
        "error_acc": n_err_correct / max(n_err, 1),
        "pred_err_frac": n_pred_err / max(n_notes, 1),
        "copies_acc": copies_correct / max(n_clip, 1),
        "note_wise_f1": note_f1,
        "note_wise_precision": note_precision,
        "note_wise_recall": note_recall,
        "set_f1": f1_sum / n_set,
        "set_precision": prec_sum / n_set,
        "set_recall": rec_sum / n_set,
        "legacy_pitch_similarity_f1": f1_sum / n_set,
        "mean_n_pred": n_pred_sum / n_set,
        "mean_n_gold": n_gold_sum / n_set,
        "n_notes": n_notes,
        "n_err": n_err,
    }


def _notes_from_tensors(pitch, onset, duration, n: int) -> list[ScoreSoundingNote]:
    notes: list[ScoreSoundingNote] = []
    for i in range(n):
        start = float(onset[i])
        dur = float(duration[i])
        notes.append(
            ScoreSoundingNote(
                index=i,
                pitch=int(pitch[i]),
                start=start,
                end=start + max(dur, 0.04),
                ql_start=0.0,
                ql_end=0.0,
                measure=None,
                note_id=f"note_{i:04d}",
            )
        )
    return notes


def train_melody(cfg: MelodyTrainConfig) -> Path:
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

    model = build_train_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"params={n_params / 1e6:.2f}M device={device_label(device)} "
        f"train={n_train} val={n_val} variant={_variant_name(cfg)}"
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
            loss, parts = compute_train_loss(out, batch_dev, cfg)
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
                    f"ema={ema:.4f}"
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
        val_metrics = evaluate(model, val_loader, device, cfg)
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
            f"note_wise_f1={val_metrics['note_wise_f1']:.3f} "
            f"legacy_set_f1={val_metrics['set_f1']:.3f} "
            f"set_p={val_metrics['set_precision']:.3f} "
            f"set_r={val_metrics['set_recall']:.3f} "
            f"n_pred={val_metrics['mean_n_pred']:.2f} "
            f"n_gold={val_metrics['mean_n_gold']:.2f} "
            f"note_acc={val_metrics['note_acc']:.3f} "
            f"error_acc={val_metrics['error_acc']:.3f} "
            f"pred_err={val_metrics['pred_err_frac']:.3f} "
            f"copies_acc={val_metrics['copies_acc']:.3f}"
        )
        ckpt = {
            "model": model.state_dict(),
            "config": cfg.model.__dict__,
            "variant": getattr(cfg, "variant", "v1"),
            "epoch": epoch,
            "step": global_step,
            "metrics": val_metrics,
        }
        torch.save(ckpt, last_path)
        if val_metrics["note_wise_f1"] >= best_f1:
            best_f1 = val_metrics["note_wise_f1"]
            torch.save(ckpt, best_path)
        if val_metrics["note_wise_f1"] >= patience_best + cfg.es_f1_delta:
            patience_best = val_metrics["note_wise_f1"]
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
    print(f"Wrote {best_path} stopped_reason={stopped_reason} epoch={last_epoch} step={global_step}")
    return best_path
