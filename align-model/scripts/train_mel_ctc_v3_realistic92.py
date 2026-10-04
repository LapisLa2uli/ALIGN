"""Train the dual-resolution long-context CTC transcriber (v3) on a dual-mel 9.2 cache.

Loss: CTC over the crop's aligned-midpoint pitch tokens + weighted frame
losses. Each epoch the model is scored on a fixed stratified val subset
(every third val clip) with greedy CTC decoding; best.pt keeps the epoch with
the highest val pitch-sequence LCS F1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.ctc_decode_v2 import greedy_decode_notes
from alignmodel.transcription.mel_ctc_v1 import ctc_loss
from alignmodel.transcription.mel_ctc_v3 import (
    DualMelConfig,
    DualMelCTCTranscriber,
    augment_dual_batch,
    extract_dual_mel,
    infer_dual_outputs,
    load_dual_checkpoint,
    save_dual_checkpoint,
)
from alignmodel.transcription.mel_v1 import load_audio_mono, mel_transcriber_loss
from alignmodel.transcription.realism_augment_v4 import augment_realism_v4
from alignmodel.transcription.mel_v1_data import MelPackedCache
from realistic92_transcriber_breakdown import Breakdown, load_gold
from train_mel_ctc_realistic92 import CTCCropDataset


def _val_score(model, items, device, midi_min: int, blank_scale: float) -> dict[str, Any]:
    breakdown = Breakdown()
    for mel, gold in items:
        outputs = infer_dual_outputs(model, mel, device)
        notes = greedy_decode_notes(outputs["ctc"], midi_min, blank_scale)
        breakdown.add([pitch for pitch, _frame in notes], gold)
    report = breakdown.report()
    recall_by = report["recall_by"]
    support = report["support_by"]
    short_hits = recall_by.get("dur_lt50", 0) * support.get("dur_lt50", 0) + \
        recall_by.get("dur_50to80", 0) * support.get("dur_50to80", 0)
    short_support = support.get("dur_lt50", 0) + support.get("dur_50to80", 0)
    return {
        "f1": report["f1"], "precision": report["precision"], "recall": report["recall"],
        "lt80_recall": short_hits / max(short_support, 1),
        "repeat_recall": recall_by.get("same_pitch_neighbor"),
        "false_positives": report["false_positives"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--extra-cache", type=Path, action="append", default=[],
                        help="additional dual-mel caches whose train records are mixed in")
    parser.add_argument("--extra-weight", type=float, default=1.0,
                        help="repeat factor for extra-cache train records")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-frames", type=int, default=2048)
    parser.add_argument("--crops-per-clip", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--frame-loss-weight", type=float, default=0.5)
    parser.add_argument("--augmentation-probability", type=float, default=0.6)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--temporal-blocks", type=int, default=12)
    parser.add_argument("--val-stride", type=int, default=3)
    parser.add_argument("--val-blank-scale", type=float, default=0.3)
    parser.add_argument("--extra-val-root", type=Path)
    parser.add_argument("--extra-val-split", type=Path)
    parser.add_argument("--extra-val-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--realism-v4", action="store_true",
                        help="add transition blips, attack scoops and mid-note dips (labels unchanged)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    cache = MelPackedCache(args.cache, deep=False)
    if cache.frontend.n_mels != 192:
        raise ValueError("Expected a dual-mel (192-band) cache")
    if args.init_checkpoint is not None:
        model, _payload = load_dual_checkpoint(args.init_checkpoint, device)
    else:
        model = DualMelCTCTranscriber(DualMelConfig(
            temporal_dim=args.temporal_dim, temporal_blocks=args.temporal_blocks,
        )).to(device)
    config = model.config

    datasets = [(cache, cache.records("train", include_targets=True), 1.0)]
    for extra in args.extra_cache:
        extra_cache = MelPackedCache(extra, deep=False)
        datasets.append((extra_cache, extra_cache.records("train", include_targets=True), args.extra_weight))

    split = json.loads(args.split.read_text(encoding="utf-8"))["splits"]["val"]
    val_names = split[::args.val_stride]
    val_items = []
    for name in val_names:
        audio = load_audio_mono(args.root / name / "performance_audio.wav", 22050)
        mel, _ = extract_dual_mel(audio, device)
        val_items.append((np.asarray(mel, np.float32), load_gold(args.root, name)))
    extra_val_items = []
    if args.extra_val_root is not None and args.extra_val_split is not None:
        extra_names = json.loads(args.extra_val_split.read_text(encoding="utf-8"))["splits"]["val"]
        for name in extra_names[::args.extra_val_stride]:
            audio = load_audio_mono(args.extra_val_root / name / "performance_audio.wav", 22050)
            mel, _ = extract_dual_mel(audio, device)
            extra_val_items.append((np.asarray(mel, np.float32), load_gold(args.extra_val_root, name)))
        print(f"extra_val_items={len(extra_val_items)}", flush=True)
    total_records = sum(len(records) * weight for _c, records, weight in datasets)
    print(f"val_items={len(val_items)} train_records={int(total_records)} "
          f"context_each_side={config.context_frames_each_side} "
          f"params={sum(p.numel() for p in model.parameters())}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = int(math.ceil(total_records * args.crops_per_clip / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=steps_per_epoch * args.epochs, pct_start=0.08,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({
        "schema_version": "align-mel-ctc-v3-realistic92-train-config-v1",
        "args": {key: str(value) for key, value in vars(args).items()},
        "model": config.to_dict(),
        "cache_pack_id": cache.pack_id,
        "split_sha256": sha256_file(args.split),
        "val_clips": val_names,
        "locked_test_materialized": False,
    }, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    with resource_lease(args.resource_status, "gpu", track="mel-ctc-v3-realistic92",
                        command=[sys.executable, *sys.argv],
                        metadata={"output_dir": str(args.output_dir.resolve())}):
        for epoch in range(1, args.epochs + 1):
            loaders = []
            for cache_item, records, weight in datasets:
                repeated = list(records) * max(1, int(round(weight)))
                dataset = CTCCropDataset(
                    cache_item.root, repeated, crop_frames=args.crop_frames, epoch=epoch,
                    seed=args.seed, crops_per_clip=args.crops_per_clip,
                    midi_min=config.midi_min, n_pitches=config.n_pitches,
                )
                loaders.append(dataset)
            combined = torch.utils.data.ConcatDataset(loaders)
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, generator=generator,
                                num_workers=0, pin_memory=True, drop_last=True)
            model.train()
            started = time.perf_counter()
            totals = {"loss": 0.0, "frame": 0.0, "ctc": 0.0}
            batches = 0
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                batch["mel"] = augment_dual_batch(
                    batch["mel"], batch["onset"], long_mels=config.long_mels,
                    probability=args.augmentation_probability,
                )
                if args.realism_v4:
                    batch["mel"] = augment_realism_v4(batch["mel"], batch["onset"], long_mels=config.long_mels)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch["mel"])
                    frame_loss, _parts = mel_transcriber_loss(output, batch)
                sequence_loss = ctc_loss(output["ctc_logits"], batch["frame_mask"],
                                         batch["ctc_target"], batch["ctc_length"])
                loss = args.frame_loss_weight * frame_loss.float() + sequence_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                if batches < steps_per_epoch * args.epochs:
                    scheduler.step()
                totals["loss"] += float(loss)
                totals["frame"] += float(frame_loss)
                totals["ctc"] += float(sequence_loss)
                batches += 1
                if batches % 300 == 0:
                    print(f"epoch={epoch} batch={batches}/{steps_per_epoch} "
                          f"loss={totals['loss']/batches:.4f} ctc={totals['ctc']/batches:.4f} "
                          f"rows_per_sec={batches*args.batch_size/(time.perf_counter()-started):.1f}",
                          flush=True)
            val = _val_score(model, val_items, device, config.midi_min, args.val_blank_scale)
            if extra_val_items:
                extra_val = _val_score(model, extra_val_items, device, config.midi_min, args.val_blank_scale)
                val = {**val, "primary_f1": val["f1"], "extra": extra_val,
                       "f1": 0.5 * (val["f1"] + extra_val["f1"])}
            row = {"epoch": epoch, **{f"train_{k}": v / max(batches, 1) for k, v in totals.items()},
                   "learning_rate": scheduler.get_last_lr()[0], "val": val,
                   "seconds": time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
            extra = {"epoch": epoch, "val": val, "history": history, "cache_pack_id": cache.pack_id,
                     "locked_test_materialized": False}
            save_dual_checkpoint(args.output_dir / "last.pt", model, extra)
            if val["f1"] > best_f1:
                best_f1 = val["f1"]
                save_dual_checkpoint(args.output_dir / "best.pt", model, extra)
            (args.output_dir / "history.json").write_text(
                json.dumps({"best_val_f1": best_f1, "history": history}, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_f1": best_f1}), flush=True)


if __name__ == "__main__":
    main()
