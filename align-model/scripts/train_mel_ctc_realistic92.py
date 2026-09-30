"""Train the mel CTC pitch-sequence transcriber on the aligned 9.2 cache.

Frame losses use the forced-aligned targets; the CTC loss uses the written
pitches of notes whose aligned midpoint lies inside each crop. Each epoch the
model is scored on a fixed val subset with greedy CTC decoding and
written-pitch sequence LCS F1; the best epoch is kept as best.pt.
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
from alignmodel.transcription.mel_ctc_v1 import (
    MelCTCTranscriber,
    ctc_loss,
    greedy_decode,
    infer_ctc_probabilities,
    load_ctc_checkpoint,
    pitch_tokens,
    save_ctc_checkpoint,
)
from alignmodel.transcription.mel_v1 import (
    MelTranscriberConfig,
    extract_log_mel,
    load_audio_mono,
    mel_transcriber_loss,
)
from alignmodel.transcription.mel_v1_data import (
    MelCropDataset,
    MelPackedCache,
    augment_mel_batch,
)
from eval_realistic92_transcriber import lcs_length


MAX_TOKENS = 400


class CTCCropDataset(MelCropDataset):
    def __init__(self, *args: Any, midi_min: int, n_pitches: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.midi_min = midi_min
        self.n_pitches = n_pitches

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        record = self.records[index % len(self.records)]
        hop = self.frontend.hop_sec
        first = item["crop_start"] * hop
        last = (item["crop_start"] + int(item["frame_mask"].sum())) * hop
        pitches = [
            int(note["pitch"]) for note in record.target
            if first <= 0.5 * (float(note["start_sec"]) + float(note["end_sec"])) < last
        ]
        tokens = pitch_tokens(pitches, self.midi_min, self.n_pitches)[:MAX_TOKENS]
        padded = np.zeros(MAX_TOKENS, np.int64)
        padded[:len(tokens)] = tokens
        item["ctc_target"] = torch.from_numpy(padded)
        item["ctc_length"] = torch.tensor(len(tokens), dtype=torch.long)
        return item


def _val_score(model, val_items, device, midi_min) -> dict[str, float]:
    matched = predicted = gold = 0
    for mel, gold_pitch in val_items:
        probabilities = infer_ctc_probabilities(model, mel, device)
        pitches = [pitch for pitch, _ in greedy_decode(probabilities, midi_min)]
        matched += lcs_length(pitches, gold_pitch)
        predicted += len(pitches)
        gold += len(gold_pitch)
    precision = matched / max(predicted, 1)
    recall = matched / max(gold, 1)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "count_ratio": predicted / max(gold, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--init-mel-checkpoint", type=Path)
    parser.add_argument("--init-ctc-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--crop-frames", type=int, default=1024)
    parser.add_argument("--crops-per-clip", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--frame-loss-weight", type=float, default=0.5)
    parser.add_argument("--ctc-weight", type=float, default=1.0)
    parser.add_argument("--augmentation-probability", type=float, default=0.55)
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--temporal-dim", type=int, default=128)
    parser.add_argument("--temporal-blocks", type=int, default=8)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    cache = MelPackedCache(args.cache, deep=False)
    frontend = cache.frontend

    if args.init_ctc_checkpoint is not None:
        model, _frontend, _payload = load_ctc_checkpoint(args.init_ctc_checkpoint, device)
    elif args.init_mel_checkpoint is not None:
        payload = torch.load(args.init_mel_checkpoint, map_location=device, weights_only=False)
        model = MelCTCTranscriber(MelTranscriberConfig.from_dict(payload["model_config"]))
        model.base.load_state_dict(payload["model_state_dict"])
    else:
        model = MelCTCTranscriber(MelTranscriberConfig(
            temporal_dim=args.temporal_dim,
            temporal_blocks=args.temporal_blocks,
            conv_channels=args.conv_channels,
        ))
    model.to(device)
    config = model.config
    training = cache.records("train", include_targets=True)

    split = json.loads(args.split.read_text(encoding="utf-8"))["splits"]["val"]
    val_items = []
    for name in split[:args.val_limit]:
        sample = args.root / name
        audio = load_audio_mono(sample / "performance_audio.wav", frontend.sample_rate)
        mel, _ = extract_log_mel(audio, frontend, device=device)
        rendered = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))[
            "rendered_notes"
        ]
        val_items.append((
            np.asarray(mel, np.float32),
            [int(row["pitch_midi_written"]) for row in rendered],
        ))
    print(f"val_items={len(val_items)} train_records={len(training)}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(training) * args.crops_per_clip / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate,
        total_steps=steps_per_epoch * args.epochs, pct_start=0.08,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({
        "schema_version": "align-mel-ctc-realistic92-train-config-v1",
        "args": {key: str(value) for key, value in vars(args).items()},
        "model": config.to_dict(),
        "cache_pack_id": cache.pack_id,
        "split_sha256": sha256_file(args.split),
        "locked_test_materialized": False,
    }, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    with resource_lease(
        args.resource_status, "gpu", track="mel-ctc-realistic92",
        command=[sys.executable, *sys.argv],
        metadata={"output_dir": str(args.output_dir.resolve())},
    ):
        baseline = _val_score(model, val_items, device, config.midi_min)
        print(json.dumps({"epoch": 0, "val": baseline}), flush=True)
        for epoch in range(1, args.epochs + 1):
            dataset = CTCCropDataset(
                cache.root, training, crop_frames=args.crop_frames, epoch=epoch,
                seed=args.seed, crops_per_clip=args.crops_per_clip,
                midi_min=config.midi_min, n_pitches=config.n_pitches,
            )
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True,
                generator=generator, num_workers=0, pin_memory=True, drop_last=True,
            )
            model.train()
            started = time.perf_counter()
            totals = {"loss": 0.0, "frame": 0.0, "ctc": 0.0}
            batches = 0
            for batch in loader:
                batch = {
                    key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                batch["mel"] = augment_mel_batch(
                    batch["mel"], probability=args.augmentation_probability
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch["mel"])
                    frame_loss, _parts = mel_transcriber_loss(output, batch)
                sequence_loss = ctc_loss(
                    output["ctc_logits"], batch["frame_mask"],
                    batch["ctc_target"], batch["ctc_length"],
                )
                loss = args.frame_loss_weight * frame_loss.float() + args.ctc_weight * sequence_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                scheduler.step()
                totals["loss"] += float(loss)
                totals["frame"] += float(frame_loss)
                totals["ctc"] += float(sequence_loss)
                batches += 1
                if batches % 200 == 0:
                    print(
                        f"epoch={epoch} batch={batches}/{steps_per_epoch} "
                        f"loss={totals['loss']/batches:.4f} ctc={totals['ctc']/batches:.4f} "
                        f"rows_per_sec={batches*args.batch_size/(time.perf_counter()-started):.1f}",
                        flush=True,
                    )
            val = _val_score(model, val_items, device, config.midi_min)
            row = {
                "epoch": epoch,
                **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
                "learning_rate": scheduler.get_last_lr()[0],
                "val": val,
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            extra = {"epoch": epoch, "val": val, "history": history,
                     "cache_pack_id": cache.pack_id, "locked_test_materialized": False}
            save_ctc_checkpoint(args.output_dir / "last.pt", model, frontend, extra)
            if val["f1"] > best_f1:
                best_f1 = val["f1"]
                save_ctc_checkpoint(args.output_dir / "best.pt", model, frontend, extra)
            (args.output_dir / "history.json").write_text(
                json.dumps({"best_val_f1": best_f1, "history": history}, indent=2),
                encoding="utf-8",
            )
    print(json.dumps({"best_val_f1": best_f1}), flush=True)


if __name__ == "__main__":
    main()
