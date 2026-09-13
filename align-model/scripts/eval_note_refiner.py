"""Evaluate a cached Basic Pitch refiner on procedural-only frozen splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from alignmodel.transcription import (
    TransNote,
    decode_frozen_basic_pitch,
    evaluate_note_lists,
    load_note_refiner,
    match_notes,
)
from alignmodel.transcription.refiner import decode_refined_notes
from alignmodel.transcription.fine_pitch import apply_pesto_cents
from alignmodel.transcription.refiner_data import (
    load_cached_refiner_features,
    load_refiner_examples,
    load_rendered_target_notes,
)


def _aggregate(rows: list[dict]) -> dict:
    predicted = sum(int(row["n_pred"]) for row in rows)
    target = sum(int(row["n_target"]) for row in rows)
    matched = sum(int(row["n_matched"]) for row in rows)
    precision = matched / max(predicted, 1)
    recall = matched / max(target, 1)
    cents_rows = [
        (float(row["cents_mae"]), int(row["n_matched"]))
        for row in rows
        if row.get("cents_mae") is not None and int(row["n_matched"])
    ]
    intonation_rows = [
        (
            float(row["intonation_cents_mae"]),
            int(row["n_intonation_matched"]),
        )
        for row in rows
        if row.get("intonation_cents_mae") is not None
        and int(row.get("n_intonation_matched") or 0)
    ]
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "n_pred": predicted,
        "n_target": target,
        "n_matched": matched,
        "pred_target_ratio": predicted / max(target, 1),
        "cents_mae": (
            sum(value * count for value, count in cents_rows)
            / sum(count for _value, count in cents_rows)
            if cents_rows
            else None
        ),
        "intonation_cents_mae": (
            sum(value * count for value, count in intonation_rows)
            / sum(count for _value, count in intonation_rows)
            if intonation_rows
            else None
        ),
        "n_intonation_matched": sum(
            int(row.get("n_intonation_matched") or 0) for row in rows
        ),
        "n_clips": len(rows),
    }


def _metrics(predicted: list[TransNote], target: list[TransNote]) -> dict:
    metrics = evaluate_note_lists(predicted, target)
    pairs = match_notes(predicted, target)
    errors = [
        abs(predicted[i].cents - target[j].cents)
        for i, j in pairs
        if abs(target[j].cents) > 1e-6
    ]
    metrics["intonation_cents_mae"] = (
        sum(errors) / len(errors) if errors else None
    )
    metrics["n_intonation_matched"] = len(errors)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--basic-cache-root", type=Path, required=True)
    parser.add_argument("--pesto-cache-root", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, extra = load_note_refiner(args.checkpoint, device)
    model.eval()
    report = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_extra": extra,
        "splits": {},
    }
    for split in args.split or ["val", "test_id"]:
        examples = load_refiner_examples(
            args.manifest, split, procedural_only=True
        )
        if args.max_samples:
            examples = examples[: args.max_samples]
        rows = []
        baseline_rows = []
        samples = []
        for index, example in enumerate(examples, 1):
            basic, pesto = load_cached_refiner_features(
                example, args.basic_cache_root, args.pesto_cache_root
            )
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(
                    torch.from_numpy(basic.note)[None].to(device),
                    torch.from_numpy(basic.onset)[None].to(device),
                    torch.from_numpy(basic.contour)[None].to(device),
                    torch.from_numpy(pesto)[None].to(device),
                )
            predicted = decode_refined_notes(
                {key: value.float().cpu() for key, value in outputs.items()},
                model.config,
            )
            target = [
                TransNote(pitch, start, end, 1.0, cents=cents)
                for pitch, start, end, cents in load_rendered_target_notes(
                    example
                )
            ]
            metrics = _metrics(predicted, target)
            rows.append(metrics)
            baseline_notes = apply_pesto_cents(
                decode_frozen_basic_pitch(basic), basic, pesto
            )
            baseline_rows.append(_metrics(baseline_notes, target))
            samples.append(
                {
                    "sample": example.sample_id,
                    "inference_sec": time.perf_counter() - started,
                    "metrics": metrics,
                }
            )
            if index == 1 or index % 25 == 0 or index == len(examples):
                print(f"{split} {index}/{len(examples)}", flush=True)
        report["splits"][split] = {
            "metrics": _aggregate(rows),
            "frozen_basic_pitch": _aggregate(baseline_rows),
            "samples": samples,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
