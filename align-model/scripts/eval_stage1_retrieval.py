"""Calibrate retrieval-first Stage 1 without running the alignment beam."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from alignmodel.eval_melodies import summarize_eval_rows
from alignmodel.melody import (
    gold_melodies_from_labels,
    labels_with_canonical_locations,
    load_bundle_notes,
    match_melodies_detail,
    official_label_metrics,
)
from alignmodel.pipeline import load_bundle_audio
from alignmodel.stages.repetition import find_past_repetitions, silence_intervals
from alignmodel.types import PipelineConfig
from eval_stage_dev import calib_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("E:/output"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("align-model/runs/model-a-improve/stage1-retrieval-sweep.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(device=args.device, weights_dir=None)
    clips = []
    started = time.perf_counter()
    dirs = calib_dirs(args.data, args.seed, args.n)
    for i, sample in enumerate(dirs, start=1):
        print(f"retrieve {i}/{len(dirs)} {sample.name}", flush=True)
        audio, chroma, hop_sec, duration = load_bundle_audio(sample, cfg)
        silences = silence_intervals(
            audio,
            cfg.sample_rate,
            silence_db=cfg.silence_db,
            min_silence_sec=cfg.min_silence_sec,
            hop_length=cfg.hop_length,
        )
        candidates = find_past_repetitions(
            chroma,
            hop_sec,
            duration,
            silences,
            max_lookback_sec=cfg.repetition_max_lookback_sec,
            search_step_sec=cfg.repetition_search_step_sec,
            probe_sec=cfg.repetition_probe_sec,
            min_confidence=0.65,
            max_candidates=8,
        )
        document = json.loads((sample / "labels.json").read_text(encoding="utf-8"))
        notes = load_bundle_notes(sample)
        gold = [
            item
            for item in gold_melodies_from_labels(document.get("labels") or [])
            if item.type == "repetition"
        ]
        clips.append((sample, notes, gold, candidates))

    thresholds = [0.72, 0.76, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]
    sweep = []
    for threshold in thresholds:
        rows = []
        for sample, notes, gold, candidates in clips:
            labels = labels_with_canonical_locations(
                [
                    {
                        "type": "repetition",
                        "start_time": item.start_time,
                        "end_time": item.end_time,
                        "repeats_label_range": {
                            "start_time": item.source_start,
                            "end_time": item.source_end,
                        },
                        "extra_copies": item.extra_copies,
                    }
                    for item in candidates
                    if item.confidence >= threshold
                ],
                notes,
            )
            gold_labels = [
                label
                for label in (
                    json.loads((sample / "labels.json").read_text(encoding="utf-8")).get(
                        "labels"
                    )
                    or []
                )
                if label.get("type") == "repetition"
            ]
            official = official_label_metrics(
                gold_labels, labels, score_event_count=len(notes) or None
            )
            legacy = match_melodies_detail(
                gold,
                [
                    item
                    for item in gold_melodies_from_labels(labels)
                ],
                soft=False,
            )
            rows.append(
                {
                    "sample": sample.name,
                    "n_gold": official["n_gold"],
                    "n_pred": official["n_pred"],
                    "n_matched": official["credit"],
                    "melody_f1": official["f1"],
                    "melody_precision": official["precision"],
                    "melody_recall": official["recall"],
                    "official_note_wise": official["official_note_wise"],
                    "legacy_pitch_similarity_f1": float(legacy["f1"]),
                    "per_type": {
                        "repetition": {
                            "n_gold": official["n_gold"],
                            "n_pred": official["n_pred"],
                            "n_matched": official["credit"],
                        }
                    },
                }
            )
        report = summarize_eval_rows(rows)
        sweep.append(
            {
                "threshold": threshold,
                **{key: value for key, value in report.items() if key != "samples"},
            }
        )
    usable = [
        row
        for row in sweep
        if row["mean_melody_recall"] >= 0.20
        and row["mean_n_pred"] <= max(1.5, 2.0 * row["mean_n_gold"])
    ]
    best = max(
        usable or sweep,
        key=lambda row: (
            row["mean_melody_f1"],
            row["mean_melody_precision"],
            row["threshold"],
        ),
    )
    payload = {
        "n": len(clips),
        "seed": args.seed,
        "runtime_sec": round(time.perf_counter() - started, 3),
        "sweep": sweep,
        "best": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
