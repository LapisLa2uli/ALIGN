"""Per-gold-class recall of frozen Basic Pitch on 9.2 train clips.

Separates renderer-only notes (no performed source, mostly ornament
expansions) from performed notes to test whether the gold sequence is fully
realized in the Muse Sounds audio. Test clips are never read.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from alignmodel.transcription.basic_pitch import (
    decode_frozen_basic_pitch,
    extract_sample_basic_pitch_features,
    sanitize_basic_pitch_notes,
    FROZEN_DECODE_CONFIG,
)
from diagnose_realistic92_timing import _lcs_pairs


def _gold_class(row: dict) -> str:
    relationship = str(row.get("relationship"))
    if relationship == "extra" and not row.get("performed_indices"):
        return "renderer_only"
    return relationship


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    names = list(json.loads(args.split.read_text(encoding="utf-8"))["splits"]["train"])
    random.Random(args.seed).shuffle(names)
    names = names[:args.limit]
    support = collections.Counter()
    found = collections.Counter()
    predicted = matched = gold_total = 0
    clip_rows = []
    for position, name in enumerate(names, 1):
        sample = args.root / name
        gold = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
        features = extract_sample_basic_pitch_features(
            sample, cache_path=args.cache_root / f"{name}.npz"
        )
        notes = sanitize_basic_pitch_notes(
            decode_frozen_basic_pitch(features), features, FROZEN_DECODE_CONFIG
        )
        pairs = _lcs_pairs(
            [int(n.pitch) for n in notes],
            [int(g["pitch_midi_written"]) for g in gold],
        )
        paired = {j for _, j in pairs}
        classes = [_gold_class(g) for g in gold]
        for j, label in enumerate(classes):
            support[label] += 1
            found[label] += int(j in paired)
        predicted += len(notes)
        matched += len(pairs)
        gold_total += len(gold)
        without = [j for j, label in enumerate(classes) if label != "renderer_only"]
        clip_rows.append({
            "sample": name,
            "gold": len(gold),
            "renderer_only": len(gold) - len(without),
            "predicted": len(notes),
            "matched": len(pairs),
            "recall_performed_only": sum(j in paired for j in without) / max(len(without), 1),
        })
        if position % 20 == 0:
            print(f"clips={position}/{len(names)}", flush=True)
    precision = matched / max(predicted, 1)
    recall = matched / max(gold_total, 1)
    report = {
        "clips": len(names),
        "overall_lcs": {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "predicted": predicted,
            "gold": gold_total,
        },
        "recall_by_gold_class": {
            label: {"support": support[label], "recall": found[label] / max(support[label], 1)}
            for label in sorted(support)
        },
        "clips_detail": clip_rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("clips", "overall_lcs", "recall_by_gold_class")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
