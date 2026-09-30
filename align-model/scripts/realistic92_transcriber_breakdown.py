"""Transcriber error breakdown on 9.2 clips (pitch-sequence LCS pairing).

Gold durations come from note_map (render timeline); they categorize notes and
are not compared with predicted timing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alignmodel.transcription.ctc_decode_v2 import lcs_pairs


DURATION_BINS = ((0.0, 0.05, "lt50"), (0.05, 0.08, "50to80"), (0.08, 0.12, "80to120"),
                 (0.12, 0.2, "120to200"), (0.2, 1e9, "ge200"))
REGISTER_BINS = ((0, 67, "low_lt_G4"), (67, 71, "throat_G4_Bb4"), (71, 85, "clarion_B4_C6"),
                 (85, 200, "high_gt_C6"))


def load_gold(root: Path, name: str) -> list[dict[str, Any]]:
    rendered = json.loads((root / name / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
    return [
        {
            "pitch": int(row["pitch_midi_written"]),
            "duration": float(row["end_sec"]) - float(row["start_sec"]),
            "renderer_only": row.get("relationship") == "extra" and not row.get("performed_indices"),
        }
        for row in rendered
    ]


class Breakdown:
    def __init__(self) -> None:
        self.matched = self.predicted = self.gold = 0
        self.support: dict[str, int] = {}
        self.found: dict[str, int] = {}
        self.false_positive: dict[str, int] = {}

    def _count(self, key: str, hit: bool) -> None:
        self.support[key] = self.support.get(key, 0) + 1
        self.found[key] = self.found.get(key, 0) + int(hit)

    def add(self, predicted: Sequence[int], gold: Sequence[dict[str, Any]]) -> None:
        pred = np.asarray(predicted, np.int64)
        gold_pitch = np.asarray([row["pitch"] for row in gold], np.int64)
        pairs = lcs_pairs(pred, gold_pitch) if len(pred) and len(gold_pitch) else np.zeros((0, 2), np.int64)
        matched_pred = set(pairs[:, 0].tolist())
        matched_gold = set(pairs[:, 1].tolist())
        self.matched += len(pairs)
        self.predicted += len(pred)
        self.gold += len(gold_pitch)
        for j, row in enumerate(gold):
            hit = j in matched_gold
            for lo, hi, label in DURATION_BINS:
                if lo <= row["duration"] < hi:
                    self._count(f"dur_{label}", hit)
            for lo, hi, label in REGISTER_BINS:
                if lo <= row["pitch"] < hi:
                    self._count(f"reg_{label}", hit)
            repeat = (j > 0 and gold_pitch[j - 1] == gold_pitch[j]) or (
                j + 1 < len(gold_pitch) and gold_pitch[j + 1] == gold_pitch[j]
            )
            self._count("same_pitch_neighbor" if repeat else "pitch_change", hit)
            if row["renderer_only"]:
                self._count("ornament_renderer_only", hit)
        for i in range(len(pred)):
            if i in matched_pred:
                continue
            previous = pred[i - 1] if i > 0 else None
            following = pred[i + 1] if i + 1 < len(pred) else None
            if previous == pred[i] or following == pred[i]:
                kind = "split_same_pitch"
            elif previous is not None and previous == following:
                kind = "flicker_ABA"
            else:
                kind = "other"
            self.false_positive[kind] = self.false_positive.get(kind, 0) + 1

    def report(self) -> dict[str, Any]:
        precision = self.matched / max(self.predicted, 1)
        recall = self.matched / max(self.gold, 1)
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "predicted": self.predicted,
            "gold": self.gold,
            "recall_by": {key: round(self.found[key] / max(self.support[key], 1), 4) for key in sorted(self.support)},
            "support_by": dict(sorted(self.support.items())),
            "false_positives": dict(sorted(self.false_positive.items())),
            "false_positive_rate": (self.predicted - self.matched) / max(self.predicted, 1),
        }
