"""Calibrate Basic Pitch note decoding on one frozen dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from alignmodel.transcription import TransNote, evaluate_note_lists
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    BasicPitchFeatures,
    basic_pitch_cache_path,
    decode_basic_pitch_features,
    sanitize_basic_pitch_notes,
)


def _features(path: Path) -> BasicPitchFeatures:
    with np.load(path, allow_pickle=False) as saved:
        return BasicPitchFeatures(
            np.asarray(saved["note"], np.float32),
            np.asarray(saved["onset"], np.float32),
            np.asarray(saved["contour"], np.float32),
            np.asarray(saved["frame_times"], np.float64),
            json.loads(str(saved["metadata"].item())),
        )


def _gold(row: dict) -> list[dict]:
    sample = Path(row["sample_dir"])
    document = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))
    clean = {
        int(note["clean_index"]): int(note["pitch_midi"])
        for note in document.get("clean_notes") or []
    }
    rendered = document.get("rendered_notes") or []
    offsets = [
        int(note["pitch_midi_written"]) - clean[int(note["primary_clean_index"])]
        for note in rendered
        if note.get("primary_clean_index") is not None
        and int(note["primary_clean_index"]) in clean
        and str(note.get("relationship")) in {"match", "copy"}
    ]
    correction = -int(round(float(np.median(offsets)))) if offsets else 0
    return [
        {
            "pitch": int(note["pitch_midi_written"]) + correction,
            "start": float(note["start_sec"]),
            "end": float(note["end_sec"]),
        }
        for note in rendered
    ]


def _monophonic(notes: list[TransNote]) -> list[TransNote]:
    kept: list[TransNote] = []
    for note in sorted(notes, key=lambda value: (value.start, -value.confidence)):
        conflict = next(
            (
                index
                for index, other in enumerate(kept)
                if min(note.end, other.end) - max(note.start, other.start)
                > 0.5 * min(note.end - note.start, other.end - other.start)
            ),
            None,
        )
        if conflict is None:
            kept.append(note)
        elif note.confidence > kept[conflict].confidence:
            kept[conflict] = note
    return sorted(kept, key=lambda value: value.start)


def _aggregate(rows: list[dict]) -> dict:
    predicted = sum(int(row["n_pred"]) for row in rows)
    target = sum(int(row["n_target"]) for row in rows)
    matched = sum(int(row["n_matched"]) for row in rows)
    precision = matched / max(predicted, 1)
    recall = matched / max(target, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "n_pred": predicted,
        "n_target": target,
        "n_matched": matched,
        "pred_target_ratio": predicted / max(target, 1),
        "n_clips": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text(encoding="utf-8"))

    def prepared(split: str, maximum: int):
        output = []
        for raw in list(document.get(split) or [])[:maximum]:
            row = dict(raw)
            sample = Path(row["sample_dir"])
            cache = basic_pitch_cache_path(
                args.cache_root,
                sample,
                str(row.get("corpus") or row.get("root")),
            )
            output.append((_features(cache), _gold(row)))
        return output

    validation = prepared("val", args.val_samples)
    testing = prepared("test_id", args.test_samples)
    best = None
    for onset in (0.45, 0.50, 0.55):
        for frame in (0.35, 0.40, 0.45):
            for minimum in (55.0, 90.0):
                config = BasicPitchDecodeConfig(
                    onset_threshold=onset,
                    frame_threshold=frame,
                    minimum_note_length_ms=minimum,
                )
                for monophonic in (False, True):
                    rows = []
                    for features, gold in validation:
                        notes = decode_basic_pitch_features(features, config)
                        if monophonic:
                            notes = sanitize_basic_pitch_notes(
                                notes, features, config
                            )
                        rows.append(evaluate_note_lists(notes, gold))
                    metrics = _aggregate(rows)
                    rank = (
                        metrics["f1"]
                        - 0.04
                        * abs(float(np.log(max(metrics["pred_target_ratio"], 1e-3))))
                    )
                    candidate = (rank, metrics["f1"], config, monophonic, metrics)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
    assert best is not None
    _rank, _f1, config, monophonic, validation_metrics = best
    test_rows = []
    for features, gold in testing:
        notes = decode_basic_pitch_features(features, config)
        if monophonic:
            notes = sanitize_basic_pitch_notes(notes, features, config)
        test_rows.append(evaluate_note_lists(notes, gold))
    report = {
        "config": asdict(config),
        "monophonic_pruning": monophonic,
        "validation": validation_metrics,
        "test_id": _aggregate(test_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
