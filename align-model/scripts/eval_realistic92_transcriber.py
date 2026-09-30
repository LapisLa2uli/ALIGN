"""Evaluate a mel transcriber on the frozen 9.2 split (val grid or one-shot test).

Headline metric: written-pitch sequence LCS F1 against note_map rendered_notes,
micro-averaged over clips. Val mode grids decoder settings for selection. Test
mode requires a frozen candidate file, verifies the frozen test hashes, and
refuses to evaluate the same candidate twice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_v1 import (
    MelDecodeConfig,
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


GRID: tuple[dict[str, float], ...] = (
    {},
    {"min_confidence": 0.5},
    {"min_confidence": 0.7},
    {"voice_on": 0.46, "voice_off": 0.30, "onset_threshold": 0.42},
    {"voice_on": 0.58, "voice_off": 0.40, "onset_threshold": 0.54},
    {"boundary_threshold": 0.52, "strong_rearticulation": 0.70, "merge_gap_sec": 0.055},
    {"boundary_threshold": 0.40, "strong_rearticulation": 0.58, "min_note_sec": 0.035},
    {"boundary_threshold": 0.40, "strong_rearticulation": 0.58, "min_note_sec": 0.030,
     "onset_threshold": 0.42},
    {"min_note_sec": 0.030},
    {"min_note_sec": 0.050},
    {"pitch_change_frames": 2},
    {"pitch_change_frames": 3, "min_note_sec": 0.030},
)


def lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
    if not left or not right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    masks: dict[int, int] = {}
    for position, value in enumerate(left):
        masks[value] = masks.get(value, 0) | (1 << position)
    full = (1 << len(left)) - 1
    vector = full
    for value in right:
        match = vector & masks.get(value, 0)
        vector = ((vector + match) | (vector - match)) & full
    return len(left) - bin(vector).count("1")


def _prf(matched: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = matched / max(predicted, 1)
    recall = matched / max(gold, 1)
    return {
        "matched": matched,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "count_ratio": predicted / max(gold, 1),
    }


def _score(
    probabilities: dict[str, dict[str, np.ndarray]],
    gold: dict[str, list[int]],
    decode: MelDecodeConfig,
    midi_min: int,
    hop_sec: float,
    *,
    keep_per_clip: bool = False,
) -> dict[str, Any]:
    totals = {"all": [0, 0, 0], "procedural": [0, 0, 0], "rawdata": [0, 0, 0]}
    per_clip = []
    for name in sorted(probabilities):
        notes = decode_mel_notes(
            probabilities[name], midi_min=midi_min, hop_sec=hop_sec, config=decode
        )
        predicted = [int(note.pitch) for note in notes]
        matched = lcs_length(predicted, gold[name])
        group = "procedural" if name.startswith("synth_gen_") else "rawdata"
        for key in ("all", group):
            totals[key][0] += matched
            totals[key][1] += len(predicted)
            totals[key][2] += len(gold[name])
        if keep_per_clip:
            per_clip.append({"sample": name, **_prf(matched, len(predicted), len(gold[name]))})
    report = {key: _prf(*value) for key, value in totals.items()}
    if keep_per_clip:
        report["per_clip"] = per_clip
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--mode", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--test-hashes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.split) != args.expected_split_sha256:
        raise ValueError("Frozen split mismatch")
    split = json.loads(args.split.read_text(encoding="utf-8"))["splits"]

    candidate = None
    if args.mode == "test":
        if not (args.candidate and args.freeze and args.test_hashes):
            raise ValueError("Test mode needs --candidate, --freeze, --test-hashes")
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        if freeze["split_sha256"] != args.expected_split_sha256:
            raise ValueError("Freeze/split mismatch")
        if sha256_file(args.test_hashes) != freeze["test_hashes_sha256"]:
            raise ValueError("Test hash manifest changed")
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        candidate_sha = sha256_file(args.candidate)
        log_path = args.freeze.parent / "TEST_EVALUATIONS.jsonl"
        if log_path.exists() and any(
            json.loads(line)["candidate_sha256"] == candidate_sha
            for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ):
            raise ValueError("This candidate was already evaluated on test")
        checkpoint = Path(candidate["checkpoint"])
        if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError("Candidate checkpoint changed")
        names = list(split["test"])
        expected_hashes = json.loads(args.test_hashes.read_text(encoding="utf-8"))
    else:
        if args.checkpoint is None:
            raise ValueError("Val mode needs --checkpoint")
        checkpoint = args.checkpoint
        names = list(split["val"])
        expected_hashes = None
    if args.limit:
        names = names[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, base_decode, _payload = load_mel_checkpoint(checkpoint, device)
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    gold: dict[str, list[int]] = {}
    for position, name in enumerate(names, 1):
        sample = args.root / name
        if expected_hashes is not None:
            if (
                sha256_file(sample / "performance_audio.wav")
                != expected_hashes[name]["performance_audio.wav"]
                or sha256_file(sample / "note_map.json")
                != expected_hashes[name]["note_map.json"]
            ):
                raise ValueError(f"Test input changed after freeze: {name}")
        audio = load_audio_mono(sample / "performance_audio.wav", frontend.sample_rate)
        mel, _ = extract_log_mel(audio, frontend, device=device)
        output = infer_mel_probabilities(
            model, np.asarray(mel, np.float32), device,
            window_frames=2048, overlap_frames=512, batch_size=4,
        )
        probabilities[name] = {
            key: output[key] for key in
            ("voiced", "pitch", "onset", "boundary", "rearticulation", "confidence")
        }
        rendered = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))[
            "rendered_notes"
        ]
        gold[name] = [int(row["pitch_midi_written"]) for row in rendered]
        if position == 1 or position % 200 == 0 or position == len(names):
            print(f"infer={position}/{len(names)}", flush=True)

    midi_min = model.config.midi_min
    if args.mode == "val":
        variants = []
        for index, override in enumerate(GRID):
            decode = replace(base_decode, **override)
            report = _score(probabilities, gold, decode, midi_min, frontend.hop_sec)
            variants.append({"variant": index, "override": override,
                             "decode_config": decode.to_dict(), **report})
            print(json.dumps({"variant": index, "override": override,
                              "f1": report["all"]["f1"],
                              "precision": report["all"]["precision"],
                              "recall": report["all"]["recall"],
                              "procedural_f1": report["procedural"]["f1"],
                              "rawdata_f1": report["rawdata"]["f1"]}), flush=True)
        best = max(variants, key=lambda row: row["all"]["f1"])
        result = {
            "schema_version": "align-realistic92-transcriber-val-v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "clips": len(names),
            "variants": variants,
            "best_variant": best["variant"],
            "best": best,
        }
        print(json.dumps({"best_variant": best["variant"], "override": best["override"],
                          "f1": best["all"]["f1"]}, indent=2), flush=True)
    else:
        decode = MelDecodeConfig.from_dict(candidate["decode_config"])
        report = _score(probabilities, gold, decode, midi_min, frontend.hop_sec,
                        keep_per_clip=True)
        result = {
            "schema_version": "align-realistic92-transcriber-test-v1",
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_sha256": candidate_sha,
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "clips": len(names),
            **report,
        }
        with (args.freeze.parent / "TEST_EVALUATIONS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "candidate_sha256": candidate_sha,
                "evaluated_utc": result["evaluated_utc"],
                "f1": report["all"]["f1"],
                "precision": report["all"]["precision"],
                "recall": report["all"]["recall"],
                "output": str(args.output),
            }) + "\n")
        print(json.dumps({"test_f1": report["all"]["f1"],
                          "precision": report["all"]["precision"],
                          "recall": report["all"]["recall"],
                          "procedural_f1": report["procedural"]["f1"],
                          "rawdata_f1": report["rawdata"]["f1"]}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
