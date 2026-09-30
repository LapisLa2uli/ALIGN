"""Evaluate a mel CTC transcriber on the frozen 9.2 split (val grid or one-shot test).

Headline metric: written-pitch sequence LCS F1 against note_map rendered_notes,
micro-averaged over clips. Val mode grids the blank-probability scale used by
greedy decoding. Test mode requires a frozen candidate, verifies the frozen
test hashes, and refuses to evaluate the same candidate twice.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_ctc_v1 import (
    BLANK,
    greedy_decode,
    infer_ctc_probabilities,
    load_ctc_checkpoint,
)
from alignmodel.transcription.mel_v1 import extract_log_mel, load_audio_mono
from eval_realistic92_transcriber import _prf, lcs_length


BLANK_SCALES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2)


def _decode(probabilities: np.ndarray, midi_min: int, blank_scale: float) -> list[int]:
    if blank_scale != 1.0:
        probabilities = probabilities.copy()
        probabilities[:, BLANK] *= blank_scale
    return [pitch for pitch, _ in greedy_decode(probabilities, midi_min)]


def _score(probabilities, gold, midi_min, blank_scale, keep_per_clip=False) -> dict[str, Any]:
    totals = {"all": [0, 0, 0], "procedural": [0, 0, 0], "rawdata": [0, 0, 0]}
    per_clip = []
    for name in sorted(probabilities):
        predicted = _decode(probabilities[name], midi_min, blank_scale)
        matched = lcs_length(predicted, gold[name])
        group = "procedural" if name.startswith("synth_gen_") else "rawdata"
        for key in ("all", group):
            totals[key][0] += matched
            totals[key][1] += len(predicted)
            totals[key][2] += len(gold[name])
        if keep_per_clip:
            per_clip.append({"sample": name, **_prf(matched, len(predicted), len(gold[name]))})
    report: dict[str, Any] = {key: _prf(*value) for key, value in totals.items()}
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
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.split) != args.expected_split_sha256:
        raise ValueError("Frozen split mismatch")
    split = json.loads(args.split.read_text(encoding="utf-8"))["splits"]

    candidate = None
    candidate_sha = None
    expected_hashes = None
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _payload = load_ctc_checkpoint(checkpoint, device)
    probabilities: dict[str, np.ndarray] = {}
    gold: dict[str, list[int]] = {}
    for position, name in enumerate(names, 1):
        sample = args.root / name
        if expected_hashes is not None and (
            sha256_file(sample / "performance_audio.wav")
            != expected_hashes[name]["performance_audio.wav"]
            or sha256_file(sample / "note_map.json") != expected_hashes[name]["note_map.json"]
        ):
            raise ValueError(f"Test input changed after freeze: {name}")
        audio = load_audio_mono(sample / "performance_audio.wav", frontend.sample_rate)
        mel, _ = extract_log_mel(audio, frontend, device=device)
        probabilities[name] = infer_ctc_probabilities(model, np.asarray(mel, np.float32), device)
        rendered = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))[
            "rendered_notes"
        ]
        gold[name] = [int(row["pitch_midi_written"]) for row in rendered]
        if position == 1 or position % 200 == 0 or position == len(names):
            print(f"infer={position}/{len(names)}", flush=True)

    midi_min = model.config.midi_min
    if args.mode == "val":
        variants = []
        for scale in BLANK_SCALES:
            report = _score(probabilities, gold, midi_min, scale)
            variants.append({"blank_scale": scale, **report})
            print(json.dumps({
                "blank_scale": scale, "f1": report["all"]["f1"],
                "precision": report["all"]["precision"], "recall": report["all"]["recall"],
                "procedural_f1": report["procedural"]["f1"], "rawdata_f1": report["rawdata"]["f1"],
            }), flush=True)
        best = max(variants, key=lambda row: row["all"]["f1"])
        result = {
            "schema_version": "align-realistic92-ctc-val-v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "clips": len(names),
            "variants": variants,
            "best_blank_scale": best["blank_scale"],
            "best": best,
        }
    else:
        scale = float(candidate["blank_scale"])
        report = _score(probabilities, gold, midi_min, scale, keep_per_clip=True)
        result = {
            "schema_version": "align-realistic92-ctc-test-v1",
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_sha256": candidate_sha,
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "blank_scale": scale,
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
        print(json.dumps({
            "test_f1": report["all"]["f1"], "precision": report["all"]["precision"],
            "recall": report["all"]["recall"], "procedural_f1": report["procedural"]["f1"],
            "rawdata_f1": report["rawdata"]["f1"],
        }, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
