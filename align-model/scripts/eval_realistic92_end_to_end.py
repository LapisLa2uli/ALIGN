"""End-to-end 9.2 test: frozen CTC transcriber -> frozen DP aligner.

No weights or costs change. Clips are a seeded uniform sample from all 9.2
clips with repaired gold (ALIGNER_FREEZE eligibility). Scoring is the official
exclusive note-wise metric on canonical verified_score events. Span-less
predicted extras receive the rendered index of the gold note they pair with
along the pitch-sequence LCS; unpaired notes get an index that never matches.
The same clips are also scored for transcription (pitch-sequence LCS F1) and
for the aligner on perfect transcriptions, as reference points.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.packed_data import sha256_file
from alignmodel.joint.perfect_dp_aligner_v1 import PerfectDPCosts, align_perfect
from alignmodel.transcription.mel_ctc_v1 import (
    BLANK,
    greedy_decode,
    infer_ctc_probabilities,
    load_ctc_checkpoint,
)
from alignmodel.transcription.mel_v1 import extract_log_mel, load_audio_mono
from diagnose_realistic92_timing import _lcs_pairs
from eval_realistic92_transcriber import lcs_length
from realistic92_aligner_common import load_clip, metric_sample, perfect_transcription


UNPAIRED_OFFSET = 1_000_000
_STATE: dict[str, Any] = {}


def _init(root: str, lineage_dir: str, costs: dict[str, Any]) -> None:
    _STATE.update(root=Path(root), lineage_dir=Path(lineage_dir), costs=PerfectDPCosts(**costs))


def _align(payload: tuple[str, list[tuple[int, float, float]]]) -> dict[str, Any]:
    name, notes = payload
    clip = load_clip(_STATE["root"], name, _STATE["lineage_dir"])
    gold_pitch = [event.pitch for event in clip.rendered]
    rows: dict[str, Any] = {"name": name}
    try:
        result = align_perfect(notes, clip.index.events, clip.score_path, _STATE["costs"])
        pairs = dict(_lcs_pairs([pitch for pitch, _s, _e in notes], gold_pitch))
        events = [
            replace(event, rendered_index=pairs.get(event.rendered_index,
                                                   UNPAIRED_OFFSET + event.rendered_index))
            for event in result.events
        ]
        rows["e2e"] = metric_sample(clip, events, result.deletions)
    except Exception as error:  # noqa: BLE001
        rows["e2e"] = metric_sample(clip, (), frozenset())
        rows["error"] = f"{type(error).__name__}: {str(error)[:200]}"
    perfect = align_perfect(perfect_transcription(clip), clip.index.events, clip.score_path, _STATE["costs"])
    rows["perfect"] = metric_sample(clip, perfect.events, perfect.deletions)
    predicted_pitch = [pitch for pitch, _s, _e in notes]
    rows["lcs"] = (lcs_length(predicted_pitch, gold_pitch), len(predicted_pitch), len(gold_pitch))
    return rows


def _prf(matched: int, predicted: int, gold: int) -> dict[str, float]:
    precision = matched / max(predicted, 1)
    recall = matched / max(gold, 1)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    e2e = baseline._full_report([row["e2e"] for row in rows], seed=20260926, replicates=1000)
    perfect = baseline._aggregate([row["perfect"] for row in rows])
    lcs = np.sum([row["lcs"] for row in rows], axis=0)
    return {
        "clips": len(rows),
        "end_to_end_note_wise": {
            "f1": e2e["f1"], "precision": e2e["precision"], "recall": e2e["recall"],
            "bootstrap_95": e2e["bootstrap_95"],
            "per_type_f1": {key: value["f1"] for key, value in e2e["per_type"].items()},
        },
        "transcriber_pitch_sequence_lcs": _prf(int(lcs[0]), int(lcs[1]), int(lcs[2])),
        "aligner_perfect_transcription_note_wise_f1": perfect["f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aligner-freeze", type=Path, required=True)
    parser.add_argument("--expected-aligner-freeze-sha256", required=True)
    parser.add_argument("--transcriber-candidate", type=Path, required=True)
    parser.add_argument("--aligner-candidate", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.aligner_freeze) != args.expected_aligner_freeze_sha256:
        raise ValueError("Aligner freeze mismatch")
    freeze = json.loads(args.aligner_freeze.read_text(encoding="utf-8"))
    transcriber = json.loads(args.transcriber_candidate.read_text(encoding="utf-8"))
    aligner = json.loads(args.aligner_candidate.read_text(encoding="utf-8"))
    if sha256_file(Path(transcriber["checkpoint"])) != transcriber["checkpoint_sha256"]:
        raise ValueError("Transcriber checkpoint changed")
    aligner_path = Path(__file__).resolve().parents[1] / aligner["aligner_path"]
    if sha256_file(aligner_path) != aligner["aligner_sha256"]:
        raise ValueError("Aligner code changed")

    split_of = {name: split for split in ("train", "val", "test") for name in freeze["eligible"][split]}
    population = sorted(split_of)
    names = sorted(random.Random(args.seed).sample(population, args.count))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _payload = load_ctc_checkpoint(Path(transcriber["checkpoint"]), device)
    blank_scale = float(transcriber["blank_scale"])
    transcriptions: list[tuple[str, list[tuple[int, float, float]]]] = []
    for position, name in enumerate(names, 1):
        audio = load_audio_mono(args.root / name / "performance_audio.wav", frontend.sample_rate)
        mel, _ = extract_log_mel(audio, frontend, device=device)
        probabilities = infer_ctc_probabilities(model, np.asarray(mel, np.float32), device)
        probabilities[:, BLANK] *= blank_scale
        decoded = greedy_decode(probabilities, model.config.midi_min)
        starts = [frame * frontend.hop_sec for _pitch, frame in decoded]
        notes = [
            (pitch, start, max(start + 0.01, starts[k + 1] if k + 1 < len(starts) else start + 0.1))
            for k, ((pitch, _frame), start) in enumerate(zip(decoded, starts))
        ]
        transcriptions.append((name, notes))
        if position % 200 == 0 or position == len(names):
            print(f"transcribed={position}/{len(names)}", flush=True)

    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init,
        initargs=(str(args.root), freeze["lineage_dir"], aligner["costs"]),
    ) as pool:
        for position, row in enumerate(pool.map(_align, transcriptions, chunksize=4), 1):
            row["split"] = split_of[row["name"]]
            rows.append(row)
            if position % 200 == 0 or position == len(names):
                print(f"aligned={position}/{len(names)}", flush=True)

    by_split = {split: _summary([row for row in rows if row["split"] == split])
                for split in ("train", "val", "test") if any(row["split"] == split for row in rows)}
    by_group = {
        group: _summary([row for row in rows if (row["name"].startswith("synth_gen_")) == (group == "procedural")])
        for group in ("procedural", "rawdata")
    }
    result = {
        "schema_version": "align-realistic92-end-to-end-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "population": "all 9.2 clips with repaired gold (ALIGNER_FREEZE eligibility)",
        "population_size": len(population),
        "sample": {"count": args.count, "seed": args.seed, "method": "uniform without replacement"},
        "transcriber_candidate_sha256": sha256_file(args.transcriber_candidate),
        "aligner_candidate_sha256": sha256_file(args.aligner_candidate),
        "blank_scale": blank_scale,
        "extra_identity_policy": "gold rendered index via pitch-sequence LCS pairing; unpaired never matches",
        "errors": dict(collections.Counter(row["error"] for row in rows if "error" in row)),
        "overall": _summary(rows),
        "by_split": by_split,
        "by_group": by_group,
        "clips": [row["name"] for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("errors", "overall", "by_split", "by_group")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
