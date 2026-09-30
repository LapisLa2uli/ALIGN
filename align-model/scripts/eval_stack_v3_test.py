"""One-shot test of a frozen v3 transcriber + robust aligner stack.

Verifies the candidate (checkpoint, decoder, aligner costs, code hashes),
refuses to evaluate a candidate twice on the same dataset, transcribes the
frozen test population, reports the transcriber breakdown and the combined
official note-wise F1, and appends the result to TEST_EVALUATIONS.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import eval_orn_phase2_baseline_v1 as baseline
import e2e_eval_realistic92 as e2e
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.ctc_decode_v2 import rich_decode
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
from alignmodel.transcription.mel_v1 import load_audio_mono
from realistic92_transcriber_breakdown import Breakdown, load_gold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="key in candidate['datasets']")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    candidate_sha = sha256_file(args.candidate)
    align = Path(__file__).resolve().parents[1]
    for relative, digest in candidate["code_sha256"].items():
        if sha256_file(align / relative) != digest:
            raise ValueError(f"Code changed after freeze: {relative}")
    checkpoint = Path(candidate["checkpoint"])
    if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after freeze")
    if args.log.exists() and any(
        json.loads(line)["candidate_sha256"] == candidate_sha and json.loads(line)["dataset"] == args.dataset
        for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()
    ):
        raise ValueError("Candidate already evaluated on this test population")
    dataset = candidate["datasets"][args.dataset]
    root = Path(dataset["root"])
    freeze_path = Path(dataset["aligner_freeze"])
    if sha256_file(freeze_path) != dataset["aligner_freeze_sha256"]:
        raise ValueError("Aligner freeze changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    lineage_dir = Path(freeze["lineage_dir"])
    names = list(freeze["eligible"]["test"])
    for name in names:
        if sha256_file(lineage_dir / f"{name}.json") != freeze["test_lineage_sha256"][name]:
            raise ValueError(f"Test lineage changed: {name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(checkpoint, device)
    decoder = candidate["decoder"]
    hop = 256 / 22050
    breakdown = Breakdown()
    notes: dict[str, list[list[float]]] = {}
    for position, name in enumerate(names, 1):
        audio = load_audio_mono(root / name / "performance_audio.wav", 22050)
        mel, _ = extract_dual_mel(audio, device)
        outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
        decoded = rich_decode(outputs["ctc"], model.config.midi_min, **decoder)
        breakdown.add([note["pitch"] for note in decoded if not note["optional"]], load_gold(root, name))
        starts = [note["frame"] * hop for note in decoded]
        notes[name] = [
            [note["pitch"], round(start, 6),
             round(max(start + 0.01, starts[k + 1] if k + 1 < len(starts) else start + 0.1), 6),
             round(note["confidence"], 4), int(note["optional"]),
             note["alternative_pitch"], round(note["alternative_confidence"], 4)]
            for k, (note, start) in enumerate(zip(decoded, starts))
        ]
        if position % 200 == 0 or position == len(names):
            print(f"transcribed={position}/{len(names)}", flush=True)

    samples, errors = [], collections.Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=e2e._init,
                             initargs=(str(root), str(lineage_dir), candidate["aligner_costs"], "dp_v2")) as pool:
        for _name, sample, error in pool.map(e2e._run, [(name, notes[name]) for name in names], chunksize=4):
            samples.append(sample)
            if error:
                errors[error] += 1
    report = baseline._full_report(samples, seed=20260927, replicates=1000)
    transcription = breakdown.report()
    result = {
        "schema_version": "align-stack-v3-test-v1",
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": candidate_sha,
        "dataset": args.dataset,
        "clips": len(names),
        "combined_note_wise": {
            "f1": report["f1"], "precision": report["precision"], "recall": report["recall"],
            "bootstrap_95": report["bootstrap_95"],
            "per_type_f1": {key: value["f1"] for key, value in report["per_type"].items()},
        },
        "transcriber_pitch_sequence": transcription,
        "errors": dict(errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "candidate_sha256": candidate_sha, "dataset": args.dataset,
            "evaluated_utc": result["evaluated_utc"], "combined_f1": report["f1"],
            "transcriber_f1": transcription["f1"], "output": str(args.output),
        }) + "\n")
    print(json.dumps({
        "combined_f1": report["f1"], "precision": report["precision"], "recall": report["recall"],
        "ci": report["bootstrap_95"],
        "per_type_f1": result["combined_note_wise"]["per_type_f1"],
        "transcriber_f1": transcription["f1"],
        "lt50_recall": transcription["recall_by"].get("dur_lt50"),
        "50to80_recall": transcription["recall_by"].get("dur_50to80"),
        "repeat_recall": transcription["recall_by"].get("same_pitch_neighbor"),
        "false_positive_rate": transcription["false_positive_rate"],
        "errors": dict(errors),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
