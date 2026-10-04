"""One-shot test of a frozen stack v4 candidate (v3 transcriber + aligner v3 with gates).

Verifies the candidate file (checkpoint, decoder, aligner/gate config, code
hashes), refuses to evaluate a candidate twice on one test population,
transcribes the frozen test clips, aligns and gates them, and reports combined
and per-type official note-wise metrics with clip-bootstrap intervals for the
extra and missed-note precision. Appends to TEST_EVALUATIONS.jsonl.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignmodel.joint.metrics import evaluate_joint_dataset
from alignmodel.joint.packed_data import sha256_file
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
from alignmodel.joint.robust_dp_aligner_v3 import AlignerV3Config, GateConfig, align_v3, gate_alignment
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
from alignmodel.transcription.mel_v1 import load_audio_mono
from audit_stack_v3_val_errors import _type_samples
from precision_harness_v4 import HOP, decode_rows, remap, rms_db
from realistic92_aligner_common import load_clip, metric_sample

_STATE: dict[str, Any] = {}


def _init(payload: dict[str, Any]) -> None:
    _STATE.update(payload)
    if payload.get("verifier"):
        torch.set_num_threads(1)
        from alignmodel.joint.presence_verifier_v1 import load_verifier

        _STATE["verifier_model"] = load_verifier(Path(payload["verifier"]), "cpu")


def _align(job):
    name, cache_path = job
    clip = load_clip(Path(_STATE["root"]), name, Path(_STATE["lineage_dir"]))
    cache = np.load(cache_path)
    ctc = cache["ctc"].astype(np.float32)
    evidence = {"ctc": ctc, "voiced": cache["voiced"].astype(np.float32), "onset": cache["onset"].astype(np.float32),
                "rms_db": cache["rms_db"], "hop": HOP, "midi_min": _STATE["midi_min"]}
    if _STATE.get("verifier_model") is not None:
        from alignmodel.joint.presence_verifier_v1 import score_presence

        audio = load_audio_mono(Path(_STATE["root"]) / name / "performance_audio.wav", 22050)
        mel, _ = extract_dual_mel(audio, "cpu")
        mel = np.asarray(mel, np.float32)
        model = _STATE["verifier_model"]
        evidence["presence"] = lambda queries: score_presence(model, mel, queries, "cpu")
    rows = decode_rows(ctc, _STATE["midi_min"], _STATE["decoder"])
    try:
        alignment = align_v3(rows, clip.index.events, clip.score_path,
                             AlignerV3Config(costs=RobustDPCosts(**_STATE["costs"]), **_STATE["aligner"]))
        events, deletions, _info = gate_alignment(alignment, clip.index.events, evidence, GateConfig(**_STATE["gate"]))
        return name, metric_sample(clip, remap(events, clip), deletions), None
    except Exception as error:  # noqa: BLE001
        return name, metric_sample(clip, (), frozenset()), f"{type(error).__name__}: {str(error)[:160]}"


def _official(samples) -> dict[str, float]:
    value = evaluate_joint_dataset(samples, tolerances_sec=())["aggregate"]["official_note_wise"]
    return {key: float(value[key]) for key in ("precision", "recall", "f1", "credit", "predicted", "gold")}


def _bootstrap_precision(samples, kind: str, seed: int, replicates: int) -> dict[str, float]:
    per_clip = []
    for sample in _type_samples(samples, kind):
        value = evaluate_joint_dataset([sample], tolerances_sec=())["aggregate"]["official_note_wise"]
        per_clip.append((float(value["credit"]), int(value["predicted"]), int(value["gold"])))
    data = np.array(per_clip, np.float64)
    generator = np.random.default_rng(seed)
    precisions, recalls = [], []
    for _ in range(replicates):
        pick = data[generator.integers(0, len(data), len(data))]
        precisions.append(pick[:, 0].sum() / max(pick[:, 1].sum(), 1))
        recalls.append(pick[:, 0].sum() / max(pick[:, 2].sum(), 1))
    return {"precision_lower_95": float(np.quantile(precisions, 0.025)),
            "precision_upper_95": float(np.quantile(precisions, 0.975)),
            "recall_lower_95": float(np.quantile(recalls, 0.025)),
            "recall_upper_95": float(np.quantile(recalls, 0.975))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="scratch dir for test transcriber outputs")
    parser.add_argument("--workers", type=int, default=6)
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
    verifier = candidate.get("verifier")
    if verifier and sha256_file(Path(verifier)) != candidate["verifier_sha256"]:
        raise ValueError("Verifier changed after freeze")
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
    if "test_lineage_sha256" not in freeze or len(freeze["test_lineage_sha256"]) != len(names):
        raise ValueError("Freeze does not hash every test clip")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(checkpoint, device)
    model.eval()
    cache_dir = args.cache / args.dataset
    if candidate["checkpoint_sha256"] != "60663218eba3a43b35691ec95c116f50eafbc2fd749428af35e566891c50cad9":
        cache_dir = args.cache / f"{args.dataset}-{candidate['checkpoint_sha256'][:12]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for position, name in enumerate(names, 1):
        path = cache_dir / f"{name}.npz"
        if not path.is_file():
            audio = load_audio_mono(root / name / "performance_audio.wav", 22050)
            with torch.inference_mode():
                mel, _ = extract_dual_mel(audio, device)
                outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
            np.savez_compressed(path, ctc=outputs["ctc"].astype(np.float16), voiced=outputs["voiced"].astype(np.float16),
                                onset=outputs["onset"].astype(np.float16), rms_db=rms_db(audio, len(outputs["ctc"])))
        jobs.append((name, str(path)))
        if position % 200 == 0 or position == len(names):
            print(f"transcribed={position}/{len(names)}", flush=True)

    payload = {"root": str(root), "lineage_dir": str(lineage_dir), "decoder": candidate["decoder"],
               "costs": candidate["aligner_costs"], "aligner": candidate["aligner_v3"], "gate": candidate["gate"],
               "midi_min": int(model.config.midi_min), "verifier": verifier}
    samples, errors = [], {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(payload,)) as pool:
        for name, sample, error in pool.map(_align, jobs, chunksize=4):
            samples.append(sample)
            if error:
                errors[name] = error
    per_type = {kind: _official(_type_samples(samples, kind))
                for kind in ("match", "copy", "substitute", "extra", "missed_note")}
    result = {
        "schema_version": "align-stack-v4-test-v1",
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": candidate_sha,
        "dataset": args.dataset,
        "clips": len(names),
        "combined_note_wise": _official(samples),
        "per_type": per_type,
        "bootstrap_95": {kind: _bootstrap_precision(samples, kind, 20261002, 1000) for kind in ("extra", "missed_note")},
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"candidate_sha256": candidate_sha, "dataset": args.dataset,
                                 "evaluated_utc": result["evaluated_utc"],
                                 "combined_f1": result["combined_note_wise"]["f1"],
                                 "extra_precision": per_type["extra"]["precision"],
                                 "missed_precision": per_type["missed_note"]["precision"],
                                 "output": str(args.output)}) + "\n")
    print(json.dumps({key: result[key] for key in ("combined_note_wise", "per_type", "bootstrap_95")}, indent=1))
    print("errors", len(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
