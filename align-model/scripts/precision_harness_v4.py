"""Aligner v3 experiments on cached v3 transcriber outputs (synthetic val only).

Per clip: cached CTC + frame heads -> frozen rich decoder -> aligner v3 ->
gate -> official note-wise scoring against repaired gold. Reports combined
F1 and per-type precision/recall/F1 on the val halves (even = tuning half,
odd = check half). With --features it also writes one row per pre-gate extra
and missed call with its gold credit, for threshold selection.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from alignmodel.joint.metrics import evaluate_joint_dataset
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
from alignmodel.joint.robust_dp_aligner_v3 import (
    AlignerV3Config, GateConfig, align_v3, extra_features, gate_alignment, missed_features,
)
from alignmodel.transcription.ctc_decode_v2 import lcs_pairs, rich_decode
from alignmodel.transcription.mel_v1 import load_audio_mono
from audit_stack_v3_val_errors import _type_samples
from realistic92_aligner_common import load_clip, metric_sample

ALIGN = Path(__file__).resolve().parents[1]
CANDIDATE = ALIGN / "runs/realistic92-stack-v2/CANDIDATE_STACK_V3.json"
CACHES = {
    "realistic92": ALIGN / "runs/realistic92-stack-v2/cache-val-v3c",
    "fast102": ALIGN / "runs/fast102-v1/cache-val-v3c",
}
EXTRA_DATASETS = {
    "dclike11": {"root": "E:\\outputRaw_dclike_11",
                 "aligner_freeze": str(ALIGN / "runs/dclike11-v1/ALIGNER_FREEZE.json")},
}


def dataset_spec(candidate: dict, dataset: str) -> dict:
    return candidate["datasets"].get(dataset) or EXTRA_DATASETS[dataset]
HOP = 256 / 22050
UNPAIRED = 1_000_000
_STATE: dict[str, Any] = {}
VERIFIER: dict[str, str] = {}


def _init(payload: dict[str, Any]) -> None:
    _STATE.update(payload)
    if payload.get("verifier"):
        import torch

        from alignmodel.joint.presence_verifier_v1 import load_verifier

        torch.set_num_threads(1)
        _STATE["verifier_model"] = load_verifier(Path(payload["verifier"]), "cpu")


def presence_scorer(audio: np.ndarray):
    """Callable for evidence["presence"] (dual mel computed on CPU)."""

    from alignmodel.joint.presence_verifier_v1 import score_presence
    from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel

    model = _STATE.get("verifier_model")
    if model is None:
        return None
    mel, _ = extract_dual_mel(audio, "cpu")
    mel = np.asarray(mel, np.float32)
    return lambda queries: score_presence(model, mel, queries, "cpu")


def rms_db(audio: np.ndarray, frames: int, hop: int = 256, window: int = 1024) -> np.ndarray:
    """Frame level in dB, centred like the mel frames."""

    padded = np.pad(np.asarray(audio, np.float64), (window // 2, window // 2 + frames * hop))
    squares = np.concatenate([[0.0], np.cumsum(padded * padded)])
    starts = np.arange(frames) * hop
    energy = (squares[starts + window] - squares[starts]) / window
    return (10.0 * np.log10(energy + 1e-10)).astype(np.float32)


def decode_rows(ctc: np.ndarray, midi_min: int, decoder: dict[str, Any]) -> list[list[float]]:
    decoded = rich_decode(ctc, midi_min, **decoder)
    starts = [note["frame"] * HOP for note in decoded]
    return [
        [note["pitch"], start, max(start + 0.01, starts[k + 1] if k + 1 < len(starts) else start + 0.1),
         note["confidence"], int(note["optional"]), note["alternative_pitch"], note["alternative_confidence"]]
        for k, (note, start) in enumerate(zip(decoded, starts))
    ]


def remap(events, clip):
    gold_pitch = np.asarray([event.pitch for event in clip.rendered], np.int64)
    ordered = sorted(events, key=lambda value: value.rendered_index)
    pred_pitch = np.asarray([event.pitch for event in ordered], np.int64)
    pairs = ({int(i): int(j) for i, j in lcs_pairs(pred_pitch, gold_pitch)}
             if len(pred_pitch) and len(gold_pitch) else {})
    position = {id(event): k for k, event in enumerate(ordered)}
    return [replace(event, rendered_index=pairs.get(position[id(event)], UNPAIRED + position[id(event)]))
            for event in events]


def _run(name: str):
    root = Path(_STATE["root"])
    clip = load_clip(root, name, Path(_STATE["lineage_dir"]))
    cache = np.load(Path(_STATE["cache"]) / f"{name}.npz")
    ctc = cache["ctc"].astype(np.float32)
    evidence = {"ctc": ctc, "voiced": cache["voiced"].astype(np.float32),
                "onset": cache["onset"].astype(np.float32), "hop": HOP, "midi_min": _STATE["midi_min"]}
    audio = load_audio_mono(root / name / "performance_audio.wav", 22050)
    evidence["rms_db"] = rms_db(audio, len(ctc))
    scorer = presence_scorer(audio)
    if scorer is not None:
        evidence["presence"] = scorer
    rows = decode_rows(ctc, _STATE["midi_min"], _STATE["decoder"])
    aligner = AlignerV3Config(costs=RobustDPCosts(**_STATE["costs"]), **_STATE["aligner"])
    gate = GateConfig(**_STATE["gate"])
    score = clip.index.events
    try:
        alignment = align_v3(rows, score, clip.score_path, aligner)
    except Exception as error:  # noqa: BLE001
        return name, metric_sample(clip, (), frozenset()), [], f"{type(error).__name__}: {error}", {}
    events, deletions, info = gate_alignment(alignment, score, evidence, gate)
    remapped = remap(events, clip)
    sample = metric_sample(clip, remapped, deletions)
    features = []
    if _STATE["features"]:
        raw = remap(alignment.events, clip)
        gold = clip.rendered
        for extra in alignment.extras:
            if extra.origin == "repeat_pass":
                continue
            event = raw[extra.event_position]
            j = event.rendered_index
            target = gold[j] if j < UNPAIRED and j < len(gold) else None
            credit = 0.0
            if target is not None and target.score_span is None:
                credit = 1.0 if target.relationship == "extra" else 0.5
            features.append({"kind": "extra", "clip": name, "credit": credit, "origin": extra.origin,
                             "ornament": float(extra.origin == "ornament"),
                             "gold_linked": bool(target is not None and target.score_span is not None),
                             **extra_features(alignment, extra, evidence)})
        gold_deleted = sorted(clip.index.deleted_event_indices)
        for unit in alignment.missed:
            k = unit.score_index
            distance = min((abs(k - d) for d in gold_deleted), default=99)
            same_neighbor = any(0 <= k + s < len(score) and score[k + s].pitch == score[k].pitch for s in (-1, 1))
            features.append({"kind": "missed", "clip": name, "score_index": k,
                             "credit": 1.0 if k in clip.index.deleted_event_indices else 0.0,
                             "gold_deletion_distance": distance, "same_pitch_score_neighbor": same_neighbor,
                             **missed_features(alignment, unit, score, evidence)})
    return name, sample, features, None, info


def _per_type(samples) -> dict[str, Any]:
    output = {}
    for kind in ("match", "copy", "substitute", "extra", "missed_note"):
        value = evaluate_joint_dataset(_type_samples(samples, kind), tolerances_sec=())["aggregate"]["official_note_wise"]
        output[kind] = {key: round(float(value[key]), 4) for key in ("precision", "recall", "f1")}
        output[kind]["predicted"] = int(value["predicted"])
        output[kind]["gold"] = int(value["gold"])
    return output


def run_dataset(dataset: str, half: str, aligner: dict, gate: dict, workers: int, features: bool,
                limit: int | None = None) -> dict[str, Any]:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    spec = dataset_spec(candidate, dataset)
    freeze = json.loads(Path(spec["aligner_freeze"]).read_text(encoding="utf-8"))
    names = [name for name in freeze["eligible"]["val"] if (CACHES[dataset] / f"{name}.npz").is_file()]
    if half == "even":
        names = names[0::2]
    elif half == "odd":
        names = names[1::2]
    if limit:
        names = names[:limit]
    payload = {"root": spec["root"], "lineage_dir": freeze["lineage_dir"], "cache": str(CACHES[dataset]),
               "decoder": candidate["decoder"], "costs": candidate["aligner_costs"], "midi_min": 52,
               "aligner": aligner, "gate": gate, "features": features, "verifier": VERIFIER.get("path")}
    samples, rows, errors = [], [], {}
    totals: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(payload,)) as pool:
        for name, sample, feature_rows, error, info in pool.map(_run, names, chunksize=4):
            samples.append(sample)
            rows.extend(feature_rows)
            if error:
                errors[name] = error
            for key, value in info.items():
                totals[key] = totals.get(key, 0) + int(value)
    combined = evaluate_joint_dataset(samples, tolerances_sec=())["aggregate"]["official_note_wise"]
    return {
        "dataset": dataset, "half": half, "clips": len(names),
        "combined": {key: round(float(combined[key]), 4) for key in ("precision", "recall", "f1")},
        "per_type": _per_type(samples), "gate_totals": totals, "errors": errors, "features": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["realistic92", "fast102"])
    parser.add_argument("--half", choices=("even", "odd", "all"), default="even")
    parser.add_argument("--aligner", default="{}")
    parser.add_argument("--gate", default='{"enabled": false}')
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache", action="append", default=[], help="dataset=dir override for cached outputs")
    parser.add_argument("--verifier", type=Path, help="note-presence verifier checkpoint (adds the presence feature)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for item in args.cache:
        key, value = item.split("=", 1)
        CACHES[key] = Path(value)
    if args.verifier:
        VERIFIER["path"] = str(args.verifier)
    aligner = json.loads(args.aligner)
    gate = json.loads(args.gate)
    report = {"aligner": aligner, "gate": gate, "half": args.half, "datasets": {}}
    for dataset in args.datasets:
        result = run_dataset(dataset, args.half, aligner, gate, args.workers, args.features, args.limit)
        report["datasets"][dataset] = result
        summary = {key: result[key] for key in ("clips", "combined", "gate_totals")}
        summary["extra"] = result["per_type"]["extra"]
        summary["missed_note"] = result["per_type"]["missed_note"]
        summary["errors"] = len(result["errors"])
        print(dataset, json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
