"""Read-only model/data audit; fresh outputs stay beside this script."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ALIGN = ROOT / "align-model"
for folder in (ALIGN / "src", ALIGN / "scripts", ROOT / "DataCreate/src", ROOT / "synth-pipeline/src"):
    sys.path.insert(0, str(folder))
os.environ.setdefault("NUMBA_CACHE_DIR", str(OUT / "numba-cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import evaluate_joint_dataset
from alignmodel.joint.presence_verifier_v1 import load_verifier, score_presence
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
from alignmodel.joint.robust_dp_aligner_v3 import AlignerV3Config, GateConfig, align_v3, gate_alignment
from alignmodel.melody import labels_with_canonical_locations, load_bundle_notes, micro_note_wise
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
from alignmodel.transcription.mel_v1 import load_audio_mono
from audit_stack_v3_val_errors import _type_samples
from datacreate.melody import match_note_wise_labels_detail
from eval_datacreate_vs_human import ERROR_TYPES, human_population, labels_from_stack_alignment, lenient
from precision_harness_v4 import HOP, decode_rows, remap, rms_db
from realistic92_aligner_common import load_clip, metric_sample

STATE = {}

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def init(candidate):
    torch.set_num_threads(1)
    STATE.update(candidate=candidate, verifier=load_verifier(Path(candidate["verifier"]), "cpu"))

def work(job):
    dataset, name, sample_path, lineage_dir = job
    sample = Path(sample_path)
    c = STATE["candidate"]
    try:
        with np.load(OUT / "cache" / dataset / f"{name}.npz") as cache:
            evidence = {k: cache[k].astype(np.float32) for k in ("ctc", "voiced", "onset", "rms_db")}
            mel = cache["mel"].astype(np.float32)
        evidence.update(hop=HOP, midi_min=52)
        evidence["presence"] = lambda queries: score_presence(STATE["verifier"], mel, queries, "cpu")
        score_path = sample / "verified_score.musicxml"
        # Inference score index never receives synthetic lineage.
        index = ScoreEventIndex.from_musicxml(score_path)
        rows = decode_rows(evidence["ctc"], 52, c["decoder"])
        alignment = align_v3(rows, index.events, score_path, AlignerV3Config(costs=RobustDPCosts(**c["aligner_costs"]), **c["aligner_v3"]))
        events, deletions, info = gate_alignment(alignment, index.events, evidence, GateConfig(**c["gate"]))
        payload = {
            "sample": name, "score_event_count": len(index.events), "match_fraction": alignment.match_fraction,
            "gate_info": info, "repeat_hypothesis": {"source_span": alignment.source_span, "extra_copies": alignment.copies},
            "missed_score_event_indices": sorted(deletions), "raw_missed_indices": sorted(alignment.deletions),
            "events": [{"note_index": e.rendered_index, "pitch": e.pitch, "start": e.start, "end": e.end,
                        "relationship": e.relationship, "score_span": e.score_span, "copy_pass": e.copy_pass,
                        "confidence": e.confidence} for e in events],
            "transcription_count": len(rows),
        }
        write(OUT / "alignments" / dataset / f"{name}.json", payload)
        post = pre = None
        detail = {}
        if lineage_dir:
            clip = load_clip(sample.parent, name, Path(lineage_dir))
            if [(e.pitch, e.source_indices) for e in clip.index.events] != [(e.pitch, e.source_indices) for e in index.events]:
                raise ValueError("Inference and gold canonical event indices disagree")
            post = metric_sample(clip, remap(events, clip), deletions)
            pre = metric_sample(clip, remap(alignment.events, clip), alignment.deletions)
            gold_missed = set(clip.index.deleted_event_indices)
            detail = {"gold_missed": sorted(gold_missed), "missed_detected": sorted(gold_missed & set(deletions)),
                      "missed_removed_by_gate": sorted(gold_missed & (set(alignment.deletions) - set(deletions))),
                      "missed_not_proposed": sorted(gold_missed - set(alignment.deletions)),
                      "missed_false_positive": sorted(set(deletions) - gold_missed)}
        return dataset, name, post, pre, info, detail, None
    except Exception as error:
        import traceback
        return dataset, name, None, None, {}, {}, traceback.format_exc()

def metric(samples):
    def score(rows):
        return evaluate_joint_dataset(rows, tolerances_sec=())["aggregate"]["official_note_wise"]
    return {"combined": score(samples), "per_type": {k: score(_type_samples(samples, k)) for k in ("match", "copy", "substitute", "extra", "missed_note")}}

def score_dc():
    population = [(s, g) for s, g in human_population(ROOT / "DataCreate/samples") if s.name.isdigit() and 1 <= int(s.name) <= 94]
    metrics = collections.defaultdict(list)
    totals = collections.defaultdict(collections.Counter)
    details = []
    unsupported = collections.Counter()
    for sample, manual in population:
        alignment = read(OUT / "alignments/datacreate" / f"{sample.name}.json")
        notes = load_bundle_notes(sample)
        gold = labels_with_canonical_locations(manual, notes)
        pred = labels_with_canonical_locations(labels_from_stack_alignment(alignment, notes), notes)
        unsupported.update(l["type"] for l in gold if l["type"] not in ERROR_TYPES)
        row = {"sample": sample.name, "gold": gold, "predicted": pred, "metrics": {}, "lenient": {}}
        for kind in ERROR_TYPES:
            g = [l for l in gold if l["type"] == kind]
            p = [l for l in pred if l["type"] == kind]
            result = match_note_wise_labels_detail(g, p, score_event_count=len(notes))
            metrics[kind].append(result)
            row["metrics"][kind] = result
            counts = lenient(gold, pred, kind)
            row["lenient"][kind] = counts
            totals[kind].update(counts)
        details.append(row)
    summary = {k: {"official": micro_note_wise(metrics[k]), "lenient": dict(totals[k])} for k in ERROR_TYPES}
    result = {"range": "001-094", "population": len(population), "sample_ids": [s.name for s, _ in population],
              "unsupported_gold_types": dict(unsupported), "summary": summary, "per_sample": details}
    write(OUT / "datacreate.json", result)
    print("DATACREATE", json.dumps({"population": len(population), "summary": summary}), flush=True)

def main():
    torch.set_num_threads(4)
    cpath = ALIGN / "runs/precision-v4/CANDIDATE_STACK_V6.json"
    c = read(cpath)
    hashes = {"checkpoint": sha(c["checkpoint"]) == c["checkpoint_sha256"], "verifier": sha(c["verifier"]) == c["verifier_sha256"]}
    hashes.update({p: sha(ALIGN / p) == h or hashlib.sha256((ALIGN / p).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest() == h for p, h in c["code_sha256"].items()})
    if not all(hashes.values()):
        raise ValueError(f"Frozen model/code drift: {hashes}")
    jobs = []
    for dataset, spec in c["datasets"].items():
        freeze = read(spec["aligner_freeze"])
        names = sorted(random.Random(20261003).sample(list(freeze["eligible"]["val"]), 30))
        jobs.extend((dataset, name, str(Path(spec["root"]) / name), freeze["lineage_dir"]) for name in names)
    dc = ROOT / "DataCreate/samples"
    jobs.extend(("datacreate", p.name, str(p), None) for p in sorted(dc.iterdir()) if p.name.isdigit() and 1 <= int(p.name) <= 94 and (p / "performance_audio.wav").is_file())
    manifest = {"candidate": str(cpath), "candidate_sha256": sha(cpath), "hash_checks": hashes,
                "synthetic_split": "validation; diagnostic only, already used in model development", "seed": 20261003,
                "jobs": jobs, "source_hashes": {}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_dual_checkpoint(c["checkpoint"], device)
    model.eval()
    for i, (dataset, name, path, _) in enumerate(jobs, 1):
        sample = Path(path)
        wav = sample / "performance_audio.wav"
        manifest["source_hashes"][f"{dataset}/{name}"] = {"audio": sha(wav), "score": sha(sample / "verified_score.musicxml")}
        cache = OUT / "cache" / dataset / f"{name}.npz"
        # Only this audit's own freshly generated cache can be resumed.
        if not cache.exists():
            audio = load_audio_mono(wav, 22050)
            with torch.inference_mode():
                mel, _ = extract_dual_mel(audio, device)
                outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, mel=np.asarray(mel, np.float32), ctc=outputs["ctc"].astype(np.float16),
                                voiced=outputs["voiced"].astype(np.float16), onset=outputs["onset"].astype(np.float16),
                                rms_db=rms_db(audio, len(outputs["ctc"])))
        print(f"transcribed {i}/{len(jobs)} {dataset}/{name}", flush=True)
    write(OUT / "manifest.json", manifest)
    del model
    torch.cuda.empty_cache()
    post, pre = collections.defaultdict(list), collections.defaultdict(list)
    gate_totals = collections.defaultdict(collections.Counter)
    details, errors = [], {}
    with ProcessPoolExecutor(max_workers=4, initializer=init, initargs=(c,)) as pool:
        for i, (dataset, name, after, before, info, detail, error) in enumerate(pool.map(work, jobs), 1):
            if error:
                errors[f"{dataset}/{name}"] = error
            if after is not None:
                post[dataset].append(after)
                pre[dataset].append(before)
            gate_totals[dataset].update(info)
            details.append({"dataset": dataset, "name": name, **detail})
            print(f"aligned {i}/{len(jobs)} {dataset}/{name} {'ERROR '+str(error) if error else ''}", flush=True)
    result = {"datasets": {d: {"clips": len(post[d]), "post_gate": metric(post[d]), "pre_gate": metric(pre[d]), "gate_totals": dict(gate_totals[d])} for d in post},
              "errors": errors, "details": details, "datacreate_gate_totals": dict(gate_totals["datacreate"])}
    write(OUT / "synthetic.json", result)
    for d, values in result["datasets"].items():
        print(d, json.dumps(values), flush=True)
    score_dc()

if __name__ == "__main__":
    main()

