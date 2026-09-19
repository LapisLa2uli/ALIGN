"""Read-only annotation audit; acoustic estimates are review aids, never gold."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "DataCreate/src"))
from datacreate.note_alignment import _extract_score_events, _annotate_sounding_indices


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def structure():
    reports = {}
    for bundle in sorted((HERE.parent / "bundles").iterdir()):
        events = _annotate_sounding_indices(_extract_score_events(bundle / "verified_score.musicxml"))
        sounding = [e for e in events if not e["is_rest"]]
        doc = load(bundle / "labels.json")
        rows = []
        duplicate_groups = defaultdict(list)
        for i, label in enumerate(doc["labels"]):
            part = label.get("score_part") or {}
            first, last = part.get("start_note_index"), part.get("end_note_index")
            core0, core1 = part.get("core_start_note_index"), part.get("core_end_note_index")
            issues = []
            selected = [e for e in events if e["id"] in (label.get("core_note_ids") or [])]
            core_from_ids = [e["sounding_index"] for e in selected if not e["is_rest"]]
            core_from_part = list(range(core0, core1 + 1)) if core0 is not None and core1 is not None else []
            if core_from_ids != core_from_part:
                issues.append("ui_core_ids_and_core_range_disagree")
            expected = [e["midi"] for e in sounding[first:last + 1]] if first is not None and last is not None else []
            if expected != label.get("pitches"):
                issues.append("stored_pitches_disagree_with_current_score_range")
            if label["type"] == "repetition" and label.get("extra_copies") is None:
                issues.append("copy_count_missing_for_canonical_scoring")
            # Ignore label ID and UI source; content duplication remains exact.
            signature = {k: v for k, v in label.items() if k not in ("id", "source")}
            duplicate_groups[json.dumps(signature, sort_keys=True)].append(i)
            rows.append(dict(index=i, id=label["id"], type=label["type"], start=label["start_time"],
                end=label["end_time"], issues=issues, core_indices_from_ui_ids=core_from_ids,
                core_indices_from_range=core_from_part,
                expected_written_pitches_from_core_range=[sounding[k]["midi"] for k in core_from_part if k < len(sounding)],
                stored_pitches=label.get("pitches"), expected_padded_pitches=expected))
        nested = []
        for i, a in enumerate(doc["labels"]):
            for j, b in enumerate(doc["labels"]):
                if i == j or a["type"] != b["type"]:
                    continue
                if a["start_time"] <= b["start_time"] and b["end_time"] <= a["end_time"] and (
                    a["start_time"] < b["start_time"] or b["end_time"] < a["end_time"]):
                    nested.append(dict(outer=i, inner=j, type=a["type"]))
        reports[bundle.name] = dict(labels=rows, score_notes=sounding, exact_duplicate_groups=[v for v in duplicate_groups.values() if len(v) > 1],
            nested_same_type=nested, label_file_sha256=hashlib.sha256((bundle / "labels.json").read_bytes()).hexdigest())
    save(HERE / "structure.json", reports)
    return reports


def pitch_one(sid):
    import librosa
    from scipy.signal import resample_poly
    from scipy.ndimage import median_filter
    path = HERE.parent / "bundles" / sid / "performance_audio.wav"
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    from math import gcd
    div = gcd(sr, 16000)
    y = resample_poly(y, 16000 // div, sr // div)
    frame, hop = 1024, 160
    f0 = librosa.yin(y, fmin=110, fmax=2000, sr=16000, frame_length=frame, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    frames = librosa.util.frame(np.pad(y, (frame // 2, frame // 2)), frame_length=frame, hop_length=hop)
    ac = librosa.autocorrelate(frames, axis=0)
    lags = np.clip(np.rint(16000 / f0).astype(int), 1, frame - 1)
    periodicity = ac[lags, np.arange(len(lags))] / np.maximum(ac[0], 1e-12)
    gate = max(.001, float(np.percentile(rms, 90)) * .06)
    voiced = (rms > gate) & (periodicity > .65)
    midi = librosa.hz_to_midi(f0)
    smooth = median_filter(midi, size=3)
    rounded = np.rint(smooth).astype(int)
    rounded[~voiced] = -1
    runs = []
    start = 0
    for end in range(1, len(rounded) + 1):
        if end < len(rounded) and rounded[end] == rounded[start]:
            continue
        if rounded[start] >= 0 and (end - start) * hop / 16000 >= .06:
            runs.append(dict(start=start * hop / 16000, end=end * hop / 16000,
                pitch=int(rounded[start]), periodicity=float(np.median(periodicity[start:end])),
                cents_mad=float(np.median(np.abs(midi[start:end] - np.median(midi[start:end]))) * 100)))
        start = end
    (HERE / "pitch").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "pitch" / (sid + ".npz"), time=np.arange(len(f0)) * hop / 16000,
        midi=midi, rms=rms, periodicity=periodicity, voiced=voiced)
    save(HERE / "pitch" / (sid + ".json"), dict(sample=sid, runs=runs, voiced_fraction=float(voiced.mean()),
        policy="Independent YIN estimate and periodicity/rms gate; not a verified transcription"))
    return sid, len(runs)


def align(ref, heard):
    n, m = len(ref), len(heard)
    cost = np.zeros((n + 1, m + 1))
    parent = np.zeros((n + 1, m + 1), np.int8)
    cost[:, 0] = np.arange(n + 1); cost[0] = np.arange(m + 1)
    parent[1:, 0] = 1; parent[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (cost[i - 1, j - 1] + (0 if ref[i - 1] == heard[j - 1] else 1.1),
                       cost[i - 1, j] + 1, cost[i, j - 1] + 1)
            p = int(np.argmin(choices)); parent[i, j] = p; cost[i, j] = choices[p]
    ops = []
    i, j = n, m
    while i or j:
        p = parent[i, j]
        if p == 0:
            ops.append(dict(kind="match" if ref[i - 1] == heard[j - 1] else "sub", score=i - 1, heard=j - 1));i -= 1;j -= 1
        elif p == 1:
            ops.append(dict(kind="missing_candidate", score=i - 1, heard=None));i -= 1
        else:
            ops.append(dict(kind="extra_candidate", score=None, heard=j - 1));j -= 1
    return list(reversed(ops))


def review_pitch():
    audit = load(HERE / "structure.json")
    report = {}
    for sid, item in audit.items():
        data = load(HERE / "pitch" / (sid + ".json"));runs = data["runs"]
        # Merge immediately repeated stable pitch runs separated by <=80ms.
        merged = []
        for run in runs:
            if merged and run["pitch"] == merged[-1]["pitch"] and run["start"] - merged[-1]["end"] <= .08:
                merged[-1]["end"] = run["end"]
            else:
                merged.append(dict(run))
        score = item["score_notes"]
        ref = [x["midi"] - 2 for x in score]
        ops = align(ref, [x["pitch"] for x in merged])
        candidates = []
        for k, op in enumerate(ops):
            if op["kind"] != "sub" or k < 2 or k + 2 >= len(ops):
                continue
            if not all(ops[t]["kind"] == "match" for t in [k - 2, k - 1, k + 1, k + 2]):
                continue
            run = merged[op["heard"]]
            if run["end"] - run["start"] < .10 or run["periodicity"] < .75 or run["cents_mad"] > 25:
                continue
            candidates.append(dict(**run, score_index=op["score"], expected_sounding_pitch=ref[op["score"]],
                expected_written_pitch=ref[op["score"]] + 2,
                overlaps_existing_label=any(l["start"] < run["end"] and run["start"] < l["end"] for l in item["labels"])))
        frame = np.load(HERE / "pitch" / (sid + ".npz"))
        label_checks = []
        for label in item["labels"]:
            select = (frame["time"] >= label["start"] + .015) & (frame["time"] <= label["end"] - .015) & frame["voiced"]
            observed = frame["midi"][select]
            expected = [p - 2 for p in label["expected_written_pitches_from_core_range"]]
            label_checks.append(dict(label_index=label["index"], type=label["type"],
                reliable_frames=int(select.sum()), dominant_sounding_pitches=Counter(np.rint(observed).astype(int).tolist()).most_common(6),
                expected_sounding_pitches_from_selected_core=expected))
        report[sid] = dict(empty_labels=not item["labels"], stable_runs=len(merged), score_notes=len(score),
            sequence_operations=dict(Counter(o["kind"] for o in ops)), strong_pitch_review_candidates=candidates,
            label_pitch_checks=label_checks)
    save(HERE / "pitch_review.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["structure", "pitch", "review"])
    args = parser.parse_args()
    if args.mode == "structure":
        structure()
    elif args.mode == "pitch":
        ids = sorted(load(HERE / "structure.json"))
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(pitch_one, sid) for sid in ids]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    else:
        review_pitch()
