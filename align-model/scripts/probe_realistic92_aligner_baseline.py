"""Probe gold projection and the frozen identity CRF on a few 9.2 train clips."""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import calibrate_v2_acoustic_crf as acoustic
import calibrate_v2_template_rescue as rescue
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_v1 import IdentityCandidate
from realistic92_aligner_common import load_clip, metric_sample, perfect_transcription


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    names = list(json.loads(args.split.read_text(encoding="utf-8"))["splits"]["train"])
    random.Random(args.seed).shuffle(names)
    model = acoustic._load_model(args.identity_checkpoint)[0]
    failures = collections.Counter()
    samples = []
    for name in names[:args.limit]:
        try:
            clip = load_clip(args.root, name)
        except Exception as error:  # noqa: BLE001
            failures[f"projection:{type(error).__name__}:{str(error)[:80]}"] += 1
            continue
        notes = perfect_transcription(clip)
        candidates = tuple(
            IdentityCandidate(n["pitch"], n["start"], n["end"], 1.0, (n["pitch"],), (1.0,))
            for n in notes
        )
        started = time.perf_counter()
        try:
            sample, diagnostics = rescue._map(model, candidates, clip.index, clip.score_path)
        except Exception as error:  # noqa: BLE001
            failures[f"identity_crf:{type(error).__name__}:{str(error)[:80]}"] += 1
            continue
        seconds = time.perf_counter() - started
        scored = metric_sample(clip, sample.predicted, sample.predicted_deletions)
        report = baseline._aggregate([scored])
        samples.append(scored)
        kinds = collections.Counter(e.relationship for e in clip.rendered)
        print(json.dumps({
            "sample": name, "notes": len(notes), "events": len(clip.index.events),
            "deleted": len(clip.index.deleted_event_indices), "kinds": dict(kinds),
            "f1": round(report["f1"], 4), "seconds": round(seconds, 2),
            "copies": diagnostics.get("copies"),
        }), flush=True)
    if samples:
        print(json.dumps({"aggregate_f1": baseline._aggregate(samples)["f1"],
                          "clips": len(samples), "failures": dict(failures)}, indent=2))
    else:
        print(json.dumps({"failures": dict(failures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
