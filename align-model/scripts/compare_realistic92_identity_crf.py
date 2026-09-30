"""Score the frozen identity CRF on repaired 9.2 val gold with perfect transcriptions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import calibrate_v2_acoustic_crf as acoustic
import calibrate_v2_template_rescue as rescue
import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.identity_crf_v1 import IdentityCandidate
from realistic92_aligner_common import load_clip, metric_sample, perfect_transcription


_STATE: dict[str, Any] = {}


def _init(root: str, lineage_dir: str, checkpoint: str) -> None:
    _STATE.update(root=Path(root), lineage_dir=Path(lineage_dir),
                  model=acoustic._load_model(Path(checkpoint))[0])


def _run(name: str):
    clip = load_clip(_STATE["root"], name, _STATE["lineage_dir"])
    candidates = tuple(
        IdentityCandidate(p, s, e, 1.0, (p,), (1.0,)) for p, s, e in perfect_transcription(clip)
    )
    try:
        sample, _ = rescue._map(_STATE["model"], candidates, clip.index, clip.score_path)
        return metric_sample(clip, sample.predicted, sample.predicted_deletions)
    except Exception:  # noqa: BLE001
        return metric_sample(clip, (), frozenset())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    names = freeze["eligible"]["val"][:args.limit]
    with ProcessPoolExecutor(
        max_workers=6, initializer=_init,
        initargs=(str(args.root), freeze["lineage_dir"], str(args.checkpoint)),
    ) as pool:
        samples = list(pool.map(_run, names, chunksize=2))
    report = baseline._full_report(samples, seed=1, replicates=500)
    result = {
        "clips": len(names), "f1": report["f1"], "precision": report["precision"],
        "recall": report["recall"],
        "per_type_f1": {k: v["f1"] for k, v in report["per_type"].items()},
        "names": names,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "names"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
