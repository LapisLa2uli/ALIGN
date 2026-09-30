"""Evaluate perfect-transcription aligners on the repaired 9.2 population.

Gold: repaired rendered lineage projected onto verified_score.musicxml
canonical events. Metric: official exclusive one-to-one note-wise F1 (same
location and type 1.0, same location other type 0.5). Input: the gold
rendered notes (pitch, start, end) in rendered order, i.e. a perfect
transcription. Test mode requires a frozen candidate and refuses to evaluate
the same candidate twice.
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import eval_orn_phase2_baseline_v1 as baseline
from alignmodel.joint.packed_data import sha256_file
from alignmodel.joint.perfect_dp_aligner_v1 import PerfectDPCosts, align_perfect
from realistic92_aligner_common import load_clip, metric_sample, perfect_transcription


_STATE: dict[str, Any] = {}


def _init(root: str, lineage_dir: str, costs: dict[str, Any]) -> None:
    _STATE["root"] = Path(root)
    _STATE["lineage_dir"] = Path(lineage_dir)
    _STATE["costs"] = PerfectDPCosts(**costs)


def _run(name: str) -> tuple[str, Any, str | None]:
    try:
        clip = load_clip(_STATE["root"], name, _STATE["lineage_dir"])
        result = align_perfect(
            perfect_transcription(clip), clip.index.events, clip.score_path, _STATE["costs"]
        )
        return name, metric_sample(clip, result.events, result.deletions), None
    except Exception as error:  # noqa: BLE001
        return name, None, f"{type(error).__name__}: {str(error)[:200]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--mode", choices=("train", "val", "test"), required=True)
    parser.add_argument("--costs", default="{}")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_file(args.freeze) != args.expected_freeze_sha256:
        raise ValueError("Aligner freeze mismatch")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    lineage_dir = Path(freeze["lineage_dir"])
    names = list(freeze["eligible"][args.mode])
    costs = json.loads(args.costs)
    candidate_sha = None
    if args.mode == "test":
        if args.candidate is None:
            raise ValueError("Test mode needs a frozen --candidate")
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        candidate_sha = sha256_file(args.candidate)
        if candidate["freeze_sha256"] != args.expected_freeze_sha256:
            raise ValueError("Candidate was frozen against another population")
        if candidate["aligner_sha256"] != sha256_file(
            Path(__file__).resolve().parents[1] / "src/alignmodel/joint/perfect_dp_aligner_v1.py"
        ):
            raise ValueError("Aligner code changed after candidate freeze")
        log_path = args.freeze.parent / "TEST_EVALUATIONS.jsonl"
        if log_path.exists() and any(
            json.loads(line)["candidate_sha256"] == candidate_sha
            for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ):
            raise ValueError("This candidate was already evaluated on test")
        costs = dict(candidate["costs"])
        hashes = freeze["test_lineage_sha256"]
        for name in names:
            if sha256_file(lineage_dir / f"{name}.json") != hashes[name]:
                raise ValueError(f"Test lineage changed after freeze: {name}")
    if args.limit:
        names = names[:args.limit]

    samples = []
    failures = collections.Counter()
    failed_names = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init,
        initargs=(str(args.root), str(lineage_dir), costs),
    ) as pool:
        for position, (name, sample, error) in enumerate(pool.map(_run, names, chunksize=4), 1):
            if sample is None:
                failures[error] += 1
                failed_names.append(name)
            else:
                samples.append(sample)
            if position % 250 == 0 or position == len(names):
                print(f"aligned={position}/{len(names)} failures={sum(failures.values())}", flush=True)
    # A clip the aligner cannot process scores as an empty prediction.
    for name in failed_names:
        clip = load_clip(args.root, name, lineage_dir)
        samples.append(metric_sample(clip, (), frozenset()))
    report = baseline._full_report(samples, seed=20260926, replicates=1000)
    by_group = collections.defaultdict(list)
    for sample in samples:
        by_group[sample.source.split(":")[0]].append(sample)
    result = {
        "schema_version": "align-realistic92-perfect-aligner-eval-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "clips": len(names),
        "scored_clips": len(samples),
        "failures": dict(failures),
        "costs": costs,
        "candidate_sha256": candidate_sha,
        "official_note_wise": {
            key: report[key] for key in report if key != "per_source"
        },
        "by_group_f1": {key: baseline._aggregate(value)["f1"] for key, value in by_group.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.mode == "test":
        with (args.freeze.parent / "TEST_EVALUATIONS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "candidate_sha256": candidate_sha,
                "evaluated_utc": result["created_utc"],
                "f1": report["f1"], "precision": report["precision"],
                "recall": report["recall"], "output": str(args.output),
            }) + "\n")
    print(json.dumps({
        "f1": report["f1"], "precision": report["precision"], "recall": report["recall"],
        "by_group_f1": result["by_group_f1"],
        "per_type_f1": {key: value["f1"] for key, value in report["per_type"].items()},
        "failures": sum(failures.values()),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
