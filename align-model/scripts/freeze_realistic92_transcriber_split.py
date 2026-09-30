"""Freeze a score-grouped train/val/test split of dataset 9.2 for transcription.

RawData snippets overlap heavily within a source score, so whole scores are
assigned to one split. Procedural clips each have their own generated score.
Test audio and gold hashes are recorded so later evaluation can prove the test
population did not change after the freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from alignmodel.joint.packed_data import sha256_file


SCHEMA_VERSION = "align-realistic92-transcriber-split-v1"
PROCEDURAL = re.compile(r"^synth_gen_\d+$")
RAWDATA = re.compile(r"^synth_(.+)_\d+$")


def _group(name: str) -> str:
    if PROCEDURAL.match(name):
        return f"procedural:{name}"
    match = RAWDATA.match(name)
    if not match:
        raise ValueError(f"Unrecognized sample name: {name}")
    return f"score:{match.group(1)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--test-scores", type=int, default=2)
    parser.add_argument("--val-scores", type=int, default=1)
    parser.add_argument("--test-procedural", type=int, default=600)
    parser.add_argument("--val-procedural", type=int, default=400)
    parser.add_argument("--dataset-version", default="9.2")
    args = parser.parse_args()

    freeze_path = args.output_dir / "FREEZE.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite {freeze_path}")
    samples = sorted(
        path for path in args.root.iterdir()
        if path.is_dir() and path.name.startswith("synth_")
    )
    usable = []
    for position, sample in enumerate(samples, 1):
        metadata_path = sample / "metadata.json"
        if not (
            metadata_path.is_file()
            and (sample / "performance_audio.wav").is_file()
            and (sample / "note_map.json").is_file()
        ):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("dataset_version")) != args.dataset_version:
            continue
        usable.append(sample.name)
        if position % 1000 == 0:
            print(f"scanned={position}/{len(samples)}", flush=True)

    rng = random.Random(args.seed)
    scores = sorted({
        _group(name) for name in usable if _group(name).startswith("score:")
    })
    rng.shuffle(scores)
    test_scores = set(scores[:args.test_scores])
    val_scores = set(scores[args.test_scores:args.test_scores + args.val_scores])
    procedural = sorted(name for name in usable if PROCEDURAL.match(name))
    rng.shuffle(procedural)
    test_procedural = set(procedural[:args.test_procedural])
    val_procedural = set(
        procedural[args.test_procedural:args.test_procedural + args.val_procedural]
    )

    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for name in usable:
        group = _group(name)
        if group in test_scores or name in test_procedural:
            split["test"].append(name)
        elif group in val_scores or name in val_procedural:
            split["val"].append(name)
        else:
            split["train"].append(name)

    groups = {
        key: sorted({_group(name) for name in names if _group(name).startswith("score:")})
        for key, names in split.items()
    }
    if set(groups["train"]) & (set(groups["val"]) | set(groups["test"])):
        raise ValueError("Score group leaked across splits")
    if set(groups["val"]) & set(groups["test"]):
        raise ValueError("Score group leaked between val and test")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_document = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": args.dataset_version,
        "root": str(args.root),
        "seed": args.seed,
        "grouping": "whole RawData source score; one group per procedural clip",
        "score_groups": groups,
        "counts": {key: len(value) for key, value in split.items()},
        "splits": split,
    }
    split_path = args.output_dir / "split.json"
    split_path.write_text(json.dumps(split_document, indent=2) + "\n", encoding="utf-8")

    test_hashes = {}
    for position, name in enumerate(split["test"], 1):
        sample = args.root / name
        test_hashes[name] = {
            "performance_audio.wav": sha256_file(sample / "performance_audio.wav"),
            "note_map.json": sha256_file(sample / "note_map.json"),
        }
        if position % 200 == 0:
            print(f"hashed_test={position}/{len(split['test'])}", flush=True)
    test_hash_path = args.output_dir / "test_hashes.json"
    test_hash_path.write_text(json.dumps(test_hashes, indent=2) + "\n", encoding="utf-8")

    freeze = {
        "schema_version": f"{SCHEMA_VERSION}-freeze",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": sha256_file(split_path),
        "test_hashes_sha256": sha256_file(test_hash_path),
        "counts": split_document["counts"],
        "score_groups": groups,
        "test_evaluations": [],
        "rules": [
            "train only on the train split",
            "select checkpoints and decoders on the val split only",
            "evaluate a frozen candidate on the test split once",
            "headline metric: written-pitch sequence LCS F1 against note_map rendered_notes",
        ],
        "membership_selected_without_model_outcomes": True,
        "selection_digest": hashlib.sha256(
            json.dumps(split, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "counts": split_document["counts"],
        "score_groups": groups,
        "split_sha256": freeze["split_sha256"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
