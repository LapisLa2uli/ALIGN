from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from alignmodel.joint.grammar_mapper_v2 import (
    decode_grammar_mapper,
    operation_features,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset
from alignmodel.training_resources import resource_lease


TYPES = ("match", "copy", "substitute", "extra")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args()
    lease = resource_lease(
        args.resource_status,
        "heavy_cpu",
        track="mel-mapper-v2-operation-core",
        command=[str(value) for value in __import__("sys").argv],
        metadata={"rows": args.rows, "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    ready = verify_data_ready(args.ready_marker)
    manifest = json.loads(
        Path(str(ready["paths"]["manifest"])).read_text(encoding="utf-8")
    )
    source_root = Path(str(manifest["source_root"]))
    with args.predictions.open("r", encoding="utf-8") as stream:
        predictions = {
            row["sample"]: row["notes"]
            for row in (json.loads(line) for line in stream if line.strip())
        }
    train_features, train_labels = [], []
    heldout_features, heldout_labels = [], []
    coverage = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        ordinals = sorted(
            dataset.ordinals("train"),
            key=lambda value: hashlib.sha256(
                f"{args.seed}:{value}".encode()
            ).digest(),
        )[: args.rows]
        for position, ordinal in enumerate(ordinals, 1):
            packed = dataset[ordinal]
            example = packed.training_example()
            candidates = tuple(
                JointCandidate(
                    pitch=int(row["pitch"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    confidence=float(row["confidence"]),
                )
                for row in predictions[packed.sample]
            )
            score = ScoreEventIndex.from_musicxml(
                source_root / packed.sample / "verified_score.musicxml"
            ).events
            mapped, grammar = decode_grammar_mapper(candidates, score)
            matcher = SequenceMatcher(
                None,
                [candidate.pitch for candidate in candidates],
                [event.pitch for event in example.target_events],
                autojunk=False,
            )
            pairs = [
                (left + offset, right + offset)
                for left, right, count in matcher.get_matching_blocks()
                for offset in range(count)
            ]
            coverage.append(len(pairs) / max(len(candidates), len(example.target_events), 1))
            destination_features = (
                heldout_features if position % 5 == 0 else train_features
            )
            destination_labels = (
                heldout_labels if position % 5 == 0 else train_labels
            )
            for candidate_index, target_index in pairs:
                target = example.target_events[target_index]
                destination_features.append(
                    operation_features(
                        candidates, mapped, score, candidate_index, grammar
                    )
                )
                destination_labels.append(
                    TYPES.index("copy" if target.is_copy else target.relationship)
                )
            if position == 1 or position % 25 == 0:
                print(f"operation_rows={position}/{len(ordinals)}", flush=True)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(np.stack(train_features), np.asarray(train_labels))
    heldout_prediction = model.predict(np.stack(heldout_features))
    report = classification_report(
        heldout_labels,
        heldout_prediction,
        labels=list(range(len(TYPES))),
        target_names=list(TYPES),
        output_dict=True,
        zero_division=0,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    joblib.dump(model, temporary)
    os.replace(temporary, output / "operation_core.joblib")
    document = {
        "schema_version": "align-grammar-operation-core-v2",
        "model": "operation_core.joblib",
        "model_sha256": _sha256(output / "operation_core.joblib"),
        "data_fingerprint": ready["hashes"]["pack_id"],
        "prediction_cache_sha256": _sha256(args.predictions),
        "train_rows": args.rows - args.rows // 5,
        "heldout_train_rows": args.rows // 5,
        "pitch_lcs_pair_coverage": float(np.mean(coverage)),
        "heldout": report,
        "timestamp_metrics_used": False,
        "locked_test_touched": False,
    }
    (output / "report.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    atexit.unregister(lease.__exit__)
    lease.__exit__(None, None, None)


if __name__ == "__main__":
    main()
