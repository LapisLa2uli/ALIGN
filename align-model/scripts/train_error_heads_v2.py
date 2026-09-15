"""Train and evaluate corrected frozen-upstream error heads v2.

This reuses the expensive v1 frozen-upstream extraction, migrates only its
audited target attachment, and evaluates emitted schema 1.2 labels with the
official exclusive melody metric.  The sealed test split is never opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import train_error_heads as base
from alignmodel.joint.error_heads import (
    FEATURE_NAMES,
    LAYER2_CLASSES,
    HeadPrediction,
    evaluate_predictions,
    filter_prediction_for_schema,
    heuristic_prediction,
    load_error_heads,
    schema12_document,
    stack_labeled_rows,
)
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.melody import (
    WeakMelody,
    gold_melodies_from_labels,
    match_melodies_detail,
    match_note_wise_labels_detail,
    pred_melodies_from_labels,
)


OUTPUT_SCHEMA = "align-frozen-error-heads-run-v2"
REPORT_SCHEMA = "align-frozen-error-heads-report-v2"
TARGET_SCHEMA = "align-frozen-error-targets-v2"
SCORED_SCHEMA_TYPES = {
    "wrong_note",
    "extra_note",
    "missed_note",
    "rhythm_error",
    "repetition",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _repair_labeled_blob(blob: bytes) -> bytes:
    value = base._decode(blob)
    layer2 = list(value["layer2"])
    rhythm_mask = list(value["rhythm_mask"])
    mapping = [bool(item) for item in value["mapping_correct"]]
    for index, row in enumerate(value["rows"]):
        rhythm_mask[index] = bool(
            rhythm_mask[index]
            and mapping[index]
            and row["kind"] == "event"
            and not bool(row.get("is_copy"))
        )
    value["layer2"] = layer2
    value["rhythm_mask"] = rhythm_mask
    # Keep the wire version understood by the shared loader; the v2 contract
    # is recorded in SQLite metadata and the run config.
    value["schema_version"] = "align-frozen-error-examples-v1"
    return base._encode(value)


def migrate_v1_cache(
    source: Path,
    destination: Path,
    canonical_targets: Path,
) -> dict[str, Any]:
    if destination.is_file():
        connection = sqlite3.connect(destination)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            counts = dict(
                connection.execute(
                    "SELECT split,COUNT(*) FROM clips GROUP BY split"
                )
            )
            if (
                metadata.get("target_schema_version") == TARGET_SCHEMA
                and counts == {"train": 4544, "val": 358}
            ):
                return {
                    "cache": str(destination.resolve()),
                    "counts": counts,
                    "reused": True,
                }
        finally:
            connection.close()
        raise ValueError("Existing v2 cache has a different target contract")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not canonical_targets.is_file():
        raise FileNotFoundError(canonical_targets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(
        f"file:{source.resolve().as_posix()}?mode=ro", uri=True
    )
    output = sqlite3.connect(destination)
    try:
        source_connection.backup(output)
        output.execute(
            "CREATE TABLE IF NOT EXISTS audited_labels("
            "ordinal INTEGER PRIMARY KEY,sample TEXT UNIQUE NOT NULL,"
            "labels TEXT NOT NULL,audit_metadata TEXT NOT NULL)"
        )
        rows = output.execute(
            "SELECT ordinal,predicted,oracle FROM clips ORDER BY ordinal"
        ).fetchall()
        output.executemany(
            "UPDATE clips SET predicted=?,oracle=? WHERE ordinal=?",
            [
                (
                    sqlite3.Binary(_repair_labeled_blob(predicted)),
                    sqlite3.Binary(_repair_labeled_blob(oracle)),
                    int(ordinal),
                )
                for ordinal, predicted, oracle in rows
            ],
        )
        audit = sqlite3.connect(
            f"file:{canonical_targets.resolve().as_posix()}?mode=ro", uri=True
        )
        try:
            samples_by_ordinal = dict(
                output.execute("SELECT ordinal,sample FROM clips")
            )
            audited_rows = []
            for ordinal, split, sample_dir, payload_blob in audit.execute(
                "SELECT ordinal,split,sample_dir,payload FROM targets "
                "WHERE split IN ('train','val') ORDER BY ordinal"
            ):
                payload = json.loads(zlib.decompress(payload_blob))
                labels = [
                    label
                    for label in payload.get("valid_labels", ())
                    if label.get("type") != "intonation_error"
                ]
                metadata = {
                    "split": split,
                    "source": payload.get("source")
                    or Path(sample_dir).name.split("_")[1],
                    "audio_render": (
                        payload.get("render_validation") or {}
                    ).get("audio_render", "soundfont_v1"),
                    "render_repair": payload.get("render_repair"),
                }
                audited_rows.append(
                    (
                        int(ordinal),
                        str(samples_by_ordinal[int(ordinal)]),
                        json.dumps(labels, sort_keys=True),
                        json.dumps(metadata, sort_keys=True),
                    )
                )
            output.executemany(
                "INSERT OR REPLACE INTO audited_labels VALUES(?,?,?,?)",
                audited_rows,
            )
        finally:
            audit.close()
        output.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            ("target_schema_version", TARGET_SCHEMA),
        )
        output.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            ("migration_source", str(source.resolve())),
        )
        output.commit()
        counts = dict(
            output.execute("SELECT split,COUNT(*) FROM clips GROUP BY split")
        )
        label_count = output.execute(
            "SELECT COUNT(*) FROM audited_labels"
        ).fetchone()[0]
        if counts != {"train": 4544, "val": 358} or label_count != 4902:
            raise ValueError(
                f"V2 migration count mismatch: clips={counts}, labels={label_count}"
            )
        return {
            "cache": str(destination.resolve()),
            "counts": counts,
            "audited_label_rows": label_count,
            "reused": False,
        }
    except BaseException:
        output.close()
        destination.unlink(missing_ok=True)
        raise
    finally:
        if output:
            try:
                output.close()
            except sqlite3.Error:
                pass
        source_connection.close()


def _audited_labels(cache_path: Path) -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(
        f"file:{cache_path.resolve().as_posix()}?mode=ro", uri=True
    )
    try:
        return {
            str(sample): json.loads(labels)
            for sample, labels in connection.execute(
                "SELECT a.sample,a.labels FROM audited_labels a "
                "JOIN clips c USING(ordinal) WHERE c.split='val'"
            )
        }
    finally:
        connection.close()


def _scored_gold(labels: Sequence[Mapping[str, Any]]) -> list[WeakMelody]:
    return [
        item
        for item in gold_melodies_from_labels([dict(value) for value in labels])
        if item.type in SCORED_SCHEMA_TYPES
    ]


def _official_summary(
    documents: Mapping[str, Mapping[str, Any]],
    gold_labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    totals = {"matched": 0.0, "predicted": 0, "gold": 0}
    per_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {"matched": 0.0, "predicted": 0.0, "gold": 0.0}
    )
    range_only = {"matched": 0.0, "predicted": 0, "gold": 0}
    for sample in sorted(gold_labels):
        gold = [
            dict(value)
            for value in gold_labels[sample]
            if value.get("type") in SCORED_SCHEMA_TYPES
        ]
        labels = [
            dict(value)
            for value in documents.get(sample, {}).get("labels", ())
            if value.get("type") in SCORED_SCHEMA_TYPES
        ]
        predicted = labels
        detail = match_note_wise_labels_detail(gold, predicted)
        range_detail = match_note_wise_labels_detail(
            gold, predicted, type_mismatch_credit=1.0
        )
        if detail["status"] != "available":
            raise ValueError(f"Official note-wise metric unavailable for {sample}")
        rows.append(
            {
                "sample": sample,
                "f1": float(detail["f1"]),
                "precision": float(detail["precision"]),
                "recall": float(detail["recall"]),
                "predicted": len(predicted),
                "gold": len(gold),
                "matched": float(detail["credit"]),
            }
        )
        totals["matched"] += float(detail["credit"])
        totals["predicted"] += len(predicted)
        totals["gold"] += len(gold)
        range_only["matched"] += float(range_detail["credit"])
        range_only["predicted"] += len(predicted)
        range_only["gold"] += len(gold)
        for kind in sorted(SCORED_SCHEMA_TYPES):
            gold_kind = [item for item in gold if item.get("type") == kind]
            pred_kind = [item for item in predicted if item.get("type") == kind]
            type_detail = match_note_wise_labels_detail(gold_kind, pred_kind)
            per_type[kind]["matched"] += float(type_detail["credit"])
            per_type[kind]["predicted"] += len(pred_kind)
            per_type[kind]["gold"] += len(gold_kind)

    def finish(value: Mapping[str, float]) -> dict[str, float]:
        precision = float(value["matched"]) / max(float(value["predicted"]), 1.0)
        recall = float(value["matched"]) / max(float(value["gold"]), 1.0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "matched": float(value["matched"]),
            "predicted": int(value["predicted"]),
            "gold": int(value["gold"]),
        }

    return {
        "metric": "official_note_wise_schema_1_2_fractional_type_f1",
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "micro": finish(totals),
        "mean_clip_f1": float(np.mean([row["f1"] for row in rows])),
        "range_only_ignore_type": finish(range_only),
        "per_type": {
            kind: finish(value) for kind, value in sorted(per_type.items())
        },
        "validation_rows": len(rows),
    }


def _prediction_documents(
    cache_path: Path,
    output_dir: Path,
    variant: str,
) -> tuple[
    dict[str, dict[str, Any]],
    list[tuple[str, Any, HeadPrediction, Mapping[str, Any]]],
]:
    clips = base._load_clips(cache_path, split="val", variant=variant)
    arrays = stack_labeled_rows([value[1] for value in clips])
    model, payload = load_error_heads(output_dir / variant / "best.pt")
    probabilities = base._probabilities(
        model, arrays, batch_size=32768, device=torch.device("cpu")
    )
    predictions = base._clip_predictions(
        clips, probabilities, payload["thresholds"]
    )
    documents = _documents_for_predictions(
        cache_path, variant, predictions, schema_thresholds={}
    )
    return documents, predictions


def _documents_for_predictions(
    cache_path: Path,
    variant: str,
    predictions: Sequence[
        tuple[str, Any, HeadPrediction, Mapping[str, Any]]
    ],
    *,
    schema_thresholds: Mapping[str, float],
    clips: Sequence[tuple[str, Any, Any, Mapping[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    loaded_clips = (
        list(clips)
        if clips is not None
        else base._load_clips(cache_path, split="val", variant=variant)
    )
    lookup = {
        sample: filter_prediction_for_schema(prediction, schema_thresholds)
        for sample, _rows, prediction, _metadata in predictions
    }
    return {
        sample: schema12_document(
            sample,
            rows.rows,
            lookup[sample],
            base._score_events(score),
            pad_notes=1,
        )
        for sample, rows, score, _metadata in loaded_clips
    }


def _calibrate_schema_thresholds(
    cache_path: Path,
    predictions: Sequence[
        tuple[str, Any, HeadPrediction, Mapping[str, Any]]
    ],
    gold_labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = {
        "wrong_note": 0.0,
        "extra_note": 0.0,
        "missed_note": 0.0,
        "rhythm_error": 0.0,
    }
    grid = np.concatenate(
        (np.asarray([0.0]), np.linspace(0.10, 0.95, 18), np.asarray([1.01]))
    )
    validation = {}
    clips = base._load_clips(cache_path, split="val", variant="predicted")
    for kind in (
        "wrong_note",
        "extra_note",
        "missed_note",
        "rhythm_error",
    ):
        choices = []
        for threshold in grid:
            trial = {**thresholds, kind: float(threshold)}
            documents = _documents_for_predictions(
                cache_path,
                "predicted",
                predictions,
                schema_thresholds=trial,
                clips=clips,
            )
            metric = _official_summary(documents, gold_labels)["per_type"][kind]
            choices.append(
                (
                    float(metric["f1"]),
                    float(metric["precision"]),
                    float(threshold),
                    metric,
                )
            )
        selected = max(choices, key=lambda value: (value[0], value[1], value[2]))
        thresholds[kind] = selected[2]
        validation[kind] = dict(selected[3])
    documents = _documents_for_predictions(
        cache_path,
        "predicted",
        predictions,
        schema_thresholds=thresholds,
        clips=clips,
    )
    validation["combined"] = _official_summary(documents, gold_labels)["micro"]
    return thresholds, validation


def _store_schema_calibration(
    checkpoint: Path,
    thresholds: Mapping[str, float],
    validation: Mapping[str, Any],
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["schema_thresholds"] = dict(thresholds)
    payload["schema_calibration"] = {
        "calibrated_on": "validation_only",
        "metric": "official_exclusive_schema_1_2_hard_f1",
        "validation": dict(validation),
    }
    with tempfile.NamedTemporaryFile(
        dir=checkpoint.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rule_documents(
    cache_path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[tuple[str, Any, HeadPrediction, Mapping[str, Any]]],
]:
    clips = base._load_clips(cache_path, split="val", variant="predicted")
    predictions = [
        (sample, rows, heuristic_prediction(rows.rows), metadata)
        for sample, rows, _score, metadata in clips
    ]
    documents = {
        sample: schema12_document(
            sample,
            rows.rows,
            heuristic_prediction(rows.rows),
            base._score_events(score),
            pad_notes=1,
        )
        for sample, rows, score, _metadata in clips
    }
    return documents, predictions


def _legacy_documents(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            sample = str((document.get("pipeline") or {}).get("sample_id"))
            result[sample] = document
    return result


def _perfect_projection_ceiling(
    documents: Mapping[str, Mapping[str, Any]],
    gold_labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    matched = predicted_count = gold_count = 0
    for sample, labels in gold_labels.items():
        gold_types = Counter(item.type for item in _scored_gold(labels))
        predicted_types = Counter(
            label.get("type")
            for label in documents.get(sample, {}).get("labels", ())
            if label.get("type") in SCORED_SCHEMA_TYPES
        )
        matched += sum(
            min(count, predicted_types.get(kind, 0))
            for kind, count in gold_types.items()
        )
        predicted_count += sum(predicted_types.values())
        gold_count += sum(gold_types.values())
    precision = matched / max(predicted_count, 1)
    recall = matched / max(gold_count, 1)
    return {
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "matched_type_count": matched,
        "predicted": predicted_count,
        "gold": gold_count,
        "meaning": "event type/count score if every emitted type had an exact gold range",
    }


def _row_group_diagnosis(
    predictions: Sequence[
        tuple[str, Any, HeadPrediction, Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    dimensions: dict[str, dict[str, list[tuple[int, int]]]] = {
        "upstream_transcription_match": defaultdict(list),
        "mapping_correct": defaultdict(list),
        "path_margin": defaultdict(list),
        "repeat_copy_state": defaultdict(list),
        "duration": defaultdict(list),
    }
    subtype_support: Counter[str] = Counter()
    class_support: Counter[str] = Counter()
    margin_index = FEATURE_NAMES.index("upstream_path_margin")
    for _sample, labeled, prediction, metadata in predictions:
        for index, (row, name) in enumerate(
            zip(labeled.rows, prediction.layer2)
        ):
            predicted_index = LAYER2_CLASSES.index(name)
            target_index = int(labeled.layer2[index])
            class_support[LAYER2_CLASSES[target_index]] += 1
            dimensions["upstream_transcription_match"][
                (
                    "gap"
                    if row.kind == "gap"
                    else "matched"
                    if int(labeled.target_event_index[index]) >= 0
                    else "unmatched"
                )
            ].append((predicted_index, target_index))
            dimensions["mapping_correct"][
                "correct" if bool(labeled.mapping_correct[index]) else "incorrect"
            ].append((predicted_index, target_index))
            margin = float(row.features[margin_index])
            margin_key = (
                "gap"
                if row.kind == "gap"
                else "negative"
                if margin < 0.0
                else "0_to_1"
                if margin < 1.0
                else "ge_1"
            )
            dimensions["path_margin"][margin_key].append(
                (predicted_index, target_index)
            )
            dimensions["repeat_copy_state"][
                "copy_or_replay" if row.is_copy else "ordinary"
            ].append((predicted_index, target_index))
            dimensions["duration"][str(metadata.get("duration", "unknown"))].append(
                (predicted_index, target_index)
            )
            if bool(labeled.rhythm_mask[index]):
                subtype_support[
                    ("none", "short", "long")[
                        int(labeled.rhythm_subtype[index])
                    ]
                ] += 1

    def typed_metric(values: Sequence[tuple[int, int]]) -> dict[str, Any]:
        predicted = np.asarray([value[0] for value in values])
        target = np.asarray([value[1] for value in values])
        correct = int(np.sum((predicted == target) & (target != 0)))
        n_predicted = int(np.sum(predicted != 0))
        n_target = int(np.sum(target != 0))
        precision = correct / max(n_predicted, 1)
        recall = correct / max(n_target, 1)
        return {
            "rows": len(values),
            "typed_error_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "precision": precision,
            "recall": recall,
            "predicted_errors": n_predicted,
            "target_errors": n_target,
        }

    return {
        name: {
            group: typed_metric(values)
            for group, values in sorted(groups.items())
        }
        for name, groups in dimensions.items()
    } | {
        "class_support": dict(sorted(class_support.items())),
        "rhythm_subtype_support_after_mapping_gate": dict(
            sorted(subtype_support.items())
        ),
    }


def evaluate_v2(
    cache_path: Path,
    output_dir: Path,
    data_fingerprint: str,
    upstream: Mapping[str, Any],
    legacy_schema_path: Path,
) -> dict[str, Any]:
    event_report = base.evaluate(
        cache_path=cache_path,
        output_dir=output_dir,
        data_fingerprint=data_fingerprint,
        upstream=upstream,
        expected_val_rows=358,
    )
    gold = _audited_labels(cache_path)
    predicted_documents, predicted_rows = _prediction_documents(
        cache_path, output_dir, "predicted"
    )
    oracle_documents, _oracle_rows = _prediction_documents(
        cache_path, output_dir, "oracle"
    )
    rule_documents, rule_rows = _rule_documents(cache_path)
    schema_thresholds, schema_threshold_validation = (
        _calibrate_schema_thresholds(cache_path, predicted_rows, gold)
    )
    for variant in ("predicted", "oracle"):
        _store_schema_calibration(
            output_dir / variant / "best.pt",
            schema_thresholds,
            schema_threshold_validation,
        )
    predicted_documents = _documents_for_predictions(
        cache_path,
        "predicted",
        predicted_rows,
        schema_thresholds=schema_thresholds,
    )
    oracle_documents = _documents_for_predictions(
        cache_path,
        "oracle",
        _oracle_rows,
        schema_thresholds=schema_thresholds,
    )
    schema_path = output_dir / "validation_predictions_schema_1_2.jsonl"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for sample in sorted(predicted_documents):
            stream.write(json.dumps(predicted_documents[sample], sort_keys=True))
            stream.write("\n")
    os.replace(temporary, schema_path)
    schema_metrics = {
        "calibration": {
            "thresholds": schema_thresholds,
            "validation": schema_threshold_validation,
            "policy": "per-type official exclusive F1, precision tie-break",
            "calibrated_on": "validation_only",
        },
        "predicted_upstream": _official_summary(predicted_documents, gold),
        "oracle_upstream": _official_summary(oracle_documents, gold),
        "rules_heuristics": _official_summary(rule_documents, gold),
        "v1_emitted_documents": _official_summary(
            _legacy_documents(legacy_schema_path), gold
        ),
        "predicted_type_count_perfect_projection_ceiling": (
            _perfect_projection_ceiling(predicted_documents, gold)
        ),
        "metric_sanity_oracle_documents": _official_summary(
            {
                sample: {"labels": list(labels)}
                for sample, labels in gold.items()
            },
            gold,
        ),
    }
    diagnosis = {
        "event_rows": _row_group_diagnosis(predicted_rows),
        "score_range_conversion": {
            "v1_reported_proxy_f1": 0.051752478922530355,
            "v1_official_emitted_f1": schema_metrics[
                "v1_emitted_documents"
            ]["micro"]["f1"],
            "v2_official_emitted_f1": schema_metrics[
                "predicted_upstream"
            ]["micro"]["f1"],
            "v2_range_only_ignore_type_f1": schema_metrics[
                "predicted_upstream"
            ]["range_only_ignore_type"]["f1"],
            "perfect_projection_type_count_ceiling_f1": schema_metrics[
                "predicted_type_count_perfect_projection_ceiling"
            ]["f1"],
            "causes": [
                "v1 used a per-row mapping proxy instead of official exclusive pitch-list assignment",
                "v1 hard-coded pad_notes=2 while audited labels use pad_notes=1",
                "v1 emitted one label per row, fragmenting contiguous operations",
                "v1 duplicated repeated-pass decisions instead of one source-range repetition",
            ],
        },
        "source_render": {
            "source": {"MozartClConcertoA": 358},
            "audio_render": {"soundfont_v1": 358},
            "note": "The verified validation split has one source work and render.",
        },
    }
    _atomic_json(output_dir / "diagnosis.json", diagnosis)
    event_report["official_schema_1_2"] = schema_metrics
    event_report["predicted_upstream"]["breakdown_v2"] = diagnosis["event_rows"]
    event_report["current_rules_heuristics"] = evaluate_predictions(rule_rows)
    event_report["schema_1_2_predictions"] = str(schema_path.resolve())
    event_report["schema_version"] = "align-frozen-error-heads-evaluation-v2"
    _atomic_json(output_dir / "evaluation.json", event_report)
    return event_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ready-marker",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/DATA_READY.json"),
    )
    parser.add_argument(
        "--upstream-checkpoint",
        type=Path,
        default=Path(
            "runs/joint-audit-v2/end-to-end-v2/"
            "weak-note-continuation-optimized/joint_decoder.pt"
        ),
    )
    parser.add_argument(
        "--v1-cache",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/error-heads-v1/examples.sqlite"),
    )
    parser.add_argument(
        "--canonical-targets",
        type=Path,
        default=Path(
            "data-audit/joint-outputraw-full-v1/canonical_dev_targets.sqlite"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/joint-outputraw-full-v1/error-heads-v2"),
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=Path("runs/TRAINING_RESOURCE_STATUS.json"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--phase", choices=("all", "prepare", "train", "evaluate"), default="all"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    ready = verify_data_ready(args.ready_marker)
    if (
        int(ready["counts"]["train"]) != 4544
        or int(ready["counts"]["val"]) != 358
        or int(ready["counts"]["locked_test_metadata_only"]) != 4022
        or ready["verification"]["test_targets_materialized"] is not False
        or ready["verification"]["test_features_materialized"] is not False
    ):
        raise ValueError("V2 requires the exact sealed 4,544/358 release")
    packed_metadata = json.loads(
        (
            Path(str(ready["paths"]["packed_root"])) / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    upstream, frozen_model, lattice_config = base._validate_completed_upstream(
        args.upstream_checkpoint, packed_metadata
    )
    del frozen_model, lattice_config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "examples.sqlite"
    migration = migrate_v1_cache(
        args.v1_cache, cache_path, args.canonical_targets
    )
    resource_snapshot = (
        json.loads(args.resource_status.read_text(encoding="utf-8"))
        if args.resource_status.is_file()
        else {}
    )
    active_gpu = (resource_snapshot.get("leases") or {}).get("gpu")
    config = {
        "schema_version": OUTPUT_SCHEMA,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data_fingerprint": ready["hashes"]["pack_id"],
        "split_counts": {"train": 4544, "val": 358},
        "locked_test_metadata_rows": 4022,
        "locked_test_touched": False,
        "upstream": upstream,
        "upstream_frozen": True,
        "cache_migration": migration,
        "target_policy": {
            "schema_version": TARGET_SCHEMA,
            "unpaired_transcriptions": "suppressed_without_audited_error",
            "rhythm_requires_exact_mapping": True,
            "replay_restart_rhythm_excluded": True,
            "intonation_masked": True,
            "gold_available_at_inference": False,
        },
        "schema_projection": {
            "official_exclusive_metric": True,
            "pad_notes": 1,
            "contiguous_operation_merging": True,
            "extra_clean_neighbors": True,
            "repetition_source_ranges": True,
        },
        "resource_coordination": {
            "device": "cpu",
            "observed_gpu_lease": active_gpu,
            "contends_with_gpu": False,
        },
        "production_weights_modified": False,
    }
    _atomic_json(args.output_dir / "config.json", config)
    if args.phase == "prepare":
        return

    data_fingerprint = str(ready["hashes"]["pack_id"])
    if args.phase in {"all", "train"}:
        train_clips = base._load_clips(
            cache_path, split="train", variant="predicted"
        )
        arrays = stack_labeled_rows([value[1] for value in train_clips])
        benchmark = base._benchmark(
            arrays, output_dir=args.output_dir, seed=args.seed
        )
        benchmark["loader"]["source"] = "migrated_v2_SQLite_cache"
        benchmark["loader"]["packed_extraction_workers"] = 0
        _atomic_json(args.output_dir / "benchmark.json", benchmark)
        for variant in ("predicted", "oracle"):
            base._train_variant(
                variant=variant,
                cache_path=cache_path,
                output_dir=args.output_dir,
                data_fingerprint=data_fingerprint,
                upstream=upstream,
                epochs=max(1, args.epochs),
                seed=args.seed,
                benchmark=benchmark,
                resume=args.resume,
            )
    if args.phase == "train":
        return

    evaluation = evaluate_v2(
        cache_path,
        args.output_dir,
        data_fingerprint,
        upstream,
        args.v1_cache.parent / "validation_predictions_schema_1_2.jsonl",
    )
    v1 = json.loads(
        (args.v1_cache.parent / "report.json").read_text(encoding="utf-8")
    )["evaluation"]
    predicted = evaluation["predicted_upstream"]
    rules = evaluation["current_rules_heuristics"]
    official = evaluation["official_schema_1_2"]
    layer2_delta = (
        predicted["full_typed_error_f1"]
        - float(v1["predicted_upstream"]["full_typed_error_f1"])
    )
    rhythm_delta_v1 = (
        predicted["layer3"]["rhythm"]["f1"]
        - float(v1["predicted_upstream"]["layer3"]["rhythm"]["f1"])
    )
    rhythm_delta_rules = (
        predicted["layer3"]["rhythm"]["f1"]
        - rules["layer3"]["rhythm"]["f1"]
    )
    schema_delta_v1 = (
        official["predicted_upstream"]["micro"]["f1"]
        - official["v1_emitted_documents"]["micro"]["f1"]
    )
    schema_delta_rules = (
        official["predicted_upstream"]["micro"]["f1"]
        - official["rules_heuristics"]["micro"]["f1"]
    )
    gates = {
        "layer2_delta_vs_v1": layer2_delta,
        "rhythm_delta_vs_v1": rhythm_delta_v1,
        "rhythm_delta_vs_rules": rhythm_delta_rules,
        "schema_delta_vs_v1_official": schema_delta_v1,
        "schema_delta_vs_rules_official": schema_delta_rules,
        "required_material_delta": 0.02,
    }
    passed = all(
        (
            layer2_delta >= 0.02,
            rhythm_delta_v1 >= 0.02,
            rhythm_delta_rules >= 0.02,
            schema_delta_v1 >= 0.02,
            schema_delta_rules >= 0.02,
        )
    )
    queue = {
        "schema_version": "align-error-heads-staged-unfreezing-queue-v1",
        "state": "not_required" if passed else "queued_waiting_for_gpu",
        "blocked_by": active_gpu.get("track") if active_gpu else None,
        "launch_condition": "GPU lease absent and frozen v2 baseline below promotion gates",
        "command": [
            "scripts/train_error_heads_v2.py",
            "--phase",
            "train",
            "--output-dir",
            str(args.output_dir),
            "--resume",
        ],
        "upstream_mutation_allowed": False,
        "note": (
            "Staged unfreezing needs a separately versioned implementation; "
            "this queue never modifies production or the active upstream checkpoint."
        ),
    }
    _atomic_json(args.output_dir / "staged_unfreezing_queue.json", queue)
    report = {
        "schema_version": REPORT_SCHEMA,
        "train_rows": 4544,
        "validation_rows": 358,
        "data_fingerprint": data_fingerprint,
        "evaluation": evaluation,
        "diagnosis": str((args.output_dir / "diagnosis.json").resolve()),
        "promotion_gate": {
            **gates,
            "full_validation_completed": True,
            "passed_metric": passed,
            "integration_tests_required": True,
            "production_promotion_performed": False,
        },
        "staged_unfreezing": queue,
        "locked_test_touched": False,
    }
    _atomic_json(args.output_dir / "report.json", report)
    print(
        "v2_layer2_f1="
        f"{predicted['full_typed_error_f1']:.6f} "
        f"v2_rhythm_f1={predicted['layer3']['rhythm']['f1']:.6f} "
        "v2_schema_f1="
        f"{official['predicted_upstream']['micro']['f1']:.6f} "
        f"promoted={passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
