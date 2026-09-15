"""Replace DataCreate agent labels with the current ALIGN error pipeline.

This writes ``labels_agent.json`` for every real-audio sample. Previous agent
documents are archived under the run directory, then overwritten. Human
``labels.json`` files are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import eval_datacreate_current as base
import train_error_heads_v5 as v5
from alignmodel.joint.candidates import (
    add_score_repeat_hints,
    basic_pitch_candidate_union,
)
from alignmodel.joint.error_heads import (
    build_inference_rows,
    direct_operation_probabilities,
    direct_rhythm_probabilities,
    infer_error_heads,
    schema12_document_v3,
)
from alignmodel.joint.index import ScoreEventIndex
from train_error_heads_v3 import _config_from_json


AGENT_LABEL_FILENAME = "labels_agent.json"
AGENT_ANNOTATOR_ID = "cursor_agent_error_heads_v5"
POLICY_NAME = "hybrid_max_f1"
LABEL_FIELDS = (
    "id",
    "source",
    "start_time",
    "end_time",
    "type",
    "severity",
    "deviation_cents",
    "deviation_ms",
    "measure_number",
    "note_id",
    "comment",
    "repeats_label_range",
    "score_part",
    "pitches",
    "note_ids",
    "core_note_ids",
    "extra_copies",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def agent_label_from_prediction(
    raw: Mapping[str, Any], index: int
) -> dict[str, Any]:
    start = float(raw["start_time"])
    end = max(float(raw["end_time"]), start + 0.001)
    score_part = raw.get("score_part")
    note_ids = list(raw.get("note_ids") or [])
    core_ids = list(raw.get("core_note_ids") or [])
    if not core_ids and isinstance(score_part, Mapping):
        core_start = score_part.get("core_start_note_index")
        core_end = score_part.get("core_end_note_index")
        if core_start is not None and core_end is not None and note_ids:
            padded_start = int(score_part.get("start_note_index") or 0)
            lo = max(0, int(core_start) - padded_start)
            hi = max(lo, int(core_end) - padded_start + 1)
            core_ids = note_ids[lo:hi]
    label = {
        "id": f"agent_{index:04d}",
        "source": "agent",
        "start_time": round(start, 4),
        "end_time": round(end, 4),
        "type": str(raw["type"]),
        "severity": 2,
        "deviation_cents": raw.get("deviation_cents"),
        "deviation_ms": raw.get("deviation_ms"),
        "measure_number": raw.get("measure_number"),
        "note_id": raw.get("note_id") or (core_ids[0] if core_ids else None),
        "comment": (
            "ALIGN error-heads v5 hybrid_max_f1 prediction on the DataCreate "
            "test take."
        ),
        "repeats_label_range": raw.get("repeats_label_range"),
        "score_part": dict(score_part) if isinstance(score_part, Mapping) else None,
        "pitches": list(raw["pitches"]) if raw.get("pitches") is not None else None,
        "note_ids": note_ids or None,
        "core_note_ids": core_ids or None,
        "extra_copies": raw.get("extra_copies"),
    }
    if label["measure_number"] is None and isinstance(score_part, Mapping):
        label["measure_number"] = score_part.get("start_measure")
    return {key: label[key] for key in LABEL_FIELDS}


def agent_document_from_prediction(
    document: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    labels = [
        agent_label_from_prediction(label, index)
        for index, label in enumerate(document.get("labels") or [])
        if isinstance(label, Mapping)
    ]
    counts = Counter(label["type"] for label in labels)
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": AGENT_ANNOTATOR_ID,
        "self_reported": [],
        "labels": labels,
        "agent_labeling": {
            "method": "align_error_heads_v5_hybrid_max_f1",
            "transcriber": "basic-pitch-frozen",
            "uses_project_alignment_or_error_models": True,
            "policy": POLICY_NAME,
            "replaced_previous_agent_labels": True,
            "kept_counts_by_type": dict(sorted(counts.items())),
            "pipeline": dict(document.get("pipeline") or {}),
            **dict(metadata),
        },
    }


def _sample_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "performance_audio.wav").is_file()
        and (path / "verified_score.musicxml").is_file()
    )


def _label_one(
    sample: Path,
    output: Path,
    stack: Mapping[str, Any],
    selector: Mapping[str, Any],
    decode_config: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    features, cache = base._cache_features(sample, output)
    score = tuple(
        ScoreEventIndex.from_musicxml(sample / "verified_score.musicxml").events
    )
    if not score:
        raise ValueError("verified score has no sounding events")
    admitted = tuple(
        add_score_repeat_hints(
            [
                value
                for value in basic_pitch_candidate_union(
                    features,
                    configs=base._frontend_configs(stack["joint_payload"]),
                    minimum_confidence=0.0,
                )
                if value.confidence >= stack["minimum_confidence"]
            ],
            score,
        )
    )
    if not admitted:
        raise ValueError("score/audio mismatch: no admitted candidates")
    path = stack["lattice"].decode(admitted, score)
    rows = build_inference_rows(stack["lattice"], admitted, score, path)
    events = tuple(path.joint_events(admitted))
    learned = infer_error_heads(
        stack["heads_model"],
        rows,
        stack["heads_payload"]["thresholds"],
        device="cpu",
    )
    direct, _diagnostics = direct_operation_probabilities(rows, events, score)
    clip = {
        "rows": rows,
        "learned_prediction": learned,
        "direct_probabilities": direct,
        "direct_rhythm_probabilities": direct_rhythm_probabilities(rows),
    }
    prediction = v5._blend(
        clip,
        weight=float(metadata["direct_weight"]),
        direct_source=str(metadata["direct_source"]),
        selector=selector,
    )
    document = schema12_document_v3(
        sample.name, rows, prediction, score, decode_config
    )
    agent = agent_document_from_prediction(document, metadata=metadata)
    prediction_path = output / "predictions" / f"{sample.name}.json"
    base._atomic_json(prediction_path, document)
    return {
        "sample": sample.name,
        "document": agent,
        "prediction_path": prediction_path,
        "feature_cache": cache,
        "runtime_seconds": time.perf_counter() - started,
        "label_count": len(agent["labels"]),
        "counts_by_type": dict(
            agent["agent_labeling"]["kept_counts_by_type"]
        ),
    }


def _archive_previous(sample: Path, archive_root: Path) -> dict[str, Any] | None:
    current = sample / AGENT_LABEL_FILENAME
    if not current.is_file():
        return None
    destination = archive_root / f"{sample.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current, destination)
    previous = json.loads(current.read_text(encoding="utf-8"))
    return {
        "sample": sample.name,
        "archived": str(destination),
        "annotator_id": previous.get("annotator_id"),
        "method": (previous.get("agent_labeling") or {}).get("method"),
        "label_count": len(previous.get("labels") or []),
    }


def main(argv: Sequence[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[2]
    align = root / "align-model"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=root / "DataCreate" / "samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=align
        / "runs"
        / "datacreate-agent-labels-error-heads-v5-20260915",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-audit-v2"
        / "end-to-end-v2"
        / "weak-note-continuation-optimized"
        / "joint_decoder.pt",
    )
    parser.add_argument(
        "--error-heads-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "error-heads-v2"
        / "predicted"
        / "best.pt",
    )
    parser.add_argument(
        "--active-checkpoint",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "training-v1-optimized"
        / "last_checkpoint.pt",
    )
    parser.add_argument(
        "--v5-selector",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "error-heads-v5"
        / "selector.json",
    )
    parser.add_argument(
        "--v5-decode-config",
        type=Path,
        default=align
        / "runs"
        / "joint-outputraw-full-v1"
        / "error-heads-v5"
        / "decode_config.json",
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    marker = output / "label_manifest.json"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {marker}")
    samples = _sample_dirs(args.samples.resolve())
    if not samples:
        raise FileNotFoundError(f"No DataCreate samples in {args.samples}")

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    stack = base._load_completed_stack(args)
    selector = base._json(args.v5_selector.resolve())
    decode_payload = base._json(args.v5_decode_config.resolve())
    policy = decode_payload[POLICY_NAME]
    decode_config = _config_from_json(policy["schema_config"])
    metadata = {
        "direct_source": policy["direct_source"],
        "direct_weight": policy["direct_weight"],
        "joint_checkpoint": str(args.joint_checkpoint.resolve()),
        "joint_checkpoint_sha256": stack["selection"]["joint_aligner_decoder"][
            "checkpoint_sha256"
        ],
        "error_heads_checkpoint": str(args.error_heads_checkpoint.resolve()),
        "error_heads_checkpoint_sha256": stack["selection"]["error_heads_v2"][
            "checkpoint_sha256"
        ],
        "selector": str(args.v5_selector.resolve()),
        "selector_sha256": base._sha256(args.v5_selector.resolve()),
        "decode_config": str(args.v5_decode_config.resolve()),
        "decode_config_sha256": base._sha256(args.v5_decode_config.resolve()),
    }

    archive_root = output / "previous-agent-labels"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    totals: Counter[str] = Counter()
    archived = []
    failures: list[dict[str, str]] = []
    sys.path.insert(0, str(root / "DataCreate" / "src"))
    from datacreate.validation import validate_labels_file

    for index, sample in enumerate(samples, 1):
        try:
            previous = _archive_previous(sample, archive_root)
            if previous is not None:
                archived.append(previous)
            result = _label_one(
                sample, output, stack, selector, decode_config, metadata
            )
            destination = sample / AGENT_LABEL_FILENAME
            destination.write_text(
                json.dumps(result["document"], indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            errors = validate_labels_file(destination)
            if errors:
                raise ValueError("; ".join(errors))
            totals.update(result["counts_by_type"])
            rows.append(
                {
                    "sample": sample.name,
                    "status": "succeeded",
                    "label_count": result["label_count"],
                    "counts_by_type": result["counts_by_type"],
                    "prediction": str(result["prediction_path"]),
                    "prediction_sha256": base._sha256(result["prediction_path"]),
                    "agent_labels_sha256": base._sha256(destination),
                    "runtime_seconds": result["runtime_seconds"],
                    "previous": previous,
                }
            )
            print(
                f"{index:02d}/{len(samples)} {sample.name}: "
                f"{result['label_count']} agent labels",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep remaining samples
            failures.append({"sample": sample.name, "error": str(exc)})
            print(
                f"{index:02d}/{len(samples)} {sample.name}: ERROR {exc}",
                flush=True,
            )

    report = {
        "schema_version": "align-datacreate-agent-labels-v5-v1",
        "created_utc": _utc(),
        "annotator_id": AGENT_ANNOTATOR_ID,
        "output_filename": AGENT_LABEL_FILENAME,
        "policy": POLICY_NAME,
        "sample_count": len(samples),
        "succeeded": len(samples) - len(failures),
        "failed": len(failures),
        "kept_labels_by_type": dict(sorted(totals.items())),
        "archived_previous_agent_labels": len(archived),
        "human_labels_modified": False,
        "paper_updated": False,
        "failures": failures,
        "model_selection": stack["selection"],
        "metadata": metadata,
        "samples": rows,
    }
    base._atomic_json(marker, report)
    report_copy = args.samples.resolve() / "agent_labeling_report.json"
    report_copy.write_text(
        json.dumps(
            {
                "sample_count": report["sample_count"],
                "succeeded": report["succeeded"],
                "failed": report["failed"],
                "output_filename": AGENT_LABEL_FILENAME,
                "annotator_id": AGENT_ANNOTATOR_ID,
                "method": "align_error_heads_v5_hybrid_max_f1",
                "kept_labels_by_type": report["kept_labels_by_type"],
                "archived_previous_agent_labels": report[
                    "archived_previous_agent_labels"
                ],
                "failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json.loads(report_copy.read_text(encoding="utf-8")), indent=2), flush=True)
    print(marker, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
