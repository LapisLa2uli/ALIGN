from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import audit_training_data as audit
from alignmodel.joint.grammar_mapper_v2 import decode_grammar_mapper
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import JointMetricSample, evaluate_joint_dataset
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.training_resources import resource_lease
from rebuild_monotonic_render_supervision_v3 import _monotonic_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_dict(event: JointEvent) -> dict:
    return {
        "pitch": event.pitch,
        "start": event.start,
        "end": event.end,
        "score_span": event.score_span,
        "relationship": event.relationship,
        "copy_pass": event.copy_pass,
        "origin_relationship": event.origin_relationship,
        "rendered_index": event.rendered_index,
        "source_indices": event.source_indices,
        "confidence": event.confidence,
    }


def _event(row: dict) -> JointEvent:
    return JointEvent(
        pitch=int(row["pitch"]),
        start=float(row["start"]),
        end=float(row["end"]),
        score_span=(
            tuple(int(value) for value in row["score_span"])
            if row["score_span"] is not None
            else None
        ),
        relationship=str(row["relationship"]),
        copy_pass=int(row["copy_pass"]),
        origin_relationship=row.get("origin_relationship"),
        rendered_index=row.get("rendered_index"),
        source_indices=tuple(int(value) for value in row.get("source_indices") or []),
        confidence=float(row.get("confidence", 1.0)),
    )


def _lease(args: argparse.Namespace, role: str):
    expected_rows = 4022 if args.split == "test" else 358
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track=f"mel-mapper-v6-{role}",
        command=[str(value) for value in __import__("sys").argv],
        metadata={
            "rows": expected_rows,
            "locked_test": args.split == "test",
        },
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    return lease


def freeze(args: argparse.Namespace) -> None:
    lease = _lease(args, "freeze")
    ready = verify_data_ready(args.ready_marker)
    manifest = json.loads(
        Path(str(ready["paths"]["manifest"])).read_text(encoding="utf-8")
    )
    with args.frontend_predictions.open("r", encoding="utf-8") as stream:
        predictions = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }
    selected_rows = manifest[args.split]
    if set(predictions) != {row["sample"] for row in selected_rows}:
        raise ValueError("Frozen frontend validation population mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "predictions.jsonl"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output_dir, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for position, row in enumerate(selected_rows, 1):
            sample = row["sample"]
            score = ScoreEventIndex.from_musicxml(
                Path(row["sample_dir"]) / "verified_score.musicxml"
            ).events
            candidates = tuple(
                JointCandidate(
                    int(note["pitch"]),
                    float(note["start"]),
                    float(note["end"]),
                    float(note["confidence"]),
                )
                for note in predictions[sample]["notes"]
            )
            mapped, grammar = decode_grammar_mapper(candidates, score)
            stream.write(
                json.dumps(
                    {
                        "sample": sample,
                        "source": row["source"],
                        "score_event_count": len(score),
                        "events": [_event_dict(event) for event in mapped],
                        "grammar": grammar,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if position == 1 or position % 25 == 0:
                print(f"freeze={position}/{len(selected_rows)}", flush=True)
    os.replace(temporary, output)
    document = {
        "schema_version": "align-mel-mapper-v6-freeze",
        "rows": len(selected_rows),
        "split": args.split,
        "predictions": str(output.resolve()),
        "predictions_sha256": _sha256(output),
        "frontend_predictions_sha256": _sha256(args.frontend_predictions),
        "manifest_sha256": ready["hashes"]["manifest_sha256"],
        "score_only_inference": True,
        "gold_accessed": False,
        "locked_test_touched": args.split == "test",
    }
    (args.output_dir / "freeze_manifest.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    atexit.unregister(lease.__exit__)
    lease.__exit__(None, None, None)


def _type_samples(samples: list[JointMetricSample], kind: str):
    output = []
    for sample in samples:
        predicted = tuple(
            event
            for event in sample.predicted
            if ("copy" if event.is_copy else event.relationship) == kind
        )
        target = tuple(
            event
            for event in sample.target
            if ("copy" if event.is_copy else event.relationship) == kind
        )
        output.append(
            JointMetricSample(
                predicted=predicted,
                target=target,
                source=sample.source,
                score_event_count=sample.score_event_count,
            )
        )
    return output


def score(args: argparse.Namespace) -> None:
    lease = _lease(args, "score")
    ready = verify_data_ready(args.ready_marker)
    manifest = json.loads(
        Path(str(ready["paths"]["manifest"])).read_text(encoding="utf-8")
    )
    freeze_manifest = json.loads(
        (args.output_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    prediction_path = Path(freeze_manifest["predictions"])
    if _sha256(prediction_path) != freeze_manifest["predictions_sha256"]:
        raise ValueError("Frozen mapper prediction hash mismatch")
    with prediction_path.open("r", encoding="utf-8") as stream:
        predicted = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }
    samples = []
    repair_totals = {}
    per_clip = []
    selected_rows = manifest[args.split]
    for position, row in enumerate(selected_rows, 1):
        sample_dir = Path(row["sample_dir"])
        note_map = json.loads(
            (sample_dir / "note_map.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (sample_dir / "metadata.json").read_text(encoding="utf-8")
        )
        midi = audit._midi_events(sample_dir / "performance_audio.mid")
        if midi is None:
            raise ValueError(f"Unreadable validation MIDI: {row['sample']}")
        shift = audit._inferred_midi_shift(note_map, midi, metadata)
        rendered, repair = _monotonic_rows(note_map, midi, shift)
        if repair["edit_cost"] != 0 or repair["unmapped_performed_notes"] != 0:
            raise ValueError(f"Validation repair is not exact: {row['sample']}")
        rebuilt = dict(note_map)
        rebuilt["rendered_notes"] = rendered
        rebuilt["rendered_note_count"] = len(rendered)
        target_index = ScoreEventIndex.from_musicxml(
            sample_dir / "verified_score.musicxml", rebuilt
        )
        frozen = predicted[row["sample"]]
        sample = JointMetricSample(
            predicted=tuple(_event(value) for value in frozen["events"]),
            target=target_index.rendered_events,
            source=row["source"],
            predicted_deletions=frozenset(),
            target_deletions=target_index.deleted_event_indices,
            score_event_count=len(target_index.events),
        )
        samples.append(sample)
        metric = evaluate_joint_dataset([sample])["aggregate"][
            "official_note_wise"
        ]
        per_clip.append(
            (
                float(metric["credit"]),
                int(metric["predicted"]),
                int(metric["gold"]),
            )
        )
        for key, value in repair.items():
            if isinstance(value, int):
                repair_totals[key] = repair_totals.get(key, 0) + value
        if position == 1 or position % 25 == 0:
            print(f"score={position}/{len(selected_rows)}", flush=True)
    metrics = evaluate_joint_dataset(samples)
    generator = np.random.default_rng(20260917)
    bootstrap = []
    for _ in range(args.bootstrap_replicates):
        selected = generator.integers(0, len(per_clip), len(per_clip))
        credit = sum(per_clip[index][0] for index in selected)
        prediction_count = sum(per_clip[index][1] for index in selected)
        gold_count = sum(per_clip[index][2] for index in selected)
        precision = credit / max(prediction_count, 1)
        recall = credit / max(gold_count, 1)
        bootstrap.append(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    per_type = {
        kind: evaluate_joint_dataset(_type_samples(samples, kind))["aggregate"][
            "official_note_wise"
        ]
        for kind in ("match", "copy", "substitute", "extra")
    }
    report = {
        "schema_version": "align-mel-mapper-v6-validation",
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "matching_policy": "exclusive one-to-one exact canonical identity; exact type 1.0; wrong type at exact location 0.5",
        "rows": len(selected_rows),
        "split": args.split,
        "metrics": metrics,
        "per_type": per_type,
        "bootstrap_95": {
            "lower": float(np.quantile(bootstrap, 0.025)),
            "median": float(np.quantile(bootstrap, 0.5)),
            "upper": float(np.quantile(bootstrap, 0.975)),
            "replicates": args.bootstrap_replicates,
        },
        "repair_totals": repair_totals,
        "freeze_manifest_sha256": _sha256(
            args.output_dir / "freeze_manifest.json"
        ),
        "timestamp_metrics_used_for_selection": False,
        "locked_test_touched": args.split == "test",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    integrity = {
        "schema_version": "align-mel-mapper-v6-integrity",
        "report_sha256": _sha256(report_path),
        "predictions_sha256": _sha256(prediction_path),
        "candidate_sha256": _sha256(args.candidate),
        "locked_test_touched": args.split == "test",
    }
    (args.output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    atexit.unregister(lease.__exit__)
    lease.__exit__(None, None, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "score"))
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--frontend-predictions", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    {"freeze": freeze, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
