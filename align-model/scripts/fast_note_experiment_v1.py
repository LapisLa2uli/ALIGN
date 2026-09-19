"""Leakage-safe audit, caching, inference, and scoring for fast-note data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import audit_training_data as audit
from alignmodel.joint.baseline import current_note_aligner_baseline
from alignmodel.joint.grammar_mapper_v2 import decode_grammar_mapper
from alignmodel.joint.global_ornament_lineage_v2 import (
    reconstruct_global_ornament_lineage,
)
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    evaluate_joint_events,
    pair_exact_pitch_onset,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import (
    CACHE_SCHEMA_VERSION,
    MelDecodeConfig,
    MelFrontendConfig,
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)
from alignmodel.transcription.mel_v1_data import (
    MelPackedCache,
    _canonical_json,
    _new_index,
)
from alignmodel.transcription.ornament_multipitch_v1 import (
    SCHEMA_VERSION as MULTIPITCH_SCHEMA_VERSION,
    MultiPitchConfig,
    MultiPitchDecodeConfig,
    OrnamentMultiPitchTranscriber,
    decode_multipitch_notes,
    infer_multipitch_probabilities,
)
SCHEMA_VERSION = "align-fast-note-experiment-v1"
SEED = 20260918
REQUIRED = (
    "metadata.json",
    "labels.json",
    "candidates.json",
    "note_map.json",
    "verified_score.musicxml",
    "performance_score.musicxml",
    "performance_audio.wav",
    "reference_audio.wav",
    "performance_audio.mid",
)
HASHED = REQUIRED + ("reference_audio.mid",)


def _atomic_json(path: Path, value: Any) -> None:
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


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return _hash_text(f"{seed}:{value}")


def _event_dict(event: JointEvent) -> dict[str, Any]:
    return {
        "pitch": event.pitch,
        "start": event.start,
        "end": event.end,
        "score_span": list(event.score_span) if event.score_span else None,
        "relationship": event.relationship,
        "copy_pass": event.copy_pass,
        "origin_relationship": event.origin_relationship,
        "rendered_index": event.rendered_index,
        "source_indices": list(event.source_indices),
        "confidence": event.confidence,
    }


def _event(value: Mapping[str, Any]) -> JointEvent:
    return JointEvent(
        pitch=int(value["pitch"]),
        start=float(value["start"]),
        end=float(value["end"]),
        score_span=(
            tuple(int(item) for item in value["score_span"])
            if value.get("score_span") is not None
            else None
        ),
        relationship=str(value["relationship"]),
        copy_pass=int(value.get("copy_pass") or 0),
        origin_relationship=value.get("origin_relationship"),
        rendered_index=value.get("rendered_index"),
        source_indices=tuple(int(item) for item in value.get("source_indices") or ()),
        confidence=float(value.get("confidence", 1.0)),
    )


def _duration_band(seconds: float) -> str:
    if seconds < 20:
        return "lt20"
    if seconds < 30:
        return "20to30"
    if seconds < 45:
        return "30to45"
    return "ge45"


def _select_stratified(
    rows: Sequence[dict[str, Any]], count: int, *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["stratum"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: _stable_key(seed, str(row["source_id"])))
    selected: list[dict[str, Any]] = []
    names = sorted(groups)
    while len(selected) < count:
        progressed = False
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"Requested {count} rows but only selected {len(selected)}")
    ids = {str(row["sample"]) for row in selected}
    return selected, [row for row in rows if str(row["sample"]) not in ids]


def _polyphonic_onsets(events: Sequence[JointEvent]) -> int:
    """Count true simultaneous different-pitch attacks, not legato overlap."""

    return sum(
        abs(left.start - right.start) <= 0.001
        and left.pitch != right.pitch
        and min(left.end, right.end) - max(left.start, right.start) >= 0.010
        for index, left in enumerate(events)
        for right in events[index + 1 :]
        if right.start <= left.start + 0.001
    )


def _audit_one(sample_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample = sample_dir.name
    reasons: list[str] = []
    missing = [name for name in REQUIRED if not (sample_dir / name).is_file()]
    reasons.extend(f"required_file_missing:{name}" for name in missing)
    metadata: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    note_map: dict[str, Any] = {}
    for name, destination in (
        ("metadata.json", metadata),
        ("labels.json", labels),
        ("note_map.json", note_map),
    ):
        if name in missing:
            continue
        try:
            value = json.loads((sample_dir / name).read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("top level is not an object")
            destination.update(value)
        except Exception as error:
            reasons.append(f"{name}_unreadable:{type(error).__name__}")
    if metadata.get("schema_version") != "1.2":
        reasons.append("metadata_schema_not_1.2")
    if labels.get("schema_version") != "1.2":
        reasons.append("labels_schema_not_1.2")
    if note_map.get("kind") != "synth_note_lineage":
        reasons.append("note_map_kind_invalid")

    hashes: dict[str, str] = {}
    for name in HASHED:
        path = sample_dir / name
        if path.is_file():
            try:
                hashes[name] = sha256_file(path)
            except OSError:
                reasons.append(f"hash_failed:{name}")
    wav = (
        audit._read_wave_info(sample_dir / "performance_audio.wav")
        if "performance_audio.wav" not in missing
        else None
    )
    reference_wav = (
        audit._read_wave_info(sample_dir / "reference_audio.wav")
        if "reference_audio.wav" not in missing
        else None
    )
    if wav is None:
        reasons.append("performance_wav_unreadable")
    elif (
        int(wav["sample_rate"]) != 22050
        or int(wav["channels"]) != 1
        or float(wav["duration_sec"]) <= 0
    ):
        reasons.append("performance_wav_format_invalid")
    if reference_wav is None:
        reasons.append("reference_wav_unreadable")

    midi = (
        audit._midi_events(sample_dir / "performance_audio.mid")
        if "performance_audio.mid" not in missing
        else None
    )
    if midi is None:
        reasons.append("performance_midi_unreadable")
    if wav is not None and midi is not None:
        tail = float(wav["duration_sec"]) - float(midi["duration_sec"])
        if not -0.10 <= tail <= 8.00:
            reasons.append("wav_midi_duration_mismatch")

    verified_pitches: list[int] = []
    performance_pitches: list[int] = []
    for name, output in (
        ("verified_score.musicxml", verified_pitches),
        ("performance_score.musicxml", performance_pitches),
    ):
        try:
            output.extend(audit._musicxml_pitches(sample_dir / name))
        except Exception:
            reasons.append(f"{name}_unreadable")

    issue_log = audit.IssueLog()
    validated = None
    legacy_valid = False
    if note_map and midi is not None:
        validated, legacy_valid, _ = audit._validate_note_map(
            sample,
            sample_dir / "note_map.json",
            verified_pitches,
            performance_pitches,
            midi,
            metadata,
            issue_log,
        )
        if not legacy_valid:
            reasons.append("canonical_note_map_invalid")

    target: list[dict[str, Any]] = []
    deleted: list[int] = []
    repair_stats: dict[str, Any] = {}
    score_event_count = 0
    if validated is not None and midi is not None:
        try:
            shift = audit._inferred_midi_shift(validated, midi, metadata)
            reconstruction = reconstruct_global_ornament_lineage(
                validated,
                sample_dir / "performance_score.musicxml",
                sample_dir / "verified_score.musicxml",
                midi,
                written_shift=shift,
            )
            rebuilt = reconstruction.lineage
            rendered = list(rebuilt["rendered_notes"])
            repair_stats = reconstruction.stats
            if int(shift) != 2:
                reasons.append("bb_written_pitch_shift_not_plus2")
            if (
                repair_stats["raw_unmatched"] != 0
                or repair_stats["template_unmatched"] != 0
                or repair_stats["unmapped_performed_notes"] != 0
                or repair_stats["canonical_backward_steps"] != 0
                or not repair_stats["unique_rendered_identity"]
            ):
                reasons.append("global_monotonic_reconstruction_not_exact")
            index = ScoreEventIndex.from_musicxml(
                sample_dir / "verified_score.musicxml", rebuilt
            )
            events = index.rendered_events
            score_event_count = len(index.events)
            if [event.rendered_index for event in events] != list(range(len(events))):
                reasons.append("rendered_identity_not_contiguous")
            if len(events) != len(rendered) or any(
                event.pitch != int(row["pitch_midi_written"])
                or abs(event.start - float(row["start_sec"])) > 1e-6
                or abs(event.end - float(row["end_sec"])) > 1e-6
                for event, row in zip(events, rendered)
            ):
                reasons.append("canonical_target_reconstruction_mismatch")
            if any(not 52 <= event.pitch <= 100 for event in events):
                reasons.append("target_pitch_outside_track_b_range")
            target = [_event_dict(event) for event in events]
            deleted = sorted(index.deleted_event_indices)
        except Exception as error:
            reasons.append(
                "canonical_reconstruction_failed:"
                f"{type(error).__name__}:{str(error)[:160]}"
            )

    for key, expected in (
        ("sample_rate", 22050),
        ("sounding_transpose", -2),
        ("midi_pitch_space", "sounding"),
        ("audio_pitch_space", "sounding"),
        ("effective_audio_transpose", 2),
    ):
        if metadata.get(key) != expected:
            reasons.append(f"pitch_metadata_invalid:{key}")
    xml = (
        (sample_dir / "performance_score.musicxml").read_text(
            encoding="utf-8", errors="replace"
        )
        if (sample_dir / "performance_score.musicxml").is_file()
        else ""
    )
    ornament = any(
        token in xml for token in ("<grace", "<ornaments", "<trill-mark", "<mordent", "<turn")
    )
    target_events = tuple(_event(value) for value in target)
    copy_count = sum(event.is_copy for event in target_events)
    if target and bool(metadata.get("repeated")) != bool(copy_count):
        reasons.append("repeat_lineage_mismatch")
    polyphonic_onsets = _polyphonic_onsets(target_events)
    label_rows = [row for row in labels.get("labels") or () if isinstance(row, dict)]
    row = {
        "sample": sample,
        "sample_dir": str(sample_dir.resolve()),
        "eligible": not reasons,
        "exclusion_reasons": sorted(set(reasons)),
        "source_id": f"score:{hashes.get('verified_score.musicxml', sample)}",
        "duration_sec": float(wav["duration_sec"]) if wav else None,
        "sample_rate": int(wav["sample_rate"]) if wav else None,
        "audio_render": str(metadata.get("audio_render") or "unknown"),
        "effective_audio_transpose": int(
            metadata.get("effective_audio_transpose", 999)
        ),
        "ornament_present": ornament,
        "repeated": bool(copy_count),
        "copy_events": copy_count,
        "polyphonic_onsets": polyphonic_onsets,
        "score_event_count": score_event_count,
        "rendered_event_count": len(target),
        "label_count": len(label_rows),
        "label_types": sorted(
            {str(item.get("type")) for item in label_rows if item.get("type")}
        ),
        "repair_stats": repair_stats,
        "hashes": hashes,
    }
    row["stratum"] = (
        f"{_duration_band(float(row['duration_sec'] or 0.0))}:"
        f"ornament={int(ornament)}"
    )
    return row, [{"sample": sample, "target": target, "deleted": deleted}]


def command_audit(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        print(manifest_path)
        return
    output.mkdir(parents=True, exist_ok=True)
    audit_cache = output / ".audited_rows.json.gz"
    if audit_cache.exists():
        with gzip.open(audit_cache, "rt", encoding="utf-8") as stream:
            cached = json.load(stream)
        rows = list(cached["rows"])
        targets = list(cached["targets"])
        bundles = [None] * int(cached["bundles_discovered"])
    else:
        bundles = sorted(
            (
                path
                for path in args.dataset.resolve().iterdir()
                if path.is_dir() and (path / "metadata.json").is_file()
            ),
            key=lambda path: path.name,
        )
        rows = []
        targets = []
        with resource_lease(
            args.resource_status,
            "cpu_validation",
            track="fast-note-v1-audit-freeze",
            command=[sys.executable, *sys.argv],
            metadata={"dataset": str(args.dataset.resolve()), "rows": len(bundles)},
        ):
            for position, path in enumerate(bundles, 1):
                row, target = _audit_one(path)
                rows.append(row)
                targets.extend(target)
                if position == 1 or position % 25 == 0 or position == len(bundles):
                    print(
                        f"audit={position}/{len(bundles)} sample={path.name}",
                        flush=True,
                    )
        with gzip.open(audit_cache, "wt", encoding="utf-8") as stream:
            json.dump(
                {
                    "bundles_discovered": len(bundles),
                    "rows": rows,
                    "targets": targets,
                },
                stream,
            )

    # Fail closed on any score, content, or audio duplicate.  A connected
    # duplicate component contributes one deterministic canonical row.
    eligible = [row for row in rows if row["eligible"]]
    parent = {str(row["sample"]): str(row["sample"]) for row in eligible}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for hash_name in (
        "verified_score.musicxml",
        "performance_score.musicxml",
        "performance_audio.wav",
    ):
        seen: dict[str, str] = {}
        for row in eligible:
            value = str(row["hashes"].get(hash_name) or "")
            if value in seen:
                union(str(row["sample"]), seen[value])
            else:
                seen[value] = str(row["sample"])
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        components[root(str(row["sample"]))].append(row)
    duplicate_rows = 0
    for values in components.values():
        values.sort(key=lambda row: str(row["sample"]))
        for duplicate in values[1:]:
            duplicate["eligible"] = False
            duplicate["exclusion_reasons"] = ["duplicate_score_content_or_audio"]
            duplicate_rows += 1

    eligible = [row for row in rows if row["eligible"]]
    source_counts = Counter(str(row["source_id"]) for row in eligible)
    for row in eligible:
        if source_counts[str(row["source_id"])] != 1:
            row["eligible"] = False
            row["exclusion_reasons"] = ["ambiguous_source_group"]
    eligible = [row for row in rows if row["eligible"]]
    reason_counts = Counter(
        reason
        for row in rows
        for reason in row["exclusion_reasons"]
    )
    if len(eligible) < 50:
        output.mkdir(parents=True, exist_ok=True)
        blocker = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "reason": "fewer_than_50_exact_eligible_unique_source_rows",
            "bundles_discovered": len(bundles),
            "eligible_unique_rows": len(eligible),
            "duplicate_rows_excluded": duplicate_rows,
            "exclusion_reasons": dict(reason_counts),
            "global_ornament_lineage_policy": (
                "align-global-ornament-lineage-v2 exact generator-template "
                "round trip with zero unmatched units and no backward identity"
            ),
            "rows": rows,
        }
        _atomic_json(output / "BLOCKED_AUDIT.json", blocker)
        raise RuntimeError(f"Only {len(eligible)} eligible unique-source rows")

    benchmark, remainder = _select_stratified(eligible, 50, seed=args.seed)
    heldout_count = min(100, max(8, len(remainder) // 6))
    open_validation, remainder = _select_stratified(
        remainder, heldout_count, seed=args.seed + 1
    )
    calibration, training = _select_stratified(
        remainder, heldout_count, seed=args.seed + 2
    )
    split_by_sample = {
        **{str(row["sample"]): "benchmark50" for row in benchmark},
        **{str(row["sample"]): "open_validation" for row in open_validation},
        **{str(row["sample"]): "calibration" for row in calibration},
        **{str(row["sample"]): "train" for row in training},
    }
    for row in rows:
        row["split"] = split_by_sample.get(str(row["sample"]), "excluded")

    target_by_sample = {str(row["sample"]): row for row in targets}
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "canonical_targets.jsonl.gz"
    with gzip.open(target_path, "wt", encoding="utf-8") as stream:
        for row in sorted(eligible, key=lambda item: str(item["sample"])):
            stream.write(json.dumps(target_by_sample[str(row["sample"])], sort_keys=True))
            stream.write("\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "seed": args.seed,
        "selection_policy": (
            "eligibility-only then deterministic source-unique round-robin over "
            "fixed duration/ornament strata; no labels, outcomes, or predictions"
        ),
        "dedupe_policy": "connected exact verified-score/performance-score/audio hashes",
        "benchmark_openings": 1,
        "benchmark_tuning_allowed": False,
        "orn_v1_v2_lockboxes_accessed": False,
        "mapper_v6_lockbox_accessed": False,
        "prespecified_final_gate": {
            "standalone_open_validation_pitch_sequence_f1_delta_min": 0.02,
            "open_validation_lt120ms_recall_delta_min": 0.05,
            "combined_canonical_f1_regression_max": 0.02,
        },
        "counts": Counter(str(row["split"]) for row in rows),
        "rows": rows,
        "canonical_targets": str(target_path.resolve()),
        "canonical_targets_sha256": sha256_file(target_path),
    }
    manifest["counts"] = dict(manifest["counts"])
    _atomic_json(manifest_path, manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "bundles_discovered": len(bundles),
        "eligible_unique_rows": len(eligible),
        "duplicate_rows_excluded": duplicate_rows,
        "exclusion_reasons": dict(reason_counts),
        "split_counts": manifest["counts"],
        "ornament_rows": sum(bool(row["ornament_present"]) for row in eligible),
        "repeat_rows": sum(bool(row["repeated"]) for row in eligible),
        "polyphonic_onset_rows": sum(
            int(row["polyphonic_onsets"]) > 0 for row in eligible
        ),
        "global_monotonic_policy": True,
        "fail_closed": True,
        "manifest_sha256": sha256_file(manifest_path),
        "targets_sha256": sha256_file(target_path),
    }
    _atomic_json(output / "audit_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported fast-note manifest")
    target = Path(manifest["canonical_targets"])
    if sha256_file(target) != manifest["canonical_targets_sha256"]:
        raise ValueError("Canonical target hash mismatch")
    return manifest


def _load_targets(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    with gzip.open(Path(manifest["canonical_targets"]), "rt", encoding="utf-8") as stream:
        return {
            str(row["sample"]): row
            for row in (json.loads(line) for line in stream if line.strip())
        }


def command_cache(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    targets = _load_targets(manifest)
    requested = tuple(args.splits.split(","))
    rows = [
        row for row in manifest["rows"]
        if str(row["split"]) in requested
    ]
    split_map = {
        "train": "train",
        "calibration": "val",
        "open_validation": "val",
        "benchmark50": "val",
    }
    destination = args.output.resolve()
    frontend = MelFrontendConfig()
    if destination.exists():
        cache = MelPackedCache(destination, deep=True)
        print(json.dumps(cache.metadata, indent=2))
        cache.close()
        return
    staging = destination.with_name(f".{destination.name}.building")
    staging.mkdir(parents=True, exist_ok=True)
    index_path = staging / "index.sqlite"
    connection = _new_index(index_path)
    committed = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    shard_rows = max(1, args.shard_rows)
    if committed % shard_rows:
        raise ValueError("Interrupted cache is not at a shard boundary")
    with resource_lease(
        args.resource_status,
        "gpu",
        track=f"fast-note-v1-cache-{'-'.join(requested)}",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(rows), "splits": requested},
    ):
        try:
            for chunk_start in range(committed, len(rows), shard_rows):
                chunk = rows[chunk_start : chunk_start + shard_rows]
                arrays: list[np.ndarray] = []
                records = []
                frame_offset = 0
                for row in chunk:
                    sample = str(row["sample"])
                    source = Path(row["sample_dir"])
                    audio_path = source / "performance_audio.wav"
                    if sha256_file(audio_path) != row["hashes"]["performance_audio.wav"]:
                        raise ValueError(f"Audio hash changed: {sample}")
                    audio = load_audio_mono(audio_path, frontend.sample_rate)
                    mel, normalization = extract_log_mel(
                        audio, frontend, device=args.device
                    )
                    contiguous = np.ascontiguousarray(mel.T, dtype="<f2")
                    raw = contiguous.tobytes(order="C")
                    arrays.append(contiguous)
                    target = [
                        {
                            "pitch": event["pitch"],
                            "start_sec": event["start"],
                            "end_sec": event["end"],
                            "rendered_index": event["rendered_index"],
                        }
                        for event in targets[sample]["target"]
                    ]
                    records.append(
                        (
                            sample,
                            split_map[str(row["split"])],
                            str(row["source_id"]),
                            chunk_start // shard_rows,
                            frame_offset,
                            int(contiguous.shape[0]),
                            float(row["duration_sec"]),
                            str(row["audio_render"]),
                            int(row["effective_audio_transpose"]),
                            str(row["hashes"]["performance_audio.wav"]),
                            hashlib.sha256(raw).hexdigest(),
                            json.dumps(normalization, sort_keys=True),
                            sqlite3.Binary(zlib.compress(_canonical_json(target), level=1)),
                        )
                    )
                    frame_offset += int(contiguous.shape[0])
                shard = chunk_start // shard_rows
                shard_path = staging / f"shard-{shard:05d}.mel.float16.bin"
                temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
                with temporary.open("wb") as stream:
                    for array in arrays:
                        stream.write(array.tobytes(order="C"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, shard_path)
                connection.executemany(
                    "INSERT INTO records(sample,split,source,shard,frame_offset,"
                    "frame_count,duration_sec,audio_render,effective_audio_transpose,"
                    "audio_sha256,mel_sha256,normalization,target) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    records,
                )
                connection.commit()
                done = chunk_start + len(chunk)
                print(f"cache={done}/{len(rows)}", flush=True)
        finally:
            connection.close()
    shards = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(staging.glob("shard-*.bin"))
    ]
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "frontend_config": frontend.to_dict(),
        "dtype": "float16",
        "layout": "time_major_contiguous",
        "shard_rows": shard_rows,
        "split_counts": {
            split: sum(split_map[str(row["split"])] == split for row in rows)
            for split in ("train", "val")
        },
        "record_count": len(rows),
        "source": {
            "release": SCHEMA_VERSION,
            "pack_id": sha256_file(args.manifest),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "source_splits": requested,
            "locked_test_materialized": False,
            "targets_included": True,
        },
        "index": {
            "name": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
        },
        "shards": shards,
    }
    metadata["pack_id"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
    _atomic_json(staging / "metadata.json", metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    print(json.dumps(metadata, indent=2), flush=True)


def command_infer(args: argparse.Namespace) -> None:
    cache = MelPackedCache(args.cache, deep=False)
    model, frontend, decode, _payload = load_mel_checkpoint(
        args.checkpoint, args.device
    )
    if frontend != cache.frontend:
        raise ValueError("Checkpoint/cache frontend mismatch")
    if args.min_confidence is not None:
        decode = replace(decode, min_confidence=args.min_confidence)
    key = hashlib.sha256(
        _canonical_json(
            {
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "cache_pack_id": cache.pack_id,
                "decode": decode.to_dict(),
                "window_frames": args.window_frames,
                "overlap_frames": args.overlap_frames,
            }
        )
    ).hexdigest()
    output = args.output_dir.resolve() / f"{key}.predictions.jsonl"
    freeze = output.with_suffix(".manifest.json")
    if output.exists() and freeze.exists():
        print(freeze)
        return
    records = cache.records("val", include_targets=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with resource_lease(
        args.resource_status,
        "gpu",
        track=f"fast-note-v1-infer-{args.name}",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(records), "checkpoint": str(args.checkpoint.resolve())},
    ):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=args.output_dir, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            started = time.perf_counter()
            for position, record in enumerate(records, 1):
                probability = infer_mel_probabilities(
                    model,
                    np.asarray(cache.mel(record), np.float32),
                    args.device,
                    window_frames=args.window_frames,
                    overlap_frames=args.overlap_frames,
                    batch_size=args.batch_size,
                )
                notes = decode_mel_notes(
                    probability,
                    midi_min=model.config.midi_min,
                    hop_sec=frontend.hop_sec,
                    config=decode,
                )
                stream.write(
                    json.dumps(
                        {
                            "sample": record.sample,
                            "source": record.source,
                            "notes": [note.to_dict() for note in notes],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                if position == 1 or position % 25 == 0 or position == len(records):
                    rate = position / max(time.perf_counter() - started, 1e-9)
                    print(
                        f"infer={position}/{len(records)} rows_per_sec={rate:.2f}",
                        flush=True,
                    )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    cache.close()
    document = {
        "schema_version": SCHEMA_VERSION,
        "name": args.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cache": str(args.cache.resolve()),
        "cache_pack_id": cache.pack_id,
        "decode_config": decode.to_dict(),
        "predictions": str(output),
        "predictions_sha256": sha256_file(output),
        "rows": len(records),
        "score_input": False,
        "gold_input": False,
        "pitch_output_space": "Bb written MIDI",
    }
    _atomic_json(freeze, document)
    print(json.dumps(document, indent=2), flush=True)


def command_infer_multipitch(args: argparse.Namespace) -> None:
    cache = MelPackedCache(args.cache, deep=False)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if payload.get("schema_version") != MULTIPITCH_SCHEMA_VERSION:
        raise ValueError("Unsupported multi-pitch checkpoint")
    frontend = MelFrontendConfig.from_dict(payload["frontend_config"])
    if frontend != cache.frontend:
        raise ValueError("Checkpoint/cache frontend mismatch")
    model = OrnamentMultiPitchTranscriber(
        MultiPitchConfig.from_dict(payload["model_config"])
    ).to(args.device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    decode = MultiPitchDecodeConfig.from_dict(payload.get("decode_config"))
    key = hashlib.sha256(
        _canonical_json(
            {
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "cache_pack_id": cache.pack_id,
                "decode": decode.to_dict(),
                "window_frames": args.window_frames,
                "overlap_frames": args.overlap_frames,
            }
        )
    ).hexdigest()
    output = args.output_dir.resolve() / f"{key}.predictions.jsonl"
    freeze = output.with_suffix(".manifest.json")
    if output.exists() and freeze.exists():
        print(freeze)
        return
    records = cache.records("val", include_targets=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with resource_lease(
        args.resource_status,
        "gpu",
        track=f"fast-note-v1-infer-multipitch-{args.name}",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(records), "checkpoint": str(args.checkpoint.resolve())},
    ):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=args.output_dir, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for position, record in enumerate(records, 1):
                probability = infer_multipitch_probabilities(
                    model,
                    np.asarray(cache.mel(record), np.float32),
                    args.device,
                    window_frames=args.window_frames,
                    overlap_frames=args.overlap_frames,
                    batch_size=args.batch_size,
                )
                notes = decode_multipitch_notes(
                    probability,
                    midi_min=model.midi_min,
                    hop_sec=frontend.hop_sec,
                    config=decode,
                )
                stream.write(
                    json.dumps(
                        {
                            "sample": record.sample,
                            "source": record.source,
                            "notes": [note.to_dict() for note in notes],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                if position == 1 or position % 10 == 0 or position == len(records):
                    print(f"infer={position}/{len(records)}", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    cache_pack_id = cache.pack_id
    cache.close()
    document = {
        "schema_version": SCHEMA_VERSION,
        "model_schema_version": MULTIPITCH_SCHEMA_VERSION,
        "name": args.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cache": str(args.cache.resolve()),
        "cache_pack_id": cache_pack_id,
        "decode_config": decode.to_dict(),
        "predictions": str(output),
        "predictions_sha256": sha256_file(output),
        "rows": len(records),
        "score_input": False,
        "gold_input": False,
        "pitch_output_space": "Bb written MIDI",
    }
    _atomic_json(freeze, document)
    print(json.dumps(document, indent=2), flush=True)


def _candidates(notes: Sequence[Mapping[str, Any]]) -> list[JointCandidate]:
    return [
        JointCandidate(
            pitch=int(note["pitch"]),
            start=float(note["start"]),
            end=float(note["end"]),
            confidence=float(note.get("confidence", 1.0)),
        )
        for note in notes
    ]


def _transcription_event(event: JointEvent, rendered_index: int) -> JointEvent:
    if event.score_span is None:
        return replace(
            event, relationship="extra", copy_pass=0, rendered_index=rendered_index
        )
    return replace(
        event, relationship="match", origin_relationship="match", copy_pass=0
    )


def _target_transcription_event(event: JointEvent) -> JointEvent:
    if event.score_span is None:
        return event
    return replace(
        event, relationship="match", origin_relationship="match", copy_pass=0
    )


def _lcs(left: Sequence[int], right: Sequence[int]) -> int:
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            row[column] = (
                previous + 1 if item == other else max(row[column], row[column - 1])
            )
            previous = saved
    return row[-1]


def _prf(credit: float, predicted: int, support: int) -> dict[str, Any]:
    precision = credit / predicted if predicted else 0.0
    recall = credit / support if support else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "credit": credit,
        "predicted": predicted,
        "support": support,
    }


def _bootstrap(
    counts: Sequence[tuple[float, int, int]], seed: int, replicates: int
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = generator.integers(0, len(counts), len(counts))
        values.append(
            _prf(
                sum(counts[index][0] for index in selected),
                sum(counts[index][1] for index in selected),
                sum(counts[index][2] for index in selected),
            )["f1"]
        )
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _typed(samples: Sequence[JointMetricSample], kind: str) -> dict[str, Any]:
    selected = []
    for sample in samples:
        if kind == "missed_note":
            selected.append(
                JointMetricSample(
                    predicted=(),
                    target=(),
                    source=sample.source,
                    predicted_deletions=sample.predicted_deletions,
                    target_deletions=sample.target_deletions,
                    score_event_count=sample.score_event_count,
                )
            )
            continue
        predicted = tuple(
            event for event in sample.predicted
            if ("copy" if event.is_copy else event.relationship) == kind
        )
        target = tuple(
            event for event in sample.target
            if ("copy" if event.is_copy else event.relationship) == kind
        )
        selected.append(
            JointMetricSample(
                predicted=predicted,
                target=target,
                source=sample.source,
                score_event_count=sample.score_event_count,
            )
        )
    return evaluate_joint_dataset(selected)["aggregate"]["official_note_wise"]


def command_score(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    targets = _load_targets(manifest)
    selected_rows = {
        str(row["sample"]): row
        for row in manifest["rows"]
        if str(row["split"]) == args.split
    }
    freeze = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    prediction_path = Path(freeze["predictions"])
    if sha256_file(prediction_path) != freeze["predictions_sha256"]:
        raise ValueError("Prediction hash mismatch")
    predictions = {
        str(row["sample"]): row
        for row in (
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if set(predictions) != set(selected_rows):
        raise ValueError("Prediction population does not equal frozen split")

    canonical_samples: list[JointMetricSample] = []
    combined_samples: list[JointMetricSample] = []
    oracle_samples: list[JointMetricSample] = []
    canonical_counts: list[tuple[float, int, int]] = []
    combined_counts: list[tuple[float, int, int]] = []
    oracle_counts: list[tuple[float, int, int]] = []
    sequence = [0, 0, 0]
    duration = {
        "lt80ms": [0, 0],
        "80to120ms": [0, 0],
        "120to180ms": [0, 0],
        "ge180ms": [0, 0],
    }
    same_pitch_splits = 0
    grouped: dict[str, list[JointMetricSample]] = defaultdict(list)
    ornament_grouped: dict[str, list[JointMetricSample]] = defaultdict(list)
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track=f"fast-note-v1-score-{args.name}-{args.split}",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(selected_rows), "split": args.split},
    ):
        for position, sample in enumerate(sorted(selected_rows), 1):
            row = selected_rows[sample]
            target_row = targets[sample]
            target = tuple(_event(value) for value in target_row["target"])
            score = ScoreEventIndex.from_musicxml(
                Path(row["sample_dir"]) / "verified_score.musicxml"
            ).events
            notes = predictions[sample]["notes"]
            candidates = _candidates(notes)
            primary = tuple(
                JointEvent(
                    pitch=item.pitch,
                    start=item.start,
                    end=item.end,
                    score_span=None,
                    relationship="extra",
                    confidence=item.confidence,
                )
                for item in candidates
            )
            baseline, _ = current_note_aligner_baseline(candidates, score)
            canonical_sample = JointMetricSample(
                predicted=tuple(
                    _transcription_event(event, index)
                    for index, event in enumerate(baseline)
                ),
                target=tuple(_target_transcription_event(event) for event in target),
                source=str(row["source_id"]),
                score_event_count=len(score),
            )
            canonical_samples.append(canonical_sample)
            combined, _ = decode_grammar_mapper(candidates, score)
            combined_sample = JointMetricSample(
                predicted=combined,
                target=target,
                source=str(row["source_id"]),
                target_deletions=frozenset(target_row["deleted"]),
                score_event_count=len(score),
            )
            combined_samples.append(combined_sample)
            oracle_candidates = [
                JointCandidate(
                    pitch=event.pitch,
                    start=event.start,
                    end=event.end,
                    confidence=1.0,
                )
                for event in target
            ]
            oracle, _ = decode_grammar_mapper(oracle_candidates, score)
            oracle_sample = JointMetricSample(
                predicted=oracle,
                target=target,
                source=str(row["source_id"]),
                target_deletions=frozenset(target_row["deleted"]),
                score_event_count=len(score),
            )
            oracle_samples.append(oracle_sample)
            grouped[str(row["stratum"])].append(combined_sample)
            ornament_grouped[
                f"ornament={int(bool(row['ornament_present']))}"
            ].append(combined_sample)
            for metric_sample, counts in (
                (canonical_sample, canonical_counts),
                (combined_sample, combined_counts),
                (oracle_sample, oracle_counts),
            ):
                metric = evaluate_joint_events(
                    metric_sample.predicted,
                    metric_sample.target,
                    predicted_deletions=metric_sample.predicted_deletions,
                    target_deletions=metric_sample.target_deletions,
                    score_event_count=metric_sample.score_event_count,
                )["official_note_wise"]
                counts.append(
                    (
                        float(metric["credit"]),
                        int(metric["predicted"]),
                        int(metric["gold"]),
                    )
                )
            pred_pitch = [event.pitch for event in primary]
            gold_pitch = [event.pitch for event in target]
            sequence[0] += _lcs(pred_pitch, gold_pitch)
            sequence[1] += len(pred_pitch)
            sequence[2] += len(gold_pitch)
            pairs = pair_exact_pitch_onset(primary, target, tolerance_sec=0.050)
            paired_gold = {right for _, right in pairs}
            for label, predicate in (
                ("lt80ms", lambda value: value < 0.080),
                ("80to120ms", lambda value: 0.080 <= value < 0.120),
                ("120to180ms", lambda value: 0.120 <= value < 0.180),
                ("ge180ms", lambda value: value >= 0.180),
            ):
                indices = {
                    index
                    for index, event in enumerate(target)
                    if predicate(event.end - event.start)
                }
                duration[label][0] += len(indices & paired_gold)
                duration[label][1] += len(indices)
            same_pitch_splits += sum(
                left.pitch == right.pitch and right.start - left.end <= 0.100
                for left, right in zip(primary, primary[1:])
            )
            if position == 1 or position % 10 == 0 or position == len(selected_rows):
                print(f"score={position}/{len(selected_rows)}", flush=True)
    canonical = evaluate_joint_dataset(canonical_samples)["aggregate"][
        "official_note_wise"
    ]
    combined_evaluation = evaluate_joint_dataset(combined_samples)
    combined = combined_evaluation["aggregate"]["official_note_wise"]
    oracle = evaluate_joint_dataset(oracle_samples)["aggregate"]["official_note_wise"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "name": args.name,
        "split": args.split,
        "rows": len(selected_rows),
        "metric": {
            "unit": "canonical score-event identity",
            "matching": (
                "exclusive one-to-one; exact location/type=1.0; "
                "exact location/wrong type=0.5; wrong location=0"
            ),
            "timestamp_metrics": "diagnostic_only",
        },
        "score_agnostic_transcriber": {
            "pitch_sequence": _prf(*sequence),
            "count_ratio": sequence[1] / max(sequence[2], 1),
            "short_note_recall": {
                key: {
                    "matched": value[0],
                    "support": value[1],
                    "recall": value[0] / max(value[1], 1),
                }
                for key, value in duration.items()
            },
            "same_pitch_split_count": same_pitch_splits,
            "onset_tolerance_sec": 0.050,
        },
        "canonical_transcriber": {
            **canonical,
            "bootstrap": _bootstrap(canonical_counts, SEED, args.bootstrap_replicates),
        },
        "oracle_note_mapper": {
            **oracle,
            "bootstrap": _bootstrap(oracle_counts, SEED + 1, args.bootstrap_replicates),
        },
        "combined_mapper_v6": {
            **combined,
            "bootstrap": _bootstrap(
                combined_counts, SEED + 2, args.bootstrap_replicates
            ),
            "per_type": {
                kind: _typed(combined_samples, kind)
                for kind in ("match", "copy", "substitute", "extra", "missed_note")
            },
            "per_stratum": {
                name: evaluate_joint_dataset(samples)["aggregate"][
                    "official_note_wise"
                ]
                for name, samples in sorted(grouped.items())
            },
            "per_ornament": {
                name: evaluate_joint_dataset(samples)["aggregate"][
                    "official_note_wise"
                ]
                for name, samples in sorted(ornament_grouped.items())
            },
            "per_source": {
                name: metrics["official_note_wise"]
                for name, metrics in sorted(
                    combined_evaluation["per_source"].items()
                )
            },
        },
        "prediction_manifest": str(args.prediction_manifest.resolve()),
        "prediction_manifest_sha256": sha256_file(args.prediction_manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "selection_use": False,
        "tuning_performed": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


def command_select(args: argparse.Namespace) -> None:
    candidates = []
    for name, checkpoint in (
        ("track_b_initialized", args.initialized),
        ("from_scratch", args.scratch),
    ):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        history = payload.get("history") or []
        calibration = [
            row.get("train_fold_calibration") or row.get("calibration")
            for row in history
            if row.get("train_fold_calibration") or row.get("calibration")
        ]
        if not calibration:
            raise ValueError(f"No calibration metrics in {checkpoint}")
        selected = calibration[-1].get("selected") or calibration[-1]
        score = float(selected["f1"])
        candidates.append(
            {
                "name": name,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "calibration": selected,
                "rank": score
                - 0.04 * abs(
                    float(np.log(max(float(selected["count_ratio"]), 1e-4)))
                ),
            }
        )
    chosen = max(candidates, key=lambda row: (row["rank"], row["name"]))
    document = {
        "schema_version": SCHEMA_VERSION,
        "selection_population": "frozen calibration only",
        "timestamp_metrics_used": False,
        "open_validation_accessed": False,
        "benchmark50_accessed_for_selection": False,
        "candidates": candidates,
        "chosen": chosen,
    }
    _atomic_json(args.output, document)
    print(json.dumps(document, indent=2), flush=True)


def command_freeze_protocol(args: argparse.Namespace) -> None:
    validation = json.loads(args.mapper_validation.read_text(encoding="utf-8"))
    document = {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_benchmark_inference": True,
        "track_b": {
            "checkpoint": str(args.track_b.resolve()),
            "checkpoint_sha256": sha256_file(args.track_b),
            "confidence": 0.8,
            "unchanged": True,
        },
        "mapper_v6": {
            "validation_artifact": str(args.mapper_validation.resolve()),
            "validation_artifact_sha256": sha256_file(args.mapper_validation),
            "declared_mapper": validation["mapper"],
            "declared_global_lineage_policy": validation["target_repair_policy"],
            "grammar_code": str(args.grammar_code.resolve()),
            "grammar_code_sha256": sha256_file(args.grammar_code),
            "global_lineage_code": str(args.lineage_code.resolve()),
            "global_lineage_code_sha256": sha256_file(args.lineage_code),
        },
        "benchmark50": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "openings_authorized": 1,
            "tuning_allowed": False,
        },
        "production_mutation_allowed": False,
        "orn_lockboxes_allowed": False,
    }
    _atomic_json(args.output, document)
    print(json.dumps(document, indent=2), flush=True)


def command_compare(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    manifest = _load_manifest(args.manifest)

    def short_under_120(report: Mapping[str, Any]) -> dict[str, Any]:
        bands = report["score_agnostic_transcriber"]["short_note_recall"]
        matched = int(bands["lt80ms"]["matched"]) + int(
            bands["80to120ms"]["matched"]
        )
        support = int(bands["lt80ms"]["support"]) + int(
            bands["80to120ms"]["support"]
        )
        return {
            "matched": matched,
            "support": support,
            "recall": matched / max(support, 1),
        }

    baseline_short = short_under_120(baseline)
    candidate_short = short_under_120(candidate)
    deltas = {
        "pitch_sequence_f1": (
            candidate["score_agnostic_transcriber"]["pitch_sequence"]["f1"]
            - baseline["score_agnostic_transcriber"]["pitch_sequence"]["f1"]
        ),
        "lt120ms_recall": candidate_short["recall"] - baseline_short["recall"],
        "combined_canonical_f1": (
            candidate["combined_mapper_v6"]["f1"]
            - baseline["combined_mapper_v6"]["f1"]
        ),
    }
    gate = manifest["prespecified_final_gate"]
    checks = {
        "pitch_sequence": deltas["pitch_sequence_f1"]
        >= gate["standalone_open_validation_pitch_sequence_f1_delta_min"],
        "short_note": deltas["lt120ms_recall"]
        >= gate["open_validation_lt120ms_recall_delta_min"],
        "combined": deltas["combined_canonical_f1"]
        >= -gate["combined_canonical_f1_regression_max"],
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "population": "frozen open_validation",
        "rows": baseline["rows"],
        "baseline_report": str(args.baseline.resolve()),
        "baseline_report_sha256": sha256_file(args.baseline),
        "candidate_report": str(args.candidate.resolve()),
        "candidate_report_sha256": sha256_file(args.candidate),
        "baseline": {
            "pitch_sequence_f1": baseline["score_agnostic_transcriber"][
                "pitch_sequence"
            ]["f1"],
            "lt120ms": baseline_short,
            "canonical_transcriber_f1": baseline["canonical_transcriber"]["f1"],
            "combined_mapper_v6_f1": baseline["combined_mapper_v6"]["f1"],
        },
        "candidate": {
            "pitch_sequence_f1": candidate["score_agnostic_transcriber"][
                "pitch_sequence"
            ]["f1"],
            "lt120ms": candidate_short,
            "canonical_transcriber_f1": candidate["canonical_transcriber"]["f1"],
            "combined_mapper_v6_f1": candidate["combined_mapper_v6"]["f1"],
        },
        "deltas": deltas,
        "prespecified_gate": gate,
        "gate_checks": checks,
        "gate_met": all(checks.values()),
        "benchmark50_comparison_authorized": all(checks.values()),
        "selection_changed_after_open_validation": False,
    }
    _atomic_json(args.output, document)
    print(json.dumps(document, indent=2), flush=True)


def command_finalize(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    relative_artifacts = (
        "audit-v1/audit_report.json",
        "audit-v1/split_manifest.json",
        "audit-v1/canonical_targets.jsonl.gz",
        "FROZEN_PROTOCOL.json",
        "FROZEN_CANDIDATE.json",
        "phase-a-current-benchmark50/report.json",
        "cache-benchmark50/metadata.json",
        "cache-train-calibration/metadata.json",
        "cache-open-validation/metadata.json",
        "training/track-b-initialized/best.pt",
        "training/track-b-initialized/history.json",
        "training/from-scratch/best.pt",
        "training/from-scratch/history.json",
        "open-validation/current/report.json",
        "open-validation/selected/report.json",
        "open-validation/COMPARISON.json",
    )
    paths = {name: root / name for name in relative_artifacts}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing final artifacts: {missing}")
    audit_report = json.loads(paths["audit-v1/audit_report.json"].read_text())
    phase_a = json.loads(
        paths["phase-a-current-benchmark50/report.json"].read_text()
    )
    candidate = json.loads(paths["FROZEN_CANDIDATE.json"].read_text())
    selected = json.loads(
        paths["open-validation/selected/report.json"].read_text()
    )
    comparison = json.loads(
        paths["open-validation/COMPARISON.json"].read_text()
    )
    protocol = json.loads(paths["FROZEN_PROTOCOL.json"].read_text())
    status = json.loads(args.resource_status.read_text(encoding="utf-8"))
    fast_leases = {
        name: lease
        for name, lease in status.get("leases", {}).items()
        if str(lease.get("track", "")).startswith("fast-note")
    }
    repo = args.repo_root.resolve()
    commands = [
        (
            "python scripts/fast_note_experiment_v1.py audit --dataset "
            "E:\\outputRaw_fast_1k --output-dir "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\audit-v1 "
            "--resource-status runs\\TRAINING_RESOURCE_STATUS.json --seed 20260918"
        ),
        (
            "python scripts/fast_note_experiment_v1.py cache --manifest "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\audit-v1\\"
            "split_manifest.json --splits benchmark50 --output "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "cache-benchmark50 --resource-status runs\\TRAINING_RESOURCE_STATUS.json "
            "--device cuda --shard-rows 25"
        ),
        (
            "python scripts/fast_note_experiment_v1.py infer --name "
            "current-track-b-benchmark50 --checkpoint runs\\joint-outputraw-full-v1\\"
            "mel-transcriber-v1\\full-training-all4544-v2\\candidate-epoch-018.pt "
            "--cache runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "cache-benchmark50 --output-dir runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\phase-a-current-benchmark50 "
            "--resource-status runs\\TRAINING_RESOURCE_STATUS.json --device cuda "
            "--min-confidence 0.8"
        ),
        (
            "python scripts/fast_note_experiment_v1.py score --name "
            "current-track-b-benchmark50 --manifest runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\audit-v1\\split_manifest.json --split "
            "benchmark50 --prediction-manifest runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\phase-a-current-benchmark50\\"
            "32bfb06cf8847a65d4a8a183aed168d9156101f81aaff5f61654de063042d0e1."
            "predictions.manifest.json "
            "--output runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "phase-a-current-benchmark50\\report.json --resource-status "
            "runs\\TRAINING_RESOURCE_STATUS.json --bootstrap-replicates 1000"
        ),
        (
            "python scripts/fast_note_experiment_v1.py cache --manifest "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\audit-v1\\"
            "split_manifest.json --splits train,calibration --output "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "cache-train-calibration --resource-status "
            "runs\\TRAINING_RESOURCE_STATUS.json --device cuda --shard-rows 37"
        ),
        (
            "python scripts/fast_note_experiment_v1.py cache --manifest "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\audit-v1\\"
            "split_manifest.json --splits open_validation --output "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "cache-open-validation --resource-status "
            "runs\\TRAINING_RESOURCE_STATUS.json --device cuda --shard-rows 8"
        ),
        (
            "python scripts/train_fast_note_multipitch_v1.py --cache "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "cache-train-calibration --track-b runs\\joint-outputraw-full-v1\\"
            "mel-transcriber-v1\\full-training-all4544-v2\\candidate-epoch-018.pt "
            "--output-dir runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "training\\track-b-initialized --resource-status "
            "runs\\TRAINING_RESOURCE_STATUS.json --epochs 12 --batch-size 12 "
            "--crop-frames 1024 --crops-per-clip 6 --learning-rate 0.0003 "
            "--backbone-learning-rate 0.00005 --workers 0 "
            "--checkpoint-every-steps 25 --augmentation-probability 0.35 "
            "--amp bf16 --seed 20260918"
        ),
        (
            "python scripts/train_fast_note_multipitch_v1.py --cache "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "cache-train-calibration --output-dir runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\training\\from-scratch --resource-status "
            "runs\\TRAINING_RESOURCE_STATUS.json --epochs 12 --batch-size 12 "
            "--crop-frames 1024 --crops-per-clip 6 --learning-rate 0.0003 "
            "--workers 0 --checkpoint-every-steps 25 "
            "--augmentation-probability 0.35 --amp bf16 --seed 20260918"
        ),
        (
            "python scripts/fast_note_experiment_v1.py select --initialized "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\training\\"
            "track-b-initialized\\best.pt --scratch runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\training\\from-scratch\\best.pt --output "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "FROZEN_CANDIDATE.json"
        ),
        (
            "python scripts/fast_note_experiment_v1.py infer --name "
            "current-track-b-open-validation --checkpoint runs\\"
            "joint-outputraw-full-v1\\mel-transcriber-v1\\"
            "full-training-all4544-v2\\candidate-epoch-018.pt --cache runs\\"
            "joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "cache-open-validation --output-dir runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\open-validation\\current --resource-status "
            "runs\\TRAINING_RESOURCE_STATUS.json --device cuda --min-confidence 0.8"
        ),
        (
            "python scripts/fast_note_experiment_v1.py infer-multipitch --name "
            "selected-track-b-init-open-validation --checkpoint runs\\"
            "joint-outputraw-full-v1\\fast-note-transcriber-v1\\training\\"
            "track-b-initialized\\best.pt --cache runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\cache-open-validation --output-dir runs\\"
            "joint-outputraw-full-v1\\fast-note-transcriber-v1\\open-validation\\"
            "selected --resource-status runs\\TRAINING_RESOURCE_STATUS.json "
            "--device cuda"
        ),
        (
            "python scripts/fast_note_experiment_v1.py compare --baseline "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "open-validation\\current\\report.json --candidate runs\\"
            "joint-outputraw-full-v1\\fast-note-transcriber-v1\\open-validation\\"
            "selected\\report.json --manifest runs\\joint-outputraw-full-v1\\"
            "fast-note-transcriber-v1\\audit-v1\\split_manifest.json --output "
            "runs\\joint-outputraw-full-v1\\fast-note-transcriber-v1\\"
            "open-validation\\COMPARISON.json"
        ),
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_without_final_benchmark_comparison",
        "audit": audit_report,
        "phase_a": {
            "rows": phase_a["rows"],
            "pitch_sequence": phase_a["score_agnostic_transcriber"][
                "pitch_sequence"
            ],
            "count_ratio": phase_a["score_agnostic_transcriber"]["count_ratio"],
            "short_note_recall": phase_a["score_agnostic_transcriber"][
                "short_note_recall"
            ],
            "canonical_transcriber": phase_a["canonical_transcriber"],
            "oracle_note_mapper": phase_a["oracle_note_mapper"],
            "combined_mapper_v6": phase_a["combined_mapper_v6"],
        },
        "phase_b": {
            "architecture": {
                "frontend": "22050 Hz, 128 mel, hop 256",
                "backbone": "Track B time-preserving spectral encoder + 8-block TCN",
                "output": (
                    "independent 49-pitch activity/onset/offset heads plus "
                    "0..8 polyphony head"
                ),
                "pitch_space": "Bb written MIDI 52..100",
            },
            "training": {
                "train_rows": audit_report["split_counts"]["train"],
                "calibration_rows": audit_report["split_counts"]["calibration"],
                "open_validation_rows": audit_report["split_counts"][
                    "open_validation"
                ],
                "epochs": 12,
                "batch_size": 12,
                "crop_frames": 1024,
                "crops_per_clip": 6,
                "head_learning_rate": 3e-4,
                "initialized_backbone_learning_rate": 5e-5,
                "scratch_backbone_learning_rate": 3e-4,
                "weight_decay": 1e-3,
                "augmentation_probability": 0.35,
                "precision": "bf16",
                "short_note_loss_weights": "<80ms=5, <120ms=3, <180ms=1.8",
            },
            "selection": candidate,
            "open_validation": {
                "pitch_sequence": selected["score_agnostic_transcriber"][
                    "pitch_sequence"
                ],
                "count_ratio": selected["score_agnostic_transcriber"][
                    "count_ratio"
                ],
                "short_note_recall": selected["score_agnostic_transcriber"][
                    "short_note_recall"
                ],
                "canonical_transcriber": selected["canonical_transcriber"],
                "oracle_note_mapper": selected["oracle_note_mapper"],
                "combined_mapper_v6": selected["combined_mapper_v6"],
            },
            "comparison": comparison,
            "final_benchmark_comparison": (
                "not run because the prespecified open-validation gate failed"
            ),
        },
        "integrity": {
            "artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "code": {
                str(path.relative_to(repo)): sha256_file(path)
                for path in (
                    repo / "scripts" / "fast_note_experiment_v1.py",
                    repo / "scripts" / "train_fast_note_multipitch_v1.py",
                    repo / "scripts" / "run_with_resource_lease.py",
                    repo
                    / "src"
                    / "alignmodel"
                    / "transcription"
                    / "ornament_multipitch_v1.py",
                )
            },
            "active_fast_note_leases": fast_leases,
            "track_b_checkpoint_hash_unchanged": (
                sha256_file(Path(protocol["track_b"]["checkpoint"]))
                == protocol["track_b"]["checkpoint_sha256"]
            ),
            "production_weights_written": False,
            "production_config_written": False,
            "orn_v1_v2_lockboxes_accessed": False,
            "mapper_v6_lockbox_accessed": False,
            "benchmark50_used_for_training_or_selection": False,
        },
        "verification": {
            "tests": [
                {
                    "command": (
                        "pytest tests/test_ornament_multipitch_v1.py "
                        "tests/test_mel_transcriber_v1.py "
                        "tests/test_joint_infer.py -q"
                    ),
                    "result": "17 passed",
                },
                {
                    "command": (
                        "pytest tests/test_mel_transcriber_v1.py "
                        "tests/test_joint_infer.py -q"
                    ),
                    "result": "14 passed",
                },
                {
                    "command": (
                        "pytest tests/test_ornament_multipitch_v1.py "
                        "tests/test_mel_transcriber_v1.py -q"
                    ),
                    "result": "14 passed",
                },
            ],
            "ide_lints": "no errors in experiment scripts",
        },
        "reproduction_commands": commands,
    }
    _atomic_json(args.output, document)
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--dataset", type=Path, required=True)
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.add_argument("--resource-status", type=Path, required=True)
    audit_parser.add_argument("--seed", type=int, default=SEED)

    cache_parser = commands.add_parser("cache")
    cache_parser.add_argument("--manifest", type=Path, required=True)
    cache_parser.add_argument("--splits", required=True)
    cache_parser.add_argument("--output", type=Path, required=True)
    cache_parser.add_argument("--resource-status", type=Path, required=True)
    cache_parser.add_argument("--device", default="cuda")
    cache_parser.add_argument("--shard-rows", type=int, default=50)

    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--name", required=True)
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--cache", type=Path, required=True)
    infer_parser.add_argument("--output-dir", type=Path, required=True)
    infer_parser.add_argument("--resource-status", type=Path, required=True)
    infer_parser.add_argument("--device", default="cuda")
    infer_parser.add_argument("--min-confidence", type=float)
    infer_parser.add_argument("--batch-size", type=int, default=4)
    infer_parser.add_argument("--window-frames", type=int, default=2048)
    infer_parser.add_argument("--overlap-frames", type=int, default=512)

    multipitch_parser = commands.add_parser("infer-multipitch")
    multipitch_parser.add_argument("--name", required=True)
    multipitch_parser.add_argument("--checkpoint", type=Path, required=True)
    multipitch_parser.add_argument("--cache", type=Path, required=True)
    multipitch_parser.add_argument("--output-dir", type=Path, required=True)
    multipitch_parser.add_argument("--resource-status", type=Path, required=True)
    multipitch_parser.add_argument("--device", default="cuda")
    multipitch_parser.add_argument("--batch-size", type=int, default=4)
    multipitch_parser.add_argument("--window-frames", type=int, default=2048)
    multipitch_parser.add_argument("--overlap-frames", type=int, default=512)

    score_parser = commands.add_parser("score")
    score_parser.add_argument("--name", required=True)
    score_parser.add_argument("--manifest", type=Path, required=True)
    score_parser.add_argument(
        "--split", choices=("benchmark50", "open_validation"), required=True
    )
    score_parser.add_argument("--prediction-manifest", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--resource-status", type=Path, required=True)
    score_parser.add_argument("--bootstrap-replicates", type=int, default=1000)

    select_parser = commands.add_parser("select")
    select_parser.add_argument("--initialized", type=Path, required=True)
    select_parser.add_argument("--scratch", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)

    protocol_parser = commands.add_parser("freeze-protocol")
    protocol_parser.add_argument("--track-b", type=Path, required=True)
    protocol_parser.add_argument("--mapper-validation", type=Path, required=True)
    protocol_parser.add_argument("--grammar-code", type=Path, required=True)
    protocol_parser.add_argument("--lineage-code", type=Path, required=True)
    protocol_parser.add_argument("--manifest", type=Path, required=True)
    protocol_parser.add_argument("--output", type=Path, required=True)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--run-root", type=Path, required=True)
    finalize_parser.add_argument("--repo-root", type=Path, required=True)
    finalize_parser.add_argument("--resource-status", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {
        "audit": command_audit,
        "cache": command_cache,
        "infer": command_infer,
        "infer-multipitch": command_infer_multipitch,
        "score": command_score,
        "select": command_select,
        "freeze-protocol": command_freeze_protocol,
        "compare": command_compare,
        "finalize": command_finalize,
    }[args.command](args)


if __name__ == "__main__":
    main()
