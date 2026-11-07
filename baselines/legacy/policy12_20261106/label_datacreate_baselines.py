"""Gold-free conversion of baseline MIDI into DataCreate schema 1.2 label sets.

Writes ``labels_polytune.json`` / ``labels_laddersym.json`` beside each sample.
Human ``labels.json`` is never opened or modified. Error spans use the frozen
``notes_to_spans`` parameters from the 2026-09-18 real-test conversion; score
locations come from ``canonical_adapter.adapt`` plus pad-2 GUI fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "baselines" / "common"),
    str(ROOT / "DataCreate" / "src"),
]

from canonical_adapter import PARAMETERS as ADAPTER_PARAMETERS
from canonical_adapter import Note, VERSION as ADAPTER_VERSION, adapt
from datacreate.melody import padded_melody, parse_sounding_notes
from datacreate.validation import validate_labels_file
from eval_bridge import Warner, class_from_name, notes_to_spans, read_pred_midi_tracks

CONVERSION_VERSION = "baseline-datacreate-gui-labels-v1"
PAD_NOTES = 2
SPAN_PARAMETERS = dict(
    tau_wrong=0.10,
    rep_gap=0.35,
    rep_gap_mode="onset",
    rep_gap_ioi_mult=0.0,
    rep_break_on_correct=True,
    rep_trim=3,
    rep_window_min=0.0,
    rep_min_run=3,
    rep_lcs=0.8,
    rep_window_mult=2.0,
    rep_merge_gap=2.0,
)
SPAN_KINDS = {
    "wrong_note": {"wrong"},
    "missed_note": {"missing"},
    "extra_note": {"extra"},
    "repetition": {"copy", "extra"},
}
LABEL_FILES = {"polytune": "labels_polytune.json", "laddersym": "labels_laddersym.json"}
GUI_TYPES = ("wrong_note", "missed_note", "extra_note", "repetition")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deny_gold(event: str, values: tuple[Any, ...]) -> None:
    if event != "open" or not values:
        return
    name = str(values[0]).replace("\\", "/")
    if name.endswith("/labels.json") or name.endswith("\\labels.json"):
        raise PermissionError(f"Gold labels.json must not be read during conversion: {name}")


def load_native(midi_path: Path) -> tuple[dict[str, list], list[Note], list[tuple], list[str]]:
    warn = Warner(quiet=True)
    tracks, _, _ = read_pred_midi_tracks(str(midi_path), warn)
    native = {kind: [] for kind in ("extra", "missing", "correct")}
    adapter_notes: list[Note] = []
    unclassified: list[tuple[float, float, int]] = []
    shift = int(ADAPTER_PARAMETERS["sounding_to_written"])
    for track in tracks:
        kind = class_from_name(track["name"])
        if kind is None and str(track["name"]).strip() and track["notes"]:
            raise ValueError(f"Unknown named MIDI class: {track['name']}")
        for start, end, pitch in track["notes"]:
            sounding = int(pitch)
            written = sounding + shift
            if kind is None:
                unclassified.append((float(start), float(end), sounding))
                adapter_notes.append(Note(float(start), float(end), written, "unclassified"))
                continue
            native[kind].append((float(start), float(end), sounding))
            adapter_notes.append(Note(float(start), float(end), written, kind))
    return native, adapter_notes, unclassified, list(warn.messages)


def _score_fields(notes, indices: Sequence[int]) -> dict[str, Any]:
    core = sorted({int(value) for value in indices if 0 <= int(value) < len(notes)})
    if not core:
        return {}
    first, last = core[0], core[-1]
    span = padded_melody(notes, first, last + 1, PAD_NOTES)
    fields = span.as_fields()
    fields["score_part"]["core_start_note_index"] = first
    fields["score_part"]["core_end_note_index"] = last
    fields["core_note_ids"] = [item.note_id for item in notes[first : last + 1]]
    fields["note_id"] = fields["core_note_ids"][0]
    fields["measure_number"] = span.start_measure
    fields["score_event_indices"] = list(
        range(span.start_note_index, span.end_note_index + 1)
    )
    return fields


def _indices_for_span(kind: str, start: float, end: float, events: Sequence[Mapping[str, Any]]) -> tuple[list[int], int]:
    wanted = SPAN_KINDS[kind]
    indices = []
    copies = 0
    for event in events:
        if event["kind"] not in wanted:
            continue
        if event["end"] <= start - 1e-4 or event["start"] >= end + 1e-4:
            continue
        if event.get("score_index") is not None:
            indices.append(int(event["score_index"]))
        copies = max(copies, int(event.get("copy_pass") or 0))
    return indices, copies


def document_from_midi(
    midi_path: Path,
    score_path: Path,
    *,
    model: str,
    sample_id: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    native, adapter_notes, unclassified, warnings = load_native(midi_path)
    diagnostic: dict[str, Any] = {}
    spans = notes_to_spans(native, SPAN_PARAMETERS, diagnostic)
    sounding = parse_sounding_notes(score_path)
    converted = adapt(adapter_notes, [note.pitch for note in sounding])
    labels = []
    unlocated = 0
    for index, (kind, start, end) in enumerate(spans):
        if kind not in GUI_TYPES:
            continue
        start_t = round(float(start), 4)
        end_t = round(max(float(end), start_t + 0.001), 4)
        label: dict[str, Any] = {
            "id": f"{model}_{index:04d}",
            "source": "agent",
            "start_time": start_t,
            "end_time": end_t,
            "type": kind,
            "severity": 2,
            "deviation_cents": None,
            "deviation_ms": None,
            "measure_number": None,
            "note_id": None,
            "comment": f"{model} baseline prediction converted with {CONVERSION_VERSION}.",
            "repeats_label_range": None,
            "score_part": None,
            "pitches": None,
            "note_ids": None,
            "core_note_ids": None,
            "extra_copies": None,
        }
        indices, copies = _indices_for_span(kind, start_t, end_t, converted["events"])
        fields = _score_fields(sounding, indices)
        if fields:
            label.update(fields)
        else:
            unlocated += 1
        if kind == "repetition":
            label["extra_copies"] = copies if copies > 0 else 1
        labels.append(label)
    counts = Counter(label["type"] for label in labels)
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": f"baseline_{model}",
        "self_reported": [],
        "labels": labels,
        "baseline_labeling": {
            "method": CONVERSION_VERSION,
            "model": model,
            "adapter_version": ADAPTER_VERSION,
            "span_parameters": SPAN_PARAMETERS,
            "pad_notes": PAD_NOTES,
            "replaced_previous_baseline_labels": True,
            "kept_counts_by_type": dict(sorted(counts.items())),
            "native_counts": {key: len(value) for key, value in native.items()},
            "unclassified_count": len(unclassified),
            "unlocated_events": unlocated,
            "adapter_unlocated_events": converted["unlocated_events"],
            "conversion_warnings": warnings,
            "conversion_diagnostics": diagnostic,
            "sample_id": sample_id,
            "written_utc": _utc(),
            **dict(provenance or {}),
        },
    }


def write_sample_document(sample_dir: Path, model: str, document: Mapping[str, Any]) -> Path:
    path = sample_dir / LABEL_FILES[model]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    errors = validate_labels_file(path)
    if errors:
        raise ValueError("; ".join(errors))
    return path


def _sample_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "performance_audio.wav").is_file()
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(LABEL_FILES), required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=ROOT / "DataCreate" / "samples")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--inference-manifest", type=Path)
    args = parser.parse_args(argv)

    sys.addaudithook(deny_gold)
    pred_dir = args.pred_dir.resolve()
    ids_path = pred_dir / "evaluated_ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    samples = {path.name: path for path in _sample_dirs(args.samples.resolve())}
    missing = [sample_id for sample_id in ids if sample_id not in samples]
    if missing:
        raise FileNotFoundError(f"Missing sample directories: {missing}")
    provenance = {
        "prediction_dir": str(pred_dir),
        "checkpoint_sha256": args.checkpoint_sha256,
    }
    if args.inference_manifest and args.inference_manifest.is_file():
        provenance["inference_manifest"] = str(args.inference_manifest.resolve())
        provenance["inference_manifest_sha256"] = sha256(args.inference_manifest.resolve())
        payload = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
        provenance["checkpoint"] = payload.get("checkpoint")
        provenance["checkpoint_sha256"] = payload.get("checkpoint_sha256") or args.checkpoint_sha256

    output = args.output.resolve()
    marker = output / f"{args.model}_label_manifest.json"
    if marker.exists():
        raise FileExistsError(marker)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    totals: Counter[str] = Counter()
    for sample_id in ids:
        midi = pred_dir / sample_id / "mix.mid"
        sample_dir = samples[sample_id]
        document = document_from_midi(
            midi,
            sample_dir / "verified_score.musicxml",
            model=args.model,
            sample_id=sample_id,
            provenance={
                **provenance,
                "prediction_midi_sha256": sha256(midi),
                "score_sha256": sha256(sample_dir / "verified_score.musicxml"),
            },
        )
        path = write_sample_document(sample_dir, args.model, document)
        archive = output / "documents" / args.model / f"{sample_id}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        counts = document["baseline_labeling"]["kept_counts_by_type"]
        totals.update(counts)
        rows.append(
            {
                "sample": sample_id,
                "path": str(path),
                "archive": str(archive),
                "label_count": len(document["labels"]),
                "counts_by_type": counts,
                "unlocated_events": document["baseline_labeling"]["unlocated_events"],
                "prediction_midi_sha256": sha256(midi),
                "labels_sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": CONVERSION_VERSION,
        "model": args.model,
        "n_samples": len(rows),
        "total_labels": int(sum(row["label_count"] for row in rows)),
        "counts_by_type": dict(sorted(totals.items())),
        "human_labels_json_touched": False,
        "rows": rows,
        "written_utc": _utc(),
        **provenance,
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(marker)


if __name__ == "__main__":
    main()
