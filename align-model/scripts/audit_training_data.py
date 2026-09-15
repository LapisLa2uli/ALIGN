"""Audit ALIGN corpora and build a leakage-safe, hash-locked data release.

The utility is deliberately read-only with respect to source bundles.  It
writes a versioned report, a source-group-isolated split, a compact exact
target cache, and a held-out evaluation protocol beneath ``--out``.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
import wave
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover - the production audit environment has numpy
    np = None

try:
    import mido
except ImportError:  # pragma: no cover - MIDI checks degrade explicitly
    mido = None


RELEASE_VERSION = "align-data-audit-v2"
DEFAULT_SEED = 365
SPLITS = ("train", "val", "test_id")
ERROR_TYPES = (
    "wrong_note",
    "missed_note",
    "extra_note",
    "rhythm_error",
    "intonation_error",
    "repetition",
)
SYNTH_REQUIRED = (
    "metadata.json",
    "labels.json",
    "verified_score.musicxml",
    "performance_score.musicxml",
    "performance_audio.wav",
    "reference_audio.wav",
    "performance_audio.mid",
)
MAX_EXAMPLES = 12


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _stable_hex(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _sha256(path: Path, cache: dict[str, dict[str, Any]]) -> str:
    stat = path.stat()
    key = _norm(path)
    prior = cache.get(key)
    if (
        prior
        and prior.get("size") == stat.st_size
        and prior.get("mtime_ns") == stat.st_mtime_ns
        and prior.get("sha256")
    ):
        return str(prior["sha256"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    value = digest.hexdigest()
    cache[key] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": value,
    }
    return value


@dataclass
class IssueLog:
    rows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def add(
        self,
        code: str,
        severity: str,
        sample: str,
        detail: str,
        *,
        path: str | None = None,
    ) -> None:
        key = (code, severity)
        row = self.rows.setdefault(
            key,
            {"code": code, "severity": severity, "count": 0, "examples": []},
        )
        row["count"] += 1
        if len(row["examples"]) < MAX_EXAMPLES:
            example = {"sample": sample, "detail": detail}
            if path is not None:
                example["path"] = path
            row["examples"].append(example)

    def report(self) -> list[dict[str, Any]]:
        rank = {"critical": 0, "high": 1, "warning": 2, "info": 3}
        return sorted(
            self.rows.values(),
            key=lambda row: (
                rank.get(str(row["severity"]), 9),
                -int(row["count"]),
                str(row["code"]),
            ),
        )


def _discover_bundles(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    candidates: set[Path] = set()
    # ``performance_audio_original/`` is a non-destructive WAV backup sidecar
    # nested inside each real bundle.  A WAV alone does not make a bundle.
    for name in ("metadata.json", "labels.json"):
        for path in root.rglob(name):
            candidates.add(path.parent)
    return sorted(candidates, key=lambda value: _norm(value))


def _discover_manifests(repo: Path) -> list[Path]:
    runs = repo / "align-model" / "runs"
    if not runs.is_dir():
        return []
    return sorted(runs.rglob("split.json"), key=lambda value: _norm(value))


def _manifest_maps(
    paths: Iterable[Path],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    maps: dict[str, Path] = {}
    overrides: dict[str, dict[str, Any]] = {}
    documents: list[dict[str, Any]] = []
    run_roots: set[Path] = set()
    for path in paths:
        run_roots.add(path.parents[1])
        try:
            document = _json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        documents.append({"path": path, "document": document})
        for split in ("train", "val", "test_id", "test_ood", "test"):
            for raw in document.get(split) or []:
                if not isinstance(raw, dict):
                    continue
                sample_dir = raw.get("sample_dir")
                note_map = raw.get("note_map")
                if sample_dir and note_map and Path(str(note_map)).is_file():
                    maps[_norm(str(sample_dir))] = Path(str(note_map))
                if sample_dir:
                    explicit = {
                        key: raw[key]
                        for key in (
                            "audio_pitch_space",
                            "midi_pitch_space",
                            "effective_audio_transpose",
                        )
                        if raw.get(key) is not None
                    }
                    if explicit:
                        overrides[_norm(str(sample_dir))] = explicit
    for runs in run_roots:
        for note_map in runs.rglob("note_map.json"):
            parts = [part.casefold() for part in note_map.parts]
            if "targets" not in parts:
                continue
            target_index = parts.index("targets")
            if target_index + 1 >= len(note_map.parts):
                continue
            corpus = note_map.parts[target_index + 1].casefold()
            sample = note_map.parent.name.casefold()
            maps[f"sample::{corpus}::{sample}"] = note_map
    return maps, overrides, documents


def _read_wave_info(path: Path) -> dict[str, Any] | None:
    try:
        with wave.open(str(path), "rb") as stream:
            frames = int(stream.getnframes())
            rate = int(stream.getframerate())
            return {
                "sample_rate": rate,
                "channels": int(stream.getnchannels()),
                "sample_width": int(stream.getsampwidth()),
                "frames": frames,
                "duration_sec": frames / max(rate, 1),
            }
    except (OSError, EOFError, wave.Error):
        try:
            import soundfile

            info = soundfile.info(str(path))
            return {
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "sample_width": None,
                "frames": int(info.frames),
                "duration_sec": float(info.duration),
            }
        except Exception:
            return None


def _musicxml_pitches(path: Path) -> list[int]:
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    output: list[int] = []
    for note in root.iter(f"{ns}note"):
        if note.find(f"{ns}grace") is not None:
            continue
        pitch = note.find(f"{ns}pitch")
        if pitch is None:
            continue
        step = pitch.findtext(f"{ns}step")
        octave = pitch.findtext(f"{ns}octave")
        if step not in steps or octave is None:
            continue
        alter = float(pitch.findtext(f"{ns}alter") or 0)
        output.append(12 * (int(octave) + 1) + steps[str(step)] + int(round(alter)))
    return output


def _midi_events(path: Path) -> dict[str, Any] | None:
    try:
        from tinysoundfont.midi import NoteOff, NoteOn, PitchBend, load

        events = load(str(path), persistent=False)
        active: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        bends: dict[int, float] = defaultdict(float)
        notes = []
        pitchwheels = []
        last_time = 0.0
        for event in events:
            now = float(event.t)
            last_time = max(last_time, now)
            channel = int(getattr(event, "channel", 0) or 0)
            action = event.action
            if isinstance(action, PitchBend):
                raw = int(getattr(action, "pitch_bend", 8192))
                cents = (raw - 8192) / 8192.0 * 200.0
                bends[channel] = cents
                pitchwheels.append(
                    {"time": now, "channel": channel, "cents": cents}
                )
                continue
            if isinstance(action, NoteOn):
                pitch = int(action.key)
                velocity = int(getattr(action, "velocity", 90) or 0)
                if velocity > 0:
                    active[(channel, pitch)].append((now, bends[channel]))
                    continue
            elif isinstance(action, NoteOff):
                pitch = int(action.key)
            else:
                continue
            stack = active.get((channel, pitch))
            if not stack:
                continue
            start, cents = stack.pop(0)
            notes.append(
                {
                    "pitch": pitch,
                    "start": start,
                    "end": max(now, start + 0.001),
                    "channel": channel,
                    "cents": cents,
                }
            )
        for (channel, pitch), stack in active.items():
            for start, cents in stack:
                notes.append(
                    {
                        "pitch": pitch,
                        "start": start,
                        "end": max(last_time, start + 0.001),
                        "channel": channel,
                        "cents": cents,
                    }
                )
        notes.sort(key=lambda row: (row["start"], row["pitch"], row["end"]))
        return {
            "notes": notes,
            "pitchwheels": pitchwheels,
            "duration_sec": max(
                [last_time, *(float(note["end"]) for note in notes)], default=0.0
            ),
            "clock": "tinysoundfont_render_space",
        }
    except Exception:
        pass
    if mido is None:
        return None
    try:
        midi = mido.MidiFile(str(path))
        merged = mido.merge_tracks(midi.tracks)
    except Exception:
        return None
    tempo = 500000
    now = 0.0
    bends: dict[int, float] = defaultdict(float)
    active: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    notes: list[dict[str, Any]] = []
    pitchwheels: list[dict[str, Any]] = []
    for message in merged:
        now += float(mido.tick2second(message.time, midi.ticks_per_beat, tempo))
        if message.type == "set_tempo":
            tempo = int(message.tempo)
            continue
        channel = int(getattr(message, "channel", 0))
        if message.type == "pitchwheel":
            cents = float(message.pitch) / 8192.0 * 200.0
            bends[channel] = cents
            pitchwheels.append({"time": now, "channel": channel, "cents": cents})
            continue
        if message.type == "note_on" and int(message.velocity) > 0:
            active[(channel, int(message.note))].append((now, bends[channel]))
            continue
        if message.type not in {"note_off", "note_on"}:
            continue
        stack = active.get((channel, int(message.note)))
        if not stack:
            continue
        start, cents = stack.pop(0)
        notes.append(
            {
                "pitch": int(message.note),
                "start": start,
                "end": max(now, start + 0.001),
                "channel": channel,
                "cents": cents,
            }
        )
    for (channel, pitch), stack in active.items():
        for start, cents in stack:
            notes.append(
                {
                    "pitch": pitch,
                    "start": start,
                    "end": max(now, start + 0.001),
                    "channel": channel,
                    "cents": cents,
                }
            )
    notes.sort(key=lambda row: (row["start"], row["pitch"], row["end"]))
    return {
        "notes": notes,
        "pitchwheels": pitchwheels,
        "duration_sec": max(
            [now, *(float(note["end"]) for note in notes)], default=0.0
        ),
        "clock": "mido_fallback",
    }


def _effective_audio_shift(metadata: Mapping[str, Any]) -> int:
    if metadata.get("effective_audio_transpose") is not None:
        return int(metadata["effective_audio_transpose"])
    if metadata.get("audio_pitch_space") == "written":
        return 0
    return -int(metadata.get("sounding_transpose", -2))


def _midi_to_written_shift(metadata: Mapping[str, Any]) -> int:
    explicit = metadata.get("midi_pitch_space")
    if explicit == "written":
        return 0
    if explicit == "sounding":
        return -int(metadata.get("sounding_transpose", -2))
    if metadata.get("audio_render") == "soundfont_rerender":
        return 0
    return -int(metadata.get("sounding_transpose", -2))


def _inferred_midi_shift(
    document: Mapping[str, Any],
    midi: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> int:
    """Infer written-minus-MIDI pitch by sequence agreement with performed notes."""

    if midi is None:
        return _midi_to_written_shift(metadata)
    performed = list(document.get("performed_notes") or [])
    source = [int(row["pitch"]) for row in midi.get("notes") or []]
    target = [
        int(row["pitch_midi"])
        for row in performed
        if row.get("pitch_midi") is not None
    ]
    if source and target:
        fallback = _midi_to_written_shift(metadata)
        ranked = []
        for shift in range(-4, 5):
            shifted = [pitch + shift for pitch in source]
            ratio = difflib.SequenceMatcher(
                None, shifted, target, autojunk=False
            ).ratio()
            ranked.append((ratio, -abs(shift - fallback), -abs(shift), shift))
        best = max(ranked)
        if best[0] >= 0.50:
            return int(best[-1])

    # Fall back to the old render map only when sequence evidence is weak.
    rendered = list(document.get("rendered_notes") or [])
    offsets: list[int] = []
    old_cursor = 0
    for actual in midi.get("notes") or []:
        candidates = []
        for old_index in range(old_cursor, min(len(rendered), old_cursor + 10)):
            old = rendered[old_index]
            if int(old.get("pitch_midi_sounding", -999)) != int(actual["pitch"]):
                continue
            distance = abs(float(old.get("start_sec", -999)) - float(actual["start"]))
            if distance <= 0.08:
                candidates.append((distance, old_index, old))
        if not candidates:
            continue
        _distance, old_index, old = min(candidates)
        old_cursor = old_index + 1
        for raw_index in old.get("performed_indices") or []:
            index = int(raw_index)
            if 0 <= index < len(performed):
                offsets.append(
                    int(performed[index].get("pitch_midi", -999))
                    - int(actual["pitch"])
                )
                break
    plausible = [value for value in offsets if -4 <= value <= 4]
    if plausible:
        return Counter(plausible).most_common(1)[0][0]
    return _midi_to_written_shift(metadata)


def _rebuilt_rendered_events(
    document: Mapping[str, Any],
    midi: Mapping[str, Any],
    written_shift: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map raw MIDI events back to existing performed-note lineage.

    Existing maps remain the first source of lineage.  Raw MIDI is authoritative
    for audible event count, key, and time.  Any stale/ghost rendered rows are
    dropped and unmapped performed rows are recovered monotonically by pitch.
    """

    performed = list(document.get("performed_notes") or [])
    old_rendered = list(document.get("rendered_notes") or [])
    raw_notes = list(midi.get("notes") or [])
    rows: list[dict[str, Any]] = []
    claimed: set[int] = set()
    old_cursor = 0
    performed_cursor = 0
    transferred = recovered = 0

    for rendered_index, actual in enumerate(raw_notes):
        sounding = int(actual["pitch"])
        written = sounding + int(written_shift)
        candidates = []
        for old_index in range(max(0, old_cursor - 2), min(len(old_rendered), old_cursor + 12)):
            old = old_rendered[old_index]
            if int(old.get("pitch_midi_sounding", -999)) != sounding:
                continue
            distance = abs(float(old.get("start_sec", -999)) - float(actual["start"]))
            if distance <= 0.10:
                candidates.append((distance, abs(old_index - old_cursor), old_index, old))
        indices: list[int] = []
        if candidates:
            _distance, _index_distance, old_index, old = min(candidates)
            old_cursor = max(old_cursor, old_index + 1)
            indices = [
                int(value)
                for value in old.get("performed_indices") or []
                if 0 <= int(value) < len(performed)
                and int(value) not in claimed
                and int(performed[int(value)].get("pitch_midi", -999)) == written
            ]
            transferred += len(indices)
        if not indices:
            nearby = [
                index
                for index in range(performed_cursor, min(len(performed), performed_cursor + 12))
                if index not in claimed
                and int(performed[index].get("pitch_midi", -999)) == written
            ]
            if not nearby:
                nearby = [
                    index
                    for index in range(performed_cursor, len(performed))
                    if index not in claimed
                    and int(performed[index].get("pitch_midi", -999)) == written
                ]
            if nearby:
                indices = [nearby[0]]
                recovered += 1
        if indices:
            claimed.update(indices)
            performed_cursor = max(performed_cursor, max(indices) + 1)
        rows.append(
            {
                "rendered_index": rendered_index,
                "pitch_midi_sounding": sounding,
                "pitch_midi_written": written,
                "start_sec": round(float(actual["start"]), 9),
                "end_sec": round(max(float(actual["end"]), float(actual["start"]) + 0.001), 9),
                "performed_indices": sorted(indices),
            }
        )

    # Fold notated tie continuations into the closest already mapped event.
    folded = 0
    for performed_index, item in enumerate(performed):
        if performed_index in claimed:
            continue
        pitch = int(item.get("pitch_midi", -999))
        candidates = [
            row
            for row in rows
            if row["performed_indices"]
            and int(row["pitch_midi_written"]) == pitch
            and min(
                abs(performed_index - existing)
                for existing in row["performed_indices"]
            )
            <= 4
        ]
        if not candidates:
            continue
        target = min(
            candidates,
            key=lambda row: min(
                abs(performed_index - existing)
                for existing in row["performed_indices"]
            ),
        )
        target["performed_indices"].append(performed_index)
        target["performed_indices"].sort()
        claimed.add(performed_index)
        folded += 1

    for row in rows:
        indices = row["performed_indices"]
        clean_indices = list(
            dict.fromkeys(
                int(performed[index]["clean_index"])
                for index in indices
                if performed[index].get("clean_index") is not None
            )
        )
        relationships = [
            str(performed[index].get("relationship") or "extra")
            for index in indices
        ]
        row["clean_indices"] = clean_indices
        row["primary_clean_index"] = clean_indices[0] if clean_indices else None
        row["relationship"] = (
            "copy"
            if "copy" in relationships
            else (relationships[0] if relationships else "extra")
        )
    return rows, {
        "raw_midi_events": len(raw_notes),
        "old_rendered_events": len(old_rendered),
        "transferred_performed_indices": transferred,
        "recovered_performed_indices": recovered,
        "folded_tied_notes": folded,
        "unmapped_performed_notes": len(performed) - len(claimed),
        "written_shift": int(written_shift),
    }


def _validate_note_map(
    sample: str,
    path: Path,
    verified_pitches: list[int],
    performance_pitches: list[int],
    midi: dict[str, Any] | None,
    metadata: Mapping[str, Any],
    issues: IssueLog,
) -> tuple[dict[str, Any] | None, bool, dict[str, Any]]:
    try:
        document = _json(path)
    except Exception as exc:
        issues.add("note_map_unreadable", "critical", sample, str(exc), path=str(path))
        return None, False, {}
    if not isinstance(document, dict):
        issues.add(
            "note_map_not_object", "critical", sample, "top level is not an object", path=str(path)
        )
        return None, False, {}
    valid = True
    clean = list(document.get("clean_notes") or [])
    performed = list(document.get("performed_notes") or [])
    rendered = list(document.get("rendered_notes") or [])
    render_repair: dict[str, Any] | None = None
    if midi is not None and rendered:
        inferred_shift = _inferred_midi_shift(document, midi, metadata)
        rebuilt, render_repair = _rebuilt_rendered_events(
            document, midi, inferred_shift
        )
        if rebuilt != rendered:
            document = dict(document)
            document["rendered_notes_original_count"] = len(rendered)
            document["rendered_notes"] = rebuilt
            document["rendered_note_count"] = len(rebuilt)
            document["render_repair"] = {
                "policy": "raw_midi_authoritative_v1",
                **render_repair,
            }
            rendered = rebuilt
            issues.add(
                "rendered_map_repaired_in_sidecar",
                "info",
                sample,
                f"old={render_repair['old_rendered_events']} raw={render_repair['raw_midi_events']} shift={render_repair['written_shift']}",
                path=str(path),
            )

    def exact_indices(rows: list[dict], field: str) -> bool:
        try:
            return [int(row[field]) for row in rows] == list(range(len(rows)))
        except (KeyError, TypeError, ValueError):
            return False

    for rows, field, code in (
        (clean, "clean_index", "clean_index_invalid"),
        (performed, "performed_index", "performed_index_invalid"),
        (rendered, "rendered_index", "rendered_index_invalid"),
    ):
        if not exact_indices(rows, field):
            valid = False
            issues.add(code, "critical", sample, f"{field} is not contiguous from zero", path=str(path))

    if int(document.get("clean_note_count", -1)) != len(clean):
        valid = False
        issues.add(
            "clean_count_mismatch",
            "critical",
            sample,
            f"declared={document.get('clean_note_count')} actual={len(clean)}",
            path=str(path),
        )
    if int(document.get("performed_note_count", -1)) != len(performed):
        valid = False
        issues.add(
            "performed_count_mismatch",
            "critical",
            sample,
            f"declared={document.get('performed_note_count')} actual={len(performed)}",
            path=str(path),
        )
    if not rendered:
        valid = False
        issues.add("rendered_notes_missing", "critical", sample, "no rendered target events", path=str(path))

    if verified_pitches and [int(row.get("pitch_midi", -999)) for row in clean] != verified_pitches:
        valid = False
        issues.add(
            "clean_map_xml_mismatch",
            "critical",
            sample,
            f"map={len(clean)} XML={len(verified_pitches)} or pitches differ",
            path=str(path),
        )
    if performance_pitches and [
        int(row.get("pitch_midi", -999)) for row in performed
    ] != performance_pitches:
        valid = False
        issues.add(
            "performed_map_xml_mismatch",
            "critical",
            sample,
            f"map={len(performed)} XML={len(performance_pitches)} or pitches differ",
            path=str(path),
        )

    clean_by_index = {
        int(row["clean_index"]): row
        for row in clean
        if isinstance(row, dict) and row.get("clean_index") is not None
    }
    mapped_clean: set[int] = set()
    performed_owners: Counter[int] = Counter()
    pitch_faults = 0
    for row in performed:
        clean_index = row.get("clean_index")
        relationship = str(row.get("relationship") or "")
        if clean_index is not None:
            try:
                clean_index = int(clean_index)
            except (TypeError, ValueError):
                clean_index = -1
            if clean_index not in clean_by_index:
                valid = False
                issues.add(
                    "performed_clean_reference_invalid",
                    "critical",
                    sample,
                    f"unknown clean index {clean_index}",
                    path=str(path),
                )
            else:
                mapped_clean.add(clean_index)
                origin = str(row.get("origin_relationship") or relationship)
                if origin == "match" and int(row.get("pitch_midi", -999)) != int(
                    clean_by_index[clean_index]["pitch_midi"]
                ):
                    pitch_faults += 1
        if relationship not in {"match", "substitute", "extra", "copy"}:
            valid = False
            issues.add(
                "relationship_invalid",
                "critical",
                sample,
                f"unknown relationship {relationship!r}",
                path=str(path),
            )
    if pitch_faults:
        valid = False
        issues.add(
            "lineage_match_pitch_mismatch",
            "critical",
            sample,
            f"{pitch_faults} match-origin performed notes differ from clean pitch",
            path=str(path),
        )

    declared_deleted = sorted(int(value) for value in document.get("deleted_clean_notes") or [])
    actual_deleted = sorted(set(clean_by_index) - mapped_clean)
    flagged_deleted = sorted(
        int(row["clean_index"]) for row in clean if bool(row.get("deleted"))
    )
    if declared_deleted != actual_deleted or flagged_deleted != actual_deleted:
        valid = False
        issues.add(
            "deleted_lineage_mismatch",
            "critical",
            sample,
            f"declared={declared_deleted[:8]} actual={actual_deleted[:8]}",
            path=str(path),
        )

    shift = (
        int(render_repair["written_shift"])
        if render_repair is not None
        else _midi_to_written_shift(metadata)
    )
    rendered_pitch_faults = 0
    rendered_time_faults = 0
    tied_events = 0
    previous_start = -math.inf
    for row in rendered:
        try:
            rendered_index = int(row["rendered_index"])
            sounding = int(row["pitch_midi_sounding"])
            written = int(row["pitch_midi_written"])
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            indices = [int(value) for value in row.get("performed_indices") or []]
        except (KeyError, TypeError, ValueError):
            valid = False
            issues.add(
                "rendered_event_malformed", "critical", sample, repr(row)[:240], path=str(path)
            )
            continue
        if written != sounding + shift or end <= start or start + 1e-6 < previous_start:
            rendered_time_faults += 1
        previous_start = max(previous_start, start)
        tied_events += max(0, len(indices) - 1)
        for performed_index in indices:
            if not 0 <= performed_index < len(performed):
                valid = False
                issues.add(
                    "rendered_performed_reference_invalid",
                    "critical",
                    sample,
                    f"rendered {rendered_index} -> performed {performed_index}",
                    path=str(path),
                )
                continue
            performed_owners[performed_index] += 1
            expected = int(performed[performed_index].get("pitch_midi", -999))
            if written != expected:
                rendered_pitch_faults += 1
        expected_clean = list(
            dict.fromkeys(
                int(performed[index]["clean_index"])
                for index in indices
                if 0 <= index < len(performed)
                and performed[index].get("clean_index") is not None
            )
        )
        actual_clean = [int(value) for value in row.get("clean_indices") or []]
        if expected_clean != actual_clean:
            valid = False
            issues.add(
                "rendered_clean_reference_mismatch",
                "critical",
                sample,
                f"rendered {rendered_index}: expected={expected_clean} actual={actual_clean}",
                path=str(path),
            )
    if rendered_time_faults:
        valid = False
        issues.add(
            "rendered_pitch_or_time_invalid",
            "critical",
            sample,
            f"{rendered_time_faults} events violate pitch-space or time invariants",
            path=str(path),
        )
    if rendered_pitch_faults:
        valid = False
        issues.add(
            "rendered_performed_pitch_mismatch",
            "critical",
            sample,
            f"{rendered_pitch_faults} rendered events disagree with performed score",
            path=str(path),
        )
    duplicate_owners = sum(value > 1 for value in performed_owners.values())
    if duplicate_owners:
        valid = False
        issues.add(
            "performed_event_mapped_multiple_times",
            "critical",
            sample,
            f"{duplicate_owners} performed notes belong to multiple rendered events",
            path=str(path),
        )

    midi_faults = 0
    if midi is not None:
        midi_notes = list(midi["notes"])
        if len(midi_notes) != len(rendered):
            midi_faults += abs(len(midi_notes) - len(rendered)) or 1
        for actual, target in zip(midi_notes, rendered):
            if (
                int(actual["pitch"]) != int(target.get("pitch_midi_sounding", -999))
                or abs(float(actual["start"]) - float(target.get("start_sec", -999))) > 0.04
                or abs(float(actual["end"]) - float(target.get("end_sec", -999))) > 0.08
            ):
                midi_faults += 1
        if midi_faults:
            valid = False
            issues.add(
                "rendered_map_midi_mismatch",
                "critical",
                sample,
                f"{midi_faults} count/pitch/time discrepancies",
                path=str(path),
            )

    repeat_rows = [row for row in rendered if str(row.get("relationship")) == "copy"]
    copy_passes = Counter()
    repeat_boundary_faults = 0
    previous_copy = False
    copy_runs = 0
    for row in performed:
        is_copy = str(row.get("relationship")) == "copy"
        if is_copy and not previous_copy:
            copy_runs += 1
        previous_copy = is_copy
        if is_copy:
            copy_passes[int(row.get("copy_pass") or 0)] += 1
            if int(row.get("copy_pass") or 0) <= 0:
                repeat_boundary_faults += 1
    if repeat_rows and not copy_passes:
        repeat_boundary_faults += 1
    if repeat_boundary_faults:
        valid = False
        issues.add(
            "repeat_copy_boundary_invalid",
            "critical",
            sample,
            f"{repeat_boundary_faults} copy rows lack a positive copy pass",
            path=str(path),
        )
    declared_repeated = bool(metadata.get("repeated"))
    if declared_repeated != bool(repeat_rows):
        issues.add(
            "repeat_metadata_map_mismatch",
            "high",
            sample,
            f"metadata={declared_repeated} rendered_copy_events={len(repeat_rows)}",
            path=str(path),
        )

    return document, valid, {
        "clean_notes": len(clean),
        "performed_notes": len(performed),
        "rendered_notes": len(rendered),
        "tied_notes_folded": tied_events,
        "copy_events": len(repeat_rows),
        "copy_runs": copy_runs,
        "repeat_consistent": declared_repeated == bool(repeat_rows),
        "midi_to_written_shift": shift,
    }


def _intonation_midi_status(
    labels: list[dict[str, Any]], midi: dict[str, Any] | None
) -> tuple[str, list[dict[str, Any]]]:
    targets = [
        row
        for row in labels
        if row.get("type") == "intonation_error"
        and row.get("deviation_cents") is not None
    ]
    if not targets:
        return "not_applicable", []
    if midi is None:
        return "not_checked", []
    measured = []
    for label in targets:
        start = float(label.get("start_time", 0.0))
        end = float(label.get("end_time", start))
        values = [
            float(note["cents"])
            for note in midi["notes"]
            if float(note["start"]) < end + 0.03
            and start - 0.03 < float(note["end"])
        ]
        value = statistics.median(values) if values else 0.0
        expected = float(label["deviation_cents"])
        measured.append(
            {
                "start": start,
                "end": end,
                "expected_cents": expected,
                "midi_cents": value,
                "absolute_error": abs(value - expected),
            }
        )
    good = sum(row["absolute_error"] <= 8.0 for row in measured)
    return ("midi_verified" if good == len(measured) else "midi_mismatch"), measured


def _corpus_specs(repo: Path) -> list[tuple[str, Path, int]]:
    return [
        ("rawsf10k", Path("E:/outputRaw_sf_10k"), 0),
        ("procedural12k", Path("E:/output"), 1),
        ("raw2k_e", Path("E:/output_2k_rawdata"), 2),
        ("raw2k_local", repo / "synth-pipeline" / "output_2k_rawdata", 3),
        ("datacreate_real", repo / "DataCreate" / "samples", 4),
    ]


def _audit_bundle(
    corpus: str,
    root: Path,
    priority: int,
    sample_dir: Path,
    note_map_lookup: Mapping[str, Path],
    metadata_overrides: Mapping[str, Mapping[str, Any]],
    hashes: dict[str, dict[str, Any]],
    issues: IssueLog,
) -> dict[str, Any]:
    sample = sample_dir.name
    metadata: dict[str, Any] = {}
    labels_document: dict[str, Any] = {}
    json_ok = True
    for name, destination in (
        ("metadata.json", "metadata"),
        ("labels.json", "labels"),
    ):
        path = sample_dir / name
        if not path.is_file():
            issues.add("required_file_missing", "critical", sample, name, path=str(sample_dir))
            json_ok = False
            continue
        try:
            value = _json(path)
            if not isinstance(value, dict):
                raise ValueError("top level is not an object")
            if destination == "metadata":
                metadata = value
            else:
                labels_document = value
        except Exception as exc:
            issues.add("json_unreadable", "critical", sample, f"{name}: {exc}", path=str(path))
            json_ok = False
    synthetic = (
        corpus != "datacreate_real"
        or metadata.get("recording_kind") == "synthetic"
        or metadata.get("mode") == "synth-pipeline"
    )
    pitch_override = dict(metadata_overrides.get(_norm(sample_dir), {}))
    effective_metadata = dict(metadata)
    for key, value in pitch_override.items():
        if effective_metadata.get(key) is None:
            effective_metadata[key] = value
    missing = [
        name for name in SYNTH_REQUIRED if synthetic and not (sample_dir / name).is_file()
    ]
    for name in missing:
        issues.add("required_file_missing", "critical", sample, name, path=str(sample_dir))

    file_hashes: dict[str, str] = {}
    for name in (
        "performance_audio.wav",
        "performance_audio.mid",
        "verified_score.musicxml",
        "performance_score.musicxml",
        "labels.json",
        "metadata.json",
    ):
        path = sample_dir / name
        if path.is_file():
            try:
                file_hashes[name] = _sha256(path, hashes)
            except OSError as exc:
                issues.add("file_hash_failed", "critical", sample, f"{name}: {exc}", path=str(path))

    wav = (
        _read_wave_info(sample_dir / "performance_audio.wav")
        if (sample_dir / "performance_audio.wav").is_file()
        else None
    )
    reference_wav = (
        _read_wave_info(sample_dir / "reference_audio.wav")
        if (sample_dir / "reference_audio.wav").is_file()
        else None
    )
    if (sample_dir / "performance_audio.wav").is_file() and wav is None:
        issues.add(
            "wav_header_unreadable",
            "critical",
            sample,
            "performance_audio.wav",
            path=str(sample_dir / "performance_audio.wav"),
        )
    if wav:
        declared_rate = int(metadata.get("sample_rate") or wav["sample_rate"])
        if declared_rate != int(wav["sample_rate"]):
            issues.add(
                "wav_metadata_rate_mismatch",
                "critical",
                sample,
                f"metadata={declared_rate} wav={wav['sample_rate']}",
                path=str(sample_dir),
            )
        if synthetic and (int(wav["sample_rate"]) != 22050 or int(wav["channels"]) != 1):
            issues.add(
                "wav_format_unexpected",
                "high",
                sample,
                f"rate={wav['sample_rate']} channels={wav['channels']}",
                path=str(sample_dir),
            )

    midi = (
        _midi_events(sample_dir / "performance_audio.mid")
        if (sample_dir / "performance_audio.mid").is_file()
        else None
    )
    if (sample_dir / "performance_audio.mid").is_file() and midi is None:
        issues.add(
            "midi_unreadable",
            "critical",
            sample,
            "could not parse performance_audio.mid",
            path=str(sample_dir),
        )
    duration_consistent = True
    if wav and midi:
        tail = float(wav["duration_sec"]) - float(midi["duration_sec"])
        if tail < -0.10 or tail > 3.5:
            duration_consistent = False
            issues.add(
                "wav_midi_duration_mismatch",
                "high",
                sample,
                f"wav={wav['duration_sec']:.3f}s midi={midi['duration_sec']:.3f}s tail={tail:.3f}s",
                path=str(sample_dir),
            )
    if wav and reference_wav:
        ratio = float(wav["duration_sec"]) / max(float(reference_wav["duration_sec"]), 1e-9)
        if ratio < 0.25 or ratio > 4.0:
            issues.add(
                "performance_reference_duration_ratio",
                "warning",
                sample,
                f"performance/reference={ratio:.3f}",
                path=str(sample_dir),
            )

    verified_pitches: list[int] = []
    performance_pitches: list[int] = []
    for name, output in (
        ("verified_score.musicxml", verified_pitches),
        ("performance_score.musicxml", performance_pitches),
    ):
        path = sample_dir / name
        if path.is_file():
            try:
                output.extend(_musicxml_pitches(path))
            except (OSError, ET.ParseError, ValueError) as exc:
                issues.add("musicxml_unreadable", "critical", sample, f"{name}: {exc}", path=str(path))

    note_map_path = sample_dir / "note_map.json"
    if not note_map_path.is_file():
        note_map_path = note_map_lookup.get(_norm(sample_dir), note_map_path)
    if not note_map_path.is_file():
        aliases = {
            "rawsf10k": ("rawsf10k", "procedural12k"),
            "procedural12k": ("procedural12k",),
            "raw2k_e": ("raw2k", "raw2k_e"),
            "raw2k_local": ("raw2k", "raw2k_local"),
        }.get(corpus, (corpus,))
        for alias in aliases:
            candidate = note_map_lookup.get(
                f"sample::{alias.casefold()}::{sample.casefold()}"
            )
            if candidate is not None and candidate.is_file():
                note_map_path = candidate
                break
    note_map = None
    note_map_valid = False
    map_stats: dict[str, Any] = {}
    if synthetic:
        if note_map_path.is_file():
            file_hashes["note_map.json"] = _sha256(note_map_path, hashes)
            note_map, note_map_valid, map_stats = _validate_note_map(
                sample,
                note_map_path,
                verified_pitches,
                performance_pitches,
                midi,
                effective_metadata,
                issues,
            )
        else:
            issues.add(
                "exact_note_map_missing",
                "critical",
                sample,
                "no in-bundle or frozen validated note_map.json",
                path=str(sample_dir),
            )

    labels = [
        row for row in labels_document.get("labels") or [] if isinstance(row, dict)
    ]
    label_types = sorted(
        {str(row.get("type")) for row in labels if row.get("type")}
    )
    invalid_times = sum(
        1
        for row in labels
        if row.get("start_time") is None
        or row.get("end_time") is None
        or float(row.get("end_time", 0)) <= float(row.get("start_time", 0))
        or (wav and float(row.get("end_time", 0)) > float(wav["duration_sec"]) + 0.15)
    )
    if invalid_times:
        issues.add(
            "label_time_invalid",
            "critical",
            sample,
            f"{invalid_times} label spans are invalid or outside WAV",
            path=str(sample_dir / "labels.json"),
        )

    intonation_status, intonation_rows = _intonation_midi_status(labels, midi)
    if intonation_status == "midi_mismatch":
        issues.add(
            "intonation_midi_mismatch",
            "high",
            sample,
            f"{sum(row['absolute_error'] > 8 for row in intonation_rows)}/{len(intonation_rows)} labels lack matching MIDI bend",
            path=str(sample_dir),
        )

    source = str(
        metadata.get("source")
        or Path(str(metadata.get("source_score") or "")).stem
        or sample
    )
    source_stem = Path(str(metadata.get("source_score") or source)).stem
    verified_hash = file_hashes.get("verified_score.musicxml", "")
    if source.casefold() in {"gen", "procedural", ""}:
        source_group = f"generated:{verified_hash or sample}"
    else:
        source_group = f"score:{source_stem.casefold()}"
    snippet = None
    if metadata.get("snippet_start_measure") is not None:
        snippet = [
            int(metadata["snippet_start_measure"]),
            int(metadata.get("snippet_end_measure", metadata["snippet_start_measure"])),
        ]
    content_fingerprint = hashlib.sha256(
        "|".join(
            file_hashes.get(name, "")
            for name in (
                "verified_score.musicxml",
                "performance_score.musicxml",
                "labels.json",
                "performance_audio.wav",
            )
        ).encode("ascii")
    ).hexdigest()
    inferred_midi_shift = int(
        map_stats.get("midi_to_written_shift", _midi_to_written_shift(effective_metadata))
    )
    inferred_midi_space = "written" if inferred_midi_shift == 0 else "sounding"

    hard_valid = (
        json_ok
        and not missing
        and (not synthetic or note_map_valid)
        and wav is not None
        and (not synthetic or midi is not None)
        and duration_consistent
    )
    map_repeated = bool(map_stats.get("copy_events", 0))
    repeat_consistent = bool(map_stats.get("repeat_consistent", True))
    return {
        "sample": sample,
        "sample_dir": str(sample_dir.resolve()),
        "root": str(root.resolve()),
        "corpus": corpus,
        "priority": priority,
        "synthetic": synthetic,
        "eligible": bool(hard_valid and synthetic),
        "source": source,
        "source_group": source_group,
        "snippet_measures": snippet,
        "recording_kind": metadata.get("recording_kind") or ("synthetic" if synthetic else "real"),
        "audio_render": metadata.get("audio_render") or "unknown",
        "audio_pitch_space": effective_metadata.get("audio_pitch_space") or "undeclared",
        "midi_pitch_space": inferred_midi_space,
        "effective_audio_transpose": _effective_audio_shift(effective_metadata),
        "sounding_transpose": int(effective_metadata.get("sounding_transpose", -2)),
        "pitch_metadata_override": pitch_override or None,
        "duration_sec": round(float(wav["duration_sec"]), 6) if wav else None,
        "reference_duration_sec": (
            round(float(reference_wav["duration_sec"]), 6) if reference_wav else None
        ),
        "sample_rate": int(wav["sample_rate"]) if wav else None,
        "channels": int(wav["channels"]) if wav else None,
        "error_types": label_types,
        "label_count": len(labels),
        "invalid_label_count": invalid_times,
        "repeated": map_repeated if note_map_valid else bool(
            metadata.get("repeated") or "repetition" in label_types
        ),
        "repeat_consistent": repeat_consistent,
        "intonation_status": intonation_status,
        "intonation_checks": intonation_rows,
        "note_map": str(note_map_path.resolve()) if note_map_path.is_file() else None,
        "note_map_valid": note_map_valid,
        "note_map_stats": map_stats,
        "hashes": file_hashes,
        "content_fingerprint": content_fingerprint,
        "_labels": labels,
        "_valid_labels": [
            row
            for row in labels
            if row.get("start_time") is not None
            and row.get("end_time") is not None
            and float(row.get("end_time", 0)) > float(row.get("start_time", 0))
            and (
                not wav
                or float(row.get("end_time", 0)) <= float(wav["duration_sec"]) + 0.15
            )
        ],
        "_note_map_document": note_map,
    }


def _cache_kind(saved: Any, metadata: Mapping[str, Any]) -> str | None:
    names = set(saved.files)
    if {"note", "onset", "contour", "frame_times"} <= names:
        return "basic_pitch"
    if "pesto" in names:
        return "pesto"
    if metadata.get("frontend_version"):
        return "basic_pitch"
    if metadata.get("pesto_version"):
        return "pesto"
    return None


def _cache_audit(
    repo: Path,
    records: list[dict[str, Any]],
    issues: IssueLog,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Path]]]:
    summary: dict[str, Any] = {
        "files_inspected": 0,
        "basic_pitch": Counter(),
        "pesto": Counter(),
    }
    matches: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    if np is None:
        summary["status"] = "numpy_unavailable"
        return summary, matches
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[record["sample"]].append(record)
    runs = repo / "align-model" / "runs"
    if not runs.is_dir():
        return summary, matches
    for path in runs.rglob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as saved:
                if "metadata" not in saved.files:
                    continue
                raw = np.asarray(saved["metadata"])
                if raw.shape != ():
                    continue
                metadata = json.loads(str(raw.item()))
                kind = _cache_kind(saved, metadata)
                if kind is None:
                    continue
        except Exception:
            continue
        summary["files_inspected"] += 1
        sample = path.stem
        candidates = by_sample.get(sample, [])
        expected_hash = str(metadata.get("wav_sha256") or "")
        exact = [
            row
            for row in candidates
            if row["hashes"].get("performance_audio.wav") == expected_hash
        ]
        if not candidates:
            status = "orphan"
        elif not exact:
            status = "stale"
            issues.add(
                f"{kind}_cache_stale",
                "high",
                sample,
                f"cache WAV hash {expected_hash[:12]} matches no on-disk sample",
                path=str(path),
            )
        else:
            transpose = int(metadata.get("effective_audio_transpose", metadata.get("pitch_policy", {}).get("effective_audio_transpose", 999)))
            valid_rows = [
                row for row in exact if transpose == int(row["effective_audio_transpose"])
            ]
            if not valid_rows:
                status = "pitch_policy_stale"
                issues.add(
                    f"{kind}_cache_pitch_policy_stale",
                    "high",
                    sample,
                    f"cache transpose={transpose}; bundle values={sorted({row['effective_audio_transpose'] for row in exact})}",
                    path=str(path),
                )
            else:
                status = "valid"
                for row in valid_rows:
                    matches[(row["corpus"], row["sample"])][kind] = path
        summary[kind][status] += 1
    for kind in ("basic_pitch", "pesto"):
        summary[kind] = dict(summary[kind])
    return summary, matches


def _acoustic_intonation(
    records: list[dict[str, Any]],
    cache_matches: Mapping[tuple[str, str], Mapping[str, Path]],
    issues: IssueLog,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "labels_total": 0,
        "labels_acoustically_measured": 0,
        "labels_within_20_cents": 0,
        "absolute_errors": [],
    }
    if np is None:
        summary["status"] = "numpy_unavailable"
        return summary
    for record in records:
        labels = [
            row
            for row in record["_labels"]
            if row.get("type") == "intonation_error"
            and row.get("deviation_cents") is not None
        ]
        summary["labels_total"] += len(labels)
        paths = cache_matches.get((record["corpus"], record["sample"]), {})
        if not labels or "basic_pitch" not in paths or "pesto" not in paths:
            record["intonation_acoustic_status"] = (
                "not_applicable" if not labels else "unverified_no_valid_cache"
            )
            continue
        try:
            with np.load(paths["basic_pitch"], allow_pickle=False) as basic:
                times = np.asarray(basic["frame_times"], dtype=np.float64)
            with np.load(paths["pesto"], allow_pickle=False) as fine:
                pesto = np.asarray(fine["pesto"], dtype=np.float32)
        except Exception:
            record["intonation_acoustic_status"] = "cache_read_failed"
            continue
        document = record["_note_map_document"] or {}
        rendered = list(document.get("rendered_notes") or [])
        failures = 0
        measured_count = 0
        for label in labels:
            start = float(label["start_time"])
            end = float(label["end_time"])
            target_events = [
                row
                for row in rendered
                if float(row.get("start_sec", 0)) < end
                and start < float(row.get("end_sec", 0))
            ]
            if not target_events:
                continue
            residuals = []
            for event in target_events:
                event_mask = (
                    (times >= max(start, float(event["start_sec"])))
                    & (times < min(end, float(event["end_sec"])))
                    & (pesto[:, 1] >= 0.80)
                    & (pesto[:, 0] > 0)
                )
                if np.any(event_mask):
                    residuals.extend(
                        (
                            100.0
                            * (
                                pesto[event_mask, 0]
                                - int(event["pitch_midi_written"])
                            )
                        ).tolist()
                    )
            if not residuals:
                continue
            measured = float(np.median(np.asarray(residuals, dtype=np.float64)))
            expected = float(label["deviation_cents"])
            error = abs(measured - expected)
            summary["labels_acoustically_measured"] += 1
            summary["absolute_errors"].append(error)
            measured_count += 1
            if error <= 20.0:
                summary["labels_within_20_cents"] += 1
            else:
                failures += 1
        if measured_count == 0:
            record["intonation_acoustic_status"] = "unmeasured_low_confidence"
        elif failures:
            record["intonation_acoustic_status"] = "acoustic_mismatch"
            issues.add(
                "intonation_acoustic_mismatch",
                "high",
                record["sample"],
                f"{failures}/{measured_count} measured labels differ by >20 cents",
                path=record["sample_dir"],
            )
        else:
            record["intonation_acoustic_status"] = "acoustic_verified"
    errors = summary.pop("absolute_errors")
    summary["cents_mae"] = float(statistics.mean(errors)) if errors else None
    summary["within_20_cents_fraction"] = (
        summary["labels_within_20_cents"] / summary["labels_acoustically_measured"]
        if summary["labels_acoustically_measured"]
        else None
    )
    return summary


def _row_for_manifest(record: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "sample": record["sample"],
        "sample_dir": record["sample_dir"],
        "corpus": record["corpus"],
        "root": record["corpus"],
        "source": record["source"],
        "source_group": record["source_group"],
        "split": split,
        "snippet_measures": record["snippet_measures"],
        "recording_kind": record["recording_kind"],
        "audio_render": record["audio_render"],
        "audio_pitch_space": record["audio_pitch_space"],
        "midi_pitch_space": record["midi_pitch_space"],
        "effective_audio_transpose": record["effective_audio_transpose"],
        "duration_sec": record["duration_sec"],
        "error_types": record["error_types"],
        "repeated": record["repeated"],
        "intonation_acoustic_status": record.get(
            "intonation_acoustic_status", "unverified_no_valid_cache"
        ),
        "note_map": record["note_map"],
        "content_fingerprint": record["content_fingerprint"],
        "source_hashes": record["hashes"],
    }


def _assign_groups(
    records: list[dict[str, Any]],
    seed: int,
    *,
    test_sources: Iterable[str] = ("WeberITAV",),
    val_sources: Iterable[str] = ("001",),
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["eligible"]:
            hashes = record.get("hashes") or {}
            components = [
                str(hashes.get(name, ""))
                for name in (
                    "verified_score.musicxml",
                    "performance_score.musicxml",
                    "labels.json",
                )
            ]
            symbolic = (
                hashlib.sha256("|".join(components).encode("ascii")).hexdigest()
                if any(components)
                else str(record["content_fingerprint"])
            )
            record["symbolic_fingerprint"] = symbolic
            by_fingerprint[symbolic].append(record)
    chosen = []
    duplicate_rows = 0
    duplicate_examples = []
    for fingerprint, group in sorted(by_fingerprint.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["priority"]),
                _stable_hex(seed, f"{row['corpus']}:{row['sample_dir']}"),
            ),
        )
        chosen.append(ordered[0])
        duplicate_rows += len(ordered) - 1
        if len(ordered) > 1 and len(duplicate_examples) < MAX_EXAMPLES:
            duplicate_examples.append(
                {
                    "fingerprint": fingerprint,
                    "kept": ordered[0]["sample_dir"],
                    "excluded": [row["sample_dir"] for row in ordered[1:]],
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in chosen:
        groups[record["source_group"]].append(record)
    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    locked_test = {f"score:{str(value).casefold()}" for value in test_sources}
    locked_val = {f"score:{str(value).casefold()}" for value in val_sources}
    for group_id, group in sorted(groups.items()):
        if group_id in locked_test:
            split = "test_id"
        elif group_id in locked_val:
            split = "val"
        else:
            fraction = int(_stable_hex(seed, group_id)[:16], 16) / float(16**16)
            split = "test_id" if fraction < 0.10 else ("val" if fraction < 0.20 else "train")
        assigned[split].extend(group)

    # Very small test fixtures still need three non-empty partitions.
    nonempty_groups = len(groups)
    if nonempty_groups >= 3:
        for split in SPLITS:
            if assigned[split]:
                continue
            donor = max(SPLITS, key=lambda name: len(assigned[name]))
            donor_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in assigned[donor]:
                donor_groups[row["source_group"]].append(row)
            moved_id = min(
                donor_groups,
                key=lambda value: (
                    len(donor_groups[value]),
                    _stable_hex(seed + len(split), value),
                ),
            )
            moved = donor_groups[moved_id]
            assigned[donor] = [
                row for row in assigned[donor] if row["source_group"] != moved_id
            ]
            assigned[split].extend(moved)

    for split in SPLITS:
        assigned[split] = _balanced_order(assigned[split], seed + SPLITS.index(split))
    return assigned, {
        "eligible_before_dedup": sum(len(group) for group in by_fingerprint.values()),
        "eligible_after_dedup": len(chosen),
        "duplicate_rows_excluded": duplicate_rows,
        "symbolic_duplicate_rows_excluded": duplicate_rows,
        "duplicate_examples": duplicate_examples,
        "source_groups": len(groups),
        "locked_test_sources": sorted(locked_test),
        "locked_val_sources": sorted(locked_val),
    }


def _balanced_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    source_buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        source_buckets[str(row["source"])][str(row["corpus"])].append(row)
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for source, corpus_buckets in source_buckets.items():
        for corpus, bucket in corpus_buckets.items():
            bucket.sort(
                key=lambda row: _stable_hex(
                    seed, f"{source}:{corpus}:{row['content_fingerprint']}"
                )
            )
        corpora = sorted(
            corpus_buckets,
            key=lambda value: _stable_hex(seed, f"{source}:{value}"),
        )
        ordered = []
        position = 0
        while True:
            added = False
            for corpus in corpora:
                if position < len(corpus_buckets[corpus]):
                    ordered.append(corpus_buckets[corpus][position])
                    added = True
            if not added:
                break
            position += 1
        source_rows[source] = ordered
    source_order = sorted(
        source_rows, key=lambda value: _stable_hex(seed, value)
    )
    output = []
    position = 0
    while True:
        added = False
        for source in source_order:
            if position < len(source_rows[source]):
                output.append(source_rows[source][position])
                added = True
        if not added:
            return output
        position += 1


def _distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    errors: Counter[str] = Counter()
    for row in values:
        errors.update(row.get("error_types") or [])
    durations = [float(row["duration_sec"]) for row in values if row.get("duration_sec")]
    return {
        "n": len(values),
        "corpus": dict(sorted(Counter(str(row["corpus"]) for row in values).items())),
        "source_groups": len({str(row["source_group"]) for row in values}),
        "top_sources": dict(Counter(str(row["source"]) for row in values).most_common(20)),
        "error_type_counts": dict(sorted(errors.items())),
        "error_type_fraction": {
            key: round(value / max(len(values), 1), 6)
            for key, value in sorted(errors.items())
        },
        "render": dict(sorted(Counter(str(row["audio_render"]) for row in values).items())),
        "pitch_space": dict(
            sorted(
                Counter(
                    f"{row['audio_pitch_space']}:{row['effective_audio_transpose']}"
                    for row in values
                ).items()
            )
        ),
        "repeated_fraction": round(
            sum(bool(row.get("repeated")) for row in values) / max(len(values), 1), 6
        ),
        "duration_sec": {
            "total": round(sum(durations), 3),
            "min": round(min(durations), 3) if durations else None,
            "median": round(statistics.median(durations), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
            "buckets": dict(
                sorted(
                    Counter(
                        "<15"
                        if value < 15
                        else ("15-30" if value < 30 else ("30-60" if value < 60 else ">=60"))
                        for value in durations
                    ).items()
                )
            ),
        },
    }


def _audit_existing_manifests(
    manifests: list[dict[str, Any]],
    records: list[dict[str, Any]],
    issues: IssueLog,
) -> list[dict[str, Any]]:
    record_by_path = {_norm(row["sample_dir"]): row for row in records}
    output = []
    for item in manifests:
        path: Path = item["path"]
        document: dict[str, Any] = item["document"]
        split_rows: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "val", "test_id", "test_ood", "test"):
            rows = []
            for raw in document.get(split) or []:
                if not isinstance(raw, dict):
                    continue
                record = record_by_path.get(_norm(str(raw.get("sample_dir", ""))))
                rows.append(
                    {
                        "path": _norm(str(raw.get("sample_dir", raw.get("sample", "")))),
                        "source_group": (
                            record["source_group"]
                            if record
                            else f"declared:{str(raw.get('source') or 'unknown').casefold()}"
                        ),
                        "source": str(raw.get("source") or (record or {}).get("source") or "unknown"),
                        "fingerprint": (record or {}).get("content_fingerprint"),
                    }
                )
            if rows:
                split_rows[split] = rows
        source_membership: dict[str, set[str]] = defaultdict(set)
        path_membership: dict[str, set[str]] = defaultdict(set)
        fingerprint_membership: dict[str, set[str]] = defaultdict(set)
        for split, rows in split_rows.items():
            for row in rows:
                source_membership[row["source_group"]].add(split)
                path_membership[row["path"]].add(split)
                if row["fingerprint"]:
                    fingerprint_membership[str(row["fingerprint"])].add(split)
        leaked_sources = {
            key: sorted(value) for key, value in source_membership.items() if len(value) > 1
        }
        leaked_paths = {
            key: sorted(value) for key, value in path_membership.items() if len(value) > 1
        }
        leaked_fingerprints = {
            key: sorted(value)
            for key, value in fingerprint_membership.items()
            if len(value) > 1
        }
        if leaked_sources:
            issues.add(
                "frozen_manifest_source_leakage",
                "critical",
                path.parent.name,
                f"{len(leaked_sources)} source groups occur in multiple splits",
                path=str(path),
            )
        if leaked_paths or leaked_fingerprints:
            issues.add(
                "frozen_manifest_exact_leakage",
                "critical",
                path.parent.name,
                f"paths={len(leaked_paths)} fingerprints={len(leaked_fingerprints)}",
                path=str(path),
            )
        prefixes = {}
        concentrated = False
        for split, maximum in (("train", 1000), ("val", 200), ("test_id", 200), ("test", 200)):
            rows = split_rows.get(split, [])[:maximum]
            if not rows:
                continue
            counts = Counter(row["source"] for row in rows)
            top_source, top_count = counts.most_common(1)[0]
            share = top_count / len(rows)
            prefixes[split] = {
                "n": len(rows),
                "unique_sources": len(counts),
                "top_source": top_source,
                "top_source_count": top_count,
                "top_source_fraction": round(share, 6),
            }
            if len(rows) >= 20 and share >= 0.80:
                concentrated = True
        if concentrated:
            issues.add(
                "frozen_manifest_prefix_source_concentration",
                "critical",
                path.parent.name,
                "a commonly used first-N subset is >=80% one source",
                path=str(path),
            )
        output.append(
            {
                "path": str(path),
                "split_sizes": {key: len(value) for key, value in split_rows.items()},
                "source_groups_crossing_splits": len(leaked_sources),
                "exact_paths_crossing_splits": len(leaked_paths),
                "exact_fingerprints_crossing_splits": len(leaked_fingerprints),
                "source_leak_examples": dict(list(leaked_sources.items())[:MAX_EXAMPLES]),
                "prefix_distributions": prefixes,
            }
        )
    return output


def _write_target_cache(
    path: Path, split_records: Mapping[str, list[dict[str, Any]]]
) -> tuple[str, dict[tuple[str, str], int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    positions: dict[tuple[str, str], int] = {}
    ordinal = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for split in SPLITS:
                for record in split_records[split]:
                    note_map = record["_note_map_document"] or {}
                    usable_labels = []
                    for label in record["_valid_labels"]:
                        if (
                            label.get("type") == "repetition"
                            and not record.get("repeat_consistent", True)
                        ):
                            continue
                        if label.get("type") == "intonation_error" and record.get(
                            "intonation_acoustic_status"
                        ) != "acoustic_verified":
                            continue
                        usable_labels.append(
                            {
                                key: label[key]
                                for key in (
                                    "id",
                                    "type",
                                    "start_time",
                                    "end_time",
                                    "deviation_cents",
                                    "extra_copies",
                                    "repeats_label_range",
                                    "score_part",
                                )
                                if key in label
                            }
                        )
                    payload = {
                        "schema_version": 1,
                        "release": RELEASE_VERSION,
                        "ordinal": ordinal,
                        "split": split,
                        "sample": record["sample"],
                        "corpus": record["corpus"],
                        "sample_dir": record["sample_dir"],
                        "source_group": record["source_group"],
                        "source_hashes": record["hashes"],
                        "pitch_policy": {
                            "target_space": "written_midi",
                            "audio_pitch_space": record["audio_pitch_space"],
                            "effective_audio_transpose": record["effective_audio_transpose"],
                        },
                        "clean_notes": note_map.get("clean_notes") or [],
                        "performed_notes": note_map.get("performed_notes") or [],
                        "rendered_notes": note_map.get("rendered_notes") or [],
                        "deleted_clean_notes": note_map.get("deleted_clean_notes") or [],
                        "usable_labels": usable_labels,
                        "excluded_label_counts": {
                            "invalid_time": int(record["invalid_label_count"]),
                            "inconsistent_repetition": sum(
                                label.get("type") == "repetition"
                                for label in record["_valid_labels"]
                            )
                            if not record.get("repeat_consistent", True)
                            else 0,
                            "unverified_intonation": sum(
                                label.get("type") == "intonation_error"
                                for label in record["_valid_labels"]
                            )
                            if record.get("intonation_acoustic_status") != "acoustic_verified"
                            else 0
                        },
                    }
                    compressed.write(
                        (
                            json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                            + "\n"
                        ).encode("utf-8")
                    )
                    positions[(record["corpus"], record["sample_dir"])] = ordinal
                    ordinal += 1
    os.replace(temporary, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest(), positions


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_sqlite_target_cache(archive: Path, destination: Path) -> str:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE targets (
                ordinal INTEGER PRIMARY KEY,
                split TEXT NOT NULL,
                corpus TEXT NOT NULL,
                sample_dir TEXT NOT NULL,
                source_hashes TEXT NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE UNIQUE INDEX target_identity
                ON targets(corpus, sample_dir);
            """
        )
        batch = []
        with gzip.open(archive, "rt", encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                batch.append(
                    (
                        int(payload["ordinal"]),
                        str(payload["split"]),
                        str(payload["corpus"]),
                        _norm(str(payload["sample_dir"])),
                        json.dumps(
                            payload.get("source_hashes") or {},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        zlib.compress(
                            json.dumps(
                                payload, separators=(",", ":"), ensure_ascii=False
                            ).encode("utf-8"),
                            level=6,
                        ),
                    )
                )
                if len(batch) >= 256:
                    connection.executemany(
                        "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?)", batch
            )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, destination)
    return _file_sha256(destination)


def _focused_intonation_reference(repo: Path) -> dict[str, Any] | None:
    path = (
        repo
        / "align-model"
        / "runs"
        / "basic-pitch-refiner-procedural-s365"
        / "intonation-audit.json"
    )
    if not path.is_file():
        return None
    try:
        document = _json(path)
        rows = list(document.get("rows") or [])
        summary = dict(document.get("summary") or {})
    except Exception:
        return None
    summary.update(
        {
            "source": str(path),
            "source_sha256": _file_sha256(path),
            "label_matches_within_20_cents": sum(
                float(row.get("absolute_error", math.inf)) <= 20.0 for row in rows
            ),
            "label_match_within_20_cents_fraction": (
                sum(
                    float(row.get("absolute_error", math.inf)) <= 20.0
                    for row in rows
                )
                / len(rows)
                if rows
                else None
            ),
        }
    )
    return summary


def _rewrite_cached_targets(
    path: Path,
    assigned: Mapping[str, list[dict[str, Any]]],
    payloads: Mapping[tuple[str, str], dict[str, Any]],
) -> tuple[str, dict[tuple[str, str], int]]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    positions: dict[tuple[str, str], int] = {}
    ordinal = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for split in SPLITS:
                for record in assigned[split]:
                    key = (str(record["corpus"]), _norm(str(record["sample_dir"])))
                    if key not in payloads:
                        raise KeyError(f"No audited target payload for {key}")
                    payload = dict(payloads[key])
                    payload["split"] = split
                    payload["ordinal"] = ordinal
                    compressed.write(
                        (
                            json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                            + "\n"
                        ).encode("utf-8")
                    )
                    positions[(record["corpus"], record["sample_dir"])] = ordinal
                    ordinal += 1
    os.replace(temporary, path)
    return _file_sha256(path), positions


def repartition_existing(output: Path, seed: int) -> dict[str, Path]:
    """Repartition a completed audit without rereading source corpora."""

    output = output.resolve()
    records = _json(output / "bundle_index.json")
    if not isinstance(records, list):
        raise ValueError("bundle_index.json must contain a list")
    target_path = output / "validated_targets.jsonl.gz"
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    with gzip.open(target_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            payloads[
                (str(payload["corpus"]), _norm(str(payload["sample_dir"])))
            ] = payload
    assigned, dedup = _assign_groups(records, seed)
    target_sha256, positions = _rewrite_cached_targets(
        target_path, assigned, payloads
    )
    target_db = output / "validated_targets.sqlite"
    target_db_sha256 = _write_sqlite_target_cache(target_path, target_db)

    manifest_path = output / "split.json"
    manifest = _json(manifest_path)
    for split in SPLITS:
        rows = []
        for record in assigned[split]:
            row = _row_for_manifest(record, split)
            row["target_cache"] = str(target_path)
            row["target_db"] = str(target_db)
            row["target_record"] = positions[(record["corpus"], record["sample_dir"])]
            rows.append(row)
        manifest[split] = rows
    manifest["seed"] = seed
    manifest["policy"].update(
        {
            "locked_test_sources": ["WeberITAV"],
            "locked_validation_sources": ["001"],
            "duplicate_policy": "one canonical row per symbolic target fingerprint by corpus priority",
            "ordering": "deterministic corpus/source round-robin; safe for first-N consumers",
        }
    )
    manifest["target_cache"]["sha256"] = target_sha256
    manifest["target_cache"].update(
        {
            "sqlite_path": str(target_db),
            "sqlite_sha256": target_db_sha256,
            "sqlite_format": "zlib JSON payload keyed by target_record ordinal",
        }
    )
    manifest["distribution"] = {
        split: _distribution(rows) for split, rows in assigned.items()
    }
    source_sets = {
        split: {row["source_group"] for row in rows}
        for split, rows in assigned.items()
    }
    fingerprint_sets = {
        split: {row["content_fingerprint"] for row in rows}
        for split, rows in assigned.items()
    }
    manifest["integrity"] = {
        "source_group_overlap": {
            f"{left}:{right}": sorted(source_sets[left] & source_sets[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        },
        "content_fingerprint_overlap": {
            f"{left}:{right}": sorted(
                fingerprint_sets[left] & fingerprint_sets[right]
            )
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        },
        "all_overlaps_empty": True,
    }
    if any(manifest["integrity"]["source_group_overlap"].values()) or any(
        manifest["integrity"]["content_fingerprint_overlap"].values()
    ):
        raise RuntimeError("repartitioning introduced leakage")
    manifest["deduplication"] = dedup
    _atomic_json(manifest_path, manifest)
    manifest_sha256 = _file_sha256(manifest_path)

    protocol_path = output / "heldout_protocol.json"
    protocol = _json(protocol_path)
    protocol.update(
        {
            "manifest_sha256": manifest_sha256,
            "target_cache_sha256": target_sha256,
            "target_db": str(target_db),
            "target_db_sha256": target_db_sha256,
            "split_counts": {
                split: len(rows) for split, rows in assigned.items()
            },
            "integrity": manifest["integrity"],
        }
    )
    _atomic_json(protocol_path, protocol)

    report_path = output / "audit_report.json"
    report = _json(report_path)
    report["summary"].update(
        {
            "eligible_before_dedup": dedup["eligible_before_dedup"],
            "eligible_after_dedup": dedup["eligible_after_dedup"],
            "split_sizes": {
                split: len(rows) for split, rows in assigned.items()
            },
        }
    )
    report["new_manifest_integrity"] = manifest["integrity"]
    report["deduplication"] = dedup
    focused = _focused_intonation_reference(output.parents[2])
    if focused is not None:
        report["intonation_audit"] = {
            "all_corpora_labels_total": report.get("intonation_audit", {}).get(
                "labels_total"
            ),
            "focused_hash_matched_pesto_audit": focused,
            "training_policy": "no intonation label is usable without acoustic verification",
        }
    report["artifacts"].update(
        {
            "manifest_sha256": manifest_sha256,
            "target_cache_sha256": target_sha256,
            "target_db": str(target_db),
            "target_db_sha256": target_db_sha256,
        }
    )
    _atomic_json(report_path, report)
    _atomic_json(output / "bundle_index.json", records)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return {
        "report": report_path,
        "manifest": manifest_path,
        "target_cache": target_path,
        "protocol": protocol_path,
    }


def verify_release(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest_path = output / "split.json"
    target_path = output / "validated_targets.jsonl.gz"
    target_db = output / "validated_targets.sqlite"
    protocol = _json(output / "heldout_protocol.json")
    manifest = _json(manifest_path)
    actual_manifest_hash = _file_sha256(manifest_path)
    actual_target_hash = _file_sha256(target_path)
    actual_target_db_hash = _file_sha256(target_db)
    if actual_manifest_hash != protocol.get("manifest_sha256"):
        raise ValueError("split.json SHA-256 does not match heldout_protocol.json")
    if actual_target_hash != protocol.get("target_cache_sha256"):
        raise ValueError(
            "validated_targets.jsonl.gz SHA-256 does not match heldout_protocol.json"
        )
    if actual_target_db_hash != protocol.get("target_db_sha256"):
        raise ValueError(
            "validated_targets.sqlite SHA-256 does not match heldout_protocol.json"
        )

    expected = []
    prefix_sources = {}
    source_sets = {}
    fingerprint_sets = {}
    for split in SPLITS:
        rows = list(manifest.get(split) or [])
        expected.extend((split, row) for row in rows)
        source_sets[split] = {str(row["source_group"]) for row in rows}
        fingerprint_sets[split] = {
            str(row["content_fingerprint"]) for row in rows
        }
        prefix_sources[split] = dict(
            Counter(str(row["source"]) for row in rows[:1000]).most_common()
        )
    count = 0
    usable_labels: Counter[str] = Counter()
    excluded_labels: Counter[str] = Counter()
    with gzip.open(target_path, "rt", encoding="utf-8") as stream:
        for count, line in enumerate(stream, start=1):
            payload = json.loads(line)
            split, row = expected[count - 1]
            if int(payload["ordinal"]) != count - 1:
                raise ValueError(f"target ordinal mismatch at record {count - 1}")
            if payload["split"] != split:
                raise ValueError(f"target split mismatch at record {count - 1}")
            if _norm(payload["sample_dir"]) != _norm(row["sample_dir"]):
                raise ValueError(f"target sample mismatch at record {count - 1}")
            if payload.get("source_hashes") != row.get("source_hashes"):
                raise ValueError(f"target hash set mismatch at record {count - 1}")
            usable_labels.update(
                str(label.get("type"))
                for label in payload.get("usable_labels") or []
                if label.get("type")
            )
            excluded_labels.update(
                {
                    str(key): int(value)
                    for key, value in (
                        payload.get("excluded_label_counts") or {}
                    ).items()
                }
            )
    if count != len(expected):
        raise ValueError(f"target count={count}, manifest rows={len(expected)}")
    connection = sqlite3.connect(f"file:{target_db.as_posix()}?mode=ro", uri=True)
    try:
        database_count = int(
            connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        )
        if database_count != len(expected):
            raise ValueError(
                f"SQLite target count={database_count}, manifest rows={len(expected)}"
            )
        for ordinal in {0, max(0, len(expected) // 2), max(0, len(expected) - 1)}:
            saved = connection.execute(
                "SELECT split, corpus, sample_dir, source_hashes, payload "
                "FROM targets WHERE ordinal=?",
                (ordinal,),
            ).fetchone()
            if saved is None:
                raise ValueError(f"SQLite target {ordinal} is missing")
            payload = json.loads(zlib.decompress(saved[4]).decode("utf-8"))
            split, row = expected[ordinal]
            if (
                saved[0] != split
                or saved[1] != row["corpus"]
                or saved[2] != _norm(row["sample_dir"])
                or payload["ordinal"] != ordinal
            ):
                raise ValueError(f"SQLite target {ordinal} does not match manifest")
    finally:
        connection.close()

    overlaps = {
        f"{left}:{right}": {
            "source_groups": len(source_sets[left] & source_sets[right]),
            "fingerprints": len(
                fingerprint_sets[left] & fingerprint_sets[right]
            ),
        }
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(
        value["source_groups"] or value["fingerprints"]
        for value in overlaps.values()
    ):
        raise ValueError(f"release leakage detected: {overlaps}")
    result = {
        "status": "ok",
        "manifest_sha256": actual_manifest_hash,
        "target_cache_sha256": actual_target_hash,
        "target_db_sha256": actual_target_db_hash,
        "target_records": count,
        "split_counts": {
            split: len(manifest.get(split) or []) for split in SPLITS
        },
        "overlaps": overlaps,
        "first_1000_source_counts": prefix_sources,
        "usable_label_counts": dict(sorted(usable_labels.items())),
        "excluded_label_counts": dict(sorted(excluded_labels.items())),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(args: argparse.Namespace) -> dict[str, Path]:
    repo = args.repo.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hash_cache_path = output / "hash_cache.json"
    try:
        hash_cache = _json(hash_cache_path) if hash_cache_path.is_file() else {}
    except Exception:
        hash_cache = {}
    issues = IssueLog()

    manifest_paths = _discover_manifests(repo)
    note_map_lookup, metadata_overrides, manifest_documents = _manifest_maps(
        manifest_paths
    )
    records = []
    corpus_inventory = {}
    for corpus, root, priority in _corpus_specs(repo):
        bundles = _discover_bundles(root)
        corpus_inventory[corpus] = {
            "root": str(root.resolve()),
            "mounted": root.is_dir(),
            "bundles_discovered": len(bundles),
        }
        if not root.is_dir():
            issues.add(
                "corpus_root_missing",
                "warning",
                corpus,
                "root is not mounted/present",
                path=str(root),
            )
            continue
        for index, sample_dir in enumerate(bundles, 1):
            records.append(
                _audit_bundle(
                    corpus,
                    root,
                    priority,
                    sample_dir,
                    note_map_lookup,
                    metadata_overrides,
                    hash_cache,
                    issues,
                )
            )
            if index % 500 == 0 or index == len(bundles):
                print(f"{corpus}: {index}/{len(bundles)} bundles", flush=True)

    cache_summary, cache_matches = _cache_audit(repo, records, issues)
    intonation_summary = _acoustic_intonation(records, cache_matches, issues)
    focused_intonation = _focused_intonation_reference(repo)
    if focused_intonation is not None:
        intonation_summary["focused_hash_matched_pesto_audit"] = focused_intonation
    existing_manifest_audit = _audit_existing_manifests(
        manifest_documents, records, issues
    )
    assigned, dedup = _assign_groups(records, args.seed)

    target_path = output / "validated_targets.jsonl.gz"
    target_sha256, target_positions = _write_target_cache(target_path, assigned)
    target_db = output / "validated_targets.sqlite"
    target_db_sha256 = _write_sqlite_target_cache(target_path, target_db)
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        manifest_rows[split] = []
        for record in assigned[split]:
            row = _row_for_manifest(record, split)
            row["target_cache"] = str(target_path)
            row["target_db"] = str(target_db)
            row["target_record"] = target_positions[(record["corpus"], record["sample_dir"])]
            manifest_rows[split].append(row)

    selected_groups = {
        split: {row["source_group"] for row in assigned[split]} for split in SPLITS
    }
    group_overlap = {
        f"{left}:{right}": sorted(selected_groups[left] & selected_groups[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    fingerprint_sets = {
        split: {row["content_fingerprint"] for row in assigned[split]} for split in SPLITS
    }
    fingerprint_overlap = {
        f"{left}:{right}": sorted(fingerprint_sets[left] & fingerprint_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(group_overlap.values()) or any(fingerprint_overlap.values()):
        raise RuntimeError("internal error: generated manifest is not leakage-safe")

    external_eval = [
        {
            "sample": row["sample"],
            "sample_dir": row["sample_dir"],
            "corpus": row["corpus"],
            "recording_kind": row["recording_kind"],
            "status": "audit_only_not_synthetic_gold",
            "source_hashes": row["hashes"],
        }
        for row in records
        if row["corpus"] == "datacreate_real"
    ]
    manifest = {
        "version": 2,
        "release": RELEASE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "roots": {
            corpus: details["root"] for corpus, details in corpus_inventory.items()
        },
        "policy": {
            "split_unit": "source_group",
            "source_group_rule": "raw snippets share source score; generated scores use clean-score SHA-256",
            "split_hash_fractions": {"train": 0.80, "val": 0.10, "test_id": 0.10},
            "locked_test_sources": ["WeberITAV"],
            "locked_validation_sources": ["001"],
            "ordering": "deterministic source round-robin; safe for first-N consumers",
            "duplicate_policy": "one canonical row per symbolic target fingerprint by corpus priority",
            "corpus_priority": [name for name, _root, _priority in _corpus_specs(repo)],
            "required_gold": "validated rendered note_map lineage",
            "intonation_policy": "only acoustically verified PESTO labels appear in usable_labels",
            "target_pitch_space": "written MIDI",
            "test_id_semantics": "locked source-held-out synthetic evaluation",
            "real_data_policy": "DataCreate samples are audit-only external sanity data",
        },
        "target_cache": {
            "path": str(target_path),
            "sha256": target_sha256,
            "format": "gzip JSONL; one exact record per manifest row",
            "sqlite_path": str(target_db),
            "sqlite_sha256": target_db_sha256,
            "sqlite_format": "zlib JSON payload keyed by target_record ordinal",
        },
        **manifest_rows,
        "external_eval": external_eval,
        "distribution": {
            split: _distribution(rows) for split, rows in assigned.items()
        },
        "integrity": {
            "source_group_overlap": group_overlap,
            "content_fingerprint_overlap": fingerprint_overlap,
            "all_overlaps_empty": True,
        },
        "deduplication": dedup,
        "excluded": {
            "not_eligible": sum(not row["eligible"] for row in records),
            "by_corpus": dict(
                sorted(
                    Counter(row["corpus"] for row in records if not row["eligible"]).items()
                )
            ),
        },
    }
    manifest_path = output / "split.json"
    _atomic_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path, {})

    issue_rows = issues.report()
    report = {
        "schema_version": 2,
        "release": RELEASE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "scope": corpus_inventory,
        "summary": {
            "bundles_discovered": len(records),
            "synthetic_bundles": sum(row["synthetic"] for row in records),
            "eligible_before_dedup": dedup["eligible_before_dedup"],
            "eligible_after_dedup": dedup["eligible_after_dedup"],
            "split_sizes": {split: len(rows) for split, rows in assigned.items()},
            "issue_counts_by_severity": dict(
                sorted(
                    Counter(
                        row["severity"]
                        for row in issue_rows
                        for _ in range(int(row["count"]))
                    ).items()
                )
            ),
            "issue_codes": len(issue_rows),
        },
        "issues": issue_rows,
        "corpus_distribution_all": {
            corpus: _distribution(
                row for row in records if row["corpus"] == corpus
            )
            for corpus in corpus_inventory
        },
        "note_map_totals": {
            "valid": sum(row["note_map_valid"] for row in records),
            "invalid_or_missing": sum(
                row["synthetic"] and not row["note_map_valid"] for row in records
            ),
            "clean_notes": sum(
                int(row["note_map_stats"].get("clean_notes", 0)) for row in records
            ),
            "rendered_notes": sum(
                int(row["note_map_stats"].get("rendered_notes", 0)) for row in records
            ),
            "tied_notes_folded": sum(
                int(row["note_map_stats"].get("tied_notes_folded", 0))
                for row in records
            ),
            "copy_events": sum(
                int(row["note_map_stats"].get("copy_events", 0)) for row in records
            ),
        },
        "cache_audit": cache_summary,
        "intonation_audit": intonation_summary,
        "existing_frozen_manifests": existing_manifest_audit,
        "new_manifest_integrity": manifest["integrity"],
        "artifacts": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "target_cache": str(target_path),
            "target_cache_sha256": target_sha256,
            "target_db": str(target_db),
            "target_db_sha256": target_db_sha256,
            "heldout_protocol": str(output / "heldout_protocol.json"),
        },
        "limitations": [
            "WAV duration uses headers only; audio is never loaded by this audit.",
            "MIDI pitch-bend checks validate render control, not acoustic output.",
            "Acoustic intonation is accepted only when a valid hash-matched Basic Pitch and PESTO cache exists.",
            "A target F1 >= 0.80 is a model goal, not a result established by a data audit.",
        ],
    }
    report_path = output / "audit_report.json"
    _atomic_json(report_path, report)

    protocol = {
        "version": 1,
        "release": RELEASE_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "target_cache": str(target_path),
        "target_cache_sha256": target_sha256,
        "target_db": str(target_db),
        "target_db_sha256": target_db_sha256,
        "locked_split": "test_id",
        "rules": [
            "Develop models and all thresholds using train and val only.",
            "Do not inspect per-sample test_id outputs before freezing the model, decoder, and thresholds.",
            "Verify manifest and target-cache SHA-256 values before every final evaluation.",
            "Report micro note mapping precision/recall/F1 and macro F1 across source_group.",
            "Use written-pitch note matching and the repository's frozen onset/offset tolerances.",
            "Report transcription F1 and end-to-end transcription-to-score alignment F1 separately.",
            "Report error-class F1 and repetition-span F1 as secondary metrics.",
            "Do not score unverified synthetic intonation labels; report acoustic intonation only on acoustically verified targets.",
            "Use external_eval only as a separately named real-audio sanity set, never for model selection.",
            "Any test_id-driven change creates a new release and requires a new untouched source-group holdout.",
        ],
        "success_target": {
            "metric": "end_to_end_transcription_to_score_alignment_micro_f1",
            "threshold": 0.80,
            "status": "target_not_yet_demonstrated",
        },
        "split_counts": {split: len(rows) for split, rows in assigned.items()},
        "integrity": manifest["integrity"],
    }
    _atomic_json(output / "heldout_protocol.json", protocol)
    _atomic_json(hash_cache_path, hash_cache)
    _atomic_json(
        output / "bundle_index.json",
        [_public_record(record) for record in records],
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    return {
        "report": report_path,
        "manifest": manifest_path,
        "target_cache": target_path,
        "protocol": output / "heldout_protocol.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--repartition-only",
        action="store_true",
        help="Reuse a completed bundle audit and only rebuild split/cache ordering",
    )
    parser.add_argument(
        "--verify-release",
        action="store_true",
        help="Verify release hashes, target ordinals, and split isolation",
    )
    args = parser.parse_args(argv)
    if args.verify_release:
        verify_release(args.out)
    elif args.repartition_only:
        repartition_existing(args.out, args.seed)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
