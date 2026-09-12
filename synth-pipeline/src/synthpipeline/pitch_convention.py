"""Bb-clarinet pitch-space convention shared by generation, targets, and inference.

MusicXML and transcriber outputs are written pitch. Rendered WAV is sounding
pitch: written minus two semitones. MIDI files may be either written (legacy
``soundfont_rerender``) or sounding (current generation / ``oscillator_v1``).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

DEFAULT_SOUNDING_TRANSPOSE = -2
WRITTEN_MIDI_RENDERS = frozenset({"soundfont_rerender"})
SOUNDING_MIDI_RENDERS = frozenset({"oscillator_v1", "current", None, ""})
_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def load_bundle_metadata(sample_dir: Path | str) -> dict[str, Any]:
    path = Path(sample_dir) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sounding_transpose(metadata: dict[str, Any] | None) -> int:
    raw = (metadata or {}).get("sounding_transpose", DEFAULT_SOUNDING_TRANSPOSE)
    if raw is None:
        return DEFAULT_SOUNDING_TRANSPOSE
    return int(raw)


def midi_pitch_space(metadata: dict[str, Any] | None) -> str:
    """Return ``written`` or ``sounding`` for the on-disk MIDI file."""

    meta = metadata or {}
    explicit = meta.get("midi_pitch_space")
    if explicit in {"written", "sounding"}:
        return str(explicit)
    render = meta.get("audio_render")
    if render in WRITTEN_MIDI_RENDERS:
        return "written"
    return "sounding"


def midi_to_written_shift(metadata: dict[str, Any] | None) -> int:
    """Semitones to add to MIDI keys to recover written pitch."""

    if midi_pitch_space(metadata) == "written":
        return 0
    return -sounding_transpose(metadata)


def written_to_sounding_shift(metadata: dict[str, Any] | None) -> int:
    return sounding_transpose(metadata)


def effective_audio_transpose(metadata: dict[str, Any] | None) -> int:
    """Semitones added to detected WAV pitch to recover written pitch."""

    meta = metadata or {}
    explicit = meta.get("effective_audio_transpose")
    if explicit is not None:
        return int(explicit)
    if meta.get("audio_pitch_space") == "written":
        return 0
    # Legacy bundles only recorded the written-to-sounding render transpose.
    return -sounding_transpose(meta)


def audio_pitch_space(metadata: dict[str, Any] | None) -> str:
    """Return the declared WAV pitch space, preserving legacy inference."""

    meta = metadata or {}
    explicit = meta.get("audio_pitch_space")
    if explicit in {"written", "sounding", "transposed"}:
        return str(explicit)
    shift = effective_audio_transpose(meta)
    if shift == 0:
        return "written"
    if shift == -sounding_transpose(meta):
        return "sounding"
    return "transposed"


def audio_to_written_shift(metadata: dict[str, Any] | None) -> int:
    """Semitones to add to detected WAV pitch to recover written pitch.

    This compatibility alias now honors explicit acoustic metadata and falls
    back to the legacy sounding-transpose convention.
    """

    return effective_audio_transpose(metadata)


def annotate_pitch_metadata(
    metadata: dict[str, Any] | None,
    *,
    midi_space: str | None = None,
    audio_space: str | None = None,
    effective_audio_shift: int | None = None,
    audio_render: str | None = None,
) -> dict[str, Any]:
    out = dict(metadata or {})
    out["sounding_transpose"] = sounding_transpose(out)
    space = midi_space or midi_pitch_space(out)
    if space not in {"written", "sounding"}:
        raise ValueError(f"midi_space must be written or sounding, got {space!r}")
    out["midi_pitch_space"] = space
    if audio_render is not None:
        out["audio_render"] = audio_render
    elif not out.get("audio_render"):
        out["audio_render"] = (
            "oscillator_v1" if space == "sounding" else "soundfont_rerender"
        )
    shift = (
        effective_audio_transpose(out)
        if effective_audio_shift is None
        else int(effective_audio_shift)
    )
    acoustic_space = audio_space
    if acoustic_space is None:
        acoustic_space = out.get("audio_pitch_space")
    if acoustic_space is None:
        acoustic_space = (
            "written"
            if shift == 0
            else (
                "sounding"
                if shift == -sounding_transpose(out)
                else "transposed"
            )
        )
    if acoustic_space not in {"written", "sounding", "transposed"}:
        raise ValueError(
            "audio_space must be written, sounding, or transposed, "
            f"got {acoustic_space!r}"
        )
    out["audio_pitch_space"] = str(acoustic_space)
    out["effective_audio_transpose"] = shift
    return out


def _midi_keys(path: Path, limit: int = 24) -> list[int]:
    if not path.exists():
        return []
    try:
        from tinysoundfont.midi import NoteOn, load

        keys: list[int] = []
        for ev in load(str(path), persistent=False):
            if isinstance(ev.action, NoteOn):
                keys.append(int(ev.action.key))
                if len(keys) >= limit:
                    break
        if keys:
            return keys
    except Exception:
        pass
    from synthpipeline.timing import midi_note_times

    return [pitch for pitch, _start, _end in midi_note_times(path)[:limit]]


def _xml_written_pitches(path: Path, limit: int = 24) -> list[int]:
    if not path.exists():
        return []
    tree = ET.parse(path)
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    pitches: list[int] = []
    for pitch in root.iter(f"{ns}pitch"):
        step = pitch.findtext(f"{ns}step")
        octave = pitch.findtext(f"{ns}octave")
        if step is None or octave is None or step not in _STEP:
            continue
        alter = pitch.findtext(f"{ns}alter") or "0"
        pitches.append(12 * (int(octave) + 1) + _STEP[step] + int(round(float(alter))))
        if len(pitches) >= limit:
            break
    return pitches


def _collapse_repeats(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def infer_midi_offset(midi_path: Path | str, xml_path: Path | str) -> int | None:
    """Median MIDI-minus-written-XML offset, or None if either sequence is empty."""

    mid = _collapse_repeats(_midi_keys(Path(midi_path)))
    xml = _collapse_repeats(_xml_written_pitches(Path(xml_path)))
    if not mid or not xml:
        return None
    count = min(16, len(mid), len(xml))
    diffs = sorted(mid[i] - xml[i] for i in range(count))
    return int(round(diffs[count // 2]))


def infer_midi_pitch_space(
    sample_dir: Path | str,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, int | None]:
    """Infer MIDI pitch space from files; fall back to metadata when uncertain."""

    sample_dir = Path(sample_dir)
    meta = metadata if metadata is not None else load_bundle_metadata(sample_dir)
    offset = infer_midi_offset(
        sample_dir / "performance_audio.mid",
        sample_dir / "performance_score.musicxml",
    )
    if offset is None:
        return midi_pitch_space(meta), None
    if offset == 0:
        return "written", offset
    if offset == -2:
        return "sounding", offset
    if abs(offset) == 1:
        return midi_pitch_space(meta), offset
    return ("sounding" if offset < 0 else "written"), offset


def audit_bundle_pitch(sample_dir: Path | str) -> dict[str, Any]:
    """Read-only pitch-space report for one bundle."""

    sample_dir = Path(sample_dir)
    metadata = load_bundle_metadata(sample_dir)
    declared = midi_pitch_space(metadata)
    inferred, offset = infer_midi_pitch_space(sample_dir, metadata)
    status = "ok"
    if offset is None:
        status = "unknown"
    elif declared != inferred and abs(int(offset)) >= 2:
        status = "mismatch"
    elif declared != inferred:
        status = "ambiguous"
    return {
        "sample": sample_dir.name,
        "sample_dir": str(sample_dir),
        "declared_space": declared,
        "inferred_space": inferred,
        "midi_minus_xml": offset,
        "sounding_transpose": sounding_transpose(metadata),
        "audio_pitch_space": audio_pitch_space(metadata),
        "effective_audio_transpose": effective_audio_transpose(metadata),
        "audio_render": metadata.get("audio_render"),
        "midi_to_written_shift": midi_to_written_shift(metadata),
        "status": status,
    }
