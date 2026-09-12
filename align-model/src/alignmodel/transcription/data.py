from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

FRAME_HOP_SEC = 512.0 / 22050.0


@dataclass(frozen=True)
class TranscriptionExample:
    sample_dir: Path
    split: str

    @property
    def sample_id(self) -> str:
        return self.sample_dir.name


def _bundle_ok(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "performance_mel.npy").exists()
        and (path / "performance_audio.mid").exists()
    )


def _all_bundles(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        out.extend(p for p in sorted(root.iterdir()) if _bundle_ok(p))
    return out


def _manifest_rows(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    doc = json.loads(text)
    if isinstance(doc, list):
        return [row if isinstance(row, dict) else {"sample": row} for row in doc]
    if not isinstance(doc, dict):
        raise ValueError("Split manifest must be a JSON object, list, or JSONL")
    rows: list[dict] = []
    for split, values in doc.items():
        if split in {"roots", "version", "metadata"} or not isinstance(values, list):
            continue
        for value in values:
            row = dict(value) if isinstance(value, dict) else {"sample": value}
            row.setdefault("split", split)
            rows.append(row)
    return rows


def _resolve_manifest_sample(row: dict, roots: list[Path]) -> Path:
    raw = row.get("sample_dir", row.get("path", row.get("sample", row.get("id"))))
    if raw is None:
        raise ValueError(f"Manifest row lacks sample/path/id: {row}")
    candidate = Path(str(raw))
    if candidate.is_absolute() and _bundle_ok(candidate):
        return candidate

    selected_roots = roots
    root_hint = row.get("root")
    if root_hint is not None:
        if isinstance(root_hint, int) or str(root_hint).isdigit():
            selected_roots = [roots[int(root_hint)]]
        else:
            hinted = Path(str(root_hint))
            selected_roots = [hinted] if hinted.is_absolute() else [
                r for r in roots if r.name == str(root_hint)
            ]
    matches = [root / candidate for root in selected_roots if _bundle_ok(root / candidate)]
    if not matches and candidate.parent != Path("."):
        matches = [
            root / candidate.name
            for root in selected_roots
            if _bundle_ok(root / candidate.name)
        ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Manifest sample {raw!r} not found under supplied roots")
    raise ValueError(
        f"Manifest sample {raw!r} is ambiguous across roots; add a root field"
    )


def load_split(
    roots: Iterable[Path | str],
    manifest: Path | str | None,
    split: str,
    *,
    seed: int = 365,
    val_fraction: float = 0.1,
) -> list[TranscriptionExample]:
    """Resolve one split from a manifest and any number of dataset roots.

    Accepted manifests are ``{"train": [...], "val": [...]}``, a list of
    ``{"sample": ..., "split": ...}`` records, or equivalent JSONL. A record
    may carry ``root`` (root index, root directory name, or absolute path) to
    disambiguate duplicate sample IDs. Future human recordings use the same
    schema with ``recording_kind: "human"``; do not treat those rows as
    evidence of real-audio performance until they are actually supplied.
    """

    roots_p = [Path(root) for root in roots]
    if not roots_p:
        raise ValueError("At least one data root is required")
    if manifest is not None:
        rows = _manifest_rows(Path(manifest))
        selected = [row for row in rows if str(row.get("split", "train")) == split]
        return [
            TranscriptionExample(_resolve_manifest_sample(row, roots_p), split)
            for row in selected
        ]

    # Convenient fallback for local experiments; production runs should keep a
    # manifest so transfer sets cannot leak between train and validation.
    bundles = _all_bundles(roots_p)
    rng = random.Random(seed)
    rng.shuffle(bundles)
    n_val = min(max(1, round(len(bundles) * val_fraction)), max(len(bundles) - 1, 0))
    chosen = bundles[:n_val] if split == "val" else bundles[n_val:]
    return [TranscriptionExample(path, split) for path in chosen]


def _bundle_metadata(sample_dir: Path) -> dict:
    meta_path = sample_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def written_pitch_shift(sample_dir: Path | str) -> int:
    """Semitones to add to detected WAV F0 to recover written pitch."""

    from synthpipeline.pitch_convention import (
        effective_audio_transpose,
        load_bundle_metadata,
    )

    return effective_audio_transpose(load_bundle_metadata(sample_dir))


@lru_cache(maxsize=16384)
def _load_written_notes_cached(
    sample_dir_text: str,
    allow_legacy_midi: bool = False,
) -> tuple[tuple[int, float, float, float], ...]:
    sample_dir = Path(sample_dir_text)
    note_map_path = sample_dir / "note_map.json"
    notes: list[tuple[int, float, float, float]]
    if note_map_path.exists():
        payload = json.loads(note_map_path.read_text(encoding="utf-8"))
        if "rendered_notes" not in payload:
            if not allow_legacy_midi:
                raise ValueError(f"note_map.json has no rendered_notes: {note_map_path}")
            notes = _load_legacy_midi_notes(sample_dir)
        else:
            notes = [
                (
                    int(row["pitch_midi_written"]),
                    float(row["start_sec"]),
                    max(float(row["end_sec"]), float(row["start_sec"]) + 0.01),
                    0.0,
                )
                for row in payload["rendered_notes"]
            ]
    elif allow_legacy_midi:
        notes = _load_legacy_midi_notes(sample_dir)
    else:
        raise FileNotFoundError(
            f"Canonical transcription targets require {note_map_path}"
        )
    spans = _intonation_spans(sample_dir)
    if spans:
        annotated: list[tuple[int, float, float, float]] = []
        for pitch, start, end, _cents in notes:
            cents = 0.0
            mid = 0.5 * (start + end)
            for span_start, span_end, span_cents in spans:
                if start < span_end and end > span_start or span_start <= mid <= span_end:
                    cents = float(span_cents)
                    break
            annotated.append((pitch, start, end, cents))
        notes = annotated
    notes.sort(key=lambda item: (item[1], item[0]))
    return tuple(notes)


def _load_legacy_midi_notes(
    sample_dir: Path,
) -> list[tuple[int, float, float, float]]:
    """Load fragment-MIDI targets for explicitly permitted legacy bundles."""

    from synthpipeline.pitch_convention import midi_to_written_shift
    from synthpipeline.timing import midi_note_times

    midi_to_written = midi_to_written_shift(_bundle_metadata(sample_dir))
    return [
        (
            int(pitch) + midi_to_written,
            float(start),
            max(float(end), float(start) + 0.01),
            0.0,
        )
        for pitch, start, end in midi_note_times(
            sample_dir / "performance_audio.mid"
        )
    ]


def _intonation_spans(sample_dir: Path) -> tuple[tuple[float, float, float], ...]:
    labels_path = sample_dir / "labels.json"
    if not labels_path.exists():
        return ()
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    spans: list[tuple[float, float, float]] = []
    for label in document.get("labels") or []:
        if str(label.get("type")) != "intonation_error":
            continue
        cents = label.get("deviation_cents")
        if cents is None:
            continue
        start = float(label.get("start_time") or 0.0)
        end = max(float(label.get("end_time") or start), start + 0.01)
        spans.append((start, end, float(cents)))
    return tuple(spans)


def load_written_notes(
    sample_dir: Path | str,
    *,
    allow_legacy_midi: bool = False,
) -> list[tuple[int, float, float]]:
    """Read canonical rendered-note targets in written pitch.

    ``allow_legacy_midi`` preserves the old fragment-MIDI path for callers
    intentionally processing bundles that predate ``note_map.json``.
    """

    return [
        (pitch, start, end)
        for pitch, start, end, _cents in _load_written_notes_cached(
            str(Path(sample_dir).resolve()),
            bool(allow_legacy_midi),
        )
    ]


def load_written_notes_with_cents(
    sample_dir: Path | str,
    *,
    allow_legacy_midi: bool = False,
) -> list[tuple[int, float, float, float]]:
    """Written pitch, times, and signed intonation cents for each performed note."""

    return list(
        _load_written_notes_cached(
            str(Path(sample_dir).resolve()),
            bool(allow_legacy_midi),
        )
    )


def load_fine_pitch(sample_dir: Path | str, n_frames: int) -> np.ndarray:
    """Return [2, T] written-space F0 MIDI and strength, or zeros if missing."""

    path = Path(sample_dir) / "performance_f0.npy"
    feature = np.zeros((2, max(n_frames, 0)), dtype=np.float32)
    if not path.exists() or n_frames <= 0:
        return feature
    raw = np.load(path, mmap_mode="r")
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, np.ones_like(arr)], axis=0)
    if arr.shape[0] != 2 and arr.shape[-1] == 2:
        arr = arr.T
    if arr.shape[0] != 2:
        return feature
    take = min(n_frames, arr.shape[1])
    feature[:, :take] = arr[:, :take]
    return feature


def write_fine_pitch_feature(sample_dir: Path | str) -> Path | None:
    """Cache a written-space F0 contour next to the existing mel."""

    sample_dir = Path(sample_dir)
    wav = sample_dir / "performance_audio.wav"
    dest = sample_dir / "performance_f0.npy"
    if not wav.exists():
        return None
    from .decode import spectral_pitch_frames

    midi, strength = spectral_pitch_frames(
        wav,
        written_shift=written_pitch_shift(sample_dir),
    )
    np.save(dest, np.stack([midi.astype(np.float32), strength.astype(np.float32)]))
    return dest


def make_crop_targets(
    notes: Iterable[tuple[int, float, float]],
    total_frames: int,
    crop_start: int,
    crop_frames: int,
    *,
    hop_sec: float = FRAME_HOP_SEC,
    midi_min: int = 36,
    midi_max: int = 108,
) -> dict[str, np.ndarray]:
    """Rasterize only the requested crop instead of allocating a full-clip target."""

    target_frames = max(0, int(crop_frames))
    voiced = np.zeros(target_frames, dtype=np.float32)
    pitch = np.full(target_frames, -1, dtype=np.int64)
    onset = np.zeros(target_frames, dtype=np.float32)
    offset = np.zeros(target_frames, dtype=np.float32)
    cents = np.zeros(target_frames, dtype=np.float32)
    total_frames = max(0, int(total_frames))
    source_start = max(0, int(crop_start))
    source_stop = min(total_frames, source_start + target_frames)
    if target_frames == 0 or source_start >= source_stop:
        return {
            "voiced": voiced,
            "pitch": pitch,
            "onset": onset,
            "offset": offset,
            "cents": cents,
        }
    for item in notes:
        if len(item) == 4:
            midi, start, end, note_cents = item
        else:
            midi, start, end = item[:3]
            note_cents = 0.0
        first = max(0, min(total_frames - 1, int(np.floor(start / hop_sec))))
        last_exclusive = max(first + 1, min(total_frames, int(np.ceil(end / hop_sec))))
        overlap_start = max(first, source_start)
        overlap_stop = min(last_exclusive, source_stop)
        if overlap_start < overlap_stop:
            local_start = overlap_start - source_start
            local_stop = overlap_stop - source_start
            voiced[local_start:local_stop] = 1.0
            cents[local_start:local_stop] = float(note_cents)
            if midi_min <= midi <= midi_max:
                pitch[local_start:local_stop] = int(midi) - midi_min
        onset_frame = max(0, min(total_frames - 1, int(round(start / hop_sec))))
        offset_frame = max(0, min(total_frames - 1, int(round(end / hop_sec))))
        if source_start <= onset_frame < source_stop:
            onset[onset_frame - source_start] = 1.0
        if source_start <= offset_frame < source_stop:
            offset[offset_frame - source_start] = 1.0
    return {
        "voiced": voiced,
        "pitch": pitch,
        "onset": onset,
        "offset": offset,
        "cents": cents,
    }


def make_frame_targets(
    notes: Iterable[tuple[int, float, float]],
    n_frames: int,
    *,
    hop_sec: float = FRAME_HOP_SEC,
    midi_min: int = 36,
    midi_max: int = 108,
) -> dict[str, np.ndarray]:
    return make_crop_targets(
        notes,
        n_frames,
        0,
        n_frames,
        hop_sec=hop_sec,
        midi_min=midi_min,
        midi_max=midi_max,
    )


class MelTransferAugment:
    """Synthetic-to-real perturbations performed entirely on cached log-mels."""

    def __init__(self, probability: float = 0.85) -> None:
        self.probability = probability

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        if random.random() > self.probability:
            return mel
        out = mel.copy()
        n_mels, n_frames = out.shape
        # Recording gain and broad microphone/channel coloration.
        out += random.uniform(-7.0, 4.0)
        knots = np.random.normal(0.0, 2.5, size=8).astype(np.float32)
        eq = np.interp(
            np.linspace(0, 7, n_mels), np.arange(8), knots
        ).astype(np.float32)
        out += eq[:, None]
        # A floor/noise perturbation in dB plus mild temporal smearing.
        out += np.random.normal(0.0, random.uniform(0.0, 1.5), out.shape).astype(
            np.float32
        )
        if n_frames > 2 and random.random() < 0.35:
            out[:, 1:-1] = (
                0.15 * out[:, :-2] + 0.70 * out[:, 1:-1] + 0.15 * out[:, 2:]
            )
        if random.random() < 0.45:
            width = random.randint(1, max(1, n_mels // 16))
            start = random.randint(0, max(0, n_mels - width))
            out[start : start + width] = -80.0
        if n_frames > 8 and random.random() < 0.35:
            width = random.randint(1, max(1, n_frames // 24))
            start = random.randint(0, max(0, n_frames - width))
            out[:, start : start + width] = -80.0
        return np.clip(out, -100.0, 20.0)


class NoteCropDataset(Dataset):
    """Random fixed-size training crops or deterministic validation crops."""

    def __init__(
        self,
        examples: list[TranscriptionExample],
        *,
        crop_frames: int = 768,
        midi_min: int = 36,
        midi_max: int = 108,
        hop_sec: float = FRAME_HOP_SEC,
        training: bool = True,
        augment: bool = True,
        crops_per_clip: int = 2,
    ) -> None:
        self.examples = examples
        self.crop_frames = int(crop_frames)
        self.midi_min = midi_min
        self.midi_max = midi_max
        self.hop_sec = hop_sec
        self.training = training
        self.augment = MelTransferAugment() if augment and training else None
        self.crops_per_clip = max(1, crops_per_clip if training else 1)

    def __len__(self) -> int:
        return len(self.examples) * self.crops_per_clip

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index % len(self.examples)]
        mel = np.load(
            example.sample_dir / "performance_mel.npy", mmap_mode="r"
        )
        if mel.ndim != 2:
            raise ValueError(f"Expected 2-D mel in {example.sample_dir}, got {mel.shape}")
        if mel.shape[0] != 128 and mel.shape[1] == 128:
            mel = mel.T
        if mel.shape[0] != 128:
            raise ValueError(f"Expected 128xT mel in {example.sample_dir}, got {mel.shape}")
        total_frames = int(mel.shape[1])
        if total_frames > self.crop_frames:
            if self.training:
                crop_start = random.randint(0, total_frames - self.crop_frames)
            else:
                crop_start = (total_frames - self.crop_frames) // 2
            valid_frames = self.crop_frames
            crop = np.asarray(
                mel[:, crop_start : crop_start + self.crop_frames], dtype=np.float32
            ).copy()
        else:
            crop_start = 0
            valid_frames = total_frames
            crop = np.full((128, self.crop_frames), -80.0, dtype=np.float32)
            crop[:, :total_frames] = np.asarray(mel, dtype=np.float32)

        sliced = make_crop_targets(
            load_written_notes_with_cents(example.sample_dir),
            total_frames,
            crop_start,
            self.crop_frames,
            hop_sec=self.hop_sec,
            midi_min=self.midi_min,
            midi_max=self.midi_max,
        )
        frame_mask = np.zeros(self.crop_frames, dtype=np.bool_)
        frame_mask[:valid_frames] = True
        if self.augment is not None:
            crop = self.augment(crop)
        f0 = load_fine_pitch(example.sample_dir, total_frames)
        f0_crop = np.zeros((2, self.crop_frames), dtype=np.float32)
        f0_end = crop_start + min(self.crop_frames, max(total_frames - crop_start, 0))
        if f0_end > crop_start:
            f0_crop[:, : f0_end - crop_start] = f0[:, crop_start:f0_end]
        return {
            "mel": torch.from_numpy(crop),
            "f0": torch.from_numpy(f0_crop),
            "voiced": torch.from_numpy(sliced["voiced"]),
            "pitch": torch.from_numpy(sliced["pitch"]),
            "onset": torch.from_numpy(sliced["onset"]),
            "offset": torch.from_numpy(sliced["offset"]),
            "cents": torch.from_numpy(sliced["cents"]),
            "frame_mask": torch.from_numpy(frame_mask),
            "sample_id": example.sample_id,
            "crop_start": crop_start,
        }
