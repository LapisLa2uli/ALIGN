"""Procedural-only cached feature dataset for the Basic Pitch refiner."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .basic_pitch import (
    BasicPitchFeatures,
    basic_pitch_cache_path,
)
from .fine_pitch import pesto_cache_path


@dataclass(frozen=True)
class RefinerExample:
    sample_dir: Path
    note_map: Path
    split: str
    corpus: str
    effective_audio_transpose: int

    @property
    def sample_id(self) -> str:
        return self.sample_dir.name


def load_refiner_examples(
    manifest_path: Path | str,
    split: str,
    *,
    procedural_only: bool = True,
) -> list[RefinerExample]:
    document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = document.get(split)
    if not isinstance(rows, list):
        raise ValueError(f"Manifest has no list split {split!r}")
    examples = []
    for value in rows:
        row = dict(value)
        corpus = str(row.get("corpus") or row.get("root") or "")
        if procedural_only and corpus != "procedural12k":
            raise ValueError(
                f"Procedural-only refiner rejected {corpus!r} row"
            )
        if row.get("effective_audio_transpose") is None:
            raise ValueError(
                f"{row.get('sample') or row.get('sample_dir')} lacks "
                "effective_audio_transpose"
            )
        sample = Path(str(row["sample_dir"]))
        note_map = Path(str(row.get("note_map") or sample / "note_map.json"))
        if not note_map.is_file():
            raise FileNotFoundError(note_map)
        examples.append(
            RefinerExample(
                sample_dir=sample,
                note_map=note_map,
                split=split,
                corpus=corpus,
                effective_audio_transpose=int(
                    row["effective_audio_transpose"]
                ),
            )
        )
    if not examples:
        raise ValueError(f"Refiner {split!r} split is empty")
    return examples


def load_cached_refiner_features(
    example: RefinerExample,
    basic_cache_root: Path | str,
    pesto_cache_root: Path | str,
) -> tuple[BasicPitchFeatures, np.ndarray]:
    basic_path = basic_pitch_cache_path(
        basic_cache_root, example.sample_dir, example.corpus
    )
    if not basic_path.is_file():
        raise FileNotFoundError(
            f"Missing or stale Basic Pitch cache {basic_path}"
        )
    try:
        with np.load(basic_path, allow_pickle=False) as saved:
            cache_metadata = json.loads(str(saved["metadata"].item()))
            basic = BasicPitchFeatures(
                np.asarray(saved["note"], dtype=np.float32),
                np.asarray(saved["onset"], dtype=np.float32),
                np.asarray(saved["contour"], dtype=np.float32),
                np.asarray(saved["frame_times"], dtype=np.float64),
                cache_metadata,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Basic Pitch cache {basic_path}") from exc
    pesto_path = pesto_cache_path(
        pesto_cache_root, example.sample_dir, example.corpus
    )
    if not pesto_path.is_file():
        raise FileNotFoundError(
            f"Missing or stale PESTO cache {pesto_path}"
        )
    try:
        with np.load(pesto_path, allow_pickle=False) as saved:
            pesto = np.asarray(saved["pesto"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid PESTO cache {pesto_path}") from exc
    if pesto.shape != (len(basic.frame_times), 2):
        raise ValueError(
            f"PESTO cache shape {pesto.shape} does not match "
            f"{len(basic.frame_times)} Basic Pitch frames"
        )
    return basic, pesto


def _intonation_spans(sample_dir: Path) -> list[tuple[float, float, float]]:
    path = sample_dir / "labels.json"
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    spans = []
    for label in document.get("labels") or []:
        if (
            str(label.get("type")) != "intonation_error"
            or label.get("deviation_cents") is None
        ):
            continue
        start = float(label.get("start_time") or 0.0)
        end = max(float(label.get("end_time") or start), start + 0.001)
        spans.append((start, end, float(label["deviation_cents"])))
    return spans


def load_rendered_target_notes(
    example: RefinerExample,
) -> list[tuple[int, float, float, float]]:
    document = json.loads(example.note_map.read_text(encoding="utf-8"))
    rendered = document.get("rendered_notes")
    if not isinstance(rendered, list) or not rendered:
        raise ValueError(f"{example.note_map} has no rendered_notes")
    spans = _intonation_spans(example.sample_dir)
    notes = []
    for row in rendered:
        pitch = int(row["pitch_midi_written"])
        start = float(row["start_sec"])
        end = max(float(row["end_sec"]), start + 0.001)
        cents = next(
            (
                value
                for span_start, span_end, value in spans
                if start < span_end and end > span_start
            ),
            0.0,
        )
        notes.append((pitch, start, end, cents))
    return sorted(notes, key=lambda item: (item[1], item[0]))


def rasterize_refiner_targets(
    notes: Sequence[tuple[int, float, float, float]],
    frame_times: np.ndarray,
    *,
    midi_min: int,
    midi_max: int,
) -> dict[str, np.ndarray | list[tuple[int, int, int]]]:
    frame_times = np.asarray(frame_times, dtype=np.float64)
    frames = len(frame_times)
    voiced = np.zeros(frames, np.float32)
    onset = np.zeros(frames, np.float32)
    offset = np.zeros(frames, np.float32)
    pitch = np.full(frames, -1, np.int64)
    cents = np.zeros(frames, np.float32)
    intervals: list[tuple[int, int, int]] = []
    if not frames:
        return {
            "voiced": voiced,
            "onset": onset,
            "offset": offset,
            "pitch": pitch,
            "cents": cents,
            "intervals": intervals,
        }
    hop = (
        float(np.median(np.diff(frame_times)))
        if frames > 1
        else 256.0 / 22050.0
    )
    events: list[list[int | float]] = []
    for midi, start, end, note_cents in notes:
        if not (midi_min <= midi <= midi_max):
            continue
        first = int(np.searchsorted(frame_times, start - 0.5 * hop, side="left"))
        last = int(np.searchsorted(frame_times, end - 0.5 * hop, side="left"))
        first = max(0, min(frames - 1, first))
        last = max(first + 1, min(frames, last))
        events.append([first, last, int(midi), float(note_cents)])
    events.sort(key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
    monophonic: list[list[int | float]] = []
    for event in events:
        first = int(event[0])
        if monophonic and first < int(monophonic[-1][1]):
            if first <= int(monophonic[-1][0]):
                # Duplicate/stacked events cannot both be represented by a
                # monophonic decoder; keep the longer, earlier event.
                continue
            monophonic[-1][1] = max(int(monophonic[-1][0]) + 1, first)
        monophonic.append(event)
    for first_value, last_value, midi_value, note_cents_value in monophonic:
        first = int(first_value)
        last = int(last_value)
        midi = int(midi_value)
        note_cents = float(note_cents_value)
        voiced[first:last] = 1.0
        pitch[first:last] = midi
        cents[first:last] = float(note_cents)
        onset[first] = 1.0
        offset[last - 1] = 1.0
        intervals.append((first, last, midi))
    return {
        "voiced": voiced,
        "onset": onset,
        "offset": offset,
        "pitch": pitch,
        "cents": cents,
        "intervals": intervals,
    }


class RefinerCropDataset(Dataset):
    def __init__(
        self,
        examples: list[RefinerExample],
        basic_cache_root: Path | str,
        pesto_cache_root: Path | str,
        *,
        crop_frames: int = 512,
        midi_min: int = 36,
        midi_max: int = 108,
        training: bool = True,
        crops_per_clip: int = 2,
        augment_probability: float = 0.5,
    ) -> None:
        self.examples = examples
        self.basic_cache_root = Path(basic_cache_root)
        self.pesto_cache_root = Path(pesto_cache_root)
        self.crop_frames = int(crop_frames)
        self.midi_min = int(midi_min)
        self.midi_max = int(midi_max)
        self.training = bool(training)
        self.crops_per_clip = max(
            1, int(crops_per_clip if training else 1)
        )
        self.augment_probability = float(augment_probability)

    def __len__(self) -> int:
        return len(self.examples) * self.crops_per_clip

    def _crop_start(self, total: int) -> int:
        if total <= self.crop_frames:
            return 0
        if self.training:
            return random.randint(0, total - self.crop_frames)
        return (total - self.crop_frames) // 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index % len(self.examples)]
        basic, pesto = load_cached_refiner_features(
            example, self.basic_cache_root, self.pesto_cache_root
        )
        total = int(basic.note.shape[0])
        start = self._crop_start(total)
        stop = min(total, start + self.crop_frames)
        valid_frames = stop - start

        def crop(values: np.ndarray, width: int) -> np.ndarray:
            shape = (self.crop_frames, *values.shape[1:])
            output = np.zeros(shape, dtype=np.float32)
            output[:valid_frames] = values[start:stop]
            return output

        note = crop(basic.note, 88)
        onset_map = crop(basic.onset, 88)
        contour = crop(basic.contour, 264)
        pesto_crop = crop(pesto, 2)
        if self.training and random.random() < self.augment_probability:
            scale = random.uniform(0.85, 1.12)
            note = np.clip(note * scale, 0.0, 1.0)
            onset_map = np.clip(
                onset_map * random.uniform(0.85, 1.15), 0.0, 1.0
            )
            noise = np.random.normal(0.0, 0.01, contour.shape).astype(
                np.float32
            )
            contour = np.clip(contour + noise, 0.0, 1.0)
            pesto_crop[:, 1] *= random.uniform(0.85, 1.0)

        targets = rasterize_refiner_targets(
            load_rendered_target_notes(example),
            basic.frame_times,
            midi_min=self.midi_min,
            midi_max=self.midi_max,
        )
        intervals = [
            (first - start, last - start, midi)
            for first, last, midi in targets["intervals"]  # type: ignore[index]
            if start <= first and last <= start + self.crop_frames
        ]
        frame_mask = np.zeros(self.crop_frames, dtype=np.bool_)
        frame_mask[:valid_frames] = True
        result: dict[str, Any] = {
            "note": torch.from_numpy(note),
            "onset_map": torch.from_numpy(onset_map),
            "contour": torch.from_numpy(contour),
            "pesto": torch.from_numpy(pesto_crop),
            "frame_mask": torch.from_numpy(frame_mask),
            "sample_id": example.sample_id,
            "crop_start": start,
            "intervals": intervals,
        }
        for key in ("voiced", "onset", "offset", "pitch", "cents"):
            values = np.asarray(targets[key])[start:stop]
            padded = np.zeros(
                self.crop_frames,
                dtype=np.int64 if key == "pitch" else np.float32,
            )
            if key == "pitch":
                padded.fill(-1)
            padded[:valid_frames] = values
            result[key] = torch.from_numpy(padded)
        return result


def collate_refiner_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output
