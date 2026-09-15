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
class RefinerAugmentConfig:
    """Cached-map approximations of difficult microphone/timbre conditions."""

    probability: float = 0.75
    band_attenuation_probability: float = 0.45
    band_attenuation_min: float = 0.25
    band_attenuation_max: float = 0.70
    filtered_timbre_probability: float = 0.45
    breath_noise_std: float = 0.018
    short_note_max_frames: int = 12
    short_note_attenuation_probability: float = 0.70
    short_note_attenuation_min: float = 0.25
    short_note_attenuation_max: float = 0.60
    same_pitch_split_probability: float = 0.45
    short_note_positive_weight: float = 2.0
    hard_negative_ratio: float = 1.0
    hard_negative_weight: float = 2.0


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


def augment_cached_refiner_maps(
    note: np.ndarray,
    onset_map: np.ndarray,
    contour: np.ndarray,
    pesto: np.ndarray,
    *,
    intervals: Sequence[tuple[int, int, int]],
    voiced_target: np.ndarray,
    midi_offset: int = 21,
    config: RefinerAugmentConfig = RefinerAugmentConfig(),
    rng: np.random.Generator | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray | int],
]:
    """Degrade positives and add equally weighted onset-like hard negatives."""

    generator = rng or np.random.default_rng()
    frames = int(note.shape[0])
    frame_weight = np.ones(frames, dtype=np.float32)
    onset_weight = np.ones(frames, dtype=np.float32)
    split_weight = np.ones(frames, dtype=np.float32)
    if frames == 0 or generator.random() >= config.probability:
        return note, onset_map, contour, pesto, {
            "frame_weight": frame_weight,
            "onset_weight": onset_weight,
            "offset_weight": split_weight,
            "short_positive_count": 0,
            "hard_negative_count": 0,
            "artificial_split_count": 0,
        }

    if generator.random() < config.band_attenuation_probability:
        width = int(generator.integers(4, min(19, note.shape[1] + 1)))
        first = int(generator.integers(0, max(1, note.shape[1] - width + 1)))
        scale = float(
            generator.uniform(
                config.band_attenuation_min,
                config.band_attenuation_max,
            )
        )
        note[:, first : first + width] *= scale
        onset_map[:, first : first + width] *= scale
        contour[:, first * 3 : (first + width) * 3] *= scale

    if generator.random() < config.filtered_timbre_probability:
        if frames > 2:
            note[1:-1] = (
                0.20 * note[:-2] + 0.60 * note[1:-1] + 0.20 * note[2:]
            )
            contour[1:-1] = (
                0.20 * contour[:-2]
                + 0.60 * contour[1:-1]
                + 0.20 * contour[2:]
            )
        breath = generator.normal(
            0.0, config.breath_noise_std, contour.shape
        ).astype(np.float32)
        contour[:] = np.clip(contour + breath, 0.0, 1.0)
        pesto[:, 1] *= float(generator.uniform(0.65, 0.92))

    short_positives: list[tuple[int, int, int]] = []
    artificial_splits = 0
    for first, last, midi in intervals:
        first = max(0, int(first))
        last = min(frames, int(last))
        axis = int(midi) - int(midi_offset)
        if last <= first or not 0 <= axis < note.shape[1]:
            continue
        length = last - first
        if (
            length <= config.short_note_max_frames
            and generator.random()
            < config.short_note_attenuation_probability
        ):
            scale = float(
                generator.uniform(
                    config.short_note_attenuation_min,
                    config.short_note_attenuation_max,
                )
            )
            note[first:last, axis] *= scale
            onset_map[first : min(last, first + 2), axis] *= scale
            contour[first:last, axis * 3 : axis * 3 + 3] *= scale
            frame_weight[first:last] = np.maximum(
                frame_weight[first:last],
                config.short_note_positive_weight,
            )
            onset_weight[first] = max(
                onset_weight[first],
                config.short_note_positive_weight,
            )
            short_positives.append((first, last, axis))
        elif (
            length >= 8
            and generator.random() < config.same_pitch_split_probability
        ):
            boundary = int(generator.integers(first + 3, last - 3))
            note[max(first, boundary - 1) : min(last, boundary + 1), axis] *= 0.30
            onset_map[boundary, axis] = max(
                float(onset_map[boundary, axis]), 0.70
            )
            contour[
                max(first, boundary - 1) : min(last, boundary + 2),
                axis * 3 : axis * 3 + 3,
            ] *= 0.65
            onset_weight[boundary] = max(
                onset_weight[boundary],
                config.hard_negative_weight,
            )
            split_weight[boundary] = max(
                split_weight[boundary],
                config.hard_negative_weight,
            )
            artificial_splits += 1

    requested_negatives = int(
        round(len(short_positives) * config.hard_negative_ratio)
    )
    available = np.flatnonzero(np.asarray(voiced_target)[:frames] < 0.5)
    hard_negative_count = min(requested_negatives, len(available))
    if hard_negative_count:
        chosen = generator.choice(
            available, size=hard_negative_count, replace=False
        )
        for frame in np.asarray(chosen, dtype=np.int64):
            axis = int(generator.integers(0, note.shape[1]))
            onset_map[frame, axis] = max(
                float(onset_map[frame, axis]),
                float(generator.uniform(0.28, 0.48)),
            )
            note[frame, axis] = max(
                float(note[frame, axis]),
                float(generator.uniform(0.12, 0.25)),
            )
            contour[frame, axis * 3 : axis * 3 + 3] = np.maximum(
                contour[frame, axis * 3 : axis * 3 + 3],
                generator.uniform(0.08, 0.18, 3),
            )
            frame_weight[frame] = max(
                frame_weight[frame], config.hard_negative_weight
            )
            onset_weight[frame] = max(
                onset_weight[frame], config.hard_negative_weight
            )

    return (
        np.clip(note, 0.0, 1.0),
        np.clip(onset_map, 0.0, 1.0),
        np.clip(contour, 0.0, 1.0),
        pesto,
        {
            "frame_weight": frame_weight,
            "onset_weight": onset_weight,
            "offset_weight": split_weight,
            "short_positive_count": len(short_positives),
            "hard_negative_count": hard_negative_count,
            "artificial_split_count": artificial_splits,
        },
    )


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
        augment_config: RefinerAugmentConfig | None = None,
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
        self.augment_config = augment_config or RefinerAugmentConfig(
            probability=self.augment_probability
        )

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
        augmentation: dict[str, np.ndarray | int] = {
            "frame_weight": np.ones(self.crop_frames, np.float32),
            "onset_weight": np.ones(self.crop_frames, np.float32),
            "offset_weight": np.ones(self.crop_frames, np.float32),
            "short_positive_count": 0,
            "hard_negative_count": 0,
            "artificial_split_count": 0,
        }
        if self.training:
            (
                note,
                onset_map,
                contour,
                pesto_crop,
                augmentation,
            ) = augment_cached_refiner_maps(
                note,
                onset_map,
                contour,
                pesto_crop,
                intervals=intervals,
                voiced_target=np.pad(
                    np.asarray(targets["voiced"])[start:stop],
                    (0, self.crop_frames - valid_frames),
                ),
                config=self.augment_config,
            )
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
        for key in ("frame_weight", "onset_weight", "offset_weight"):
            result[key] = torch.from_numpy(
                np.asarray(augmentation[key], dtype=np.float32)
            )
        for key in (
            "short_positive_count",
            "hard_negative_count",
            "artificial_split_count",
        ):
            result[key] = int(augmentation[key])
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
