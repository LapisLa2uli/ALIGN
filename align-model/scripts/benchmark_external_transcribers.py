"""Benchmark pretrained audio-to-note models on frozen ALIGN splits.

The external models hear sounding pitch. This script converts every prediction
to written Bb-clarinet pitch before applying ALIGN's exact-pitch, 50 ms metric.
Raw model outputs are cached locally so decoder calibration never reruns the
expensive audio model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.ndimage import median_filter

from alignmodel.transcription import TransNote, evaluate_note_lists, match_notes
from alignmodel.transcription.data import load_written_notes_with_cents
from synthpipeline.pitch_convention import (
    audio_to_written_shift,
    load_bundle_metadata,
)


@dataclass(frozen=True)
class BasicPitchDecode:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_ms: float = 55.0
    extra_written_shift: int = 0


@dataclass(frozen=True)
class F0Decode:
    confidence_threshold: float
    onset_threshold: float = 0.35
    min_note_sec: float = 0.055
    pitch_change_semitones: float = 0.65
    max_gap_frames: int = 2
    extra_written_shift: int = 0


@dataclass(frozen=True)
class PrecomputedDecode:
    extra_written_shift: int = 0


def _cache_path(cache_root: Path, model: str, row: dict[str, Any]) -> Path:
    corpus = str(row.get("corpus") or row.get("root") or "unknown")
    return cache_root / model / corpus / f"{Path(row['sample_dir']).name}.npz"


def _close_gaps(active: np.ndarray, maximum: int) -> np.ndarray:
    output = active.copy()
    index = 0
    while index < len(output):
        if output[index]:
            index += 1
            continue
        end = index
        while end < len(output) and not output[end]:
            end += 1
        if index and end < len(output) and end - index <= maximum:
            output[index:end] = True
        index = end
    return output


def _audio_features(
    audio: np.ndarray, sample_rate: int, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(round(sample_rate * 0.01)))
    onset = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop, n_fft=2048
    )
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]

    def resize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if len(values) == frame_count:
            return values
        if not len(values):
            return np.zeros(frame_count, dtype=np.float32)
        source = np.linspace(0.0, 1.0, len(values))
        target = np.linspace(0.0, 1.0, frame_count)
        return np.interp(target, source, values).astype(np.float32)

    onset = resize(onset)
    rms = resize(rms)
    onset_scale = float(np.percentile(onset, 95)) if onset.size else 0.0
    if onset_scale > 0:
        onset = np.clip(onset / onset_scale, 0.0, 2.0)
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
    return onset.astype(np.float32), rms_db.astype(np.float32)


def _segment_f0(
    feature: dict[str, np.ndarray],
    config: F0Decode,
    written_shift: int,
) -> list[TransNote]:
    frequency = np.asarray(feature["frequency"], dtype=np.float32).reshape(-1)
    confidence = np.asarray(feature["confidence"], dtype=np.float32).reshape(-1)
    onset = np.asarray(feature["onset"], dtype=np.float32).reshape(-1)
    rms_db = np.asarray(feature["rms_db"], dtype=np.float32).reshape(-1)
    count = min(len(frequency), len(confidence), len(onset), len(rms_db))
    frequency, confidence, onset, rms_db = (
        value[:count] for value in (frequency, confidence, onset, rms_db)
    )
    midi = librosa.hz_to_midi(np.maximum(frequency, 1.0)).astype(np.float32)
    active = (
        np.isfinite(midi)
        & (frequency >= 45.0)
        & (frequency <= 2600.0)
        & (confidence >= config.confidence_threshold)
        & (rms_db >= -48.0)
    )
    active = _close_gaps(active, config.max_gap_frames)
    smoothed = median_filter(midi, size=5, mode="nearest")
    min_frames = max(2, int(round(config.min_note_sec / 0.01)))
    notes: list[TransNote] = []
    index = 0
    while index < count:
        if not active[index]:
            index += 1
            continue
        run_start = index
        while index < count and active[index]:
            index += 1
        run_end = index
        boundaries = {run_start, run_end}
        candidate_onsets = np.flatnonzero(
            onset[run_start:run_end] >= config.onset_threshold
        ) + run_start
        pitch_delta = np.zeros(run_end - run_start, dtype=np.float32)
        if run_end - run_start > 3:
            pitch_delta[3:] = np.abs(
                smoothed[run_start + 3 : run_end]
                - smoothed[run_start : run_end - 3]
            )
        candidate_changes = (
            np.flatnonzero(pitch_delta >= config.pitch_change_semitones)
            + run_start
        )
        candidates = np.unique(
            np.concatenate((candidate_onsets, candidate_changes))
        )
        last_boundary = run_start
        for frame in candidates:
            frame = int(frame)
            if (
                frame - last_boundary >= min_frames
                and run_end - frame >= min_frames
            ):
                boundaries.add(frame)
                last_boundary = frame
        ordered = sorted(boundaries)
        merged = [ordered[0]]
        for boundary in ordered[1:]:
            if boundary - merged[-1] >= min_frames or boundary == run_end:
                merged.append(boundary)
        if merged[-1] != run_end:
            merged.append(run_end)
        for start, end in zip(merged[:-1], merged[1:]):
            valid = active[start:end]
            if end - start < min_frames or not np.any(valid):
                continue
            continuous = float(np.median(smoothed[start:end][valid]))
            sounding_pitch = int(round(continuous))
            note_confidence = float(np.median(confidence[start:end][valid]))
            notes.append(
                TransNote(
                    pitch=sounding_pitch + written_shift,
                    start=round(start * 0.01, 6),
                    end=round(end * 0.01, 6),
                    confidence=round(note_confidence, 6),
                    cents=round(100.0 * (continuous - sounding_pitch), 2),
                    pitch_candidates=(
                        sounding_pitch + written_shift,
                        sounding_pitch - 1 + written_shift,
                        sounding_pitch + 1 + written_shift,
                    ),
                )
            )
    return notes


class BasicPitchBackend:
    def __init__(self) -> None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        self.model = Model(ICASSP_2022_MODEL_PATH)

    def extract(self, wav: Path, cache: Path) -> tuple[dict[str, np.ndarray], float]:
        if cache.exists():
            saved = np.load(cache)
            return (
                {key: saved[key] for key in ("note", "onset", "contour")},
                float(saved["inference_sec"]),
            )
        from basic_pitch.inference import run_inference

        started = time.perf_counter()
        output = run_inference(wav, self.model)
        elapsed = time.perf_counter() - started
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **output, inference_sec=np.asarray(elapsed))
        return output, elapsed

    @staticmethod
    def decode(
        feature: dict[str, np.ndarray],
        config: BasicPitchDecode,
        written_shift: int,
    ) -> list[TransNote]:
        from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
        from basic_pitch.note_creation import model_output_to_notes

        minimum_frames = int(
            round(config.min_note_ms / 1000.0 * AUDIO_SAMPLE_RATE / FFT_HOP)
        )
        _midi, events = model_output_to_notes(
            feature,
            onset_thresh=config.onset_threshold,
            frame_thresh=config.frame_threshold,
            min_note_len=minimum_frames,
            min_freq=45.0,
            max_freq=2600.0,
            multiple_pitch_bends=True,
            melodia_trick=True,
        )
        notes = []
        for start, end, pitch, amplitude, bends in events:
            cents = 0.0
            if bends:
                cents = float(np.median(np.asarray(bends, dtype=np.float32))) * 100 / 3
            notes.append(
                TransNote(
                    pitch=int(pitch) + written_shift + config.extra_written_shift,
                    start=float(start),
                    end=float(end),
                    confidence=float(amplitude),
                    cents=round(cents, 2),
                )
            )
        return sorted(notes, key=lambda note: (note.start, note.pitch))


class F0Backend:
    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.device = device
        self._pesto_model = None

    def _predict(
        self, wav: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        import torch
        import torchaudio

        audio, sample_rate = torchaudio.load(str(wav))
        audio = audio.mean(dim=0)
        target_device = torch.device(self.device)
        if self.name == "pesto":
            from pesto.core import _predict
            from pesto.loader import load_model

            if self._pesto_model is None:
                self._pesto_model = load_model(
                    "mir-1k_g7", step_size=10.0, sampling_rate=sample_rate
                ).to(target_device)
            _times, frequency, confidence, _activations = _predict(
                audio.to(target_device),
                sample_rate,
                self._pesto_model,
                num_chunks=2,
            )
        elif self.name == "penn":
            import penn

            frequency, confidence = penn.from_audio(
                audio[None],
                sample_rate,
                hopsize=0.01,
                fmin=45.0,
                fmax=2600.0,
                batch_size=2048,
                decoder="viterbi",
                gpu=0 if target_device.type == "cuda" else None,
            )
        elif self.name in {"torchcrepe-tiny", "torchcrepe-full"}:
            import torchcrepe

            batched = audio[None].to(target_device)
            frequency, confidence = torchcrepe.predict(
                batched,
                sample_rate,
                hop_length=max(1, sample_rate // 100),
                fmin=45.0,
                fmax=2600.0,
                model=self.name.split("-")[1],
                decoder=torchcrepe.decode.viterbi,
                return_periodicity=True,
                batch_size=2048,
                device=str(target_device),
            )
        else:
            raise ValueError(self.name)
        return (
            frequency.detach().cpu().numpy().reshape(-1),
            confidence.detach().cpu().numpy().reshape(-1),
            audio.cpu().numpy(),
            sample_rate,
        )

    def extract(self, wav: Path, cache: Path) -> tuple[dict[str, np.ndarray], float]:
        if cache.exists():
            saved = np.load(cache)
            return (
                {
                    key: saved[key]
                    for key in ("frequency", "confidence", "onset", "rms_db")
                },
                float(saved["inference_sec"]),
            )
        started = time.perf_counter()
        frequency, confidence, audio, sample_rate = self._predict(wav)
        elapsed = time.perf_counter() - started
        onset, rms_db = _audio_features(audio, sample_rate, len(frequency))
        feature = {
            "frequency": frequency.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "onset": onset,
            "rms_db": rms_db,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **feature, inference_sec=np.asarray(elapsed))
        return feature, elapsed

    @staticmethod
    def decode(
        feature: dict[str, np.ndarray], config: F0Decode, written_shift: int
    ) -> list[TransNote]:
        return _segment_f0(
            feature, config, written_shift + config.extra_written_shift
        )


class PrecomputedBackend:
    def __init__(self, path: Path) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        self.rows = {
            str(Path(row["sample_dir"]).resolve()): row
            for split in (document.get("splits") or {}).values()
            for row in split.get("rows") or []
        }

    def extract(self, wav: Path, _cache: Path) -> tuple[dict[str, np.ndarray], float]:
        key = str(wav.parent.resolve())
        if key not in self.rows:
            raise KeyError(f"No precomputed prediction for {wav.parent.name}")
        row = self.rows[key]
        notes = row.get("notes") or []
        return (
            {
                "pitch": np.asarray([note["pitch"] for note in notes], np.int16),
                "start": np.asarray([note["start"] for note in notes], np.float32),
                "end": np.asarray([note["end"] for note in notes], np.float32),
                "confidence": np.asarray(
                    [note.get("confidence", 1.0) for note in notes], np.float32
                ),
            },
            float(row.get("inference_sec") or 0.0),
        )

    @staticmethod
    def decode(
        feature: dict[str, np.ndarray],
        config: PrecomputedDecode,
        written_shift: int,
    ) -> list[TransNote]:
        shift = written_shift + config.extra_written_shift
        return [
            TransNote(
                int(pitch) + shift,
                float(start),
                float(end),
                float(confidence),
            )
            for pitch, start, end, confidence in zip(
                feature["pitch"],
                feature["start"],
                feature["end"],
                feature["confidence"],
            )
        ]


def _model_configs(
    model: str,
) -> list[BasicPitchDecode | F0Decode | PrecomputedDecode]:
    if model == "tsumugi":
        return [PrecomputedDecode()]
    if model == "basic-pitch":
        return [
            BasicPitchDecode(onset, frame, duration)
            for onset in (0.3, 0.4, 0.5, 0.6)
            for frame in (0.2, 0.3, 0.4)
            for duration in (55.0, 90.0)
        ]
    confidence = {
        "pesto": (0.5, 0.7, 0.85, 0.95),
        "penn": (0.1, 0.2, 0.3, 0.4),
        "torchcrepe-tiny": (0.1, 0.2, 0.3, 0.5),
        "torchcrepe-full": (0.1, 0.2, 0.3, 0.5),
    }[model]
    return [
        F0Decode(conf, onset, duration, change)
        for conf in confidence
        for onset in (0.2, 0.35, 0.5)
        for duration in (0.05, 0.09)
        for change in (0.65,)
    ]


def _target(
    sample: Path, row: dict[str, Any], note_map_root: Path | None
) -> list[TransNote]:
    if note_map_root is not None:
        from alignmodel.note_align_train import load_exact_note_map

        corpus = str(row.get("corpus") or row.get("root") or "")
        candidates = (
            note_map_root / corpus / sample.name / "note_map.json",
            note_map_root / sample.name / "note_map.json",
            sample / "note_map.json",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No exact note map for {sample.name}")
        exact = load_exact_note_map(path)
        spans: list[tuple[float, float, float]] = []
        labels_path = sample / "labels.json"
        if labels_path.exists():
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            for label in labels.get("labels") or []:
                if (
                    str(label.get("type")) == "intonation_error"
                    and label.get("deviation_cents") is not None
                ):
                    start = float(label.get("start_time") or 0.0)
                    end = max(
                        float(label.get("end_time") or start), start + 0.01
                    )
                    spans.append((start, end, float(label["deviation_cents"])))
        return [
            TransNote(
                note.pitch,
                note.start,
                note.end,
                note.confidence,
                cents=next(
                    (
                        cents
                        for start, end, cents in spans
                        if note.start < end and note.end > start
                    ),
                    0.0,
                ),
            )
            for note in exact.notes
        ]
    return [
        TransNote(pitch, start, end, 1.0, cents=cents)
        for pitch, start, end, cents in load_written_notes_with_cents(sample)
    ]


def _metrics(predicted: list[TransNote], target: list[TransNote]) -> dict[str, Any]:
    result = evaluate_note_lists(predicted, target, onset_tolerance_sec=0.05)
    pairs = match_notes(predicted, target, onset_tolerance_sec=0.05)
    intonation_errors = [
        abs(predicted[i].cents - target[j].cents)
        for i, j in pairs
        if abs(target[j].cents) > 1e-6
    ]
    result["intonation_cents_mae"] = (
        float(np.mean(intonation_errors)) if intonation_errors else None
    )
    result["n_intonation_matched"] = len(intonation_errors)
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_pred = sum(int(row["n_pred"]) for row in rows)
    n_target = sum(int(row["n_target"]) for row in rows)
    n_matched = sum(int(row["n_matched"]) for row in rows)
    precision = n_matched / max(n_pred, 1)
    recall = n_matched / max(n_target, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    def weighted(key: str, count_key: str = "n_matched") -> float | None:
        values = [
            (float(row[key]), int(row[count_key]))
            for row in rows
            if row.get(key) is not None and int(row.get(count_key) or 0)
        ]
        if not values:
            return None
        return sum(value * count for value, count in values) / sum(
            count for _value, count in values
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": n_pred,
        "n_target": n_target,
        "n_matched": n_matched,
        "pred_target_ratio": n_pred / max(n_target, 1),
        "onset_mae_sec": weighted("onset_mae_sec"),
        "offset_mae_sec": weighted("offset_mae_sec"),
        "cents_mae": weighted("cents_mae"),
        "intonation_cents_mae": weighted(
            "intonation_cents_mae", "n_intonation_matched"
        ),
        "n_intonation_matched": sum(
            int(row.get("n_intonation_matched") or 0) for row in rows
        ),
        "n_semitone_errors": sum(
            int(row.get("n_semitone_errors") or 0) for row in rows
        ),
        "n_octave_errors": sum(
            int(row.get("n_octave_errors") or 0) for row in rows
        ),
        "n_plus_minus_2": sum(int(row.get("n_plus_minus_2") or 0) for row in rows),
    }


def _rank(metrics: dict[str, Any]) -> float:
    ratio = max(float(metrics["pred_target_ratio"]), 1e-3)
    return float(metrics["f1"]) - 0.08 * abs(math.log(ratio))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "basic-pitch",
            "pesto",
            "penn",
            "torchcrepe-tiny",
            "torchcrepe-full",
            "tsumugi",
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--note-map-root",
        type=Path,
        default=None,
        help="Exact note-map cache; required for full raw-corpus ground truth",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--calibration-samples", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Raw prediction JSON for a precomputed backend such as Tsumugi",
    )
    parser.add_argument(
        "--corpus-extra-shift",
        action="append",
        default=[],
        metavar="CORPUS=SEMITONES",
        help="Override a corpus's extra legacy acoustic correction",
    )
    args = parser.parse_args()

    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = args.split or ["val", "test_id", "test_ood"]
    backend: BasicPitchBackend | F0Backend | PrecomputedBackend
    if args.model == "basic-pitch":
        backend = BasicPitchBackend()
    elif args.model == "tsumugi":
        if args.predictions is None:
            raise ValueError("--predictions is required for Tsumugi")
        backend = PrecomputedBackend(args.predictions)
    else:
        backend = F0Backend(args.model, args.device)
    extracted: dict[
        str, tuple[dict[str, np.ndarray], float, int, str, list[TransNote]]
    ] = {}

    def feature_for(row: dict[str, Any]):
        sample = Path(row["sample_dir"])
        key = str(sample.resolve())
        if key not in extracted:
            wav = sample / "performance_audio.wav"
            if not wav.exists():
                raise FileNotFoundError(wav)
            feature, seconds = backend.extract(
                wav, _cache_path(args.cache_root, args.model, row)
            )
            shift = audio_to_written_shift(load_bundle_metadata(sample))
            corpus = str(row.get("corpus") or row.get("root") or "unknown")
            extracted[key] = (
                feature,
                seconds,
                shift,
                corpus,
                _target(sample, row, args.note_map_root),
            )
        return extracted[key]

    all_val_rows = list(document.get("val") or [])
    by_corpus: dict[str, list[dict[str, Any]]] = {}
    for row in all_val_rows:
        corpus = str(row.get("corpus") or row.get("root") or "unknown")
        by_corpus.setdefault(corpus, []).append(row)
    per_corpus = max(
        1, int(math.ceil(args.calibration_samples / max(len(by_corpus), 1)))
    )
    calibration_rows = [
        row
        for corpus in sorted(by_corpus)
        for row in by_corpus[corpus][:per_corpus]
    ][: args.calibration_samples]
    best_config = None
    best_metrics = None
    best_rank = -float("inf")
    usable_calibration: list[
        tuple[dict[str, np.ndarray], int, str, list[TransNote]]
    ] = []
    for index, row in enumerate(calibration_rows, 1):
        try:
            feature, _seconds, shift, corpus, target = feature_for(row)
            usable_calibration.append((feature, shift, corpus, target))
        except Exception as exc:  # noqa: BLE001
            print(f"calibration skip {Path(row['sample_dir']).name}: {exc}", flush=True)
        if index == 1 or index % 10 == 0 or index == len(calibration_rows):
            print(
                f"calibration extraction {index}/{len(calibration_rows)} "
                f"usable={len(usable_calibration)}",
                flush=True,
            )
    if not usable_calibration:
        raise RuntimeError("No usable calibration clips")
    configs = _model_configs(args.model)
    probe_config = configs[0]
    extra_shift_by_corpus: dict[str, int] = {}
    for value in args.corpus_extra_shift:
        corpus, separator, shift = value.partition("=")
        if not separator:
            raise ValueError(
                f"--corpus-extra-shift must be CORPUS=SEMITONES, got {value!r}"
            )
        extra_shift_by_corpus[corpus] = int(shift)
    for corpus in sorted({row[2] for row in usable_calibration}):
        if corpus in extra_shift_by_corpus:
            continue
        corpus_rows = [row for row in usable_calibration if row[2] == corpus]
        candidates = []
        for extra in (0, 2):
            rows = [
                _metrics(
                    backend.decode(feature, probe_config, shift + extra),
                    target,
                )
                for feature, shift, _corpus, target in corpus_rows
            ]
            metrics = _aggregate(rows)
            candidates.append((_rank(metrics), metrics["f1"], -extra, extra))
        extra_shift_by_corpus[corpus] = max(candidates)[-1]
    print(
        "corpus_extra_written_shift",
        json.dumps(extra_shift_by_corpus, sort_keys=True),
        flush=True,
    )
    for config in configs:
        rows = [
            _metrics(
                backend.decode(
                    feature,
                    config,
                    shift + extra_shift_by_corpus.get(corpus, 0),
                ),
                target,
            )
            for feature, shift, corpus, target in usable_calibration
        ]
        metrics = _aggregate(rows)
        rank = _rank(metrics)
        if rank > best_rank:
            best_config, best_metrics, best_rank = config, metrics, rank
    assert best_config is not None
    print(
        "calibrated",
        json.dumps(
            {"config": asdict(best_config), "metrics": best_metrics}, indent=2
        ),
        flush=True,
    )

    report: dict[str, Any] = {
        "model": args.model,
        "manifest": str(args.manifest),
        "written_pitch_policy": "detected WAV pitch - sounding_transpose",
        "calibration_samples_requested": len(calibration_rows),
        "corpus_extra_written_shift": extra_shift_by_corpus,
        "calibration": {
            "config": asdict(best_config),
            "metrics": best_metrics,
        },
        "splits": {},
    }
    for split in splits:
        source_rows = list(document.get(split) or [])
        if args.max_samples:
            source_rows = source_rows[: args.max_samples]
        results = []
        samples = []
        skipped = []
        for index, row in enumerate(source_rows, 1):
            sample = Path(row["sample_dir"])
            try:
                feature, seconds, shift, corpus, target = feature_for(row)
                predicted = backend.decode(
                    feature,
                    best_config,
                    shift + extra_shift_by_corpus.get(corpus, 0),
                )
                metrics = _metrics(predicted, target)
                results.append(metrics)
                samples.append(
                    {
                        "sample": sample.name,
                        "corpus": row.get("corpus"),
                        "source": row.get("source"),
                        "inference_sec": seconds,
                        "metrics": metrics,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"sample": sample.name, "error": str(exc)})
            if index == 1 or index % 10 == 0 or index == len(source_rows):
                print(
                    f"{split} {index}/{len(source_rows)} "
                    f"ok={len(results)} skipped={len(skipped)}",
                    flush=True,
                )
        aggregate = _aggregate(results) if results else {}
        aggregate["n_requested"] = len(source_rows)
        aggregate["n_evaluated"] = len(results)
        aggregate["n_skipped"] = len(skipped)
        aggregate["inference_sec_total"] = sum(
            float(row["inference_sec"]) for row in samples
        )
        report["splits"][split] = {
            "metrics": aggregate,
            "samples": samples,
            "skipped": skipped,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
