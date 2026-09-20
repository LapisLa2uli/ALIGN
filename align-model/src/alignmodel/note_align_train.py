"""Training data and trainer for the symbolic note-list aligner."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alignmodel.device import device_label, resolve_device
from alignmodel.stages.note_align import (
    FEATURE_DIM,
    OP_CLASSES,
    LearnedNoteScorer,
    NoteAlignConfig,
    NoteAligner,
    ObservedNote,
    alignment_metrics,
    checkpoint_payload,
    normalize_notes,
    note_features,
)
from alignmodel.stages.score_graph import build_score_graph
from alignmodel.types import ScoreGraph

CACHE_NAMES = (
    "note_map.json",
    "exact_note_map.npz",
    "_exact_note_map.npz",
    "note_map.npz",
)


@dataclass
class ExactNoteMap:
    """One exact frontend-note to original-written-note training target."""

    sample_id: str
    score: ScoreGraph
    notes: list[ObservedNote]
    target_score_indices: list[int | None]
    target_is_copy: list[bool]
    split: str | None = None
    sample_dir: Path | None = None


@dataclass
class NoteAlignTrainConfig:
    data_root: Path
    output_dir: Path
    cache_dir: Path | None = None
    manifest: Path | None = None
    epochs: int = 8
    batch_size: int = 2048
    lr: float = 8e-4
    weight_decay: float = 1e-3
    hidden_dim: int = 48
    learned_weight: float = 0.65
    device: str = "auto"
    seed: int = 365
    val_fraction: float = 0.10
    augmentations_per_map: int = 1
    drop_probability: float = 0.08
    pitch_error_probability: float = 0.10
    spurious_probability: float = 0.08
    timing_jitter_sec: float = 0.035
    max_samples: int = 0
    build_missing_cache: bool = False
    calibration_maps: int = 128
    early_stop_patience: int = 3
    transcriber_checkpoint: Path | None = None
    transcriber_onset_tolerance_sec: float = 0.12


def _array(blob: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in blob:
            return blob[name]
    return default


def load_exact_note_map(path: Path | str) -> ExactNoteMap:
    """Load an exact JSON lineage cache or compact NPZ target cache."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_lineage_note_map(path)
    blob = np.load(path, allow_pickle=False)
    pitches = _array(blob, ("observed_pitch", "performance_pitch", "pitch"))
    starts = _array(blob, ("observed_start", "performance_start", "start"))
    ends = _array(blob, ("observed_end", "performance_end", "end"))
    targets = _array(blob, ("target_score_index", "score_index", "target"))
    if pitches is None or starts is None or ends is None or targets is None:
        raise ValueError(f"{path} is not an exact note-map cache")
    confidence = _array(blob, ("confidence", "observed_confidence"), np.ones(len(pitches)))
    is_copy = _array(blob, ("target_is_copy", "is_copy"), np.zeros(len(pitches), dtype=np.bool_))

    score_path_raw = _array(blob, ("score_path",), None)
    sample_dir = path.parent
    if score_path_raw is not None:
        raw = score_path_raw.item() if np.asarray(score_path_raw).ndim == 0 else score_path_raw[0]
        score_path = Path(str(raw))
        if not score_path.is_absolute():
            score_path = sample_dir / score_path
    else:
        score_path = sample_dir / "verified_score.musicxml"
    if score_path.exists():
        score = build_score_graph(score_path)
    else:
        score_pitch = _array(blob, ("score_pitch", "written_pitch"))
        score_start = _array(blob, ("score_start", "written_start"))
        score_end = _array(blob, ("score_end", "written_end"))
        if score_pitch is None or score_start is None or score_end is None:
            raise FileNotFoundError(
                f"No score_path or embedded score arrays in {path}"
            )
        from alignmodel.types import GraphNote

        graph_notes = []
        for i, (pitch, start, end) in enumerate(zip(score_pitch, score_start, score_end)):
            start_f, end_f = float(start), max(float(end), float(start) + 0.001)
            graph_notes.append(
                GraphNote(
                    index=i,
                    pitch=int(pitch),
                    start=start_f,
                    end=end_f,
                    duration=end_f - start_f,
                    ql_start=start_f,
                    ql_end=end_f,
                )
            )
        score = ScoreGraph(
            notes=graph_notes,
            duration_sec=graph_notes[-1].end if graph_notes else 0.0,
        )
    raw_notes = [
        {
            "pitch": int(pitch),
            "start": float(start),
            "end": float(end),
            "confidence": float(conf),
        }
        for pitch, start, end, conf in zip(pitches, starts, ends, confidence)
    ]
    notes = normalize_notes(raw_notes)
    order = [n.source_index for n in notes]
    target_list = [
        None if int(targets[i]) < 0 else int(targets[i])
        for i in order
    ]
    copy_list = [bool(is_copy[i]) for i in order]
    if any(t is not None and not (0 <= t < len(score.notes)) for t in target_list):
        raise ValueError(f"Out-of-range target score index in {path}")
    return ExactNoteMap(path.parent.name, score, notes, target_list, copy_list)


def _load_lineage_note_map(path: Path) -> ExactNoteMap:
    """Load synthpipeline's generation-time exact lineage cache."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "synth_note_lineage":
        raise ValueError(f"{path} is not a synth note-lineage cache")
    from alignmodel.types import GraphNote

    clean_rows = sorted(
        payload.get("clean_notes") or [],
        key=lambda row: int(row["clean_index"]),
    )
    graph_notes = []
    for row in clean_rows:
        index = int(row["clean_index"])
        start = float(row["onset_ql"])
        duration = max(float(row["duration_ql"]), 0.001)
        graph_notes.append(
            GraphNote(
                index=index,
                pitch=int(row["pitch_midi"]),
                start=start,
                end=start + duration,
                duration=duration,
                ql_start=start,
                ql_end=start + duration,
                measure=(
                    int(row["measure"]) if row.get("measure") is not None else None
                ),
            )
        )
    if [note.index for note in graph_notes] != list(range(len(graph_notes))):
        raise ValueError(f"Non-contiguous clean note indices in {path}")
    score = ScoreGraph(
        notes=graph_notes,
        duration_sec=max((note.end for note in graph_notes), default=0.0),
    )
    rendered_rows = sorted(
        payload.get("rendered_notes") or [],
        key=lambda row: int(row["rendered_index"]),
    )
    performed_rows = sorted(
        payload.get("performed_notes") or [],
        key=lambda row: int(row["performed_index"]),
    )
    raw_notes = []
    targets: list[int | None] = []
    copies: list[bool] = []
    if rendered_rows:
        for row in rendered_rows:
            raw_notes.append(
                {
                    "pitch": int(row["pitch_midi_written"]),
                    "start": float(row["start_sec"]),
                    "end": float(row["end_sec"]),
                    "confidence": 1.0,
                }
            )
            primary = row.get("primary_clean_index")
            target = int(primary) if primary is not None else None
            targets.append(target)
            copies.append(row.get("relationship") == "copy" and target is not None)
    else:
        for row in performed_rows:
            start = float(row["onset_ql"])
            duration = max(float(row["duration_ql"]), 0.001)
            raw_notes.append(
                {
                    "pitch": int(row["pitch_midi"]),
                    "start": start,
                    "end": start + duration,
                    "confidence": 1.0,
                }
            )
            clean_index = row.get("clean_index")
            target = int(clean_index) if clean_index is not None else None
            targets.append(target)
            copies.append(row.get("relationship") == "copy" and target is not None)
    notes = normalize_notes(raw_notes)
    order = [note.source_index for note in notes]
    return ExactNoteMap(
        path.parent.name,
        score,
        notes,
        [targets[i] for i in order],
        [copies[i] for i in order],
    )


def write_exact_note_map(item: ExactNoteMap, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observed_pitch=np.asarray([n.pitch for n in item.notes], dtype=np.int16),
        observed_start=np.asarray([n.start for n in item.notes], dtype=np.float32),
        observed_end=np.asarray([n.end for n in item.notes], dtype=np.float32),
        confidence=np.asarray([n.confidence for n in item.notes], dtype=np.float32),
        target_score_index=np.asarray(
            [-1 if x is None else x for x in item.target_score_indices], dtype=np.int32
        ),
        target_is_copy=np.asarray(item.target_is_copy, dtype=np.bool_),
        score_pitch=np.asarray([n.pitch for n in item.score.notes], dtype=np.int16),
        score_start=np.asarray([n.start for n in item.score.notes], dtype=np.float32),
        score_end=np.asarray([n.end for n in item.score.notes], dtype=np.float32),
    )
    return path


def _score_notes_from_musicxml(path: Path) -> list[ObservedNote]:
    from music21 import converter, note

    parsed = converter.parse(str(path))
    flat = parsed.flatten()
    rows: list[dict[str, float | int]] = []
    try:
        seconds_map = list(flat.secondsMap)
    except Exception:
        seconds_map = []
    for row in seconds_map:
        element = row.get("element")
        if not isinstance(element, note.Note) or element.duration.isGrace:
            continue
        start = float(row.get("offsetSeconds", 0.0))
        end = max(float(row.get("endTimeSeconds", start)), start + 0.001)
        rows.append({"pitch": int(element.pitch.midi), "start": start, "end": end})
    if not rows:
        for element in flat.getElementsByClass(note.Note):
            if element.duration.isGrace:
                continue
            start = float(element.offset)
            end = start + max(float(element.duration.quarterLength), 0.001)
            rows.append({"pitch": int(element.pitch.midi), "start": start, "end": end})
    return normalize_notes(rows)


def _plain_edit_map(
    observed: Sequence[ObservedNote], score_notes: Sequence[Any]
) -> list[int | None]:
    """Independent oracle edit path used only while making synthetic caches."""

    n, m = len(observed), len(score_notes)
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[:, 0] = np.arange(n + 1, dtype=np.float32) * 0.82
    dp[0, :] = np.arange(m + 1, dtype=np.float32) * 0.82
    bt[1:, 0], bt[0, 1:] = 2, 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delta = abs(observed[i - 1].pitch - int(score_notes[j - 1].pitch))
            pair = 0.0 if delta == 0 else 0.58 + min(delta, 12) / 24.0
            dp[i, j], bt[i, j] = min(
                (
                    (dp[i - 1, j - 1] + pair, 1),
                    (dp[i - 1, j] + 0.82, 2),
                    (dp[i, j - 1] + 0.82, 3),
                ),
                key=lambda x: (x[0], x[1]),
            )
    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i or j:
        code = int(bt[i, j])
        if i and j and code == 1:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i and (not j or code == 2):
            i -= 1
        else:
            j -= 1
    return mapping


def build_exact_note_map(sample_dir: Path | str) -> ExactNoteMap:
    """Build exact synthetic targets from clean/performance score artifacts.

    Repeat labels define source and replay windows explicitly.  Remaining notes
    use a deterministic edit path; no learned model or audio DTW is involved.
    """

    sample_dir = Path(sample_dir)
    score = build_score_graph(sample_dir / "verified_score.musicxml")
    notes = _score_notes_from_musicxml(sample_dir / "performance_score.musicxml")
    targets: list[int | None] = [None] * len(notes)
    is_copy = [False] * len(notes)
    claimed_obs: set[int] = set()

    labels_path = sample_dir / "labels.json"
    labels = []
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8")).get("labels") or []
    for label in labels:
        if label.get("type") != "repetition":
            continue
        source = label.get("repeats_label_range") or {}
        if "start_time" not in source or "end_time" not in source:
            continue
        source_start, source_end = float(source["start_time"]), float(source["end_time"])
        score_indices = [
            i
            for i, snote in enumerate(score.notes)
            if snote.start < source_end + 1e-4 and snote.end > source_start - 1e-4
        ]
        if not score_indices:
            continue
        copies = max(1, int(label.get("extra_copies") or 1))
        replay_start, replay_end = float(label["start_time"]), float(label["end_time"])
        width = max((replay_end - replay_start) / copies, 0.001)
        for copy_i in range(copies):
            lo = replay_start + copy_i * width
            hi = replay_start + (copy_i + 1) * width
            obs_indices = [
                i
                for i, item in enumerate(notes)
                if i not in claimed_obs
                and item.start < hi + 1e-4
                and item.end > lo - 1e-4
            ]
            local_obs = [notes[i] for i in obs_indices]
            local_score = [score.notes[i] for i in score_indices]
            local_map = _plain_edit_map(local_obs, local_score)
            for obs_i, local_target in zip(obs_indices, local_map):
                claimed_obs.add(obs_i)
                if local_target is not None:
                    targets[obs_i] = score_indices[local_target]
                    is_copy[obs_i] = True

    remaining_obs_indices = [i for i in range(len(notes)) if i not in claimed_obs]
    main_map = _plain_edit_map([notes[i] for i in remaining_obs_indices], score.notes)
    for obs_i, target in zip(remaining_obs_indices, main_map):
        targets[obs_i] = target
    return ExactNoteMap(sample_dir.name, score, notes, targets, is_copy)


def discover_exact_note_maps(
    cfg: NoteAlignTrainConfig,
) -> list[ExactNoteMap]:
    """Load precomputed targets, optionally building missing synthetic caches."""

    root = Path(cfg.data_root)
    cache_root = Path(cfg.cache_dir) if cfg.cache_dir is not None else None
    if cfg.manifest is not None:
        manifest = json.loads(Path(cfg.manifest).read_text(encoding="utf-8"))
        selected: list[ExactNoteMap] = []
        for split in ("train", "val"):
            for row in manifest.get(split) or []:
                sample_dir = Path(row["sample_dir"])
                candidates = []
                if cache_root is not None:
                    corpus = str(row.get("corpus") or row.get("root") or "")
                    if corpus:
                        candidates.extend(
                            cache_root / corpus / sample_dir.name / name
                            for name in CACHE_NAMES
                        )
                    candidates.extend(cache_root / sample_dir.name / name for name in CACHE_NAMES)
                candidates.extend(sample_dir / name for name in CACHE_NAMES)
                path = next((candidate for candidate in candidates if candidate.exists()), None)
                if path is None:
                    continue
                item = load_exact_note_map(path)
                item.split = split
                item.sample_dir = sample_dir
                selected.append(item)
                if cfg.max_samples and len(selected) >= cfg.max_samples:
                    return selected
        if selected:
            return selected
        raise FileNotFoundError(
            f"No exact note-map caches referenced by manifest {cfg.manifest}"
        )
    paths: list[Path] = []
    search_root = cache_root or root
    for name in CACHE_NAMES:
        paths.extend(search_root.rglob(name))
    # Prefer generation-time JSON lineage when multiple formats exist.
    preference = {name: rank for rank, name in enumerate(CACHE_NAMES)}
    by_sample: dict[Path, Path] = {}
    for path in sorted(set(paths)):
        current = by_sample.get(path.parent)
        if current is None or preference[path.name] < preference[current.name]:
            by_sample[path.parent] = path
    paths = sorted(by_sample.values())
    if cfg.max_samples:
        paths = paths[: cfg.max_samples]
    if paths:
        return [load_exact_note_map(path) for path in paths]
    if not cfg.build_missing_cache:
        raise FileNotFoundError(
            f"No exact note-map caches below {search_root}. "
            "Pass --build-missing-cache for synthetic bundles."
        )
    sample_dirs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p / "verified_score.musicxml").exists()
        and (p / "performance_score.musicxml").exists()
    )
    if cfg.max_samples:
        sample_dirs = sample_dirs[: cfg.max_samples]
    items: list[ExactNoteMap] = []
    for i, sample in enumerate(sample_dirs, start=1):
        try:
            item = build_exact_note_map(sample)
        except Exception as exc:
            print(f"note-map cache skip {sample.name}: {exc}", flush=True)
            continue
        dest = (
            cache_root / sample.name / "exact_note_map.npz"
            if cache_root is not None
            else sample / "exact_note_map.npz"
        )
        write_exact_note_map(item, dest)
        items.append(item)
        if i % 500 == 0:
            print(f"built exact note-map caches {i}/{len(sample_dirs)}", flush=True)
    if not items:
        raise FileNotFoundError(f"No usable synthetic bundles below {root}")
    return items


def _span(notes: Sequence[ObservedNote]) -> tuple[float, float]:
    if not notes:
        return (0.0, 1.0)
    return notes[0].start, max(n.end for n in notes)


def _training_rows(
    item: ExactNoteMap,
    rng: random.Random,
    cfg: NoteAlignTrainConfig,
    *,
    augment: bool,
) -> tuple[list[np.ndarray], list[int]]:
    notes = list(item.notes)
    targets = list(item.target_score_indices)
    copies = list(item.target_is_copy)
    if augment:
        new_notes: list[ObservedNote] = []
        new_targets: list[int | None] = []
        new_copies: list[bool] = []
        for obs, target, is_copy in zip(notes, targets, copies):
            if rng.random() < cfg.drop_probability:
                continue
            pitch = obs.pitch
            if target is not None and rng.random() < cfg.pitch_error_probability:
                pitch += rng.choice((-12, -2, -1, 1, 2, 12))
            jitter = rng.gauss(0.0, cfg.timing_jitter_sec)
            conf = float(np.clip(rng.uniform(0.65, 1.0), 0.0, 1.0))
            new_notes.append(
                ObservedNote(
                    pitch=pitch,
                    start=max(0.0, obs.start + jitter),
                    end=max(0.001, obs.end + jitter),
                    confidence=conf,
                )
            )
            new_targets.append(target)
            new_copies.append(is_copy)
            if rng.random() < cfg.spurious_probability:
                duration = max(obs.duration * rng.uniform(0.35, 0.8), 0.03)
                new_notes.append(
                    ObservedNote(
                        pitch=pitch + rng.choice((-7, -2, -1, 1, 2, 7)),
                        start=max(0.0, obs.start + rng.uniform(0.05, max(0.06, obs.duration))),
                        end=max(0.03, obs.start + duration),
                        confidence=rng.uniform(0.45, 0.95),
                    )
                )
                new_targets.append(None)
                new_copies.append(False)
        order = sorted(range(len(new_notes)), key=lambda i: (new_notes[i].start, new_notes[i].pitch))
        notes = [new_notes[i] for i in order]
        targets = [new_targets[i] for i in order]
        copies = [new_copies[i] for i in order]

    rows: list[np.ndarray] = []
    labels: list[int] = []
    ospan = _span(notes)
    sspan = (
        (item.score.notes[0].start, item.score.notes[-1].end)
        if item.score.notes
        else (0.0, 1.0)
    )
    used_score: set[int] = set()
    for i, (obs, target) in enumerate(zip(notes, targets)):
        if target is None:
            rows.append(
                note_features(
                    obs,
                    None,
                    observed_span=ospan,
                    score_span=sspan,
                    observed_order=i / max(len(notes) - 1, 1),
                )
            )
            labels.append(OP_CLASSES.index("extra"))
            continue
        score_note = item.score.notes[target]
        used_score.add(target)
        rows.append(
            note_features(
                obs,
                score_note,
                observed_span=ospan,
                score_span=sspan,
                observed_order=i / max(len(notes) - 1, 1),
                score_order=target / max(len(item.score.notes) - 1, 1),
            )
        )
        labels.append(
            OP_CLASSES.index("match" if obs.pitch == score_note.pitch else "substitute")
        )
        if len(item.score.notes) > 1:
            wrong = rng.randrange(len(item.score.notes) - 1)
            if wrong >= target:
                wrong += 1
            rows.append(
                note_features(
                    obs,
                    item.score.notes[wrong],
                    observed_span=ospan,
                    score_span=sspan,
                    observed_order=i / max(len(notes) - 1, 1),
                    score_order=wrong / max(len(item.score.notes) - 1, 1),
                )
            )
            labels.append(OP_CLASSES.index("reject"))
    for j, score_note in enumerate(item.score.notes):
        if j in used_score:
            continue
        rows.append(
            note_features(
                None,
                score_note,
                observed_span=ospan,
                score_span=sspan,
                score_order=j / max(len(item.score.notes) - 1, 1),
            )
        )
        labels.append(OP_CLASSES.index("deletion"))
    return rows, labels


def make_training_tensors(
    items: Sequence[ExactNoteMap],
    cfg: NoteAlignTrainConfig,
    *,
    augment: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    all_rows: list[np.ndarray] = []
    all_labels: list[int] = []
    for item in items:
        rows, labels = _training_rows(item, rng, cfg, augment=False)
        all_rows.extend(rows)
        all_labels.extend(labels)
        if augment:
            for _ in range(cfg.augmentations_per_map):
                rows, labels = _training_rows(item, rng, cfg, augment=True)
                all_rows.extend(rows)
                all_labels.extend(labels)
    if not all_rows:
        raise ValueError("No operation rows could be made from exact note maps")
    return torch.from_numpy(np.stack(all_rows)), torch.tensor(all_labels, dtype=torch.long)


def _prediction_maps(
    items: Sequence[ExactNoteMap],
    cfg: NoteAlignTrainConfig,
    device: torch.device,
) -> list[ExactNoteMap]:
    """Convert transcriber outputs into aligner training rows with exact lineage."""
    if cfg.transcriber_checkpoint is None:
        return []
    from alignmodel.transcription import infer_note_decoder, load_note_decoder

    transcriber = load_note_decoder(cfg.transcriber_checkpoint, device)
    output: list[ExactNoteMap] = []
    for index, item in enumerate(items, start=1):
        sample_dir = item.sample_dir
        if sample_dir is None:
            continue
        mel_path = sample_dir / "performance_mel.npy"
        score_path = sample_dir / "verified_score.musicxml"
        if not (mel_path.exists() and score_path.exists()):
            continue
        gold_performed = item.notes
        predicted = normalize_notes(
            infer_note_decoder(
                transcriber,
                sample_dir,
            )
        )
        available = set(range(len(gold_performed)))
        targets: list[int | None] = []
        copies: list[bool] = []
        for value in predicted:
            candidates = [
                gold_i
                for gold_i in available
                if abs(gold_performed[gold_i].start - value.start)
                <= cfg.transcriber_onset_tolerance_sec
            ]
            if candidates:
                gold_i = min(
                    candidates,
                    key=lambda i: (
                        abs(gold_performed[i].start - value.start),
                        abs(gold_performed[i].pitch - value.pitch),
                    ),
                )
                available.remove(gold_i)
                targets.append(item.target_score_indices[gold_i])
                copies.append(item.target_is_copy[gold_i])
            else:
                targets.append(None)
                copies.append(False)
        predicted_score = build_score_graph(score_path)
        if len(predicted_score.notes) != len(item.score.notes):
            continue
        output.append(
            ExactNoteMap(
                sample_id=item.sample_id,
                score=predicted_score,
                notes=predicted,
                target_score_indices=targets,
                target_is_copy=copies,
                split=item.split,
                sample_dir=sample_dir,
            )
        )
        if index == 1 or index % 250 == 0:
            print(
                f"note-align transcriber maps {index}/{len(items)} "
                f"usable={len(output)}",
                flush=True,
            )
    del transcriber
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _classification_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    accuracy = float((pred == target).float().mean().item())
    pair_mask = target < 2
    pair_accuracy = (
        float((pred[pair_mask] == target[pair_mask]).float().mean().item())
        if bool(pair_mask.any())
        else 0.0
    )
    return {"operation_accuracy": accuracy, "pair_accuracy": pair_accuracy}


def _evaluate_maps(
    scorer: LearnedNoteScorer,
    items: Sequence[ExactNoteMap],
    config: NoteAlignConfig,
    device: torch.device,
) -> dict[str, float]:
    aligner = NoteAligner(scorer, copy.deepcopy(config), device=device)
    rows = [
        alignment_metrics(
            aligner.align(item.notes, item.score),
            item.target_score_indices,
            target_is_copy=item.target_is_copy,
        )
        for item in items
    ]
    keys = ("accuracy", "precision", "recall", "f1", "copy_accuracy", "copy_f1")
    return {
        key: float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
        for key in keys
    }


def _calibrate(
    scorer: LearnedNoteScorer,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_maps: Sequence[ExactNoteMap],
    align_config: NoteAlignConfig,
    device: torch.device,
) -> dict[str, Any]:
    scorer.eval()
    with torch.no_grad():
        raw_logits = scorer(val_x.to(device)).cpu()
    temperatures = (0.60, 0.75, 0.90, 1.0, 1.15, 1.35, 1.60, 2.0)
    nll_rows = [
        (float(nn.functional.cross_entropy(raw_logits / temp, val_y).item()), temp)
        for temp in temperatures
    ]
    best_nll, temperature = min(nll_rows)
    threshold_rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.08, 0.48, 11):
        config = copy.deepcopy(align_config)
        config.temperature = temperature
        config.unattached_threshold = float(threshold)
        config.substitute_threshold = max(float(threshold), 0.24)
        metrics = _evaluate_maps(scorer, val_maps, config, device)
        threshold_rows.append({"threshold": float(threshold), **metrics})
    best = max(
        threshold_rows,
        key=lambda row: (row["f1"], row["copy_f1"], row["accuracy"]),
    )
    return {
        "temperature": float(temperature),
        "unattached_threshold": float(best["threshold"]),
        "substitute_threshold": max(float(best["threshold"]), 0.24),
        "nll": best_nll,
        "alignment_metrics": {k: v for k, v in best.items() if k != "threshold"},
        "threshold_sweep": threshold_rows,
    }


def train_note_aligner(cfg: NoteAlignTrainConfig) -> Path:
    """Train and calibrate a small scorer; writes best/last/history artifacts."""

    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    items = discover_exact_note_maps(cfg)
    explicit_train = [item for item in items if item.split == "train"]
    explicit_val = [item for item in items if item.split == "val"]
    if explicit_train and explicit_val:
        train_maps, val_maps = explicit_train, explicit_val
    else:
        rng = random.Random(cfg.seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * cfg.val_fraction)))
        if len(shuffled) > 1:
            n_val = min(n_val, len(shuffled) - 1)
        val_maps, train_maps = shuffled[:n_val], shuffled[n_val:]
    if not train_maps:
        train_maps = val_maps
    device = resolve_device(cfg.device)
    train_x, train_y = make_training_tensors(
        train_maps, cfg, augment=True, seed=cfg.seed + 1
    )
    predicted_maps = _prediction_maps(train_maps, cfg, device)
    if predicted_maps:
        pred_x, pred_y = make_training_tensors(
            predicted_maps, cfg, augment=False, seed=cfg.seed + 3
        )
        train_x = torch.cat([train_x, pred_x], dim=0)
        train_y = torch.cat([train_y, pred_y], dim=0)
    val_x, val_y = make_training_tensors(
        val_maps, cfg, augment=False, seed=cfg.seed + 2
    )
    print(
        f"note-align maps={len(items)} train_maps={len(train_maps)} val_maps={len(val_maps)} "
        f"rows={len(train_y)}/{len(val_y)}",
        flush=True,
    )

    model = LearnedNoteScorer(FEATURE_DIM, cfg.hidden_dim).to(device)
    counts = torch.bincount(train_y, minlength=len(OP_CLASSES)).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    align_config = NoteAlignConfig(learned_weight=cfg.learned_weight)
    best_f1 = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = cfg.output_dir / "note_aligner.pt"
    last_path = cfg.output_dir / "note_aligner_last.pt"
    eval_maps = val_maps[: cfg.calibration_maps]
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().item())
            batches += 1
        scheduler.step()
        model.eval()
        logits_out = []
        target_out = []
        val_loss = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = model(features)
                val_loss += float(criterion(logits, target).item())
                logits_out.append(logits.cpu())
                target_out.append(target.cpu())
        logits_cat = torch.cat(logits_out)
        target_cat = torch.cat(target_out)
        cls_metrics = _classification_metrics(logits_cat, target_cat)
        map_metrics = _evaluate_maps(model, eval_maps, align_config, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            "val_loss": val_loss / max(len(val_loader), 1),
            **cls_metrics,
            **{f"alignment_{k}": v for k, v in map_metrics.items()},
        }
        history.append(row)
        calibration = {
            "temperature": 1.0,
            "unattached_threshold": align_config.unattached_threshold,
            "substitute_threshold": align_config.substitute_threshold,
        }
        payload = checkpoint_payload(
            model,
            align_config,
            epoch=epoch,
            metrics=row,
            calibration=calibration,
            history=history,
        )
        payload["train_config"] = {
            **asdict(cfg),
            "data_root": str(cfg.data_root),
            "output_dir": str(cfg.output_dir),
            "cache_dir": str(cfg.cache_dir) if cfg.cache_dir is not None else None,
            "manifest": str(cfg.manifest) if cfg.manifest is not None else None,
        }
        torch.save(payload, last_path)
        # Component diagnostic only: alignment-map F1 cannot promote a model.
        improved = map_metrics["f1"] > best_f1 + 1e-4
        if improved:
            best_f1 = map_metrics["f1"]
            stale = 0
            torch.save(payload, best_path)
        else:
            stale += 1
        print(
            f"note-align epoch {epoch} train={row['train_loss']:.4f} "
            f"val={row['val_loss']:.4f} op_acc={row['operation_accuracy']:.3f} "
            f"align_f1={map_metrics['f1']:.3f} copy_f1={map_metrics['copy_f1']:.3f} "
            f"device={device_label(device)}",
            flush=True,
        )
        if stale >= cfg.early_stop_patience:
            break

    best_blob = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_blob["model"])
    calibration = _calibrate(
        model, val_x, val_y, eval_maps, align_config, device
    )
    align_config.temperature = float(calibration["temperature"])
    align_config.unattached_threshold = float(calibration["unattached_threshold"])
    align_config.substitute_threshold = float(calibration["substitute_threshold"])
    final_metrics = _evaluate_maps(model, eval_maps, align_config, device)
    final_payload = checkpoint_payload(
        model,
        align_config,
        epoch=int(best_blob["epoch"]),
        metrics=final_metrics,
        calibration=calibration,
        history=history,
    )
    final_payload["train_config"] = best_blob.get("train_config", {})
    torch.save(final_payload, best_path)
    (cfg.output_dir / "note_aligner_history.json").write_text(
        json.dumps(
            {
                "history": history,
                "calibration": calibration,
                "final_metrics": final_metrics,
                "operation_counts": dict(zip(OP_CLASSES, counts.int().tolist())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {best_path} align_f1={final_metrics['f1']:.3f} "
        f"copy_f1={final_metrics['copy_f1']:.3f}",
        flush=True,
    )
    return best_path


__all__ = [
    "ExactNoteMap",
    "NoteAlignTrainConfig",
    "build_exact_note_map",
    "discover_exact_note_maps",
    "load_exact_note_map",
    "make_training_tensors",
    "train_note_aligner",
    "write_exact_note_map",
]
