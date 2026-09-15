"""Fine-tune contextual alignment on real cached Basic Pitch sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from alignmodel.stages.contextual_note_aligner import (
    ContextualNoteAligner,
    load_contextual_aligner,
    save_contextual_aligner,
    sequence_features,
)
from alignmodel.stages.note_repetition_model import load_note_repetition_model
from alignmodel.stages.repetition import apply_note_repetitions
from alignmodel.transcription import match_notes
from alignmodel.transcription.basic_pitch import (
    BasicPitchFeatures,
    basic_pitch_cache_path,
    decode_frozen_basic_pitch,
    sanitize_basic_pitch_notes,
)
from alignmodel.types import (
    GraphNote,
    PipelineConfig,
    PipelineState,
    ScoreGraph,
    TranscribedNote,
)
from alignmodel.validated_targets import target_note_map


@dataclass
class CachedAlignmentSequence:
    observed: np.ndarray
    score: np.ndarray
    targets: np.ndarray


def _pitch_correction(document: dict) -> int:
    clean = {
        int(note["clean_index"]): int(note["pitch_midi"])
        for note in document.get("clean_notes") or []
    }
    offsets = [
        int(note["pitch_midi_written"])
        - clean[int(note["primary_clean_index"])]
        for note in document.get("rendered_notes") or []
        if note.get("primary_clean_index") is not None
        and int(note["primary_clean_index"]) in clean
        and str(note.get("relationship")) in {"match", "copy"}
    ]
    return -int(round(float(np.median(offsets)))) if offsets else 0


def _gold_and_score(
    note_map: Path | Mapping[str, Any], score_path: Path | None = None
):
    document = (
        target_note_map(note_map)
        if isinstance(note_map, Mapping)
        else json.loads(note_map.read_text(encoding="utf-8"))
    )
    clean_rows = sorted(
        document.get("clean_notes") or [],
        key=lambda note: int(note["clean_index"]),
    )
    score_notes = [
        GraphNote(
            int(note["clean_index"]),
            int(note["pitch_midi"]),
            float(note["onset_ql"]),
            float(note["onset_ql"]) + float(note["duration_ql"]),
            float(note["duration_ql"]),
            float(note["onset_ql"]),
            float(note["onset_ql"]) + float(note["duration_ql"]),
            measure=note.get("measure"),
        )
        for note in clean_rows
    ]
    target_index_map = {
        int(note["clean_index"]): int(note["clean_index"])
        for note in clean_rows
    }
    if score_path is not None and score_path.is_file():
        from alignmodel.stages.score_graph import build_score_graph

        graph = build_score_graph(score_path)
        score_notes = graph.notes
        target_index_map = {
            source_index: graph_note.index
            for graph_note in graph.notes
            for source_index in (
                graph_note.source_note_indices or [graph_note.index]
            )
        }
    correction = _pitch_correction(document)
    gold_notes = []
    targets = []
    for note in sorted(
        document.get("rendered_notes") or [],
        key=lambda value: int(value["rendered_index"]),
    ):
        gold_notes.append(
            {
                "pitch": int(note["pitch_midi_written"]) + correction,
                "start": float(note["start_sec"]),
                "end": float(note["end_sec"]),
            }
        )
        target = note.get("primary_clean_index")
        targets.append(
            target_index_map.get(int(target))
            if target is not None
            else None
        )
    return gold_notes, targets, score_notes


def _load_basic_cache(path: Path) -> BasicPitchFeatures:
    with np.load(path, allow_pickle=False) as saved:
        return BasicPitchFeatures(
            np.asarray(saved["note"], dtype=np.float32),
            np.asarray(saved["onset"], dtype=np.float32),
            np.asarray(saved["contour"], dtype=np.float32),
            np.asarray(saved["frame_times"], dtype=np.float64),
            json.loads(str(saved["metadata"].item())),
        )


def build_cached_alignment_sequence(
    row: dict,
    basic_cache_root: Path,
    repetition_model,
) -> CachedAlignmentSequence:
    sample = Path(str(row["sample_dir"]))
    corpus = str(row.get("corpus") or row.get("root") or "procedural12k")
    cache = basic_pitch_cache_path(basic_cache_root, sample, corpus)
    if not cache.is_file():
        raise FileNotFoundError(cache)
    basic_features = _load_basic_cache(cache)
    predicted = sanitize_basic_pitch_notes(
        decode_frozen_basic_pitch(basic_features), basic_features
    )
    note_map = (
        row
        if row.get("target_db") is not None and row.get("target_record") is not None
        else Path(str(row.get("note_map") or sample / "note_map.json"))
    )
    gold_notes, gold_targets, score_notes = _gold_and_score(
        note_map, sample / "verified_score.musicxml"
    )
    matched = dict(match_notes(predicted, gold_notes))
    prediction_targets = [
        gold_targets[matched[index]] if index in matched else None
        for index in range(len(predicted))
    ]
    transcribed = [
        TranscribedNote(
            note.pitch,
            note.start,
            note.end,
            note.confidence,
            note.cents,
            note.pitch_candidates,
        )
        for note in predicted
    ]
    score = ScoreGraph(
        notes=score_notes,
        duration_sec=max((note.end for note in score_notes), default=0.0),
    )
    state = PipelineState(
        sample_id=sample.name,
        sample_dir=str(sample),
        sr=22050,
        duration_sec=max((note.end for note in transcribed), default=0.0),
        hop_sec=256.0 / 22050.0,
        config=PipelineConfig(),
        score=score,
        transcribed_notes=transcribed,
    )
    apply_note_repetitions(
        state, learned=SimpleNamespace(note_repetition=repetition_model)
    )
    repeated = {
        index
        for repeat in state.note_repetitions
        for index in range(repeat.repeat_i0, repeat.repeat_i1)
    }
    first_notes = []
    first_targets = []
    previous_target = -1
    for index, (note, target) in enumerate(
        zip(transcribed, prediction_targets)
    ):
        if index in repeated:
            continue
        value = int(target) if target is not None else -1
        # A valid first-pass gold path must be monotonic. Layer-1 misses or
        # ambiguous duplicate matches become extra-note supervision.
        if value >= 0 and value <= previous_target:
            value = -1
        elif value >= 0:
            previous_target = value
        first_notes.append(note)
        first_targets.append(value)
    return CachedAlignmentSequence(
        sequence_features(first_notes),
        sequence_features(score_notes),
        np.asarray(first_targets, np.int64),
    )


class CachedSequenceDataset(Dataset):
    def __init__(self, rows, cache_root, repetition_model) -> None:
        self.sequences = [
            build_cached_alignment_sequence(
                dict(row), Path(cache_root), repetition_model
            )
            for row in rows
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def _collate(rows):
    batch = len(rows)
    n_max = max(len(row.observed) for row in rows)
    m_max = max(len(row.score) for row in rows)
    feature_dim = rows[0].observed.shape[-1]
    observed = torch.zeros(batch, n_max, feature_dim)
    score = torch.zeros(batch, m_max, feature_dim)
    observed_mask = torch.zeros(batch, n_max, dtype=torch.bool)
    score_mask = torch.zeros(batch, m_max, dtype=torch.bool)
    targets = torch.full((batch, n_max), -100, dtype=torch.long)
    lengths = []
    for index, row in enumerate(rows):
        n, m = len(row.observed), len(row.score)
        observed[index, :n] = torch.from_numpy(row.observed)
        score[index, :m] = torch.from_numpy(row.score)
        observed_mask[index, :n] = True
        score_mask[index, :m] = True
        value = torch.from_numpy(row.targets)
        targets[index, :n] = torch.where(
            value >= 0, value, torch.full_like(value, m_max)
        )
        lengths.append((n, m))
    return observed, score, observed_mask, score_mask, targets, lengths


def monotonic_sequence_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood over every monotonic match/extra/delete path."""

    n, width = logits.shape
    m = width - 1
    extra = logits[:, m]
    diagonals = [logits.new_zeros(1)]
    diagonal_starts = [0]
    negative = logits.new_tensor(-1e9)
    for diagonal_index in range(1, n + m + 1):
        start_i = max(0, diagonal_index - m)
        end_i = min(n, diagonal_index)
        i_values = torch.arange(
            start_i, end_i + 1, device=logits.device
        )
        j_values = diagonal_index - i_values
        previous = diagonals[-1]
        previous_start = diagonal_starts[-1]
        insert_mask = i_values > 0
        insert_index = (i_values - 1 - previous_start).clamp(
            0, len(previous) - 1
        )
        insert_score = previous[insert_index] + extra[
            (i_values - 1).clamp(0, max(n - 1, 0))
        ]
        insert_score = torch.where(insert_mask, insert_score, negative)
        delete_mask = j_values > 0
        delete_index = (i_values - previous_start).clamp(
            0, len(previous) - 1
        )
        delete_score = torch.where(
            delete_mask, previous[delete_index], negative
        )
        match_score = torch.full_like(insert_score, negative)
        if diagonal_index >= 2:
            previous_two = diagonals[-2]
            previous_two_start = diagonal_starts[-2]
            match_mask = (i_values > 0) & (j_values > 0)
            match_index = (i_values - 1 - previous_two_start).clamp(
                0, len(previous_two) - 1
            )
            pair = logits[
                (i_values - 1).clamp(0, max(n - 1, 0)),
                (j_values - 1).clamp(0, max(m - 1, 0)),
            ]
            match_score = torch.where(
                match_mask, previous_two[match_index] + pair, negative
            )
        diagonal = torch.logsumexp(
            torch.stack([insert_score, delete_score, match_score]), dim=0
        )
        diagonals.append(diagonal)
        diagonal_starts.append(start_i)
    log_partition = diagonals[-1][0]
    gold_score = logits.new_zeros(())
    for index, target in enumerate(targets.tolist()):
        if target == m:
            gold_score = gold_score + extra[index]
        elif 0 <= target < m:
            gold_score = gold_score + logits[index, target]
    # Deletion edges have fixed zero score in both gold and partition.
    return (log_partition - gold_score) / max(n + m, 1)


def train_cached_contextual_aligner(
    manifest: Path,
    cache_root: Path,
    repetition_checkpoint: Path,
    initial_checkpoint: Path,
    output: Path,
    *,
    train_samples: int = 1000,
    val_samples: int = 200,
    epochs: int = 5,
    batch_size: int = 8,
    device: str = "cuda",
) -> Path:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    train_rows = list(document.get("train") or [])[:train_samples]
    val_rows = list(document.get("val") or [])[:val_samples]
    repetition_model, _ = load_note_repetition_model(
        repetition_checkpoint, device
    )
    train_data = CachedSequenceDataset(
        train_rows, cache_root, repetition_model
    )
    val_data = CachedSequenceDataset(val_rows, cache_root, repetition_model)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    model, initial_extra = load_contextual_aligner(
        initial_checkpoint, device
    )
    model.config.deletion_cost = 0.90
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (
            observed,
            score,
            observed_mask,
            score_mask,
            targets,
            lengths,
        ) in train_loader:
            observed = observed.to(device)
            score = score.to(device)
            observed_mask = observed_mask.to(device)
            score_mask = score_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(observed, score, observed_mask, score_mask)
            losses = []
            for index, (n, m) in enumerate(lengths):
                losses.append(
                    monotonic_sequence_nll(
                        torch.cat(
                            [
                                logits[index, :n, :m],
                                logits[index, :n, -1:],
                            ],
                            dim=-1,
                        ),
                        torch.where(
                            targets[index, :n] == logits.shape[-1] - 1,
                            torch.full_like(targets[index, :n], m),
                            targets[index, :n],
                        ),
                    )
                )
            structured = torch.stack(losses).mean()
            local = nn.functional.cross_entropy(
                logits.transpose(1, 2), targets, ignore_index=-100
            )
            loss = structured + 0.25 * local
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach())
        model.eval()
        correct = count = 0
        with torch.no_grad():
            for observed, score, obs_mask, score_mask, targets, _lengths in val_loader:
                logits = model(
                    observed.to(device),
                    score.to(device),
                    obs_mask.to(device),
                    score_mask.to(device),
                ).cpu()
                valid = targets != -100
                correct += int(((logits.argmax(-1) == targets) & valid).sum())
                count += int(valid.sum())
        accuracy = correct / max(count, 1)
        history.append(
            {
                "epoch": epoch,
                "train_sequence_loss": total_loss / max(len(train_loader), 1),
                "val_note_accuracy": accuracy,
            }
        )
        print(
            f"epoch={epoch} sequence_loss={history[-1]['train_sequence_loss']:.4f} "
            f"val_note_accuracy={accuracy:.4f}",
            flush=True,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_contextual_aligner(
        output,
        model,
        extra={
            "format_version": 2,
            "training_input": "cached-basic-pitch-layer1",
            "loss": "monotonic-sequence-nll-plus-local-ce",
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "best_val_note_accuracy": best_accuracy,
            "initial_checkpoint": str(initial_checkpoint),
            "initial_extra": initial_extra,
            "history": history,
        },
    )
    output.with_suffix(".history.json").write_text(
        json.dumps(
            {
                "best_val_note_accuracy": best_accuracy,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
