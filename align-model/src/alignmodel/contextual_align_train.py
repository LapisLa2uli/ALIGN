"""Training loop for the contextual repetition-aware note aligner."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from alignmodel.stages.contextual_note_aligner import (
    ContextualNoteAligner,
    sequence_features,
    save_contextual_aligner,
)
from alignmodel.types import TranscribedNote


@dataclass
class AlignmentSequence:
    observed: np.ndarray
    score: np.ndarray
    targets: np.ndarray


def load_alignment_sequence(path: Path, *, augment: bool = False) -> AlignmentSequence:
    document = json.loads(path.read_text(encoding="utf-8"))
    clean_rows = sorted(
        document.get("clean_notes") or [],
        key=lambda row: int(row["clean_index"]),
    )
    score_notes = [
        {
            "pitch": int(row["pitch_midi"]),
            "start": float(row["onset_ql"]),
            "end": float(row["onset_ql"]) + float(row["duration_ql"]),
            "confidence": 1.0,
        }
        for row in clean_rows
    ]
    clean_pitch = {
        int(row["clean_index"]): int(row["pitch_midi"])
        for row in clean_rows
    }
    rendered = sorted(
        document.get("rendered_notes") or [],
        key=lambda row: int(row["rendered_index"]),
    )
    observed = []
    targets = []
    for row in rendered:
        if str(row.get("relationship")) == "copy":
            continue
        target = row.get("primary_clean_index")
        if (
            target is not None
            and str(row.get("relationship")) != "substitute"
            and int(target) in clean_pitch
        ):
            pitch = clean_pitch[int(target)]
        else:
            pitch = int(row["pitch_midi_written"]) - 2
        confidence = random.uniform(0.65, 1.0) if augment else 1.0
        if augment and random.random() < 0.08:
            pitch += random.choice((-12, -2, -1, 1, 2, 12))
        jitter = random.gauss(0.0, 0.025) if augment else 0.0
        observed.append(
            TranscribedNote(
                pitch,
                max(0.0, float(row["start_sec"]) + jitter),
                max(
                    0.001,
                    float(row["end_sec"]) + jitter,
                ),
                confidence,
            )
        )
        targets.append(int(target) if target is not None else -1)
    if augment and observed:
        kept_notes = []
        kept_targets = []
        for note, target in zip(observed, targets):
            if random.random() < 0.05:
                continue
            kept_notes.append(note)
            kept_targets.append(target)
            if random.random() < 0.05:
                kept_notes.append(
                    TranscribedNote(
                        note.pitch + random.choice((-2, -1, 1, 2)),
                        note.start + 0.02,
                        note.end,
                        random.uniform(0.35, 0.75),
                    )
                )
                kept_targets.append(-1)
        observed, targets = kept_notes, kept_targets
    ordered = sorted(
        zip(observed, targets),
        key=lambda item: (item[0].start, item[0].end, item[0].pitch),
    )
    observed = [item[0] for item in ordered]
    targets = [item[1] for item in ordered]
    return AlignmentSequence(
        sequence_features(observed),
        sequence_features(score_notes),
        np.asarray(targets, np.int64),
    )


class AlignmentDataset(Dataset):
    def __init__(self, paths: list[Path], *, augment: bool) -> None:
        self.paths = paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> AlignmentSequence:
        return load_alignment_sequence(
            self.paths[index], augment=self.augment
        )


def collate_alignment(rows: list[AlignmentSequence]):
    batch = len(rows)
    n_max = max(len(row.observed) for row in rows)
    m_max = max(len(row.score) for row in rows)
    observed = torch.zeros(batch, n_max, rows[0].observed.shape[-1])
    score = torch.zeros(batch, m_max, rows[0].score.shape[-1])
    observed_mask = torch.zeros(batch, n_max, dtype=torch.bool)
    score_mask = torch.zeros(batch, m_max, dtype=torch.bool)
    targets = torch.full((batch, n_max), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        n, m = len(row.observed), len(row.score)
        observed[index, :n] = torch.from_numpy(row.observed)
        score[index, :m] = torch.from_numpy(row.score)
        observed_mask[index, :n] = True
        score_mask[index, :m] = True
        row_targets = torch.from_numpy(row.targets)
        row_targets = torch.where(
            row_targets >= 0,
            row_targets,
            torch.full_like(row_targets, m_max),
        )
        targets[index, :n] = row_targets
    return observed, score, observed_mask, score_mask, targets


def _paths(manifest: Path, split: str) -> list[Path]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    paths = []
    for raw in document.get(split) or []:
        row = dict(raw)
        if str(row.get("corpus") or row.get("root")) != "procedural12k":
            raise ValueError("Contextual aligner rejected non-procedural row")
        path = Path(str(row["note_map"]))
        if path.is_file():
            paths.append(path)
    return paths


def train_contextual_aligner(
    manifest: Path,
    output: Path,
    *,
    train_samples: int = 10000,
    val_samples: int = 300,
    epochs: int = 8,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 365,
) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_paths = _paths(manifest, "train")
    val_paths = _paths(manifest, "val")
    if len(train_paths) < train_samples:
        needed = train_samples - len(train_paths)
        if len(val_paths) <= needed:
            raise ValueError(
                f"Only {len(train_paths) + len(val_paths)} maps available; "
                f"cannot reserve validation after selecting {train_samples}"
            )
        train_paths = train_paths + val_paths[:needed]
        val_paths = val_paths[needed:]
    else:
        train_paths = train_paths[:train_samples]
    val_paths = val_paths[:val_samples]
    train_loader = DataLoader(
        AlignmentDataset(train_paths, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_alignment,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        AlignmentDataset(val_paths, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_alignment,
        persistent_workers=True,
        prefetch_factor=2,
    )
    model = ContextualNoteAligner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for observed, score, observed_mask, score_mask, targets in train_loader:
            observed = observed.to(device)
            score = score.to(device)
            observed_mask = observed_mask.to(device)
            score_mask = score_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(observed, score, observed_mask, score_mask)
            loss = nn.functional.cross_entropy(
                logits.transpose(1, 2), targets, ignore_index=-100
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        model.eval()
        correct = count = 0
        with torch.no_grad():
            for observed, score, observed_mask, score_mask, targets in val_loader:
                logits = model(
                    observed.to(device),
                    score.to(device),
                    observed_mask.to(device),
                    score_mask.to(device),
                ).cpu()
                predicted = logits.argmax(dim=-1)
                valid = targets != -100
                correct += int(((predicted == targets) & valid).sum())
                count += int(valid.sum())
        accuracy = correct / max(count, 1)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "val_note_accuracy": accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
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
            "format_version": 1,
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "best_val_note_accuracy": best_accuracy,
            "history": history,
        },
    )
    output.with_suffix(".history.json").write_text(
        json.dumps(
            {
                "train_samples": len(train_paths),
                "val_samples": len(val_paths),
                "best_val_note_accuracy": best_accuracy,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
