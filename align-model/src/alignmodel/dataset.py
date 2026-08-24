from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from alignmodel.config import FRAME_HOP_SEC, SCORE_CLASSES, ModelConfig
from alignmodel.score import ScoreNote, parse_score_notes

# Overlap priority: more specific pitch/timing errors beat match.
_SPAN_TO_CLASS = {
    "missed_note": SCORE_CLASSES.index("miss"),
    "wrong_note": SCORE_CLASSES.index("wrong"),
    "rhythm_error": SCORE_CLASSES.index("rhythm"),
    "intonation_error": SCORE_CLASSES.index("intonation"),
}
_PRIORITY = {
    SCORE_CLASSES.index("wrong"): 4,
    SCORE_CLASSES.index("miss"): 3,
    SCORE_CLASSES.index("rhythm"): 2,
    SCORE_CLASSES.index("intonation"): 1,
    SCORE_CLASSES.index("match"): 0,
}


def list_sample_dirs(root: Path) -> list[Path]:
    dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "labels.json").exists() and (p / "performance_mel.npy").exists():
            dirs.append(p)
    return dirs


def _load_score_notes(sample_dir: Path) -> list[ScoreNote]:
    cache = sample_dir / "_score_notes.npz"
    score_path = sample_dir / "verified_score.musicxml"
    if cache.exists() and cache.stat().st_mtime >= score_path.stat().st_mtime:
        blob = np.load(cache)
        return [
            ScoreNote(
                pitch=int(p),
                start=float(s),
                end=float(e),
                duration=float(d),
            )
            for p, s, e, d in zip(blob["pitch"], blob["start"], blob["end"], blob["duration"])
        ]
    notes = parse_score_notes(score_path)
    if notes:
        np.savez(
            cache,
            pitch=np.array([n.pitch for n in notes], dtype=np.int16),
            start=np.array([n.start for n in notes], dtype=np.float32),
            end=np.array([n.end for n in notes], dtype=np.float32),
            duration=np.array([n.duration for n in notes], dtype=np.float32),
        )
    return notes


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


class AlignBundleDataset(Dataset):
    def __init__(self, sample_dirs: list[Path], cfg: ModelConfig):
        self.sample_dirs = sample_dirs
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.sample_dirs[idx]
        mel = np.load(sample_dir / "performance_mel.npy").astype(np.float32)
        if mel.ndim != 2:
            raise ValueError(f"Unexpected mel shape {mel.shape} in {sample_dir}")
        # librosa melspectrogram is [n_mels, T]
        if mel.shape[0] > mel.shape[1] and mel.shape[0] != self.cfg.n_mels:
            mel = mel.T
        if mel.shape[0] != self.cfg.n_mels:
            # crop or pad frequency
            if mel.shape[0] > self.cfg.n_mels:
                mel = mel[: self.cfg.n_mels]
            else:
                pad = np.zeros((self.cfg.n_mels - mel.shape[0], mel.shape[1]), dtype=np.float32)
                mel = np.concatenate([mel, pad], axis=0)

        t = min(mel.shape[1], self.cfg.max_audio_frames)
        mel = mel[:, :t]
        mel_mask = np.ones(t, dtype=np.bool_)

        notes = _load_score_notes(sample_dir)[: self.cfg.max_score_notes]
        labels = json.loads((sample_dir / "labels.json").read_text(encoding="utf-8")).get(
            "labels", []
        )
        meta_path = sample_dir / "metadata.json"
        repeated = False
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            repeated = bool(meta.get("repeated"))
        if any(lab.get("type") == "repetition" for lab in labels):
            repeated = True

        n = len(notes)
        if n == 0:
            notes = [ScoreNote(pitch=60, start=0.0, end=0.25, duration=0.25)]
            n = 1
        pitch = np.zeros(self.cfg.max_score_notes, dtype=np.int64)
        onset = np.zeros(self.cfg.max_score_notes, dtype=np.float32)
        duration = np.zeros(self.cfg.max_score_notes, dtype=np.float32)
        note_mask = np.zeros(self.cfg.max_score_notes, dtype=np.bool_)
        score_y = np.zeros(self.cfg.max_score_notes, dtype=np.int64)  # match
        for i, note in enumerate(notes):
            pitch[i] = max(0, min(127, note.pitch))
            onset[i] = note.start
            duration[i] = note.duration
            note_mask[i] = True
            cls = SCORE_CLASSES.index("match")
            pri = 0
            for lab in labels:
                kind = lab.get("type")
                if kind not in _SPAN_TO_CLASS:
                    continue
                if _overlaps(note.start, note.end, float(lab["start_time"]), float(lab["end_time"])):
                    cand = _SPAN_TO_CLASS[kind]
                    if _PRIORITY[cand] > pri:
                        cls = cand
                        pri = _PRIORITY[cand]
            score_y[i] = cls

        extra = np.zeros(t, dtype=np.float32)
        for lab in labels:
            if lab.get("type") not in {"extra_note", "repetition"}:
                continue
            i0 = max(0, int(float(lab["start_time"]) / FRAME_HOP_SEC))
            i1 = min(t, int(math.ceil(float(lab["end_time"]) / FRAME_HOP_SEC)))
            extra[i0:i1] = 1.0

        return {
            "mel": torch.from_numpy(mel),
            "mel_mask": torch.from_numpy(mel_mask),
            "pitch": torch.from_numpy(pitch),
            "onset": torch.from_numpy(onset),
            "duration": torch.from_numpy(duration),
            "note_mask": torch.from_numpy(note_mask),
            "score_y": torch.from_numpy(score_y),
            "extra_y": torch.from_numpy(extra),
            "repeat_y": torch.tensor(1.0 if repeated else 0.0, dtype=torch.float32),
            "n_notes": n,
            "sample_id": sample_dir.name,
        }


def collate_bundles(batch: list[dict]) -> dict:
    max_t = max(item["mel"].shape[-1] for item in batch)
    n_mels = batch[0]["mel"].shape[0]
    b = len(batch)
    mel = torch.zeros(b, n_mels, max_t)
    mel_mask = torch.zeros(b, max_t, dtype=torch.bool)
    extra_y = torch.zeros(b, max_t)
    pitch = torch.stack([item["pitch"] for item in batch])
    onset = torch.stack([item["onset"] for item in batch])
    duration = torch.stack([item["duration"] for item in batch])
    note_mask = torch.stack([item["note_mask"] for item in batch])
    score_y = torch.stack([item["score_y"] for item in batch])
    repeat_y = torch.stack([item["repeat_y"] for item in batch])
    for i, item in enumerate(batch):
        t = item["mel"].shape[-1]
        mel[i, :, :t] = item["mel"]
        mel_mask[i, :t] = item["mel_mask"]
        extra_y[i, :t] = item["extra_y"]
    return {
        "mel": mel,
        "mel_mask": mel_mask,
        "pitch": pitch,
        "onset": onset,
        "duration": duration,
        "note_mask": note_mask,
        "score_y": score_y,
        "extra_y": extra_y,
        "repeat_y": repeat_y,
        "sample_id": [item["sample_id"] for item in batch],
        "n_notes": [item["n_notes"] for item in batch],
    }
