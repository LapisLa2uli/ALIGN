"""Lightweight, score-free audio-to-note transcription.

The public API deliberately lives in this subpackage so the score-informed
ALIGN pipeline does not need to depend on a particular note recognizer.
"""

from .data import (
    FRAME_HOP_SEC,
    NoteCropDataset,
    TranscriptionExample,
    load_split,
    load_written_notes,
    load_written_notes_with_cents,
    write_fine_pitch_feature,
)
from .decode import (
    DecodeConfig,
    TransNote,
    decode_notes,
    infer_full_clip,
    infer_sample_notes,
    infer_sample_notes_hybrid,
    load_note_transcriber,
)
from .evaluate import evaluate_note_lists, match_notes
from .model import NoteFrameNet, NoteFrameNetConfig, note_frame_loss
from .train import NoteTrainConfig, train_note_transcriber

__all__ = [
    "FRAME_HOP_SEC",
    "DecodeConfig",
    "NoteCropDataset",
    "NoteFrameNet",
    "NoteFrameNetConfig",
    "NoteTrainConfig",
    "TransNote",
    "TranscriptionExample",
    "decode_notes",
    "evaluate_note_lists",
    "infer_full_clip",
    "infer_sample_notes",
    "load_note_transcriber",
    "load_split",
    "load_written_notes",
    "load_written_notes_with_cents",
    "match_notes",
    "note_frame_loss",
    "infer_sample_notes_hybrid",
    "train_note_transcriber",
    "write_fine_pitch_feature",
]
