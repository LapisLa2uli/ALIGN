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
from .basic_pitch import (
    BasicPitchDecodeConfig,
    BasicPitchFeatures,
    decode_frozen_basic_pitch,
    extract_sample_basic_pitch_features,
)
from .canonical import (
    CanonicalNoteDecoder,
    infer_note_decoder,
    load_note_decoder,
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
from .refiner import (
    NoteRefiner,
    NoteRefinerConfig,
    decode_refined_notes,
    load_note_refiner,
    note_refiner_loss,
)
from .refiner_train import RefinerTrainConfig, train_note_refiner
from .train import NoteTrainConfig, train_note_transcriber

__all__ = [
    "FRAME_HOP_SEC",
    "BasicPitchDecodeConfig",
    "BasicPitchFeatures",
    "CanonicalNoteDecoder",
    "DecodeConfig",
    "NoteCropDataset",
    "NoteFrameNet",
    "NoteFrameNetConfig",
    "NoteTrainConfig",
    "NoteRefiner",
    "NoteRefinerConfig",
    "RefinerTrainConfig",
    "TransNote",
    "TranscriptionExample",
    "decode_notes",
    "decode_frozen_basic_pitch",
    "decode_refined_notes",
    "evaluate_note_lists",
    "infer_full_clip",
    "infer_note_decoder",
    "infer_sample_notes",
    "load_note_transcriber",
    "load_note_decoder",
    "load_note_refiner",
    "load_split",
    "load_written_notes",
    "load_written_notes_with_cents",
    "match_notes",
    "note_frame_loss",
    "note_refiner_loss",
    "infer_sample_notes_hybrid",
    "train_note_transcriber",
    "train_note_refiner",
    "extract_sample_basic_pitch_features",
    "write_fine_pitch_feature",
]
