from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from alignmodel.stages.dc_alignment import (
    pairs_from_learned_alignment,
    transcribe_pipeline_state,
)
from alignmodel.stages.edits import run_stage2
from alignmodel.stages.note_align import AlignmentOperation, AlignmentResult
from alignmodel.stages.repetition import (
    apply_note_repetitions,
    filter_repetitions_by_score_continuation,
    find_note_sequence_repetitions,
)
from alignmodel.stages.note_repetition_model import NoteRepetitionModelConfig
from alignmodel.stages.rhythm import run_stage3
from alignmodel.types import (
    GraphNote,
    NoteRepetition,
    PairedEvent,
    PipelineConfig,
    PipelineLabel,
    PipelineState,
    ScoreGraph,
    TranscribedNote,
)


def _score(pitches: list[int]) -> ScoreGraph:
    return ScoreGraph(
        notes=[
            GraphNote(
                index,
                pitch,
                float(index),
                float(index + 1),
                1.0,
                float(index),
                float(index + 1),
                measure=1,
            )
            for index, pitch in enumerate(pitches)
        ],
        duration_sec=float(len(pitches)),
    )


def _state(pitches: list[int]) -> PipelineState:
    return PipelineState(
        sample_id="clip",
        sample_dir=".",
        sr=22050,
        duration_sec=float(len(pitches)),
        hop_sec=0.01,
        config=PipelineConfig(weights_dir=None),
        score=_score(pitches),
    )


class _IdentityAligner:
    def align(self, notes, score):
        operations = [
            AlignmentOperation("match", index, index, 1.0)
            for index in range(min(len(notes), len(score.notes)))
        ]
        return AlignmentResult(
            operations,
            len(notes),
            len(score.notes),
            0.0,
        )


class _OneNoteRepeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = NoteRepetitionModelConfig(
            threshold=0.5, max_repetitions=2
        )

    def forward(self, features):
        return features[:, 9] * 12.0 - 6.0


class NoteFirstPipelineTests(unittest.TestCase):
    def test_transcription_is_computed_only_once(self) -> None:
        state = _state([60])
        learned = SimpleNamespace(transcriber=object())
        predicted = [SimpleNamespace(
            pitch=60,
            start=0.0,
            end=0.8,
            confidence=0.9,
            cents=0.0,
            pitch_candidates=(60,),
        )]
        with patch(
            "alignmodel.transcription.infer_note_decoder",
            return_value=predicted,
        ) as infer:
            first = transcribe_pipeline_state(state, learned)
            second = transcribe_pipeline_state(state, learned)
        self.assertIs(first, second)
        self.assertEqual(infer.call_count, 1)

    def test_layer1_finds_repeated_transcribed_phrase(self) -> None:
        phrase = [60, 62, 64, 65, 67, 69, 71, 72]
        notes = [
            TranscribedNote(pitch, index * 0.5, index * 0.5 + 0.4)
            for index, pitch in enumerate(phrase + phrase)
        ]
        repetitions = find_note_sequence_repetitions(
            notes, min_notes=6, min_confidence=0.8
        )
        self.assertEqual(len(repetitions), 1)
        self.assertEqual(
            (
                repetitions[0].source_i0,
                repetitions[0].source_i1,
                repetitions[0].repeat_i0,
                repetitions[0].repeat_i1,
            ),
            (0, 8, 8, 16),
        )

    def test_trained_layer1_can_accept_one_extra_repeated_note(self) -> None:
        state = _state([60, 62])
        state.transcribed_notes = [
            TranscribedNote(60, 0.0, 0.4),
            TranscribedNote(62, 0.5, 0.9),
            TranscribedNote(60, 1.0, 1.4),
        ]
        apply_note_repetitions(
            state,
            learned=SimpleNamespace(
                note_repetition=_OneNoteRepeatModel()
            ),
        )
        self.assertEqual(len(state.note_repetitions), 1)
        repeat = state.note_repetitions[0]
        self.assertEqual((repeat.source_i0, repeat.repeat_i0), (0, 2))

    def test_continuation_rule_keeps_replay_that_resumes_score(self) -> None:
        score = _score([60, 62, 64, 65, 67])
        notes = [
            TranscribedNote(pitch, index * 0.5, index * 0.5 + 0.4)
            for index, pitch in enumerate(
                [60, 62, 64, 60, 62, 64, 65, 67]
            )
        ]
        repeat = NoteRepetition(0, 3, 3, 6, 0.0, 1.4, 1.5, 2.9, 0.98)
        self.assertEqual(
            filter_repetitions_by_score_continuation(
                notes, score.notes, [repeat]
            ),
            [repeat],
        )

    def test_continuation_rule_rejects_nonadjacent_motif(self) -> None:
        pitches = [60, 62, 64, 65, 67, 60, 62, 64, 69, 71]
        score = _score(pitches)
        notes = [
            TranscribedNote(pitch, index * 0.5, index * 0.5 + 0.4)
            for index, pitch in enumerate(pitches)
        ]
        repeat = NoteRepetition(0, 3, 5, 8, 0.0, 1.4, 2.5, 3.9, 0.98)
        self.assertEqual(
            filter_repetitions_by_score_continuation(
                notes, score.notes, [repeat]
            ),
            [],
        )

    def test_continuation_rule_rejects_replay_at_audio_end(self) -> None:
        score = _score([60, 62, 64, 65, 67])
        notes = [
            TranscribedNote(pitch, index * 0.5, index * 0.5 + 0.4)
            for index, pitch in enumerate(
                [60, 62, 64, 60, 62, 64]
            )
        ]
        repeat = NoteRepetition(0, 3, 3, 6, 0.0, 1.4, 1.5, 2.9, 0.98)
        self.assertEqual(
            filter_repetitions_by_score_continuation(
                notes, score.notes, [repeat]
            ),
            [],
        )

    def test_repeat_notes_reuse_source_score_mapping(self) -> None:
        phrase = [60, 62, 64, 65, 67, 69]
        state = _state(phrase)
        state.transcribed_notes = [
            TranscribedNote(pitch, index * 0.5, index * 0.5 + 0.4)
            for index, pitch in enumerate(phrase + phrase)
        ]
        state.note_repetitions = [
            NoteRepetition(0, 6, 6, 12, 0.0, 2.9, 3.0, 5.9, 0.98)
        ]
        learned = SimpleNamespace(
            transcriber=object(), note_aligner=_IdentityAligner()
        )
        pairs = pairs_from_learned_alignment(state, None, learned)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(state.note_mapping[:6], list(range(6)))
        self.assertEqual(state.note_mapping[6:], list(range(6)))

    def test_layer2_emits_note_errors_but_never_intonation(self) -> None:
        state = _state([60, 62, 64])
        state.transcribed_notes = [
            TranscribedNote(60, 0.0, 0.8, 0.9),
            TranscribedNote(63, 1.0, 1.8, 0.9),
            TranscribedNote(67, 2.0, 2.4, 0.9),
        ]
        state.note_mapping = [0, 1, None]
        state.rhythm_pairs = [
            PairedEvent(0, 60, 0.0, 1.0, 0.0, 0.8, "match"),
            PairedEvent(1, 62, 1.0, 2.0, 1.0, 1.8, "substitute"),
        ]
        state.labels = [
            PipelineLabel("old", "intonation_error", 0.0, 0.5)
        ]
        run_stage2(
            state,
            audio=None,
            chroma=None,
            learned=SimpleNamespace(),
        )
        kinds = [label.type for label in state.labels]
        self.assertEqual(kinds.count("wrong_note"), 1)
        self.assertEqual(kinds.count("extra_note"), 1)
        self.assertEqual(kinds.count("missed_note"), 1)
        self.assertNotIn("intonation_error", kinds)

    def test_layer3_reuses_pairs_and_flags_only_large_duration_change(self) -> None:
        state = _state([60, 62, 64, 65, 67])
        state.transcribed_notes = [
            TranscribedNote(pitch, float(i), float(i + 1))
            for i, pitch in enumerate([60, 62, 64, 65, 67])
        ]
        state.rhythm_pairs = [
            PairedEvent(i, 60 + i, float(i), float(i + 1), float(i), float(i + 1), "match")
            for i in range(4)
        ]
        state.rhythm_pairs.append(
            PairedEvent(4, 67, 4.0, 5.0, 4.0, 6.0, "match")
        )
        run_stage3(state)
        rhythm = [
            label for label in state.labels if label.type == "rhythm_error"
        ]
        self.assertEqual(len(rhythm), 1)
        self.assertEqual(rhythm[0].note_id, "note_0004")


if __name__ == "__main__":
    unittest.main()
