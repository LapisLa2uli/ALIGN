from __future__ import annotations

import json
import wave
from pathlib import Path

from music21 import note, stream

from datacreate.melody import ScoreSoundingNote
from datacreate.transcription_labeling import (
    TranscribedNote,
    align_pitch_sequences,
    build_agent_label_document,
    detect_repetitions,
)


def _score(pitches: list[int]) -> list[ScoreSoundingNote]:
    return [
        ScoreSoundingNote(
            index=index,
            pitch=pitch,
            start=index * 0.5,
            end=(index + 1) * 0.5,
            ql_start=float(index),
            ql_end=float(index + 1),
            measure=1 + index // 4,
            note_id=f"note_{index:04d}",
        )
        for index, pitch in enumerate(pitches)
    ]


def _heard(pitches: list[int]) -> list[TranscribedNote]:
    return [
        TranscribedNote(
            pitch=pitch,
            start=index * 0.5,
            end=(index + 1) * 0.5,
            confidence=0.9,
        )
        for index, pitch in enumerate(pitches)
    ]


def test_independent_alignment_finds_insert_delete_and_substitute():
    score = _score([60, 62, 64, 65, 67])
    heard = _heard([60, 61, 62, 66, 67])
    operations = align_pitch_sequences(score, heard)
    kinds = [operation.kind for operation in operations]
    assert kinds[0] == "match"
    assert kinds[-1] == "match"
    assert "insert" in kinds
    assert "delete" in kinds
    assert "substitute" in kinds


def test_insertion_run_is_recognized_as_repetition():
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    score = _score(pitches)
    heard = _heard(pitches[:4] + pitches[:4] + pitches[4:])
    operations = align_pitch_sequences(score, heard)
    repeats = detect_repetitions(score, heard, operations)
    assert len(repeats) == 1
    assert repeats[0].score_start == 0
    assert repeats[0].score_end == 4
    assert repeats[0].repeat_start > repeats[0].source_start


def _write_sample(sample: Path, score_pitches: list[int], heard: list[int]) -> None:
    sample.mkdir()
    score = stream.Score()
    part = stream.Part()
    for pitch in score_pitches:
        part.append(note.Note(pitch, quarterLength=1.0))
    score.append(part)
    score.write("musicxml", fp=str(sample / "verified_score.musicxml"))
    with wave.open(str(sample / "performance_audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\0\0" * 22050 * max(1, len(heard)))
    (sample / "transcription_notes.json").write_text(
        json.dumps(
            {
                "transcribed_notes": [
                    {
                        "pitch": pitch,
                        "start": index * 0.5,
                        "end": index * 0.5 + 0.4,
                        "confidence": 0.9,
                    }
                    for index, pitch in enumerate(heard)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_more_than_ten_labels_of_one_type_are_all_dismissed(tmp_path):
    sample = tmp_path / "sample"
    score_pitches = [60, 62] * 8
    _write_sample(sample, score_pitches, [])
    document = build_agent_label_document(sample, maximum_per_type=10)
    assert document["labels"] == []
    assert document["agent_labeling"]["raw_counts_by_type"]["missed_note"] == 16
    assert document["agent_labeling"]["dismissed_types"] == ["missed_note"]
