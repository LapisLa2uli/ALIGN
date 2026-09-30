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
    build_agent_label_document_from_note_alignment,
    detect_repetitions,
    relabel_sample_from_current_alignment,
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


def test_relabel_from_note_alignment_uses_current_mapping(tmp_path):
    sample = tmp_path / "sample"
    _write_sample(sample, [60, 62, 64, 65], [60, 61, 62, 67, 65])
    (sample / "note_alignment_v2.json").write_text(
        json.dumps(
            {
                "engine": "align-joint",
                "transcribed_notes": [
                    {"pitch": 60, "start": 0.0, "end": 0.4, "confidence": 0.9},
                    {"pitch": 61, "start": 0.5, "end": 0.9, "confidence": 0.8},
                    {"pitch": 62, "start": 1.0, "end": 1.4, "confidence": 0.9},
                    {"pitch": 67, "start": 1.5, "end": 1.9, "confidence": 0.7},
                    {"pitch": 65, "start": 2.0, "end": 2.4, "confidence": 0.9},
                ],
                "note_mapping": [0, None, 1, 2, 3],
                "repetitions": [],
            }
        ),
        encoding="utf-8",
    )
    result = relabel_sample_from_current_alignment(sample, maximum_per_type=10)
    document = build_agent_label_document_from_note_alignment(sample)
    types = [label["type"] for label in document["labels"]]
    assert result["source"] == "note_alignment_v2.json"
    assert result["label_count"] == len(document["labels"])
    assert "extra_note" in types
    assert "wrong_note" in types
    assert document["agent_labeling"]["method"] == "current_note_alignment_review_v1"
    assert (sample / "labels_agent.json").is_file()
    synced = json.loads((sample / "transcription_notes.json").read_text(encoding="utf-8"))
    assert len(synced["transcribed_notes"]) == 5


def test_relabel_skips_ignored_postprocessor_extras(tmp_path):
    sample = tmp_path / "sample"
    _write_sample(sample, [60, 62], [60, 70, 62])
    (sample / "note_alignment_v2.json").write_text(
        json.dumps(
            {
                "engine": "align-joint",
                "transcribed_notes": [
                    {"pitch": 60, "start": 0.0, "end": 0.4, "confidence": 0.9},
                    {
                        "pitch": 70,
                        "start": 0.45,
                        "end": 0.55,
                        "confidence": 0.3,
                        "ignored": True,
                        "ignored_reason": "joint_noise",
                    },
                    {"pitch": 62, "start": 0.6, "end": 1.0, "confidence": 0.9},
                ],
                "note_mapping": [0, None, 1],
                "repetitions": [],
            }
        ),
        encoding="utf-8",
    )
    document = build_agent_label_document_from_note_alignment(sample)
    types = [label["type"] for label in document["labels"]]
    assert "extra_note" not in types
    assert document["agent_labeling"]["transcribed_note_count"] == 2
