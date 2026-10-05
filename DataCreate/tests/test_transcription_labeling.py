from __future__ import annotations

import json
import wave
from pathlib import Path
from copy import deepcopy
import pytest

from music21 import note, stream

from datacreate.melody import ScoreSoundingNote
from datacreate.transcription_labeling import (
    TranscribedNote,
    align_pitch_sequences,
    build_agent_label_document,
    build_agent_label_document_from_note_alignment,
    detect_repetitions,
    relabel_sample_from_current_alignment,
    build_agent_label_document_from_alignment_payload,
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


def _v9_feedback_payload():
    repeat = {
        'id':'v9_0000','type':'repetition','source':'agent',
        'score_event_indices':[0,1,2,3],
        'note_ids':[f'note_{i:04d}' for i in range(4)],
        'core_note_ids':[f'note_{i:04d}' for i in range(4)],
        'score_part':{'start_note_index':0,'end_note_index':3,'pad_notes':0,
                      'core_start_note_index':0,'core_end_note_index':3},
        'start_time':2.,'end_time':3.9,'extra_copies':1,
        'repeats_label_range':{'start_time':0.,'end_time':1.9},
    }
    return {
        'engine':'align-joint',
        'summary':{'backend':'ALIGN v9 (experimental)','status':'ok'},
        'diagnostics':{'status':'ok','same_pitch_repair':{'schema_version':'same-pitch-repair-v2'}},
        'provenance':{'candidate_sha256':'test-candidate'},
        'labels':[repeat],'repetitions':[repeat],
        # All replay notes are mapped, so the legacy insertion detector misses them.
        'note_mapping':[0,1,2,3,0,1,2,3,4],
        'transcribed_notes':[{'pitch':p,'start':i*.5,'end':i*.5+.4,'confidence':.9}
                             for i,p in enumerate([60,62,64,65,60,62,64,65,67])],
    }


@pytest.mark.parametrize('explicit_contract',[False,True])
def test_relabel_preserves_mapped_v9_repetition_and_provenance(tmp_path,explicit_contract):
    sample=tmp_path/'sample'
    _write_sample(sample,[60,62,64,65,67],[60,62,64,65,60,62,64,65,67])
    payload=_v9_feedback_payload()
    if explicit_contract:
        payload['label_generation']={'schema_version':'datacreate-model-feedback-v1','method':'align_stack_v9'}
    (sample/'note_alignment_v2.json').write_text(json.dumps(payload))
    (sample/'labels.json').write_text('{"labels":[],"annotator_id":"human"}')
    protected={n:(sample/n).read_bytes() for n in ('labels.json','transcription_notes.json','note_alignment_v2.json')}
    result=relabel_sample_from_current_alignment(sample,maximum_per_type=0)
    saved=json.loads((sample/'labels_agent.json').read_text())
    assert saved['labels']==payload['labels']
    assert result['counts_by_type']=={'repetition':1}
    assert result['method']=='align_stack_v9' and result['dismissed_types']==[]
    assert saved['agent_labeling']['candidate_sha256']=='test-candidate'
    for name,content in protected.items():assert (sample/name).read_bytes()==content
    relabel_sample_from_current_alignment(sample)
    assert json.loads((sample/'labels_agent.json').read_text())==saved


@pytest.mark.parametrize('status',['ok','alignment_uncertain'])
def test_v9_empty_or_uncertain_feedback_never_falls_back_to_legacy_errors(tmp_path,status):
    payload=_v9_feedback_payload()
    payload['diagnostics']['status']=status
    if status=='ok':payload['labels']=[]
    original=deepcopy(payload)
    doc=build_agent_label_document_from_alignment_payload(tmp_path,payload)
    assert not doc['labels'] and not doc['agent_labeling']['kept_counts_by_type']
    assert doc['agent_labeling']['status']==status
    assert payload==original


def test_multiple_repeat_regions_keep_exact_identity_and_copy_count(tmp_path):
    payload=_v9_feedback_payload()
    second=deepcopy(payload['labels'][0])
    second.update(id='v9_0001',score_event_indices=[8,9],extra_copies=2,start_time=9.,end_time=12.)
    payload['labels'].append(second)
    doc=build_agent_label_document_from_alignment_payload(tmp_path,payload,maximum_per_type=1)
    assert doc['labels']==payload['labels']
    assert doc['agent_labeling']['kept_counts_by_type']=={'repetition':2}


def test_relabel_http_route_serves_v9_repetition_with_reference_note_ids(tmp_path):
    from fastapi.testclient import TestClient

    from datacreate.config import PipelineConfig
    from datacreate.web.app import create_app

    sample = tmp_path / "095"
    _write_sample(sample, [60, 62, 64, 65, 67], [60, 62, 64, 65, 60, 62, 64, 65, 67])
    payload = _v9_feedback_payload()
    (sample / "note_alignment_v2.json").write_text(json.dumps(payload), encoding="utf-8")
    client = TestClient(create_app(PipelineConfig(
        paths={"samples_root": str(tmp_path)}, taxonomy=["repetition"],
    )))

    response = client.post("/api/samples/095/re-label")
    assert response.status_code == 200, response.text
    assert response.json()["counts_by_type"] == {"repetition": 1}
    response = client.get("/api/samples/095?label_source=agent")
    assert response.status_code == 200, response.text
    assert response.json()["labels"] == payload["labels"]


def test_v9_relabel_upgrades_legacy_and_refreshes_changed_inputs(tmp_path, monkeypatch):
    import hashlib
    from fastapi.testclient import TestClient
    from datacreate import align_bridge
    from datacreate.config import PipelineConfig
    from datacreate.web.app import create_app

    sample = tmp_path / "095"
    _write_sample(sample, [60, 62, 64, 65, 67], [60, 62, 64, 65, 60, 62, 64, 65, 67])
    candidate = tmp_path / "candidate.json"
    candidate.write_text("candidate-v9")
    alignment = sample / "note_alignment_v2.json"
    alignment.write_text(json.dumps({"engine": "align-joint", "labels": []}))
    human = sample / "labels.json"
    human.write_text('{"labels": [], "annotator_id": "human"}')
    protected = human.read_bytes()
    config = PipelineConfig(
        paths={"samples_root": str(tmp_path), "note_alignment_candidate": str(candidate)},
        alignment={"model_version": "stack-v9"}, taxonomy=["repetition"],
    )
    calls = []

    def fresh_run(*args, **kwargs):
        calls.append(True)
        payload = _v9_feedback_payload()
        payload["provenance"] = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in {
                "candidate_sha256": candidate,
                "audio_sha256": sample / "performance_audio.wav",
                "score_sha256": sample / "verified_score.musicxml",
            }.items()
        }
        payload["provenance"]["pipeline_revision"] = "v9-passage-v1"
        alignment.write_text(json.dumps(payload))

    monkeypatch.setattr(align_bridge, "run_preferred_alignment", fresh_run)
    client = TestClient(create_app(config))
    response = client.post("/api/samples/095/re-label")
    assert response.status_code == 200, response.text
    assert response.json()["regenerated_alignment"] is True
    assert response.json()["method"] == "align_stack_v9"
    assert response.json()["counts_by_type"] == {"repetition": 1}
    assert len(calls) == 1
    response = client.post("/api/samples/095/re-label")
    assert response.status_code == 200
    assert response.json()["regenerated_alignment"] is False
    assert len(calls) == 1
    for path in [sample / "performance_audio.wav", sample / "verified_score.musicxml", candidate]:
        path.write_bytes(path.read_bytes() + b"\n")
        response = client.post("/api/samples/095/re-label")
        assert response.status_code == 200, response.text
        assert response.json()["regenerated_alignment"] is True
    assert len(calls) == 4
    assert human.read_bytes() == protected

    # A failed upgrade must not overwrite labels with legacy output.
    saved = (sample / "labels_agent.json").read_bytes()
    alignment.write_text('{"engine":"align-joint","labels":[]}')
    def failing_run(*args, **kwargs):
        raise RuntimeError("v9 checkpoint unavailable")
    monkeypatch.setattr(align_bridge, "run_preferred_alignment", failing_run)
    response = client.post("/api/samples/095/re-label")
    assert response.status_code == 400
    assert "v9 checkpoint unavailable" in response.text
    assert (sample / "labels_agent.json").read_bytes() == saved
