import json
import logging
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from datacreate.config import PipelineConfig
from datacreate.transcriber_selection import selected_config, sample_config
from datacreate.web import app as web


def test_realign_choices_are_isolated_and_invalid_choices_rejected(tmp_path, monkeypatch):
    (tmp_path / '001').mkdir()
    config = PipelineConfig(paths={'samples_root': str(tmp_path)}, alignment={'model_version': 'stack-v9'})
    calls = []
    def align(sample, request_config, logger):
        calls.append(request_config.alignment['transcriber'])
        request_config.paths['work_dir'] = 'request-only'
        return {'model_version': calls[-1]}
    monkeypatch.setattr(web, 'reprocess_alignment', align)
    client = TestClient(web.create_app(config))
    for choice in ['muscriptor', 'align']:
        response = client.post('/api/samples/001/re-align', json={'transcriber': choice})
        assert response.status_code == 200
        assert response.json()['model_version'] == choice
    assert client.post('/api/samples/001/re-align', json={'transcriber': 'unknown'}).status_code == 422
    assert calls == ['muscriptor', 'align']
    assert config.alignment == {'model_version': 'stack-v9'}
    assert 'work_dir' not in config.paths


def test_relabel_and_bodyless_realign_keep_published_choice(tmp_path, monkeypatch):
    sample = tmp_path / '001'; sample.mkdir()
    (sample / 'note_alignment_v2.json').write_text(json.dumps({'provenance': {'transcriber': 'muscriptor'}}))
    config = PipelineConfig(paths={'samples_root': str(tmp_path)})
    assert sample_config(sample, config).alignment['transcriber'] == 'muscriptor'
    seen = []
    monkeypatch.setattr(web, 'reprocess_alignment', lambda s, c, l: seen.append(c.alignment['transcriber']) or {})
    from datacreate import align_bridge, transcription_labeling
    monkeypatch.setattr(align_bridge, 'ensure_current_model_feedback', lambda s, c, l: seen.append(c.alignment['transcriber']) or False)
    monkeypatch.setattr(transcription_labeling, 'relabel_sample_from_current_alignment', lambda *a, **k: {'label_count': 0})
    monkeypatch.setattr(web, 'validate_labels_file', lambda *a: [])
    client = TestClient(web.create_app(config))
    assert client.post('/api/samples/001/re-align').status_code == 200
    assert client.post('/api/samples/001/re-label').status_code == 200
    assert seen == ['muscriptor', 'muscriptor']


def test_refresh_after_alignment_invalidation_keeps_transcriber(tmp_path, monkeypatch):
    sample = tmp_path / '001'; sample.mkdir()
    (sample / 'transcription_notes.json').write_text(json.dumps({'provenance': {'transcriber': 'muscriptor'}}))
    config = PipelineConfig(paths={'samples_root': str(tmp_path)}, alignment={'transcriber': 'align'})
    seen = []
    monkeypatch.setattr(web, 'reprocess_alignment', lambda s, c, l: seen.append(c.alignment['transcriber']))
    monkeypatch.setattr(web, 'build_note_alignment', lambda *a: {'notes': []})
    client = TestClient(web.create_app(config))
    assert client.get('/api/samples/001/note-alignment').status_code == 200
    assert seen == ['muscriptor']
    # Published alignment takes precedence over stale transcription metadata.
    (sample / 'note_alignment_v2.json').write_text('{"provenance": {}}')
    assert sample_config(sample, config).alignment['transcriber'] == 'align'


@pytest.mark.parametrize('metadata,chromatic,expected', [
    ({}, -2, 2), ({}, 0, 0), ({'effective_audio_transpose': 0}, -2, 0),
    ({'audio_pitch_space': 'written'}, -2, 0), ({'effective_audio_transpose': 9}, 0, 9),
])
def test_muscriptor_bridge_uses_acoustic_pitch_metadata(tmp_path, monkeypatch, metadata, chromatic, expected):
    from datacreate import muscriptor_bridge as bridge
    sample = tmp_path / 'sample'; sample.mkdir()
    (sample / 'performance_audio.wav').write_bytes(b'audio')
    (sample / 'verified_score.musicxml').write_text(f'<score-partwise><transpose><chromatic>{chromatic}</chromatic></transpose></score-partwise>')
    (sample / 'metadata.json').write_text(json.dumps(metadata))
    python = tmp_path / 'python.exe'; python.touch()
    output = tmp_path / 'notes.json'
    config = PipelineConfig(paths={'muscriptor_python': str(python)})
    def run(command, **kwargs):
        assert command[command.index('--written-shift') + 1] == str(expected)
        assert 'PYTHONPATH' not in kwargs['env']
        output.write_text(json.dumps({'provenance': {}, 'transcribed_notes': []}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bridge.subprocess, 'run', run)
    assert bridge.transcribe(sample, config, logging.getLogger('test'), output) == output
    p = json.loads(output.read_text())['provenance']
    assert len(p['audio_sha256']) == 64 and len(p['score_sha256']) == 64


def test_missing_muscriptor_install_does_not_change_published_artifacts(tmp_path):
    from datacreate.muscriptor_bridge import run_alignment
    (tmp_path / 'performance_audio.wav').write_bytes(b'audio')
    (tmp_path / 'verified_score.musicxml').write_text('<score-partwise/>')
    artifact = tmp_path / 'note_alignment_v2.json'; artifact.write_text('{"old": true}')
    config = PipelineConfig(paths={'work_dir': str(tmp_path / 'work'), 'muscriptor_python': str(tmp_path / 'missing.exe')})
    with pytest.raises(FileNotFoundError, match='MuScriptor is not installed'):
        run_alignment(tmp_path, config, logging.getLogger('test'), False)
    assert artifact.read_text() == '{"old": true}'


def test_selection_config_does_not_mutate_nested_values():
    config = PipelineConfig(alignment={'transcriber': 'align', 'nested': {'x': 1}})
    selected = selected_config(config, 'muscriptor'); selected.alignment['nested']['x'] = 2
    assert config.alignment['nested']['x'] == 1
    assert config.alignment['transcriber'] == 'align'
    with pytest.raises(ValueError, match='Unknown transcriber'):
        selected_config(config, 'invalid')
