"""Regression coverage for preserving MuScriptor's full-recording context."""
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from datacreate.config import PipelineConfig
from datacreate.muscriptor_audio import transcription_source, context_signature
from datacreate import muscriptor_bridge as bridge

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/transcribe_muscriptor.py'
spec = importlib.util.spec_from_file_location('transcribe_muscriptor_context_test', SCRIPT)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def bundle(tmp_path, *, start=52083, length=60000, rate=10000):
    rng = np.random.default_rng(365)
    full = rng.integers(-18000, 18000, size=140000, dtype=np.int16)
    wavfile.write(tmp_path/'performance_audio_full.wav', rate, full)
    wavfile.write(tmp_path/'performance_audio.wav', rate, full[start:start+length])
    (tmp_path/'verified_score.musicxml').write_text('<score-partwise/>')
    # Simulate old metadata's rounded origin.
    (tmp_path/'metadata.json').write_text(json.dumps({'performance_trim':{
        'trim_start':round(start/rate,4),'trim_end':round((start+length)/rate,4)}}))
    return full


def test_verified_original_and_precise_origin(tmp_path):
    bundle(tmp_path, start=114845, length=20000, rate=22050)
    path, context=transcription_source(tmp_path)
    assert path.name=='performance_audio_full.wav'
    assert context['mode']=='verified_full_recording'
    assert context['source_start_sample']==114845
    assert context['crop_start']==114845/22050
    assert context['crop_end']==134845/22050
    assert context['max_slice_difference']==0


def test_stale_full_recording_cannot_replace_current_audio(tmp_path):
    bundle(tmp_path)
    wavfile.write(tmp_path/'performance_audio_full.wav',10000,np.zeros(140000,dtype=np.int16))
    path, context=transcription_source(tmp_path)
    assert path.name=='performance_audio.wav'
    assert context['reason']=='full_recording_does_not_match_trim'


def test_without_backup_keeps_current_audio(tmp_path):
    (tmp_path/'performance_audio.wav').write_bytes(b'audio')
    path, context=transcription_source(tmp_path)
    assert path.name=='performance_audio.wav'
    assert context['crop_start']==0 and context['crop_end'] is None


def test_projection_clips_once_keeps_real_reattacks_and_absolute_times():
    notes=[dict(index=i,pitch=p,start=a,end=b) for i,(p,a,b) in enumerate([
        (70,4.,5.), (70,5.,5.5), (70,5.5,5.8), (72,10.,10.8),
        (74,11.,12.), (76,12.,13.)])]
    rows,stats=adapter.project_notes(notes,5.2,11.4)
    assert [r['index'] for r in rows]==[1,2,3,4]
    assert rows[0]['start']==0
    assert rows[2]['start']==pytest.approx(4.8)  # No double chunk offset.
    assert rows[-1]['end']==pytest.approx(6.2)
    assert stats==dict(source_note_count=6,retained_note_count=4,outside_crop_count=2,clipped_note_count=2)
    assert notes[1]['start']==5.  # Input is not mutated.


def test_duplicate_note_end_is_rejected_not_silently_appended():
    n=dict(index=1,pitch=70,start=1.,end=2.)
    with pytest.raises(ValueError,match='Duplicate'):
        adapter.project_notes([n,n])


def test_cross_chunk_end_uses_start_event_identity(monkeypatch):
    class End:
        def __init__(self,start,end):
            self.start_event=start
            self.end_time=end
    module=ModuleType('muscriptor.events'); module.NoteEndEvent=End
    monkeypatch.setitem(__import__('sys').modules,'muscriptor.events',module)
    first=SimpleNamespace(index=5,pitch=70,start_time=4.9,instrument='clarinet')
    later=SimpleNamespace(index=6,pitch=72,start_time=5.1,instrument='clarinet')
    calls=[]
    def transcribe(audio,**kwargs):
        calls.append(kwargs)
        # End events need not arrive in onset order.
        return iter([End(later,5.3),End(first,5.4)])
    result=adapter.transcribe_file(SimpleNamespace(transcribe=transcribe),'float16','sha','input',2,crop_start=5.)
    assert [n['index'] for n in result['transcribed_notes']]==[5,6]
    assert [n['pitch'] for n in result['transcribed_notes']]==[72,74]
    assert calls==[dict(batch_size=1,prelude_forcing=True)]
    assert result['transcribed_notes'][0]['start']==0


def test_bridge_passes_full_audio_and_crop_and_rejects_changed_context(tmp_path,monkeypatch):
    bundle(tmp_path)
    python=tmp_path/'python.exe'; python.touch()
    config=PipelineConfig(paths={'muscriptor_python':str(python)})
    output=tmp_path/'notes.json'
    def run(command,**kwargs):
        assert Path(command[command.index('--audio')+1]).name=='performance_audio_full.wav'
        assert float(command[command.index('--crop-start')+1])==5.2083
        assert float(command[command.index('--crop-end')+1])==11.2083
        output.write_text(json.dumps({'provenance':{},'transcribed_notes':[]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bridge.subprocess,'run',run)
    bridge.transcribe(tmp_path,config,logging.getLogger('test'),output)
    provenance=json.loads(output.read_text())['provenance']
    assert provenance['audio_context']['mode']=='verified_full_recording'
    def changed(command,**kwargs):
        response=run(command,**kwargs)
        (tmp_path/'metadata.json').write_text('{}')
        return response
    monkeypatch.setattr(bridge.subprocess,'run',changed)
    with pytest.raises(RuntimeError,match='trim changed'):
        bridge.transcribe(tmp_path,config,logging.getLogger('test'),output)


def test_retrim_preserves_original_clock(tmp_path):
    from datacreate.sample_prep import apply_performance_trim
    bundle(tmp_path)
    config=PipelineConfig(audio={'sample_rate':10000},taxonomy=['wrong_note'])
    apply_performance_trim(tmp_path,1.,4.,config,logging.getLogger('test'))
    path,context=transcription_source(tmp_path)
    assert path.name=='performance_audio_full.wav'
    assert context['crop_start']==pytest.approx(6.2083)
    assert context['crop_end']==pytest.approx(9.2083)


def test_context_signature_invalidates_backup_and_trim_changes(tmp_path):
    bundle(tmp_path)
    original=context_signature(tmp_path)
    (tmp_path/'metadata.json').write_text('{}')
    assert context_signature(tmp_path)!=original
    after=context_signature(tmp_path)
    (tmp_path/'performance_audio_full.wav').write_bytes(b'changed')
    assert context_signature(tmp_path)!=after


def test_freshness_requires_new_context_revision(tmp_path):
    from alignmodel.joint.rhythm_v1 import REVISION as RHYTHM
    bundle(tmp_path)
    candidate=tmp_path/'candidate.json'; candidate.write_text('{}')
    config=PipelineConfig(paths={'note_alignment_candidate':str(candidate)})
    keys={'candidate_sha256':candidate, 'audio_sha256':tmp_path/'performance_audio.wav',
          'score_sha256':tmp_path/'verified_score.musicxml'}
    provenance={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in keys.items()}
    provenance.update(pipeline_revision=bridge.REVISION,transcriber='muscriptor',
        rhythm_revision=RHYTHM,transcription={'written_shift':bridge.written_shift(tmp_path,config),
                                             'audio_context':context_signature(tmp_path)})
    artifact=tmp_path/'note_alignment_v2.json'
    artifact.write_text(json.dumps({'provenance':provenance}))
    assert bridge.current_feedback(tmp_path,config)
    provenance['rhythm_revision']='aligned-rhythm-v1'
    artifact.write_text(json.dumps({'provenance':provenance}))
    assert not bridge.current_feedback(tmp_path,config)
    provenance['rhythm_revision']=RHYTHM
    provenance['pipeline_revision']='muscriptor-medium-align-v9-v1'
    artifact.write_text(json.dumps({'provenance':provenance}))
    assert not bridge.current_feedback(tmp_path,config)

