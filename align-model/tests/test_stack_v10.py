from copy import deepcopy
from dataclasses import asdict, replace
import numpy as np
import pytest

from alignmodel.transcription.same_pitch_v3 import repair_same_pitch, FEATURE_NAMES
from alignmodel.joint import stack_v10
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts


def model(vote):
    return {'schema_version':'test', 'feature_names':list(FEATURE_NAMES), 'trees':[
        {'left':[-1], 'right':[-1], 'feature':[-2], 'threshold':[-2.], 'repeat_fraction':[vote]}]}


def fixture():
    sr=22050; t=np.arange(sr*2)/sr
    audio=.2*np.sin(2*np.pi*440*t)
    frames=len(audio)//256+1
    mel=np.tile(np.linspace(-.5,.5,192)[:,None],(1,frames))
    outputs={'onset':np.zeros(frames),'rearticulation':np.zeros(frames),'voiced':np.ones(frames)}
    rows=[[70,.2,.7,.99,0,-1,0.],[70,.7,1.2,.99,0,-1,0.],[72,1.2,1.6,.99,0,-1,0.]]
    return rows,outputs,mel,audio,t


def test_single_split_without_score_support_preserves_v9():
    rows,outputs,mel,audio,_=fixture()
    result,audit=repair_same_pitch(rows,outputs,mel,audio=audio,model=model(1.))
    assert len(result)==2 and audit['source_groups']==[[0,1],[2]]
    assert audit['decisions'][0]['reason']=='v9_preserved_without_score_support'


def test_score_and_acoustic_repeat_evidence_restore_boundary():
    rows,outputs,mel,audio,_=fixture()
    saved=deepcopy(rows)
    result,audit=repair_same_pitch(rows,outputs,mel,audio=audio,model=model(1.),supported_boundaries={1})
    assert result==rows==saved
    assert audit['decisions'][0]['reason']=='score_supported_rearticulation'
    assert repair_same_pitch(rows,outputs,mel,audio=audio,model=model(0.),supported_boundaries={1})[1]['merged_boundaries']==1


@pytest.mark.parametrize('offset',[0.,.0003,.0007])
def test_10ms_gap_is_never_overridden(offset):
    rows,outputs,mel,audio,t=fixture()
    audio[(t>=.695+offset)&(t<.705+offset)]=0
    result,audit=repair_same_pitch(rows,outputs,mel,audio=audio,model=model(0.),supported_boundaries={1})
    assert result==rows
    assert audit['decisions'][0]['reason']=='acoustic_gap'


def test_fast_different_pitch_candidates_are_byte_for_byte_preserved():
    rows,outputs,mel,audio,_=fixture()
    rows=[[60+i%12,.2+i*.025,.225+i*.025,.95,0,-1,0.] for i in range(50)]
    result,audit=repair_same_pitch(rows,outputs,mel,audio=audio,model=model(1.),supported_boundaries=set(range(50)))
    assert result==rows and audit['pairs_checked']==0


def run_stack(tmp_path,monkeypatch,pitches,*,corrupt=False):
    from music21 import stream,note
    from alignmodel.joint import stack_v9_passage
    rows,outputs,mel,audio,_=fixture()
    path=tmp_path/'score.musicxml'
    stream.Stream([note.Note(p,quarterLength=1) for p in pitches]).write('musicxml',fp=path)
    monkeypatch.setattr(stack_v10,'decode_v7',lambda *a,**k:(deepcopy(rows),{}))
    monkeypatch.setattr(stack_v9_passage,'decode_v7',lambda *a,**k:(deepcopy(rows),{}))
    if corrupt:
        original=stack_v10.align_notes
        def altered(*a,**kw):
            alignment,events,deletions,info=original(*a,**kw)
            changed=(*alignment.events[:-1],replace(alignment.events[-1],pitch=80))
            return replace(alignment,events=changed),events,deletions,info
        monkeypatch.setattr(stack_v10,'align_notes',altered)
    c={'decoder':{},'aligner_costs':asdict(RobustDPCosts()),'aligner_v3':{},'gate':{'enabled':False}}
    return stack_v10.align_outputs(outputs,path,c,mel=mel,audio=audio,boundary_model=model(1.))


def test_stack_restores_reference_note_identity(tmp_path,monkeypatch):
    result=run_stack(tmp_path,monkeypatch,[70,70,72])
    assert [e.score_span for e in result[2]]==[(0,1),(1,2),(2,3)]
    assert result[-1]['boundary_refinement']['accepted_restorations']==[1]
    assert stack_v10.feedback(*result)['schema_version']=='align-score-feedback-v10'


def test_stack_keeps_sustained_single_note_repair(tmp_path,monkeypatch):
    result=run_stack(tmp_path,monkeypatch,[70,72])
    assert [e.score_span for e in result[2]]==[(0,1),(1,2)]
    assert not result[-1]['boundary_refinement']['proposed_restorations']


def test_stack_rolls_back_if_an_unrelated_note_changes(tmp_path,monkeypatch):
    result=run_stack(tmp_path,monkeypatch,[70,70,72],corrupt=True)
    assert not result[-1]['boundary_refinement']['accepted_restorations']
    assert result[-1]['boundary_refinement']['rejected_reason']
    assert [e.score_span for e in result[2]]==[(0,2),(2,3)]


def test_v10_ui_contract_is_preserved_by_relabel(tmp_path):
    import json,hashlib
    from datacreate.align_bridge import current_v9_feedback
    from datacreate.config import PipelineConfig
    from datacreate.transcription_labeling import _has_model_feedback
    sample=tmp_path/'sample';sample.mkdir()
    candidate=tmp_path/'candidate.json';candidate.write_text('{}')
    paths={'candidate_sha256':candidate,'audio_sha256':sample/'performance_audio.wav','score_sha256':sample/'verified_score.musicxml'}
    for p in list(paths.values())[1:]:p.write_bytes(b'fixture')
    provenance={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    provenance['pipeline_revision']='v10-boundary-v1'
    payload={'label_generation':{'schema_version':'datacreate-model-feedback-v1','method':'align_stack_v10'},
             'labels':[],'provenance':provenance}
    (sample/'note_alignment_v2.json').write_text(json.dumps(payload))
    c=PipelineConfig(paths={'note_alignment_candidate':str(candidate)},alignment={'model_version':'stack-v10'})
    assert _has_model_feedback(payload) and current_v9_feedback(sample,c)
    c.alignment['model_version']='stack-v9'
    assert not current_v9_feedback(sample,c)
