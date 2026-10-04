import numpy as np
import pytest
from alignmodel.transcription.same_pitch_v2 import repair_same_pitch, acoustic_evidence, SamePitchConfig

SR=22050
HOP=256/SR


def fixture(frequency=440,phase=0):
    time=np.arange(SR*2)/SR
    audio=.2*(np.sin(2*np.pi*frequency*time+phase)+.25*np.sin(2*np.pi*3*frequency*time+.5))
    frames=int(len(audio)/256)+1
    mel=np.tile(np.linspace(-.5,.5,192)[:,None],(1,frames))
    outputs={'onset':np.zeros(frames),'rearticulation':np.zeros(frames),'voiced':np.ones(frames)}
    rows=[[70,.2,.7,.8,0,-1,0.],[70,.7,1.2,.9,0,-1,0.],[70,1.2,1.6,.7,0,-1,0.]]
    return rows,outputs,mel,time,audio


@pytest.mark.parametrize('frequency',[150,440,1500])
@pytest.mark.parametrize('shift',[0,.0003,.0007])
@pytest.mark.parametrize('attenuation',[0.,.05])
def test_10ms_gap_at_different_phases_and_pitches(frequency,shift,attenuation):
    rows,outputs,mel,t,audio=fixture(frequency,.7)
    audio[(t>=.695+shift)&(t<.705+shift)]*=attenuation
    result,audit=repair_same_pitch(rows,outputs,mel,audio=audio)
    assert audit['decisions'][0]['reason']=='acoustic_gap'
    assert result[0][2]==.7
    assert audit['decisions'][1]['decision']=='merge'


def test_modulated_sustain_chain_merges_despite_old_variation_limits():
    rows,outputs,mel,t,audio=fixture()
    audio*=10**((6*np.sin(2*np.pi*5*t))/20)
    mel[160:]+= .15*np.sin(2*np.pi*5*np.arange(mel.shape[1])*HOP)
    outputs['onset'][:]=.85;outputs['voiced'][:]=.1
    result,audit=repair_same_pitch(rows,outputs,mel,audio=audio)
    assert audit['merged_boundaries']==2
    assert audit['source_groups']==[[0,1,2]]
    assert rows[0][2]==.7 and result[0][:4]==[70,.2,1.6,.9]


def test_short_context_now_assessed_and_gap_preserved():
    rows,outputs,mel,t,audio=fixture()
    rows=[[70,.65,.7,.9],[70,.7,.75,.9],[72,.75,.8,.9]]
    audio[(t>=.69)&(t<.70)]=0
    result,audit=repair_same_pitch(rows,outputs,mel,audio=audio)
    assert audit['decisions'][0]['reason']=='acoustic_gap' and len(result)==3


@pytest.mark.parametrize('duration',[.010,.020,.050])
def test_longer_silence(duration):
    rows,outputs,mel,t,audio=fixture()
    audio[(t>=.7-duration/2)&(t<.7+duration/2)]=0
    assert repair_same_pitch(rows,outputs,mel,audio=audio)[1]['decisions'][0]['reason']=='acoustic_gap'


def test_missing_audio_and_other_pitch_are_not_merged():
    rows,outputs,mel,_,audio=fixture()
    assert repair_same_pitch(rows,outputs,mel)[1]['merged_boundaries']==0
    rows[1][0]=71
    assert repair_same_pitch(rows,outputs,mel,audio=audio)[1]['pairs_checked']==0
    with pytest.raises(ValueError):acoustic_evidence(audio,mel.shape[1],config=SamePitchConfig(envelope_hop_sec=.01))


def test_stack_realigns_to_score_identity(tmp_path,monkeypatch):
    from music21 import stream,note
    from alignmodel.joint import stack_v9
    from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
    from dataclasses import asdict
    rows,outputs,mel,_,audio=fixture();rows[2][0]=72
    path=tmp_path/'score.musicxml'
    stream.Stream([note.Note(70),note.Note(72)]).write('musicxml',fp=path)
    monkeypatch.setattr(stack_v9,'decode_v7',lambda *a,**k:(rows,{}))
    candidate={'decoder':{},'aligner_costs':asdict(RobustDPCosts()),'aligner_v3':{},'gate':{'enabled':False}}
    index,alignment,events,deletions,info=stack_v9.align_outputs(outputs,path,candidate,mel=mel,audio=audio)
    assert [e.score_span for e in events]==[(0,1),(1,2)]
    doc=stack_v9.feedback(index,alignment,events,deletions,info)
    assert doc['schema_version']=='align-score-feedback-v9' and not doc['labels']
