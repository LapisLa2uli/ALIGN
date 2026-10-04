import numpy as np
import torch
from dataclasses import replace
from alignmodel.transcription.transition_v7 import decode_v7,consistency_loss
from alignmodel.joint.index import ScoreEvent
from alignmodel.joint.robust_dp_aligner_v2 import TranscribedNote
from alignmodel.joint.restarts_v4 import local_hypotheses,schedule_restarts
from alignmodel.joint.robust_dp_aligner_v4 import align_v4,AlignerV4Config,gate_v4
from alignmodel.joint.robust_dp_aligner_v3 import GateConfig
from alignmodel.joint.stack_v7 import feedback


def outputs(attack=0):
    ctc=np.full((24,50),.0001,np.float32); ctc[:,0]=.99
    for f,p in [(2,60),(10,62),(12,64),(20,65)]: ctc[f,0]=.001; ctc[f,p-52+1]=.995
    return {'ctc':ctc,'onset':np.full(24,attack,np.float32),'rearticulation':np.zeros(24,np.float32)}

def test_transition_optional_but_genuine_attacked_short_note_retained():
    rows,info=decode_v7(outputs())
    assert next(r for r in rows if r[0]==62)[4]==1
    assert info['transition_candidates']==1
    rows,_=decode_v7(outputs(1))
    assert next(r for r in rows if r[0]==62)[4]==0

def test_consistency_gradient_and_padding():
    logits=torch.randn(2,5,4,requires_grad=True); teacher=torch.full((2,5,4),-8.); teacher[:,:,0]=8
    valid=torch.ones(2,5,dtype=torch.bool); valid[:,3:]=False
    loss=consistency_loss(logits,teacher,valid); loss.backward()
    assert torch.isfinite(loss) and logits.grad[:,:3].abs().sum()>0
    assert logits.grad[:,3:].abs().sum()==0

def test_two_local_restarts_have_correct_score_identity():
    score=[ScoreEvent(i,p,float(i),float(i+1),(i,),1) for i,p in enumerate([60,62,64,65,67,69,71,72])]
    expected=[60,62,64,62,64,65,67,69,71,69,71,72]
    notes=[TranscribedNote(p,i*.2,(i+1)*.2) for i,p in enumerate(expected)]
    proposals=local_hypotheses(score,notes)
    h=next(h for h in proposals if h.restart_regions==((1,3,1),(5,7,1)))
    assert [score[i].pitch for i,_ in h.units]==expected
    assert list(h.times)==sorted(h.times)

def test_aligner_local_restarts_and_optional_short_score_note(tmp_path):
    from music21 import stream,note
    pitches=[60,62,64,65,67,69,71,72]
    s=stream.Stream([note.Note(p,quarterLength=1) for p in pitches]); path=tmp_path/'score.musicxml'; s.write('musicxml',fp=path)
    from alignmodel.joint.index import ScoreEventIndex
    index=ScoreEventIndex.from_musicxml(path)
    played=[60,62,64,62,64,65,67,69,71,69,71,72]
    rows=[[p,i*.2,(i+1)*.2,1.,int(i==1),-1,0.] for i,p in enumerate(played)]
    result=align_v4(rows,index.events,path,AlignerV4Config(timing_weight=.2,artifact_ioi=0))
    assert len(result.restart_regions)==2
    assert not result.deletions
    assert any(e.score_span==(1,2) for e in result.events)
    bad=replace(result,match_fraction=.1)
    events,deletions,info=gate_v4(bad,index.events,{},GateConfig(enabled=False))
    doc=feedback(index,bad,events,deletions,info)
    assert doc['status']=='alignment_uncertain' and not doc['labels']
    assert len(doc['unassessed_score_event_indices'])==8
    events,deletions,info=gate_v4(result,index.events,{},GateConfig(enabled=False))
    doc=feedback(index,result,events,deletions,info)
    assert len([l for l in doc['labels'] if l['type']=='repetition'])==2
    assert all(l['start_time']>0 for l in doc['labels'])
