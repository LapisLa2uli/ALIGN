from types import SimpleNamespace
import pytest
from music21 import stream,note
from alignmodel.joint.index import ScoreEventIndex,JointEvent
from alignmodel.joint.datacreate_v9 import gui_documents
from datacreate.note_alignment import _normalize_transcribed_notes, build_note_alignment


def test_gui_keeps_multinote_identity_dropped_candidates_and_uncertain_state(tmp_path):
    path=tmp_path/'verified_score.musicxml'
    stream.Stream([note.Note(70),note.Note(70),note.Note(72)]).write('musicxml',fp=path)
    index=ScoreEventIndex.from_musicxml(path)
    rows=[SimpleNamespace(pitch=70,start=.2,end=.8,confidence=.9,optional=False),
          SimpleNamespace(pitch=75,start=.8,end=.82,confidence=.1,optional=True),
          SimpleNamespace(pitch=72,start=.82,end=1.,confidence=.9,optional=False)]
    alignment=SimpleNamespace(notes=rows,kept_note_indices=(0,2),match_fraction=.4,
        events=[JointEvent(70,.2,.8,(0,2),rendered_index=0),JointEvent(72,.82,1.,(2,3),rendered_index=1)])
    feedback={'status':'alignment_uncertain','labels':[],'unassessed_score_event_indices':[0,1,2],
        'diagnostics':{'same_pitch_repair':{'source_groups':[[0,1],[2],[3]],'merged_boundaries':1,'pairs_checked':1}}}
    gui,labels=gui_documents(tmp_path,index,alignment,feedback,duration=1.2,provenance={})
    assert gui['label_generation']=={
        'schema_version':'datacreate-model-feedback-v1',
        'method':'align_stack_v9','annotator_id':'align_stack_v9_review'}
    assert gui['note_mapping']==[0,None,2]
    normalized=_normalize_transcribed_notes(gui)
    assert normalized[0]['score_event_indices']==[0,1]
    assert normalized[1]['ignored'] and normalized[1]['score_event_indices']==[]
    assert [e['score_index'] for e in gui['events']]==[0,1,2]
    assert gui['summary']['status']=='alignment_uncertain' and not labels['labels']
    import json
    (tmp_path/'note_alignment_v2.json').write_text(json.dumps(gui))
    assert build_note_alignment(tmp_path)['unassessed_score_event_indices']==[0,1,2]


def test_gui_refuses_mismatched_score_identity(tmp_path):
    path=tmp_path/'verified_score.musicxml'
    stream.Stream([note.Note(70)]).write('musicxml',fp=path)
    with pytest.raises(ValueError,match='index mismatch'):
        gui_documents(tmp_path,SimpleNamespace(events=[]),None,None,duration=1.,provenance={})
