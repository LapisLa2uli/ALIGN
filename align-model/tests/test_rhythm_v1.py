from dataclasses import replace
from types import SimpleNamespace
import pytest
from alignmodel.joint.index import ScoreEvent, JointEvent
from alignmodel.joint.rhythm_v1 import RhythmConfig, classify_rhythm, REVISION
from alignmodel.joint.stack_v9_passage import feedback

POLICY = RhythmConfig(method='median', window=65, log_threshold=.20)

def sequence(count=48, scale=.6):
    score = [ScoreEvent(i,70,i,i+.75,(i,),i//4+1) for i in range(count)]
    events = [JointEvent(70,2+i*scale,2+(i+.75)*scale,(i,i+1)) for i in range(count)]
    return score,events

def ids(labels):
    return {i for row in labels for i in row['score_event_indices']}

@pytest.mark.parametrize('scale',[.25,.6,1.2])
def test_global_tempo_is_not_an_error(scale):
    score,events=sequence(scale=scale)
    for measured in (False,True):
        labels,info=classify_rhythm(score,events,measured_offsets=measured,config=POLICY)
        assert not labels
        assert info['status']=='ok'

@pytest.mark.parametrize('method',['ewma','median'])
@pytest.mark.parametrize('measured',[False,True])
@pytest.mark.parametrize('release',['short','legato','final','mixed'])
def test_articulation_does_not_create_rhythm_errors(method,measured,release):
    score,events=sequence()
    if release=='short':
        events[20]=replace(events[20],end=events[20].start+.1)
    elif release=='legato':
        events[20]=replace(events[20],end=events[21].start+.2)
    elif release=='final':
        events[-1]=replace(events[-1],end=events[-1].start+.05)
    else:
        events=[replace(e,end=e.start+(.08 if i%2 else .8)) for i,e in enumerate(events)]
    labels,info=classify_rhythm(score,events,measured_offsets=measured,
                              config=replace(POLICY,method=method))
    assert not labels
    assert info['timing_source']=='inter_onset_intervals'
    assert info['release_policy']=='not_assessed'
    assert len(score)-1 not in info['assessed_score_event_indices']


def shift_following(events, boundary, delta):
    return [replace(e,start=e.start+delta,end=e.end+delta) if i>boundary else e
            for i,e in enumerate(events)]


@pytest.mark.parametrize('delta',[-.25,.25])
@pytest.mark.parametrize('measured',[False,True])
def test_real_attack_spacing_errors_survive_short_or_overlapping_releases(delta,measured):
    score,events=sequence()
    events=shift_following(events,20,delta)
    events=[replace(e,end=e.start+(.08 if i%2 else 1.)) for i,e in enumerate(events)]
    labels,_=classify_rhythm(score,events,measured_offsets=measured,config=POLICY)
    assert ids(labels)=={20}
    assert labels[0]['deviation_ms']==pytest.approx(delta*1000)
    assert labels[0]['rhythm_subtype']==('long' if delta>0 else 'short')
    assert labels[0]['end_time']==events[21].start


def test_local_tempo_region_and_return_do_not_cascade():
    score,events=sequence()
    for i in range(18,26):
        events=shift_following(events,i,.25)
    labels,_=classify_rhythm(score,events,measured_offsets=True,config=POLICY)
    assert ids(labels)==set(range(18,26))
    # A sustained change is absorbed by EWMA, motivating the alternate method.
    ewma,_=classify_rhythm(score,events,measured_offsets=True)
    assert ids(ewma)!=set(range(18,26))


def test_ioi_skips_extra_and_missing_boundaries():
    score,events=sequence()
    events.pop(20)
    events.insert(10,JointEvent(75,events[9].end+.02,events[9].end+.05,None,'extra'))
    labels,info=classify_rhythm(score,events,config=POLICY)
    assert not labels
    assert 9 not in info['assessed_score_event_indices']
    assert 19 not in info['assessed_score_event_indices']

def test_copies_merged_and_duplicate_mappings_are_unassessed():
    score,events=sequence()
    events[10]=replace(events[10],relationship='copy',copy_pass=1)
    events[20]=replace(events[20],score_span=(20,22))
    events[30]=replace(events[30],score_span=(29,30))
    labels,info=classify_rhythm(score,events,measured_offsets=True,config=POLICY)
    assert not labels
    assert not {10,20,21,29,30} & set(info['assessed_score_event_indices'])

def test_insufficient_context():
    score,events=sequence(3)
    assert classify_rhythm(score,events)[1]['status']=='insufficient_context'

def test_uncertain_alignment_withholds_rhythm():
    score,events=sequence()
    result=feedback(SimpleNamespace(events=score),SimpleNamespace(events=events),
        (),(),{'status':'alignment_uncertain'})
    assert not result['labels']
    assert result['diagnostics']['rhythm']['status']=='alignment_uncertain'

def test_wrapper_ignores_release_only_changes_with_measured_offsets():
    score,events=sequence()
    events[20]=replace(events[20],end=events[20].start+.08)
    events[-1]=replace(events[-1],end=events[-1].start+.05)
    alignment=SimpleNamespace(events=events,missed=(),restart_regions=())
    result=feedback(SimpleNamespace(events=score),alignment,events,(),
        {'status':'ok'},measured_offsets=True)
    assert not result['labels']
    assert result['diagnostics']['rhythm']['release_policy']=='not_assessed'


def test_wrapper_uses_raw_events_not_inferred_matches():
    score,events=sequence()
    events=shift_following(events,20,-.25)
    alignment=SimpleNamespace(events=events,missed=(),restart_regions=())
    result=feedback(SimpleNamespace(events=score),alignment,events,(),
        {'status':'ok'},measured_offsets=True)
    rhythm=[l for l in result['labels'] if l['type']=='rhythm_error']
    assert ids(rhythm)=={20}
    assert result['schema_version']=='align-score-feedback-v9'



def test_rhythm_survives_gui_label_validation(tmp_path):
    from music21 import stream, note
    from alignmodel.joint.index import ScoreEventIndex
    from alignmodel.joint.datacreate_v9 import gui_documents
    from datacreate.models import LabelsDocument
    score, events = sequence()
    xml = stream.Stream()
    for _ in score:
        xml.append(note.Note(70,quarterLength=.75))
        xml.append(note.Rest(quarterLength=.25))
    xml.write('musicxml',fp=tmp_path/'verified_score.musicxml')
    index=ScoreEventIndex.from_musicxml(tmp_path/'verified_score.musicxml')
    events=shift_following(events,20,-.25)
    events=[replace(e,rendered_index=i) for i,e in enumerate(events)]
    notes=[SimpleNamespace(pitch=e.pitch,start=e.start,end=e.end,confidence=1.,optional=False) for e in events]
    alignment=SimpleNamespace(events=events,notes=notes,kept_note_indices=tuple(range(len(events))),
        match_fraction=1.,missed=(),restart_regions=())
    info={'status':'ok','same_pitch_repair':{'source_groups':[[i] for i in range(len(events))],
        'merged_boundaries':0,'pairs_checked':0}}
    result=feedback(index,alignment,events,(),info,measured_offsets=True)
    gui,document=gui_documents(tmp_path,index,alignment,result,duration=40,provenance={})
    LabelsDocument.model_validate(document)
    assert ids(document['labels'])=={20}
    assert document['labels'][0]['type']=='rhythm_error'
    assert 'next note attack' in document['labels'][0]['comment']
    assert gui['diagnostics']['rhythm']['revision']==REVISION

