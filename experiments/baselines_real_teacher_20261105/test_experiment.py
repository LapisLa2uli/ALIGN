import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from supervision import targets,canonical,valid_window

def label(kind,indices,start,end):
    return {'type':kind,'score_event_indices':indices,'start_time':start,'end_time':end}

def gui(notes,labels,unassessed=()):
    return {'diagnostics':{'status':'ok'},'transcribed_notes':notes,'labels':labels,'unassessed_score_event_indices':list(unassessed)}

def note(pitch,start,relationship,span=None):
    return {'pitch':pitch,'start':start,'end':start+.2,'relationship':relationship,'score_span':span}

def test_wrong_note_is_heard_extra_and_expected_missing():
    data=gui([note(65,1,'substitute',[0,1])],[label('wrong_note',[0],1,1.2)])
    result=targets(data,[67],2)
    assert result['usable'] and result['classes']['extra'][0]['pitch']==63
    assert result['classes']['removed'][0]['pitch']==65
    assert not result['classes']['correct'] and result['converted_label_count']==1

def test_replay_is_extra_even_when_its_pitch_matches():
    data=gui([note(60,1,'match',[0,1]),note(60,2,'copy',[0,1])],[label('repetition',[0],2,2.2)])
    result=targets(data,[60],2)
    assert result['class_counts']=={'extra':1,'removed':0,'correct':1}

def test_withheld_and_unassessed_regions_do_not_become_clean_targets():
    data=gui([note(60,1,'match',[0,1]),note(61,2,'extra'),note(64,3,'match',[2,3])],[],[1])
    result=targets(data,[60,62,64],2)
    assert result['classes']['extra']==[]
    assert not valid_window(1.5,2.5,result['unknown'])
    assert not valid_window(2.5,3,result['unknown'])

def test_abstention_is_not_an_error_free_training_example():
    data=gui([],[]);data['diagnostics']['status']='alignment_uncertain'
    assert not targets(data,[],2)['usable']

def test_core_location_excludes_gui_context_padding():
    value=label('extra_note',[0,1,2,3,4],1,2)
    value['score_part']={'start_note_index':0,'end_note_index':4,'core_start_note_index':2,'core_end_note_index':2,'pad_notes':2}
    assert canonical([value])[0]['score_event_indices']==[2]

def test_unconvertible_teacher_error_rejects_training_example():
    data=gui([note(60,1,'match',[0,1])],[label('wrong_note',[0],3,4)])
    assert targets(data,[60],2)['reason']=='published_error_has_no_native_target'
