from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'baselines/common'), str(ROOT / 'align-model/src'),
               str(ROOT / 'baselines/scripts')]
from canonical_adapter import Note, adapt
from alignmodel.melody import match_note_wise_labels_detail


def test_substitution_and_deletion_use_reference_identity():
    converted = adapt([Note(0,.4,60,'correct'), Note(1,1.4,65,'extra'),
                       Note(1,1.4,62,'missing'), Note(2,2.4,64,'missing')], [60,62,64])
    assert converted['paired_wrong_notes'] == 1
    assert converted['labels'] == [
        {'type':'match|match|no_rhythm|ordinary','score_event_indices':[0],'copy_pass':0},
        {'type':'substitute|wrong_note|no_rhythm|ordinary','score_event_indices':[1],'copy_pass':0},
        {'type':'missed_note','score_event_indices':[2]}]


def test_repeat_source_and_pass_are_derived_from_prediction_history():
    notes = [Note(i,i+.4,pitch,'correct' if i<3 else 'extra')
             for i,pitch in enumerate([60,62,64]*3)]
    result = adapt(notes,[60,62,64])
    assert [x['score_event_indices'][0] for x in result['labels']] == [0,1,2]*3
    assert [x['copy_pass'] for x in result['labels']] == [0]*3+[1]*3+[2]*3
    assert all(x['type']=='copy|match|no_rhythm|copy' for x in result['labels'][3:])


def test_unlocated_events_remain_false_positives_without_invented_ids():
    result = adapt([Note(0,.5,60,'correct'),Note(1,1.5,71,'extra'),
                    Note(2,2.5,73,'unclassified')],[60])
    assert len(result['labels']) == 3 and result['unlocated_events'] == 2
    assert all('rendered_index' not in x for x in result['labels'])
    metric = match_note_wise_labels_detail([result['labels'][0]],result['labels'],score_event_count=1)
    assert metric['predicted']==3 and metric['credit']==1 and metric['f1']==.5


def test_reference_gaps_do_not_invent_missing_predictions():
    result = adapt([Note(0,.5,60,'correct')],[60,62,64])
    assert len(result['labels'])==1
    missing = adapt([Note(i,i+.5,60,'missing') for i in range(3)],[60])
    assert len(missing['labels'])==3 and missing['unlocated_events']==2


def test_combined_target_matches_ours_full_pipeline_metric(tmp_path):
    import music21
    from alignmodel.joint.index import ScoreEventIndex
    from alignmodel.joint.outputraw_metrics import FullPipelineMetricSample, evaluate_full_pipeline
    from evaluate_outputraw_canonical import gold_labels
    score = music21.stream.Score();part = music21.stream.Part()
    part.append(music21.note.Note(60,quarterLength=1));part.append(music21.note.Note(62,quarterLength=1))
    score.append(part);path=tmp_path/'score.musicxml';score.write('musicxml',fp=path)
    target = dict(kind='synth_note_lineage',
                  clean_notes=[dict(clean_index=i,deleted=False) for i in range(2)],
                  performed_notes=[dict(performed_index=i,copy_pass=1 if i==2 else 0,
                                        origin_relationship='match') for i in range(3)],
                  rendered_notes=[dict(rendered_index=i,performed_indices=[i],clean_indices=[0 if i==2 else i],
                                       pitch_midi_written=60 if i!=1 else 62,start_sec=float(i),end_sec=i+.5,
                                       relationship='copy' if i==2 else 'match') for i in range(3)],
                  deleted_clean_notes=[],valid_labels=[dict(type='rhythm_error',start_time=1,end_time=1.5)])
    labels,index=gold_labels(path,target)
    ours=evaluate_full_pipeline([FullPipelineMetricSample(predicted=index.rendered_events,
             target=index.rendered_events,predicted_layer2=('match',)*3,target_layer2=('match',)*3,
             predicted_rhythm=(False,)*3,target_rhythm=(False,True,False),score_event_count=2)],bootstrap_replicates=5)
    predictions=[dict(x) for x in labels]
    predictions[1]['type']='match|match|no_rhythm|ordinary'
    direct=match_note_wise_labels_detail(labels,predictions,score_event_count=2)
    assert direct['credit']==2.5
    assert direct['f1']==ours['combined']['f1']
