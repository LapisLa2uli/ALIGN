import json
from copy import deepcopy

from datacreate.feedback_visibility import feedback_review, score_only_feedback
from datacreate.note_alignment import _normalize_transcribed_notes, build_note_alignment


def payload():
    return {'engine':'align-joint','label_generation':{
        'schema_version':'datacreate-model-feedback-v1','method':'align_stack_v10'},
        'labels':[], 'events':[], 'note_mapping':[0],
        'transcribed_notes':[{'pitch':70,'alignment_pitch':72,'relationship':'match',
            'score_span':[0,1],'start':.1,'end':.5,'confidence':.8}],
        'summary':{'status':'ok'},
        'diagnostics':{'status':'ok','extras_withheld':3,'missed_withheld':3},
        'provenance':{'candidate_sha256':'candidate','audio_sha256':'audio','score_sha256':'score',
                      'pipeline_revision':'v10-boundary-v1'}}


def test_review_exposes_withheld_counts_without_inventing_labels(tmp_path):
    raw=payload();old=deepcopy(raw)
    (tmp_path/'note_alignment_v2.json').write_text(json.dumps(raw))
    review=build_note_alignment(tmp_path)['feedback_review']
    assert review['extras_withheld']==review['missed_withheld']==3
    assert review['agent_label_count']==0 and not review['score_only_labels']
    assert raw==old


def test_normalization_keeps_raw_and_selected_pitch_and_replay():
    raw=payload()
    value=_normalize_transcribed_notes(raw)[0]
    assert value['midi']==70 and value['alignment_midi']==72 and value['relationship']=='match'
    raw['transcribed_notes'][0]['relationship']='copy'
    assert _normalize_transcribed_notes(raw)[0]['relationship']=='copy'


def test_old_score_only_export_requires_matching_provenance(tmp_path):
    raw=payload()
    label={'id':'v10_1','type':'missed_note','score_event_indices':[3],'start_time':None,'end_time':None}
    feedback={'status':'ok','provenance':deepcopy(raw['provenance']),'labels':[label]}
    path=tmp_path/'feedback_v10.json'
    path.write_text(json.dumps(feedback))
    assert score_only_feedback(raw,tmp_path)==[label]
    feedback['provenance']['audio_sha256']='changed'
    path.write_text(json.dumps(feedback))
    assert not score_only_feedback(raw,tmp_path)
    assert feedback_review({'labels':[]},tmp_path) is None


def test_legacy_export_relabel_keeps_score_only_identity(tmp_path):
    from datacreate.transcription_labeling import relabel_sample_from_current_alignment
    raw=payload()
    label={'id':'v10_1','type':'missed_note','score_event_indices':[3],'start_time':None,'end_time':None}
    (tmp_path/'note_alignment_v2.json').write_text(json.dumps(raw))
    (tmp_path/'feedback_v10.json').write_text(json.dumps({'status':'ok','provenance':raw['provenance'],'labels':[label]}))
    result=relabel_sample_from_current_alignment(tmp_path)
    assert result['score_only_label_count']==1
    saved=json.loads((tmp_path/'labels_agent.json').read_text())
    assert saved['agent_labeling']['score_only_labels']==[label]


def test_uncertain_alignment_does_not_publish_score_only_feedback(tmp_path):
    from datacreate.transcription_labeling import _model_feedback_document
    raw = payload()
    raw['diagnostics']['status'] = 'alignment_uncertain'
    raw['score_only_labels'] = [{'id': 'v10_1', 'type': 'missed_note', 'score_event_indices': [3]}]
    assert score_only_feedback(raw, tmp_path) == []
    assert feedback_review(raw, tmp_path)['score_only_labels'] == []
    document = _model_feedback_document(raw)
    assert document['agent_labeling']['score_only_labels'] == []
    assert document['agent_labeling']['labels_without_playback_time'] == []
