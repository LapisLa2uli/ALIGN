"""Policy 3 exact score identities must survive GUI/API saves."""
import json
from datacreate.config import PipelineConfig
from datacreate.melody import canonical_note_location
from datacreate.models import LabelsDocument
from datacreate.web.app import LabelsPayload, create_app


def test_policy3_gui_save_preserves_noncontiguous_core_and_copy_count(tmp_path):
    folder=tmp_path/'001';folder.mkdir()
    (folder/'performance_audio.wav').write_bytes(b'wav')
    label=dict(id='polytune_0000',source='agent',start_time=1.,end_time=2.,type='repetition',
        score_event_indices=[2,4],context_score_event_indices=list(range(7)),
        core_note_ids=['note_0002','note_0004'],note_ids=[f'note_{i:04d}' for i in range(7)],
        score_part=dict(start_note_index=0,end_note_index=6,core_start_note_index=2,core_end_note_index=4,pad_notes=2),
        extra_copies=2,location_policy='stored_transcription_lookup_v1',location_evidence={'method':'covered_transcript_events'})
    path=folder/'labels_polytune.json'
    path.write_text(json.dumps(dict(schema_version='1.2',labels=[label],baseline_labeling={'alignment_policy':3})),encoding='utf-8')
    human=folder/'labels.json';human.write_text('human untouched',encoding='utf-8')
    app=create_app(PipelineConfig(schema_version='1.2',paths={'samples_root':str(tmp_path)},taxonomy=['repetition']))
    endpoint=next(r.endpoint for r in app.routes if getattr(r,'name',None)=='save_labels')
    endpoint('001',LabelsPayload(labels=[label],annotator_id='baseline_polytune'),'polytune')
    saved=json.loads(path.read_text())
    assert canonical_note_location(saved['labels'][0])==('score_events_with_copies',(2,4),2)
    assert saved['labels'][0]['location_policy']=='stored_transcription_lookup_v1'
    assert saved['labels'][0]['context_score_event_indices']==list(range(7))
    assert saved['baseline_labeling']['alignment_policy']==3
    assert saved['baseline_labeling']['edited_in_gui'] is True
    assert human.read_text()=='human untouched'


def test_unlocated_policy3_label_does_not_acquire_context_identity():
    doc=LabelsDocument.model_validate(dict(schema_version='1.2',labels=[dict(id='x',source='agent',
        start_time=1.,end_time=2.,type='wrong_note',score_event_indices=[],score_part=None,
        location_policy='stored_transcription_lookup_v1')])).model_dump()
    assert doc['labels'][0]['score_event_indices']==[]
    assert canonical_note_location(doc['labels'][0]) is None
