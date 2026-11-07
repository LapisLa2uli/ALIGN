"""Preserve v1 conversion and add reference-neighbor locations to extra labels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'baselines/common'))
import label_datacreate_baselines as base
from canonical_adapter_v2 import adapt, VERSION, PARAMETERS

def document_from_midi(midi_path, score_path, *, model, sample_id, provenance=None):
    doc = base.document_from_midi(midi_path, score_path, model=model, sample_id=sample_id, provenance=provenance)
    _, notes, _, _ = base.load_native(midi_path)
    score = base.parse_sounding_notes(score_path)
    converted = adapt(notes, [n.pitch for n in score])
    for label in doc['labels']:
        if label['type'] != 'extra_note':
            continue
        indices = set()
        for event in converted['events']:
            if event['kind'] != 'extra' or event['end'] <= label['start_time'] - 1e-4 or event['start'] >= label['end_time'] + 1e-4:
                continue
            indices.update(event.get('score_event_indices', []))
        if indices:
            label.update(base._score_fields(score, sorted(indices)))
    metadata = doc['baseline_labeling']
    metadata.update(adapter_version=VERSION, adapter_parameters=PARAMETERS,
        unlocated_events=sum(not l.get('score_part') for l in doc['labels']),
        adapter_unlocated_events=converted['unlocated_events'])
    return doc
