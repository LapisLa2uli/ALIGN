from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'DataCreate/src'))
sys.path.insert(0, str(ROOT / 'align-model/scripts'))
import eval_datacreate_current as base
from datacreate.transcription_labeling import build_agent_label_document_from_note_alignment
from datacreate.validation import validate_labels_file

sample = ROOT / 'DataCreate/samples/demo_001'
document = build_agent_label_document_from_note_alignment(sample, maximum_per_type=10000)
base._atomic_json(sample / 'labels_agent.json', document)
errors = validate_labels_file(sample / 'labels_agent.json')
if errors:
    raise SystemExit('; '.join(errors))
print({'sample': sample.name, 'labels': len(document.get('labels') or [])})
