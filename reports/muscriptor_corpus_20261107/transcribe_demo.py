from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[2]
for rel in ('DataCreate/src', 'align-model/src', 'synth-pipeline/src', 'align-model/scripts', 'DataCreate/scripts'):
    sys.path.insert(0, str(ROOT / rel))
from datacreate.config import PipelineConfig
from datacreate.muscriptor_audio import transcription_source, file_hash
from datacreate.muscriptor_bridge import written_shift
from transcribe_muscriptor import load_local_model, transcribe_file

sample = ROOT / 'DataCreate/samples/demo_001'
out = ROOT / 'reports/muscriptor_corpus_20261107/transcriptions/demo_001.json'
config = PipelineConfig.load()
source, context = transcription_source(sample)
model, dtype, digest = load_local_model('cuda')
payload = transcribe_file(
    model, dtype, digest, source, written_shift(sample, config),
    crop_start=context['crop_start'], crop_end=context['crop_end'])
payload['provenance'].update(
    audio_sha256=file_hash(sample / 'performance_audio.wav'),
    score_sha256=file_hash(sample / 'verified_score.musicxml'),
    audio_context=context)
out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps({'sample': sample.name, 'notes': len(payload['transcribed_notes']),
                  'audio_mode': context['mode'], 'source': source.name}, indent=2))
