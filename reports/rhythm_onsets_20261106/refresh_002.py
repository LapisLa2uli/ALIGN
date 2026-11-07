"""Republish take 002 with the onset policy, preserving its acoustic transcript."""
import json, hashlib, sys, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for relative in ('DataCreate/src','align-model/src','synth-pipeline/src','align-model/scripts'):
    sys.path.insert(0,str(ROOT/relative))
from datacreate.config import PipelineConfig
from datacreate.align_bridge import _bridge_command_env
from datacreate.muscriptor_bridge import candidate_path, current_feedback
sample=ROOT/'DataCreate/samples/002'
out=Path(__file__).parent
before=json.loads((sample/'note_alignment_v2.json').read_text())
transcript=Path(before['provenance']['run']).parent/'muscriptor_notes.json'
assert transcript.is_file()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
protected={name:sha(sample/name) for name in ('labels.json','metadata.json','performance_audio.wav','performance_audio_full.wav','verified_score.musicxml')}
raw_hash=sha(transcript)
(out/'002_before.json').write_text(json.dumps(before,indent=2))
config=PipelineConfig.load()
python,env=_bridge_command_env(config)
command=[str(python),str(ROOT/'align-model/scripts/publish_datacreate_v9.py'),
    '--sample',str(sample),'--output',str(out/'publish002'),
    '--candidate',str(candidate_path(config)),'--transcription',str(transcript),
    '--device',str(config.alignment.get('note_alignment_device','cuda'))]
subprocess.run(command,cwd=ROOT,env=env,check=True)
after=json.loads((sample/'note_alignment_v2.json').read_text())
assert all(sha(sample/name)==digest for name,digest in protected.items())
assert sha(transcript)==raw_hash
assert before['transcribed_notes']==after['transcribed_notes']
assert before['note_mapping']==after['note_mapping']
assert [l for l in before['labels'] if l['type']!='rhythm_error']==[l for l in after['labels'] if l['type']!='rhythm_error']
assert current_feedback(sample,config)
(out/'002_after.json').write_text(json.dumps(after,indent=2))
report={'protected_sha256':protected,'protected_unchanged':True,'transcription_and_mapping_unchanged':True,
    'transcription_sha256':raw_hash,'before_rhythm_count':sum(l['type']=='rhythm_error' for l in before['labels']),
    'after_rhythm_count':sum(l['type']=='rhythm_error' for l in after['labels']),
    'rhythm_diagnostics':after['diagnostics']['rhythm'],'current_feedback':True}
(out/'002_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('protected_sha256','rhythm_diagnostics')},indent=2))
