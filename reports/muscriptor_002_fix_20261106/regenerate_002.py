"""Regenerate only take 002 and verify protected inputs byte-for-byte."""
import json, logging, hashlib, sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in ('DataCreate/src','align-model/src','synth-pipeline/src','align-model/scripts'):
    sys.path.insert(0,str(ROOT/p))
from datacreate.config import PipelineConfig
from datacreate.muscriptor_bridge import run_alignment
out=ROOT/'reports/muscriptor_002_fix_20261106'
sample=ROOT/'DataCreate/samples/002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected={f:sha(sample/f) for f in ('labels.json','performance_audio.wav','performance_audio_full.wav','verified_score.musicxml','metadata.json')}
before=json.loads((sample/'note_alignment_v2.json').read_text())
(out/'before_alignment.json').write_text(json.dumps(before,indent=2))
logging.basicConfig(level=logging.INFO)
config=PipelineConfig.load()
config.alignment['transcriber']='muscriptor'
result=run_alignment(sample,config,logging.getLogger('repair002'),True)
after=json.loads((sample/'note_alignment_v2.json').read_text())
assert all(sha(sample/f)==h for f,h in protected.items()),'Protected input changed'
(out/'after_alignment.json').write_text(json.dumps(after,indent=2))
report={'protected_sha256':protected,'protected_unchanged':True,
        'before':before['summary'],'after':after['summary'],
        'before_repetitions':before['repetitions'],'after_repetitions':after['repetitions'],
        'provenance':after['provenance']}
(out/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'before':before['summary'],'after':after['summary'],'protected_unchanged':True},indent=2))

