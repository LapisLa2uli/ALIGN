"""Batch MuScriptor transcription with verified full-recording context."""
from __future__ import annotations
import hashlib, json, logging, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for rel in ('DataCreate/src','align-model/src','synth-pipeline/src','align-model/scripts','DataCreate/scripts'):
    sys.path.insert(0,str(ROOT/rel))
from datacreate.config import PipelineConfig
from datacreate.muscriptor_audio import transcription_source, file_hash
from datacreate.muscriptor_bridge import written_shift
from transcribe_muscriptor import load_local_model, transcribe_file
OUT=ROOT/'reports/muscriptor_corpus_20261107'
SAMPLES=ROOT/'DataCreate/samples'
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
log=logging.getLogger('transcribe-all')
config=PipelineConfig.load()
config.alignment['transcriber']='muscriptor'
model,dtype,digest=load_local_model('cuda')
rows=[]
for sample in sorted(SAMPLES.iterdir(), key=lambda p:p.name):
    if not sample.is_dir() or not (sample/'performance_audio.wav').is_file() or not (sample/'verified_score.musicxml').is_file():
        continue
    if not (sample.name.isdigit() and len(sample.name)==3):
        continue
    output=OUT/'transcriptions'/f'{sample.name}.json'
    row={'sample':sample.name,'output':str(output),'status':'started'}
    try:
        source,context=transcription_source(sample)
        shift=written_shift(sample,config)
        if output.is_file():
            payload=json.loads(output.read_text(encoding='utf-8'))
            if payload.get('provenance',{}).get('audio_context')==context and payload.get('provenance',{}).get('written_shift')==shift:
                row.update(status='skipped',notes=len(payload.get('transcribed_notes',[])),audio_mode=context['mode'])
                rows.append(row); print(json.dumps(row),flush=True); continue
        output.parent.mkdir(parents=True,exist_ok=True)
        log.info('%s: %s source=%s shift=%s',sample.name,source.name,context['mode'],shift)
        payload=transcribe_file(model,dtype,digest,source,shift,crop_start=context['crop_start'],crop_end=context['crop_end'])
        payload['provenance'].update(audio_sha256=file_hash(sample/'performance_audio.wav'),score_sha256=file_hash(sample/'verified_score.musicxml'),audio_context=context)
        output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        row.update(status='ok',notes=len(payload['transcribed_notes']),audio_mode=context['mode'],source=source.name)
    except Exception as exc:
        row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        log.exception('%s failed',sample.name)
    rows.append(row)
    (OUT/'transcription_manifest.json').write_text(json.dumps({'model':'medium','model_sha256':digest,'dtype':dtype,'rows':rows},indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)
(OUT/'transcription_manifest.json').write_text(json.dumps({'model':'medium','model_sha256':digest,'dtype':dtype,'rows':rows},indent=2),encoding='utf-8')
print(json.dumps({'completed':len(rows),'ok':sum(r['status'] in ('ok','skipped') for r in rows),'errors':sum(r['status']=='error' for r in rows)},indent=2),flush=True)
