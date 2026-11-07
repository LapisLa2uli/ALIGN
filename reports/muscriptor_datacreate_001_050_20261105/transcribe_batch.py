"""Step 1: transcribe the requested range with the same settings as the GUI."""
import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT/'DataCreate/scripts'), str(ROOT/'DataCreate/src')]
from datacreate.muscriptor_bridge import written_shift
from transcribe_muscriptor import load_local_model, transcribe_file

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d, indent=2),encoding='utf-8')

if __name__ == '__main__':
    notes = OUT/'transcriptions'
    notes.mkdir(exist_ok=True)
    samples = [ROOT/'DataCreate/samples'/f'{i:03}' for i in range(1,51)]
    protected = {s.name:{f:sha(s/f) for f in ('labels.json','verified_score.musicxml','performance_audio.wav','labels_agent.json')} for s in samples}
    before = OUT/'before.json'
    if before.exists():
        assert json.loads(before.read_bytes()) == protected
    else: write(before,protected)
    model, dtype, digest = load_local_model('cuda')
    for s in samples:
        h = protected[s.name]
        shift = written_shift(s, SimpleNamespace(alignment={}))
        prov = dict(written_shift=shift, model_sha256=digest, audio_sha256=h['performance_audio.wav'], score_sha256=h['verified_score.musicxml'])
        out = notes/(s.name+'.json')
        if out.exists():
            old=json.loads(out.read_bytes())
            if all(old['provenance'].get(k)==v for k,v in prov.items()): continue
        result=transcribe_file(model,dtype,digest,s/'performance_audio.wav',shift)
        result['provenance'].update(prov)
        tmp=out.with_suffix('.tmp');write(tmp,result);tmp.replace(out)
        print('transcribed',s.name,len(result['transcribed_notes']),flush=True)
