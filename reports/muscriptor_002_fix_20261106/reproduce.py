"""Reproduce take 002 with unchanged model weights and different precision."""
import sys, json, hashlib, time, gc, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'DataCreate/scripts'))
from transcribe_muscriptor import transcribe_file
from huggingface_hub import hf_hub_download
from muscriptor.transcription_model import TranscriptionModel
import torch
p=argparse.ArgumentParser()
p.add_argument('--dtype',choices=['float16','float32'],required=True)
p.add_argument('--full',action='store_true')
p.add_argument('--instrument')
args=p.parse_args()
out=ROOT/'reports/muscriptor_002_fix_20261106'
out.mkdir(parents=True,exist_ok=True)
weights=Path(hf_hub_download('MuScriptor/muscriptor-medium','model.safetensors',local_files_only=True))
with weights.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
t=time.perf_counter()
model=TranscriptionModel.load_model(weights,device='cuda',dtype=args.dtype)
torch.set_num_threads(4)
audio=ROOT/('DataCreate/samples/002/performance_audio_full.wav' if args.full else 'DataCreate/samples/002/performance_audio.wav')
if args.instrument:
    original_transcribe=model.transcribe
    model.transcribe=lambda a, **kwargs: original_transcribe(a,instruments=[args.instrument], **kwargs)
result=transcribe_file(model,args.dtype,digest,audio,2)
result['seconds']=time.perf_counter()-t
result['audio_sha256']=hashlib.sha256(audio.read_bytes()).hexdigest()
(out/f'002-{args.dtype}{"-full" if args.full else ""}{"-"+args.instrument if args.instrument else ""}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'dtype':args.dtype,'notes':len(result['transcribed_notes']),'seconds':result['seconds']}),flush=True)

