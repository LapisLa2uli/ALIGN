"""Boundary/interior feature parity without loading a model or reading GT labels."""
import sys,random,contextlib,io,json,types
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];flavor=sys.argv[1]
sys.path.insert(0,str(ROOT/'baselines'/('Polytune' if flavor=='polytune' else 'LadderSym')))
from dataset.dataset_2_random import Dataset
from inference_error import InferenceHandler
build=Dataset._build_dataset;Dataset._build_dataset=lambda *a,**kw:[]
kw=dict(root_dir='',split_json_path='',split='train',is_deterministic=False,is_randomize_tokens=False)
if flavor=='laddersym':kw['skip_build']=True
with contextlib.redirect_stdout(io.StringIO()):ds=Dataset(**kw)
Dataset._build_dataset=build;torch.set_num_threads(1)
h=InferenceHandler(model=None,device=torch.device('cpu'),mel_norm=True)
if flavor=='laddersym':h.model=types.SimpleNamespace(config=types.SimpleNamespace(use_prompt=False))
original=h._compute_spectrograms;captured={}
def capture(a,b):
 out=original(a,b);captured['reference']=out[1].copy();return out
h._compute_spectrograms=capture
synth=json.loads((ROOT/'baselines/data/align_synth_20260914/manifest.json').read_text())['tracks']
real=json.loads((ROOT/'baselines/data/real_034_20260918/manifest.json').read_text())['tracks']
selection=json.loads((OUT/f'{flavor}_probe.json').read_text())['selection']
cases=[(sid,synth[sid]['outputs']) for sid in [selection[i] for i in [0,3,6,9]]]+[(sid,real[sid]['outputs']) for sid in ['004','033']]
rows=[]
for sid,v in cases:
 pa,sr=sf.read(v['mistake_wav'],dtype='float32');ra,rr=sf.read(v['score_wav'],dtype='float32');assert sr==rr==16000
 with contextlib.redirect_stdout(io.StringIO()):
  prep=h._preprocess(pa,ra) if flavor=='polytune' else h._preprocess(pa,ra,v['score_mid'])
 pf,pt=ds._audio_to_frames(pa);rf,rt=ds._audio_to_frames(ra)
 lastfull=(len(pf)-256)//256
 for k in sorted(set([0,1,max(1,lastfull//2),lastfull])):
  start=k*256
  raw={'mistake_inputs':pf,'score_inputs':rf,'mistake_input_times':pt,'score_input_times':rt}
  old=random.randint;random.randint=lambda lo,hi:start
  try:chunk=ds._random_chunk(raw)
  finally:random.randint=old
  chunk=ds._compute_spectrogram(chunk)
  row={'id':sid,'window':k,'start_seconds':start/125,'total_frames':len(pf)}
  for name,j in [('mistake',0),('score',1)]:
   a=chunk[f'{name}_inputs'][:256].numpy();b=prep[j][k,:256];changed=np.flatnonzero(np.any(a!=b,axis=1))
   row[name]={'max_abs':float(np.abs(a-b).max()),'mean_abs':float(np.abs(a-b).mean()),'changed_columns':changed.tolist()}
  if k==0:
   row['score_before_erroneous_mask_max_abs']=float(np.abs(chunk['score_inputs'][:256].numpy()-captured['reference'][0,:256]).max())
  rows.append(row)
(OUT/f'{flavor}_window_parity.json').write_text(json.dumps(rows,indent=2))
print(flavor,'windows',len(rows),'performance_identical',sum(r['mistake']['max_abs']==0 for r in rows),'reference_identical',sum(r['score']['max_abs']==0 for r in rows))
for r in rows:
 if r['score']['max_abs']>0:print(r['id'],r['window'],r['score']['max_abs'],r['score']['changed_columns'][:3],r['score']['changed_columns'][-3:])
