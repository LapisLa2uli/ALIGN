"""Isolated first-window padding intervention; production code/outputs untouched."""
import os,sys,contextlib,io,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch,soundfile as sf
from omegaconf import OmegaConf
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
os.environ['LADDERSYM_DETERMINISTIC_PROMPT']='1';os.environ['LADDERSYM_PROMPT_LENGTH']='1024'
sys.path[:0]=[str(ROOT/'baselines/LadderSym'),str(ROOT/'baselines/common')]
from tasks.laddersym_mt3_net import laddersym_MT3Net
from align_runtime import load_model_weights
from inference_error import InferenceHandler,dataset
from contrib import metrics_utils,note_sequences
from dataset.dataset_2_random import Dataset

torch.set_num_threads(1);torch.manual_seed(365)
cfg=OmegaConf.load(ROOT/'baselines/runs/laddersym/synth_20260914_s365/.hydra/config.yaml');cfg.model.config.use_prompt=True
wrapper=laddersym_MT3Net(cfg.model.config,cfg.optim)
checkpoint=json.loads((OUT.parent/'checkpoint_selection.json').read_text())['models']['laddersym']['path']
load_model_weights(wrapper.model,checkpoint);model=wrapper.model.eval().cuda()
h=InferenceHandler(model=model,device=torch.device('cuda'),mel_norm=True)
original=h._compute_spectrograms;captured={}
def capture(a,b):
 out=original(a,b);captured['performance']=out[0].copy();captured['reference']=out[1].copy();return out
h._compute_spectrograms=capture
report={'scope':'first 2.048s window only; generated max_length=1024; fix restores first reference feature before incorrect trailing zeroing; no gain/alignment changes','rows':[]}
DATA=ROOT/'baselines/data/real_034_20260918'
manifest=json.loads((DATA/'manifest.json').read_text())['tracks']
for sid in ['004','009','033']:
 paths=manifest[sid]['outputs'];pa,sr=sf.read(paths['mistake_wav'],dtype='float32');ra,rr=sf.read(paths['score_wav'],dtype='float32');assert sr==rr==16000
 with contextlib.redirect_stdout(io.StringIO()):
  prep=h._preprocess(pa,ra,paths['score_mid']);prompts=h._postprocess_prompt_batch(prep[4],dataset)
 a=prep[1][0,:256];fixed=captured['reference'][0,:256];changed=np.flatnonzero(np.any(a!=fixed,axis=1))
 row={'id':sid,'changed_reference_mel_columns':changed.tolist(),'first_window_mean_abs_difference':float(np.abs(a-fixed).mean()),'conditions':{}}
 for condition,sc in [('original',a),('fixed_first_window_padding',fixed)]:
  with torch.no_grad():
   gen=model.generate(mistake_inputs=torch.tensor(prep[0][0:1,:256]).cuda(),score_inputs=torch.tensor(sc[None]).cuda(),decoder_input_ids=prompts['prompt_tokens'][0].cuda(),decoder_attention_mask=prompts['prompt_masks'][0].cuda(),max_length=1024,num_beams=1,do_sample=False,use_cache=True)
   tok=h._postprocess_batch(gen)[0];eos=np.flatnonzero(tok==-1)
   if len(eos):tok=tok[:eos[0]]
   with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    ev=metrics_utils.event_predictions_to_ns([dict(est_tokens=tok,start_time=0.,raw_inputs=[])],codec=h.codec,encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
  row['conditions'][condition]={'native_counts':dict(Counter(str(n.instrument) for n in ev['est_ns'].notes)),'invalid_events':int(ev['est_invalid_events']),'notes':[[n.instrument,n.pitch,n.start_time,n.end_time] for n in ev['est_ns'].notes],'tokens':gen[0].cpu().tolist()}
 report['rows'].append(row);(OUT/'padding_control.json').write_text(json.dumps(report,indent=2));print(sid,{c:d['native_counts'] for c,d in row['conditions'].items()},flush=True)
# CPU-only exact feature parity after changing padding support accounting.
# Same actual Dataset path, synthetic selected first window.
root=ROOT/'baselines/data/align_synth_20260914';sid=json.loads((OUT/'laddersym_probe.json').read_text())['selection'][0]
v=json.loads((root/'manifest.json').read_text())['tracks'][sid]['outputs']
ds=Dataset(skip_build=True,use_prompt=True,prompt_length=1024,is_deterministic=True,is_randomize_tokens=False,num_rows_per_batch=1)
ds.df=[dict(extra_notes_midi=v['extra'],removed_notes_midi=v['removed'],correct_notes_midi=v['correct'],mistake_audio=v['mistake_wav'],score_audio=v['score_wav'])]
with contextlib.redirect_stdout(io.StringIO()):
 item=ds[0];tracks,pa,ra=ds._preprocess_inputs(ds.df[0]);h._preprocess(pa,ra,v['score_mid'])
report['corrected_feature_parity']={'sample':sid,'max_abs_difference':float(np.abs(item[1][0].numpy()-captured['reference'][0,:256]).max())}
(OUT/'padding_control.json').write_text(json.dumps(report,indent=2));print(report['corrected_feature_parity'])
