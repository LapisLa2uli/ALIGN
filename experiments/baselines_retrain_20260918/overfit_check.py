"""Tiny from-scratch memorization diagnostic; weights never used by the formal run."""
import contextlib,io,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];m=sys.argv[1]
os.environ['LADDERSYM_DETERMINISTIC_PROMPT']='1';os.environ['LADDERSYM_PROMPT_LENGTH']='1024'
sys.path[:0]=[str(RUN/'source'/('Polytune' if m=='polytune' else 'LadderSym')),str(ROOT/'baselines/common')]
from dataset.dataset_2_random import Dataset
from inference_error import InferenceHandler
from contrib import metrics_utils,note_sequences
cfg=OmegaConf.load(ROOT/f'baselines/runs/{m}/synth_20260914_s365/.hydra/config.yaml');torch.manual_seed(365);torch.set_num_threads(1)
if m=='polytune':
 from tasks.polytune_net import polytune
 model=polytune(cfg.model.config,cfg.optim).model
else:
 from tasks.laddersym_mt3_net import laddersym_MT3Net
 cfg.model.config.use_prompt=True;model=laddersym_MT3Net(cfg.model.config,cfg.optim).model
root=ROOT/'baselines/data/align_synth_20260914';manifest=json.loads((root/'manifest.json').read_text())['tracks'];sid='synth_gen_5544';p=manifest[sid]['outputs'];assert manifest[sid]['split']=='train'
kw=dict(root_dir=str(root),split_json_path=str(root/'split.json'),split='train',full_recording=True,is_deterministic=True,is_randomize_tokens=False,num_rows_per_batch=1,shuffle=False,audio_filename='mix.wav')
if m=='laddersym':kw.update(use_prompt=True,prompt_length=1024)
with contextlib.redirect_stdout(io.StringIO()):ds=Dataset(**kw)
ds.df=[dict(extra_notes_midi=p['extra'],removed_notes_midi=p['removed'],correct_notes_midi=p['correct'],mistake_audio=p['mistake_wav'],score_audio=p['score_wav'])]
with contextlib.redirect_stdout(io.StringIO()):item=ds[0]
model.cuda().train();mi,sc=item[0].cuda(),item[1].cuda();targets=item[2][:,:int((item[2][0]!=-100).sum())].cuda();prompt={}
if m=='laddersym':prompt=dict(decoder_input_ids=item[3].cuda(),decoder_attention_mask=item[4].cuda())
opt=torch.optim.AdamW(model.parameters(),lr=3e-4);history=[];start=time.time()
for step in range(150):
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16,enabled=os.environ.get('OVERFIT_PRECISION','bf16')=='bf16'):
  logits=model(mistake_inputs=mi,score_inputs=sc,labels=targets,**{k:v.clone() for k,v in prompt.items()})
  if m=='laddersym':logits=logits[:,1024:]
  loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten())
 loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 if step%25==0 or step==149:history.append({'step':step+1,'ce':float(loss.detach())});print(m,history[-1],flush=True)
model.eval();handler=InferenceHandler(model=model,device=torch.device('cuda'),mel_norm=True)
gkw={k:v.clone() for k,v in prompt.items()}
if gkw:gkw['decoder_input_ids'].masked_fill_(gkw['decoder_input_ids']==-100,0)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=os.environ.get('OVERFIT_PRECISION','bf16')=='bf16'):gen=model.generate(mistake_inputs=mi,score_inputs=sc,max_length=128,num_beams=1,do_sample=False,use_cache=m=='laddersym',**gkw)
def notes(tok):
 e=np.flatnonzero(tok==-1)
 if len(e):tok=tok[:e[0]]
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):d=metrics_utils.event_predictions_to_ns([dict(est_tokens=tok,start_time=0.,raw_inputs=[])],codec=ds.codec,encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
 return [(n.instrument,n.pitch,n.start_time,n.end_time) for n in d['est_ns'].notes]
pred=notes(handler._postprocess_batch(gen)[0]);gold=notes(np.where(targets[0].cpu().numpy()==1,-1,targets[0].cpu().numpy()-3));matrix=np.array([[a[0]==b[0] and a[1]==b[1] and abs(a[2]-b[2])<=.05+1e-9 for b in gold] for a in pred],dtype=np.int64).reshape(len(pred),len(gold));r,c=linear_sum_assignment(-matrix);tp=int(matrix[r,c].sum());f1=2*tp/max(1,len(pred)+len(gold))
comparison={}
if m=='laddersym':
 with torch.inference_mode():
  full=model.generate(mistake_inputs=mi,score_inputs=sc,max_length=128,num_beams=1,do_sample=False,use_cache=False,**{k:v.clone() for k,v in gkw.items()})
  logits=model(mistake_inputs=mi,score_inputs=sc,labels=targets,**{k:v.clone() for k,v in prompt.items()})[:,1024:]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  bfgen=model.generate(mistake_inputs=mi,score_inputs=sc,max_length=128,num_beams=1,do_sample=False,use_cache=True,**{k:v.clone() for k,v in gkw.items()})
  bflogits=model(mistake_inputs=mi,score_inputs=sc,labels=targets,**{k:v.clone() for k,v in prompt.items()})[:,1024:]
 comparison={'bf16_tokens':bfgen[0].cpu().tolist(),'bf16_teacher_tokens':bflogits.argmax(-1)[0].cpu().tolist(),'bf16_notes':notes(handler._postprocess_batch(bfgen)[0]),'cache_tokens':gen[0].cpu().tolist(),'no_cache_tokens':full[0].cpu().tolist(),'no_cache_notes':notes(handler._postprocess_batch(full)[0]),'gold_tokens':targets[0].cpu().tolist(),'teacher_eval_tokens':logits.argmax(-1)[0].cpu().tolist()}
report=dict(training_precision=os.environ.get('OVERFIT_PRECISION','bf16'),generation_comparison=comparison,model=m,sample=sid,steps=150,learning_rate=3e-4,scope='one 2.048s training window; short target prefix; diagnostic learning rate; discarded scratch weights',loss_history=history,gold=gold,pred=pred,note_f1=f1,elapsed_seconds=time.time()-start)
(RUN/f"{m}_overfit{os.environ.get('OVERFIT_SUFFIX','')}.json").write_text(json.dumps(report,indent=2));print('OVERFIT',m,f1,flush=True)
