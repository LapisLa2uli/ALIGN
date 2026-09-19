"""Frozen corrected training, generative dev selection, and final real-test inference."""
from __future__ import annotations
import argparse,contextlib,hashlib,io,json,os,random,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
p=argparse.ArgumentParser();p.add_argument('--model',choices=['polytune','laddersym'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();flavor=a.model
OUT=RUN/('smoke' if a.smoke else 'training')/flavor;OUT.mkdir(parents=True,exist_ok=True)
proto=json.loads((RUN/'protocol.json').read_text());DATA=Path(proto['train_data'])
sys.path[:0]=[str(RUN/'source'/('Polytune' if flavor=='polytune' else 'LadderSym')),str(ROOT/'baselines/common')]
os.environ['LADDERSYM_DETERMINISTIC_PROMPT']='1';os.environ['LADDERSYM_PROMPT_LENGTH']='1024';os.environ['LADDERSYM_USE_CACHE']='1'
from dataset.dataset_2_random import Dataset,collate_fn
from inference_error import InferenceHandler
from evaluate_notes import evaluate
from align_runtime import load_model_weights

def write(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def infer(model,root,ids,dest,smoke=False):
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir(parents=True);write(dest/'evaluated_ids.json',ids)
 manifest=json.loads((root/'manifest.json').read_text())['tracks']
 handler=InferenceHandler(model=model,device=model.device,mel_norm=True)
 prior_mode=model.training;model.eval()
 with (dest/'inference.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log),torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
  for i,sid in enumerate(ids):
   paths=manifest[sid]['outputs'];perf,sr=sf.read(paths['mistake_wav'],dtype='float32');ref,rr=sf.read(paths['score_wav'],dtype='float32');assert sr==rr==16000
   if smoke:perf=perf[:32768];ref=ref[:32768]
   kwargs=dict(mistake_audio=perf,score_audio=ref,audio_path=sid,outpath=str(dest/sid/'mix.mid'),batch_size=1,max_length=32 if smoke else 1024)
   if flavor=='laddersym':kwargs['prompt_path']=paths['score_mid']
   handler.inference(**kwargs)
   write(dest/'progress.json',{'completed':i+1,'total':len(ids),'updated_at':time.time()})
 model.train(prior_mode)

class Status(pl.Callback):
 def __init__(self):self.started=time.time()
 def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
  if batch_idx%100==0:
   write(OUT/'status.json',dict(state='training',epoch=trainer.current_epoch+1,epochs=trainer.max_epochs,batch=batch_idx+1,total_batches=trainer.num_training_batches,global_step=trainer.global_step,loss=float(outputs['loss']) if outputs else None,elapsed_seconds=time.time()-self.started,updated_at=time.time(),pid=os.getpid(),gpu_peak_MiB=torch.cuda.max_memory_allocated()/1024**2))

class SelectByGeneration(pl.Callback):
 def __init__(self):
  self.best=-1.;self.selected=None
  if a.resume and (OUT/'best.json').exists():
   self.selected=json.loads((OUT/'best.json').read_text());self.best=self.selected['f1']
 def on_validation_end(self,trainer,module):
  if trainer.sanity_checking:return
  epoch=trainer.current_epoch+1
  if not a.smoke and epoch not in proto['selection_epochs']:return
  write(OUT/'status.json',dict(state='generative_validation',epoch=epoch,pid=os.getpid(),updated_at=time.time()))
  dest=OUT/'dev_predictions'/f'epoch_{epoch:02d}'
  ids=proto['dev_ids'][:1] if a.smoke else proto['dev_ids']
  infer(module.model,DATA,ids,dest,smoke=a.smoke)
  report=evaluate(DATA,dest,allow_unclassified=True);write(dest/'note_metrics.json',report)
  score=report['micro']['class_aware']['F1']
  with (OUT/'dev_history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,f1=score,metrics=report['micro']))+'\n')
  if score>self.best:
   self.best=score;path=OUT/'best.pt';tmp=OUT/'best.pt.tmp';torch.save({k:v.detach().cpu() for k,v in module.model.state_dict().items()},tmp);tmp.replace(path)
   self.selected=dict(model=flavor,path=str(path),sha256=digest(path),epoch=epoch,completed_epochs=epoch,global_step=trainer.global_step,f1=score,selection=proto['selection_metric'],dev_ids=ids)
   write(OUT/'best.json',self.selected)
  print(f'GENERATION_SELECTION epoch={epoch} f1={score:.6f} best={self.best:.6f}',flush=True)

pl.seed_everything(proto['seed'],workers=True);torch.set_num_threads(1)
cfg=OmegaConf.load(ROOT/f'baselines/runs/{flavor}/synth_20260914_s365/.hydra/config.yaml')
cfg.optim.num_epochs=proto['epochs']
if a.smoke:cfg.optim.warmup_steps=1
if flavor=='polytune':
 from tasks.polytune_net import polytune
 module=polytune(cfg.model.config,cfg.optim)
else:
 from tasks.laddersym_mt3_net import laddersym_MT3Net
 cfg.model.config.use_prompt=True;module=laddersym_MT3Net(cfg.model.config,cfg.optim)
common=dict(root_dir=str(DATA),split_json_path=str(DATA/'split.json'),mel_length=256,event_length=1024,num_rows_per_batch=1,split_frame_length=2000,is_randomize_tokens=False,is_random_alignment_shift_augmentation=False,full_recording=True,shuffle=False,audio_filename='mix.wav' if flavor=='laddersym' else 'mix_16k.wav')
if flavor=='laddersym':common.update(use_prompt=True,prompt_length=1024)
train=Dataset(split='train',is_deterministic=False,validation_windows=1,**common)
val=Dataset(split='validation',is_deterministic=True,validation_windows=5,**common)
if a.smoke:train.df=train.df[:2];val.df=val.df[:1]
module.num_steps_per_epoch=len(train)
write(OUT/'config.json',dict(model=OmegaConf.to_container(cfg.model.config,resolve=True),optim=OmegaConf.to_container(cfg.optim,resolve=True),dataset_options=common,train_items=len(train),validation_items=len(val),epochs=1 if a.smoke else proto['epochs'],batch_size=1,gradient_accumulation=1 if a.smoke else 16,precision='bf16-mixed',initialized_from='scratch',source=str(RUN/'source')))
loader=lambda ds,shuffle:DataLoader(ds,batch_size=1,shuffle=shuffle,num_workers=0 if a.smoke else 4,collate_fn=collate_fn,pin_memory=True)
selector=SelectByGeneration();checkpoint=ModelCheckpoint(dirpath=str(OUT/'checkpoints'),save_last=True,save_top_k=0,every_n_epochs=1)
trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=1 if a.smoke else proto['epochs'],accumulate_grad_batches=1 if a.smoke else 16,logger=CSVLogger(str(OUT),name='metrics'),callbacks=[Status(),selector,checkpoint],num_sanity_val_steps=0 if a.smoke else 2,log_every_n_steps=1 if a.smoke else 100,enable_progress_bar=False,limit_train_batches=2 if a.smoke else 1.0,limit_val_batches=2 if a.smoke else 1.0)
trainer.fit(module,loader(train,True),loader(val,False),ckpt_path=str(OUT/'checkpoints/last.ckpt') if a.resume else None)
assert selector.selected is not None
if not a.smoke:
 write(OUT/'status.json',dict(state='final_test_inference',best=selector.selected,pid=os.getpid(),updated_at=time.time()))
 load_model_weights(module.model,selector.selected['path']);module.model.cuda().eval()
 infer(module.model,Path(proto['test_data']),proto['full_test_ids'],OUT/'test_predictions')
write(OUT/'status.json',dict(state='complete',best=selector.selected,smoke=a.smoke,updated_at=time.time(),pid=os.getpid()))
print('COMPLETE',flavor,'smoke',a.smoke,flush=True)
