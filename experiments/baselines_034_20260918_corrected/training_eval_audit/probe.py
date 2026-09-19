"""Read-only train/eval audit using saved config and unchanged checkpoints."""
import argparse,contextlib,copy,io,json,os,random,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);args=ap.parse_args()
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];flavor=args.model
SOURCE=ROOT/'baselines'/('Polytune' if flavor=='polytune' else 'LadderSym')
sys.path[:0]=[str(SOURCE),str(ROOT/'baselines/common')]
os.environ['LADDERSYM_DETERMINISTIC_PROMPT']='1';os.environ['LADDERSYM_PROMPT_LENGTH']='1024'
from dataset.dataset_2_random import Dataset
from inference_error import InferenceHandler
from contrib import metrics_utils,note_sequences,event_codec
from verify_loaders import decode,onsets_from_tokens,expected_onsets
from align_runtime import load_model_weights

random.seed(365);np.random.seed(365);torch.manual_seed(365);torch.set_num_threads(1)
DATA=ROOT/'baselines/data/align_synth_20260914'
manifest=json.loads((DATA/'manifest.json').read_text())
selection=[]
for split in ['train','validation']:
 for source in ['procedural12k','rawdata2k']:
  ids=sorted(k for k,v in manifest['tracks'].items() if v['split']==split and v['set']==source)
  selection+=random.sample(ids,3)
kwargs=dict(root_dir=str(DATA),split_json_path=str(DATA/'split.json'),split='train',
 mel_length=256,event_length=1024,num_rows_per_batch=1,split_frame_length=2000,
 is_deterministic=True,is_randomize_tokens=False,is_random_alignment_shift_augmentation=False,
 shuffle=False,audio_filename='mix.wav' if flavor=='laddersym' else 'mix_16k.wav')
if flavor=='laddersym':kwargs.update(use_prompt=True,prompt_length=1024)
with contextlib.redirect_stdout(io.StringIO()):ds=Dataset(**kwargs)
ds.df=[]
for sid in selection:
 out=manifest['tracks'][sid]['outputs']
 ds.df.append(dict(track_id=sid,extra_notes_midi=out['extra'],removed_notes_midi=out['removed'],
                   correct_notes_midi=out['correct'],mistake_audio=out['mistake_wav'],score_audio=out['score_wav']))
cfg=OmegaConf.load(ROOT/f'baselines/runs/{flavor}/synth_20260914_s365/.hydra/config.yaml')
if flavor=='polytune':
 from tasks.polytune_net import polytune
 wrapper=polytune(cfg.model.config,cfg.optim)
else:
 from tasks.laddersym_mt3_net import laddersym_MT3Net
 cfg.model.config.use_prompt=True;wrapper=laddersym_MT3Net(cfg.model.config,cfg.optim)
checkpoint=json.loads((OUT.parent/'checkpoint_selection.json').read_text())['models'][flavor]['path']
load_model_weights(wrapper.model,checkpoint)
model=wrapper.model.eval().cuda();handler=InferenceHandler(model=model,device=torch.device('cuda'),mel_norm=True)
report={'model':flavor,'checkpoint':checkpoint,'seed':365,'selection':selection,'rows':[],
        'class_tokens':{str(c):ds.vocab._encode([ds.codec.encode_event(event_codec.Event('error_class',c))])[0] for c in [1,2,3]}}

def serialized_notes(tokens):
 arr=np.array([int(t)-3 for t in tokens if int(t)>=3])
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  result=metrics_utils.event_predictions_to_ns([dict(est_tokens=arr,start_time=0.,raw_inputs=[])],
       codec=ds.codec,encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
 return sorted((n.instrument,n.pitch,round(n.start_time,4),round(n.end_time,4)) for n in result['est_ns'].notes),{k:v for k,v in result.items() if k!='est_ns'}

for idx,sid in enumerate(selection):
 with contextlib.redirect_stdout(io.StringIO()):
  tracks,pa,ra=ds._preprocess_inputs(ds.df[idx]);raw=ds._tokenize(tracks,pa,ra,None)
  item=ds[idx]
  if flavor=='polytune':infer=handler._preprocess(pa,ra)
  else:
   infer=handler._preprocess(pa,ra,ds.df[idx]['score_audio'].replace('.wav','.mid'))
   import inference_error
   prompts=handler._postprocess_prompt_batch(infer[4],inference_error.dataset)
 row={'id':sid,'split':manifest['tracks'][sid]['split'],'source':manifest['tracks'][sid]['set'],
      'duration':len(pa)/16000,'first_window_feature_difference':{},'augmentation_checks':[]}
 for name,j in [('performance',0),('reference',1)]:
  a=item[j][0].numpy();b=infer[j][0,:256];row['first_window_feature_difference'][name]={'max':float(np.abs(a-b).max()),'mean':float(np.abs(a-b).mean())}
 if flavor=='laddersym':
  row['prompt_equal']=bool(torch.equal(item[3][0].masked_fill(item[3][0]==-100,0),prompts['prompt_tokens'][0][0]))
  row['prompt_mask_equal']=bool(torch.equal(item[4][0],prompts['prompt_masks'][0][0]))
 # Compare actual randomized target construction at the same deterministic window.
 baseline_notes,_=serialized_notes(item[2][0])
 expected=expected_onsets(str(DATA),sid,0)
 row['deterministic_onsets_match_midi']=onsets_from_tokens(decode(ds,item[2][0]))==expected
 for seed in [1,365,999]:
  random.seed(seed);np.random.seed(seed);ds.is_randomize_tokens=True
  with contextlib.redirect_stdout(io.StringIO()):aug=ds[idx]
  notes,diag=serialized_notes(aug[2][0]);row['augmentation_checks'].append(dict(seed=seed,notes_identical=notes==baseline_notes,onsets_match_midi=onsets_from_tokens(decode(ds,aug[2][0]))==expected,
      removed_notes=[list(x) for x in (Counter(baseline_notes)-Counter(notes)).elements()],
      added_notes=[list(x) for x in (Counter(notes)-Counter(baseline_notes)).elements()],
      decoder_diagnostics={k:int(v) for k,v in diag.items() if isinstance(v,(int,np.integer))}))
 ds.is_randomize_tokens=False
 # Gold-prefix (teacher-forced) diagnostic, first and later 2.048-second windows.
 row['windows']=[]
 for start in [0, max(0,min(int(len(pa)/128)-512,768))]:
  chunk_raw=copy.deepcopy(raw)
  # Force the same random chunk function to the requested start, preserving context.
  ds.is_deterministic=False;old=random.randint;random.randint=lambda lo,hi:min(start,hi)
  try:chunk=ds._random_chunk(chunk_raw)
  finally:random.randint=old;ds.is_deterministic=True
  chunk=ds._extract_target_sequence_with_indices(chunk,ds.tie_token)
  if flavor=='polytune':chunk=ds._run_length_encode_shifts(chunk)
  else:
   chunk=ds.run_length_encode_shifts(chunk);chunk=ds.run_length_encode_shifts(chunk,feature_key='prompt_events')
  chunk=ds._compute_spectrogram(chunk);chunk=ds._pad_length(chunk)
  mi=chunk['mistake_inputs'].unsqueeze(0).cuda();sc=chunk['score_inputs'].unsqueeze(0).cuda()
  tar=torch.as_tensor(chunk['targets'],dtype=torch.long).unsqueeze(0).cuda()
  kw={}
  if flavor=='laddersym':kw=dict(decoder_input_ids=torch.as_tensor(chunk['prompts'],dtype=torch.long).unsqueeze(0).cuda(),decoder_attention_mask=torch.as_tensor(chunk['prompts_attention_mask']).unsqueeze(0).cuda())
  with torch.no_grad():
   logits=model(mistake_inputs=mi,score_inputs=sc,labels=tar,**{k:v.clone() for k,v in kw.items()})
   if flavor=='laddersym':logits=logits[:,1024:]
   valid=tar!=-100;pred=logits.argmax(-1);loss=torch.nn.functional.cross_entropy(logits[valid],tar[valid]).item()
   cls=(tar>=1135)&(tar<=1137)
   confusion=Counter((int(t),int(p)) for t,p in zip(tar[cls],pred[cls]))
   win={'start_seconds':start/125,'tokens':int(valid.sum()),'teacher_token_accuracy':float((pred[valid]==tar[valid]).float().mean()),'unweighted_ce':loss,'class_token_confusion':[[a,b,n] for (a,b),n in sorted(confusion.items())]}
   # Free generation only first four examples + later window, bounded to 256 tokens.
   if idx in [0,3,6,9]:
    generate_kw={k:v.clone() for k,v in kw.items()}
    if 'decoder_input_ids' in generate_kw:
     generate_kw['decoder_input_ids'].masked_fill_(generate_kw['decoder_input_ids']==-100,0)
    generation=model.generate(mistake_inputs=mi,score_inputs=sc,max_length=256,num_beams=1,do_sample=False,use_cache=flavor=='laddersym',**generate_kw)
    gen=generation[0].cpu().tolist();win['generation_tokens']=len(gen);win['generation_eos']=1 in gen
    post=handler._postprocess_batch(generation)
    tok=np.array(post[0]);end=np.flatnonzero(tok==-1)
    if len(end):tok=tok[:end[0]]
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
     ev=metrics_utils.event_predictions_to_ns([dict(est_tokens=tok,start_time=0.,raw_inputs=[])],codec=ds.codec,encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
    win['generated_note_classes']=dict(Counter(str(n.instrument) for n in ev['est_ns'].notes))
    win['decode_diagnostics']={k:int(v) for k,v in ev.items() if k!='est_ns' and isinstance(v,(int,np.integer))}
   row['windows'].append(win)
 report['rows'].append(row)
 (OUT/f'{flavor}_probe.json').write_text(json.dumps(report,indent=2))
 print(flavor,sid,'features',row['first_window_feature_difference'],'target_checks',row['deterministic_onsets_match_midi'],row['augmentation_checks'],'windows',row['windows'],flush=True)
print('COMPLETE',flavor,flush=True)
