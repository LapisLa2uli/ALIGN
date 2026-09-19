import json,math
from pathlib import Path
from collections import Counter
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
m=json.loads((ROOT/'baselines/data/align_synth_20260914/manifest.json').read_text())
stats={s:Counter() for s in ['train','validation']};offsets={s:[] for s in stats};examples=[]
for sid,v in m['tracks'].items():
 split=v['split'];c=stats[split];d=json.loads((Path(v['bundle'])/'note_labels.json').read_text());notes=d['performance_notes'];ref=d['reference_notes']
 c['clips']+=1;c['audio_seconds']+=v['written_duration_s'];c['no_extra_missing_clips']+=int(v['n_extra']==0 and v['n_removed']==0)
 frames=math.floor(round(v['written_duration_s']*16000)/128)+1
 kept=(frames-1)//2000*2000
 cutoff=kept/125 if kept else v['written_duration_s']
 c['discarded_tail_seconds']+=max(0,v['written_duration_s']-cutoff)
 for n in notes:
  cls=n['cls'];c['notes_'+cls]+=1;c['first_window_'+cls]+=int(round(n['onset']*100)<204)
  c['discarded_tail_'+cls]+=int(n['onset']>=cutoff)
 for n in d['missed_notes']:
  if n.get('copy',0)!=0:continue
  c['notes_missing']+=1;c['first_window_missing']+=int(round(n['onset']*100)<204)
  c['discarded_tail_missing']+=int(n['onset']>=cutoff)
 collisions=0
 for cls in ['correct','extra']:
  seq=sorted([n for n in notes if n['cls']==cls],key=lambda n:n['onset'])
  for prev,n in zip(seq,seq[1:]):
   if prev['sounding_pitch']==n['sounding_pitch'] and round(min(prev['offset'],n['onset'])*100)==round(n['onset']*100):
    collisions+=1
 c['same_pitch_class_boundary_pairs_before_cutoff_check']+=collisions
 c['clips_with_same_pitch_class_boundaries']+=int(collisions>0)
 for r in ref:
  if r.get('cls')!='correct' or r.get('perf_index') is None:continue
  p=notes[r['perf_index']]
  assert p['index']==r['perf_index']
  delta=p['onset']-r['onset'];offsets[split].append(delta);c['linked_correct']+=1
  s=math.floor(p['onset']/2.048)*2.048
  outside=r['offset']<s-1.024 or r['onset']>=s+3.072
  c['correct_reference_outside_inference_context']+=int(outside)
  c['correct_reference_offset_gt3s']+=int(abs(delta)>3.072)
  if outside and len(examples)<6:examples.append(dict(id=sid,performance_onset=p['onset'],reference_onset=r['onset'],reference_window=[s-1.024,s+3.072]))
report={'stats':{k:dict(v) for k,v in stats.items()},'correct_note_time_offset_quantiles':{k:dict(zip(['q0','q25','q50','q75','q100'],map(float,np.quantile(v,[0,.25,.5,.75,1])))) for k,v in offsets.items()},'examples':examples,'context_definition':'Fixed inference window at floor(performance_onset/2.048)*2.048; reference covers [window_start-1.024,window_start+3.072); outside requires no overlap with the linked correct reference note. Does not imply every mismatch is unlearnable.'}
(OUT/'population.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
