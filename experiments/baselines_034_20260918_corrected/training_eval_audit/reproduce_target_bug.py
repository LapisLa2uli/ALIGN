"""Minimal independent target augmentation round-trip; no model weights needed."""
import sys,os,copy,contextlib,io,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];flavor=sys.argv[1]
sys.path.insert(0,str(ROOT/'baselines'/('Polytune' if flavor=='polytune' else 'LadderSym')))
from dataset.dataset_2_random import Dataset
from contrib import metrics_utils,note_sequences
import note_seq
build=Dataset._build_dataset;Dataset._build_dataset=lambda *a,**k:[]
kw=dict(root_dir='',split_json_path='',split='train',is_deterministic=True,is_randomize_tokens=False)
if flavor=='laddersym':kw.update(skip_build=True,use_prompt=True,prompt_length=1024)
ds=Dataset(**kw);Dataset._build_dataset=build
correct=note_seq.NoteSequence()
for start,end in [(.2,.5),(.5,.8)]:
 n=correct.notes.add();n.pitch=65;n.velocity=90;n.start_time=start;n.end_time=end
tracks=[note_seq.NoteSequence(),note_seq.NoteSequence(),correct]
if flavor=='laddersym':tracks.append(copy.deepcopy(correct))
raw=ds._tokenize(tracks,np.zeros(64000,dtype=np.float32),np.zeros(64000,dtype=np.float32),None)
raw=ds._random_chunk(raw);raw=ds._extract_target_sequence_with_indices(raw,ds.tie_token)
def convert(shuffle,seed):
 ds.is_randomize_tokens=shuffle;row=copy.deepcopy(raw)
 row=ds._run_length_encode_shifts(row) if flavor=='polytune' else ds.run_length_encode_shifts(row)
 tokens=row['targets']
 if shuffle:
  np.random.seed(seed);tokens=np.array([ds.token_to_idx(t) for t in ds.randomize_tokens([ds.get_token_name(t) for t in tokens])])
  tokens=ds._remove_redundant_tokens(tokens) if flavor=='polytune' else ds.remove_redundant_tokens(tokens)
 tokens=np.asarray(tokens,dtype=np.int64)
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  result=metrics_utils.event_predictions_to_ns([dict(est_tokens=tokens,start_time=0.,raw_inputs=[])],codec=ds.codec,encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
 return dict(seed=seed,notes=sorted([n.instrument,n.pitch,round(n.start_time,4),round(n.end_time,4)] for n in result['est_ns'].notes),invalid=int(result['est_invalid_events']),tokens=[ds.get_token_name(t) for t in tokens])
base=convert(False,0);runs=[convert(True,i) for i in range(100)];bad=[r for r in runs if r['notes']!=base['notes']]
report=dict(model=flavor,baseline=base,shuffles=100,changed_notes=len(bad),first_failure=bad[0] if bad else None)
(OUT/f'{flavor}_target_reproducer.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
