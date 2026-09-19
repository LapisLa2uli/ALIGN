"""Measure augmentation round-trip damage on 100 seeded actual training windows."""
import sys,contextlib,io,json,random,copy
from pathlib import Path
from collections import Counter
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path[:0]=[str(ROOT/'baselines/Polytune')]
from dataset.dataset_2_random import Dataset
from contrib import metrics_utils,note_sequences
build=Dataset._build_dataset;Dataset._build_dataset=lambda *a,**k:[]
ds=Dataset(root_dir='',split_json_path='',split='train',is_deterministic=False,is_randomize_tokens=False,num_rows_per_batch=1)
Dataset._build_dataset=build
m=json.loads((ROOT/'baselines/data/align_synth_20260914/manifest.json').read_text())['tracks']
rng=random.Random(365);ids=rng.sample(sorted(k for k,v in m.items() if v['split']=='train'),100)
random.seed(365)
def decode(tokens):
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  result=metrics_utils.event_predictions_to_ns([dict(est_tokens=np.asarray(tokens,dtype=np.int64),start_time=0.,raw_inputs=[])],codec=ds.codec,encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
 return sorted((n.instrument,n.pitch,round(n.start_time,4),round(n.end_time,4)) for n in result['est_ns'].notes),int(result['est_invalid_events'])
rows=[]
for sid in ids:
 p=m[sid]['outputs'];r=dict(extra_notes_midi=p['extra'],removed_notes_midi=p['removed'],correct_notes_midi=p['correct'],mistake_audio=p['mistake_wav'],score_audio=p['score_wav'])
 tracks,pa,ra=ds._preprocess_inputs(r);raw=ds._tokenize(tracks,pa,ra,None)
 chunks=ds._split_frame(raw,length=2000);raw=ds._random_chunk(random.choice(chunks));raw=ds._extract_target_sequence_with_indices(raw,ds.tie_token)
 ds.is_randomize_tokens=False;base=ds._run_length_encode_shifts(copy.deepcopy(raw));notes,invalid=decode(base['targets'])
 trials=[]
 for seed in [1,365,999]:
  ds.is_randomize_tokens=True;row=ds._run_length_encode_shifts(copy.deepcopy(raw));np.random.seed(seed)
  tokens=ds.randomize_tokens([ds.get_token_name(t) for t in row['targets']]);tokens=ds._remove_redundant_tokens(np.array([ds.token_to_idx(t) for t in tokens]))
  n,iv=decode(tokens)
  trials.append(dict(seed=seed,changed=n!=notes,baseline_notes=len(notes),augmented_notes=len(n),added_invalid=iv-invalid,removed=[list(x) for x in (Counter(notes)-Counter(n)).elements()],added=[list(x) for x in (Counter(n)-Counter(notes)).elements()]))
 rows.append(dict(id=sid,window_start=float(raw['mistake_input_times'][0]),baseline_invalid=invalid,trials=trials))
report={'seed':365,'clips':100,'windows':100,'trials':300,'changed_trials':sum(t['changed'] for r in rows for t in r['trials']),'windows_changed_at_least_once':sum(any(t['changed'] for t in r['trials']) for r in rows),'rows':rows}
(OUT/'augmentation_sample.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='rows'})
