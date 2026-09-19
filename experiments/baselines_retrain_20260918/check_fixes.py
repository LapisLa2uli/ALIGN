import contextlib,io,json,os,sys,random,copy,types
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch,note_seq
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];flavor=sys.argv[1]
sys.path[:0]=[str(RUN/'source'/('Polytune' if flavor=='polytune' else 'LadderSym')),str(ROOT/'baselines/common')]
from dataset.dataset_2_random import Dataset
from inference_error import InferenceHandler
from contrib import metrics_utils,note_sequences
DATA=ROOT/'baselines/data/align_synth_20260914';manifest=json.loads((DATA/'manifest.json').read_text())['tracks'];sid=next(k for k,v in sorted(manifest.items()) if v['written_duration_s']>40 and v['split']=='validation');v=manifest[sid]['outputs']
kwargs=dict(root_dir=str(DATA),split_json_path=str(DATA/'split.json'),split='validation',full_recording=True,validation_windows=5,is_deterministic=True,is_randomize_tokens=False,num_rows_per_batch=1,shuffle=False,audio_filename='mix.wav')
if flavor=='laddersym':kwargs.update(use_prompt=True,prompt_length=1024)
with patch.object(Dataset,'_build_dataset',return_value=[]):ds=Dataset(**kwargs)
ds.df=[dict(extra_notes_midi=v['extra'],removed_notes_midi=v['removed'],correct_notes_midi=v['correct'],mistake_audio=v['mistake_wav'],score_audio=v['score_wav'])]
assert len(ds)==5
starts=[];original=ds._random_chunk
# Full-recording mode must never enter the old pre-chunker.
ds._split_frame=lambda *a,**k:(_ for _ in ()).throw(AssertionError('pre-chunker invoked'))
def capture(row):
 out=original(row);starts.append(float(out['mistake_input_times'][0]));return out
ds._random_chunk=capture
with contextlib.redirect_stdout(io.StringIO()):
 items=[ds[i] for i in range(5)];tracks,pa,ra=ds._preprocess_inputs(ds.df[0])
assert starts[0]==0 and all(b>a for a,b in zip(starts,starts[1:]));assert starts[-1]>len(pa)/16000-2.06
# Force training to sample the very last valid start.
ds.is_deterministic=False;ds.validation_windows=1
with patch.object(random,'randint',side_effect=lambda a,b:b),contextlib.redirect_stdout(io.StringIO()):ds[0]
assert starts[-1]==starts[-2]
h=InferenceHandler(model=None,device=torch.device('cpu'),mel_norm=True)
if flavor=='laddersym':h.model=types.SimpleNamespace(config=types.SimpleNamespace(use_prompt=False))
with contextlib.redirect_stdout(io.StringIO()):prep=h._preprocess(pa,ra) if flavor=='polytune' else h._preprocess(pa,ra,v['score_mid'])
for j in [0,1]:assert np.array_equal(items[0][j][0].numpy(),prep[j][0,:256])
# Disabled permutation round-trips adjacent same-pitch notes without corruption.
ds.is_deterministic=True;ds._validation_fraction=0
ns=note_seq.NoteSequence()
for s,e in [(.2,.5),(.5,.8)]:
 n=ns.notes.add();n.pitch=65;n.velocity=90;n.start_time=s;n.end_time=e
tracks=[note_seq.NoteSequence(),note_seq.NoteSequence(),ns]+([copy.deepcopy(ns)] if flavor=='laddersym' else [])
raw=ds._tokenize(tracks,np.zeros(64000,dtype=np.float32),np.zeros(64000,dtype=np.float32),None);raw=original(raw);raw=ds._extract_target_sequence_with_indices(raw,ds.tie_token)
row=ds._run_length_encode_shifts(raw) if flavor=='polytune' else ds.run_length_encode_shifts(raw)
ev=metrics_utils.event_predictions_to_ns([dict(est_tokens=np.asarray(row['targets'],dtype=np.int64),start_time=0.,raw_inputs=[])],codec=ds.codec,encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
notes=sorted((n.instrument,n.pitch,round(n.start_time,3),round(n.end_time,3)) for n in ev['est_ns'].notes)
assert notes==[(3,65,.2,.5),(3,65,.5,.8)] and ev['est_invalid_events']==0
report=dict(model=flavor,status='PASS',sample=sid,duration=len(pa)/16000,validation_start_seconds=starts[:5],last_training_start=starts[5],full_recording_bypasses_prechunker=True,first_window_audio_features_identical=True,adjacent_same_pitch_target_roundtrip=True,token_permutation=False)
(RUN/f'{flavor}_fix_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
