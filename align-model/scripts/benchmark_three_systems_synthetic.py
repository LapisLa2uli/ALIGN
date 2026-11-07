"""Frozen synthetic comparison: deployed AudioEval+MuScriptor vs author baselines.

Stages separate input selection, gold-free inference, and canonical error scoring.
Run with the existing baselines/envs/polytune Python for inference stages.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/three_systems_synthetic_20261105'
OLD = ROOT / 'reports/muscriptor_synthetic_20261024'
for directory in ('align-model/src', 'DataCreate/src', 'synth-pipeline/src', 'align-model/scripts',
                  'baselines/common', 'baselines/scripts', 'experiments/baselines_retrain_20260918'):
    sys.path.insert(0, str(ROOT / directory))
KINDS = ('wrong_note', 'missed_note', 'extra_note', 'repetition')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def local(path):
    text = str(path).replace('\\','/')
    if '/ALIGN/' in text: return ROOT / text.split('/ALIGN/',1)[1]
    return Path(path)

def checkpoints():
    base = ROOT/'experiments/baselines_datacreate_095_124_20260921/checkpoints'
    return dict(polytune=base/'polytune/CocoChorales-E/last.ckpt',
                laddersym=base/'laddersym/checkpoints/cocochorales/prompted/model.ckpt')

def core_labels(labels, score_count):
    """Use affected notes, excluding display-only context padding, for every system."""
    from datacreate.melody import is_repeated_pass
    output=[]
    for original in labels:
        if original['type'] not in KINDS or is_repeated_pass(original): continue
        row=dict(original)
        part=row.get('score_part') or {}
        if part:
            pad=int(part.get('pad_notes',0))
            first=int(part.get('core_start_note_index',part['start_note_index']+pad))
            last=int(part.get('core_end_note_index',part['end_note_index']-pad))
            if 0<=first<=last<score_count: row['score_event_indices']=list(range(first,last+1))
            else: raise ValueError('Invalid affected-note range')
        if row['type']=='repetition' and row.get('extra_copies') is None: row['extra_copies']=1
        output.append(row)
    return output

def prepare():
    from datacreate.melody import parse_sounding_notes, canonical_note_location
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    previous=read(OLD/'manifest.json')
    eligible=set(read(ROOT/'align-model/runs/realistic92-aligner-v1/ALIGNER_FREEZE.json')['eligible']['test'])
    pool=sorted([r for r in previous['samples'] if r['corpus']=='realistic92' and r['sample'] in eligible],key=lambda r:r['sample'])
    selected=sorted(random.Random(20261105).sample(pool,30),key=lambda r:r['sample'])
    records=[]; gold=[]
    for row in selected:
        sample=Path(row['path']);name=row['sample']
        for file,digest in row['hashes'].items(): assert sha(sample/file)==digest
        notes=parse_sounding_notes(sample/'verified_score.musicxml')
        labels=read(sample/'labels.json')['labels']
        for label in labels:
            if label['type'] not in KINDS: continue
            part=label.get('score_part') or {}
            if part and label.get('pitches'):
                a,b=part['start_note_index'],part['end_note_index']
                assert [n.pitch for n in notes[a:b+1]]==label['pitches'], f'Gold pitch/range mismatch: {name}'
        canonical=core_labels(labels,len(notes))
        assert all(canonical_note_location(l,score_event_count=len(notes)) is not None for l in canonical)
        hashes={f:sha(sample/f) for f in ['performance_audio.wav','reference_audio.wav','reference_audio.mid','verified_score.musicxml','metadata.json']}
        pred=OLD/'predictions/realistic92'/f'{name}.json'
        assert read(pred)['audio_sha256']==hashes['performance_audio.wav']
        for source,dest in [('performance_audio.wav','performance_16k.wav'),('reference_audio.wav','reference_16k.wav')]:
            audio,sr=sf.read(sample/source,dtype='float32',always_2d=True)
            audio=audio.mean(axis=1); factor=gcd(sr,16000)
            audio=resample_poly(audio,16000//factor,sr//factor).astype(np.float32)
            path=OUT/'inputs'/name/dest;path.parent.mkdir(parents=True,exist_ok=True)
            sf.write(path,audio,16000,subtype='FLOAT')
        records.append(dict(sample=name,path=str(sample),written_shift=row['written_shift'],hashes=hashes,
                            score_event_count=len(notes),muscriptor_prediction=str(pred),muscriptor_prediction_sha256=sha(pred)))
        gold.append(dict(sample=name,labels=canonical,labels_sha256=sha(sample/'labels.json'),score_event_count=len(notes),
                         omitted_types=sorted(set(l['type'] for l in labels)-set(KINDS))))
    manifest=dict(seed=20261105,selection='30 random clips from the 87 cached dataset-9.2 test clips passing prior lineage audit',
                  metric='Canonical affected-score-note content-error micro F1, exact location+type credit 1, location-only credit .5',
                  error_types=KINDS,context_padding_excluded=True,test_tuning=False,baseline_weights='author pretrained CocoChorales-E / CocoChorales prompted; not ALIGN retrained',
                  checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in checkpoints().items()},samples=records)
    write(OUT/'manifest.json',manifest);write(OUT/'gold.json',gold)
    print('Prepared and audited',len(records),'clips;',sum(len(r['labels']) for r in gold),'gold content-error labels',flush=True)

def deny_gold(event, args):
    if event=='open' and args:
        p=str(args[0]).replace('\\','/')
        if p.endswith(('/labels.json','/gold.json','/note_map.json')): raise PermissionError('Inference must not read gold: '+p)

def baseline(flavor, limit=None):
    import numpy as np
    import soundfile as sf
    import torch
    manifest=read(OUT/'manifest.json');ckpt=manifest['checkpoints'][flavor]
    assert sha(ckpt['path'])==ckpt['sha256']
    os.environ['LADDERSYM_DETERMINISTIC_PROMPT']='1'
    os.environ['LADDERSYM_PROMPT_LENGTH']='1024'
    os.environ['LADDERSYM_USE_CACHE']='1'
    # PyTorch 2.6 defaults weights_only=True; these verified local Lightning
    # artifacts include training metadata. Explicitly load the model state only.
    import align_runtime
    def strict_load(model,path):
        state=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
        assert all(k.startswith('model.') for k in state)
        model.load_state_dict({k[6:]:v for k,v in state.items()},strict=True)
    align_runtime.load_model_weights=strict_load
    from infer_datacreate import build_model
    torch.set_num_threads(4);torch.manual_seed(365);np.random.seed(365)
    model=build_model(flavor,Path(ckpt['path'])).cuda().eval()
    from inference_error import InferenceHandler
    handler=InferenceHandler(model=model,device='cuda',mel_norm=True)
    sys.addaudithook(deny_gold)
    jobs=manifest['samples'][:limit] if limit else manifest['samples']
    for i,row in enumerate(jobs,1):
        name=row['sample'];dest=OUT/'predictions'/flavor/name/'mix.mid'
        if dest.exists() and dest.with_suffix('.done.json').exists(): continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        sample=Path(row['path']);perf,sr=sf.read(OUT/'inputs'/name/'performance_16k.wav',dtype='float32')
        ref,rr=sf.read(OUT/'inputs'/name/'reference_16k.wav',dtype='float32');assert sr==rr==16000
        kwargs=dict(mistake_audio=perf,score_audio=ref,audio_path=name,outpath=dest.as_posix(),batch_size=1,max_length=1024)
        if flavor=='laddersym':kwargs['prompt_path']=str(sample/'reference_audio.mid')
        began=time.perf_counter()
        with (dest.parent/'inference.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log),torch.inference_mode():
            handler.inference(**kwargs)
        assert dest.is_file()
        seconds=time.perf_counter()-began
        write(dest.with_suffix('.done.json'),dict(seconds=seconds,sha256=sha(dest),precision='float32',checkpoint_sha256=ckpt['sha256']))
        print(flavor,i,len(jobs),name,round(seconds,1),'seconds',flush=True)

def audioeval(limit=None):
    import numpy as np
    import torch
    from alignmodel.transcription.mel_v1 import load_audio_mono
    from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
    from alignmodel.joint.presence_verifier_v1 import load_verifier,score_presence
    from alignmodel.joint.stack_v9_passage import align_notes,feedback
    from alignmodel.joint.index import ScoreEventIndex
    from precision_harness_v4 import rms_db
    manifest=read(OUT/'manifest.json');c=read(ROOT/'align-model/runs/stack-v9/CANDIDATE_STACK_V9.json');base=read(local(c['base_candidate']))
    assert sha(local(c['checkpoint']))==c['checkpoint_sha256']
    torch.set_num_threads(4);model,_=load_dual_checkpoint(local(c['checkpoint']),'cuda');model.eval()
    verifier=load_verifier(local(base['verifier']),'cuda')
    config={k:c[k] for k in ('decoder','gate','minimum_match_fraction','same_pitch')}
    write(OUT/'audioeval_config.json',dict(candidate=c,base_candidate=base,model='MuScriptor medium + AudioEval v9 passage mapper and acoustic gates',
         mu_weights_sha256=read(manifest['samples'][0]['muscriptor_prediction'])['model_sha256']))
    sys.addaudithook(deny_gold)
    jobs=manifest['samples'][:limit] if limit else manifest['samples']
    for i,row in enumerate(jobs,1):
        name=row['sample'];dest=OUT/'predictions/audioeval'/f'{name}.json'
        if dest.exists(): continue
        sample=Path(row['path']);began=time.perf_counter()
        assert sha(row['muscriptor_prediction'])==row['muscriptor_prediction_sha256']
        raw=read(row['muscriptor_prediction'])['notes']
        rows=[(n['pitch']+row['written_shift'],n['start'],n['end'],1.0) for n in raw]
        audio=load_audio_mono(sample/'performance_audio.wav',22050)
        with torch.inference_mode():
            mel,_=extract_dual_mel(audio,'cuda');outputs=infer_dual_outputs(model,np.asarray(mel,np.float32),'cuda')
        outputs={k:v.astype(np.float16).astype(np.float32) for k,v in outputs.items()}
        outputs['rms_db']=rms_db(audio,len(outputs['ctc']))
        presence=lambda queries:score_presence(verifier,np.asarray(mel,np.float32),queries,'cuda')
        index=ScoreEventIndex.from_musicxml(sample/'verified_score.musicxml')
        try:
            alignment,events,deletions,info=align_notes(rows,index,sample/'verified_score.musicxml',base,
                 evidence={**outputs,'hop':256/22050,'midi_min':52,'presence':presence},config=config)
            doc=feedback(index,alignment,events,deletions,info)
        except Exception as error:
            # Retain failures in the fixed population as no usable predictions.
            # Do not tune/fix the frozen pipeline using test outcomes.
            events=[];info=dict(status='inference_failed',error=f'{type(error).__name__}: {error}')
            doc=dict(labels=[],diagnostics=info,status=info['status'])
        write(dest,dict(sample=name,feedback=doc,seconds=time.perf_counter()-began,
                        events=[dict(pitch=e.pitch,start=e.start,end=e.end,score_span=e.score_span,relationship=e.relationship,copy_pass=e.copy_pass) for e in events]))
        print('audioeval',i,len(jobs),name,info['status'],len(doc['labels']),'labels',round(time.perf_counter()-began,1),'seconds',flush=True)

def score():
    from datacreate.melody import match_note_wise_labels_detail
    from label_datacreate_baselines import document_from_midi
    import numpy as np
    manifest=read(OUT/'manifest.json');gold={r['sample']:r for r in read(OUT/'gold.json')};results={}
    for model in ('audioeval','polytune','laddersym'):
        rows=[]
        for row in manifest['samples']:
            name=row['sample'];sample=Path(row['path']);g=gold[name]
            if model=='audioeval':labels=read(OUT/'predictions/audioeval'/f'{name}.json')['feedback']['labels'];unclassified=0
            else:
                midi=OUT/'predictions'/model/name/'mix.mid'
                assert sha(midi)==read(midi.with_suffix('.done.json'))['sha256']
                doc=document_from_midi(midi,sample/'verified_score.musicxml',model=model,sample_id=name)
                labels=doc['labels'];unclassified=doc['baseline_labeling']['unclassified_count']
                write(OUT/'converted'/model/f'{name}.json',doc)
            pred=core_labels(labels,g['score_event_count'])
            # Unclassified native notes are predictions, never silently discarded.
            pred += [dict(type='unclassified') for _ in range(unclassified)]
            metric=match_note_wise_labels_detail(g['labels'],pred,score_event_count=g['score_event_count'])
            assert metric['status']=='available'
            rows.append(dict(sample=name,**metric))
        counts=np.array([[r[k] for k in ('credit','predicted','gold')] for r in rows])
        credit,p,g=counts.sum(axis=0);rng=np.random.default_rng(20261105);boot=[]
        for _ in range(2000):
            a,b,c=counts[rng.integers(len(rows),size=len(rows))].sum(axis=0)
            boot.append(2*a/(b+c) if b+c else 1)
        results[model]=dict(clips=len(rows),credit=credit,predicted=int(p),gold=int(g),precision=credit/p if p else 0,
                            recall=credit/g if g else 0,f1=2*credit/(p+g) if p+g else 1,
                            ci95=list(map(float,np.quantile(boot,[.025,.975]))),per_clip=rows)
        print(model,{k:v for k,v in results[model].items() if k!='per_clip'},flush=True)
    write(OUT/'results.json',dict(protocol=manifest,results=results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audioeval','polytune','laddersym','score']);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='audioeval':audioeval(a.limit)
    elif a.stage=='score':score()
    else:baseline(a.stage,a.limit)
