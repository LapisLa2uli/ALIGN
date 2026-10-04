"""Select v7 by canonical error identity on synthetic validation, then inspect real data.

Splits the prior 90-clip validation audit into fixed selection/check halves.
No E: test clips or DataCreate >=095 are read. No DataCreate labels select weights.
"""
from __future__ import annotations
import os,sys,json,hashlib,random,collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'align-model/runs/stack-v7'
os.environ.setdefault('NUMBA_CACHE_DIR',str(OUT/'numba-cache'))
for p in ('align-model/src','align-model/scripts','DataCreate/src','synth-pipeline/src'): sys.path.insert(0,str(ROOT/p))
import numpy as np
import torch
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.joint.presence_verifier_v1 import load_verifier,score_presence
from alignmodel.joint.stack_v7 import align_outputs,feedback,HOP
from alignmodel.joint.robust_dp_aligner_v3 import align_v3,AlignerV3Config,gate_alignment,GateConfig
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.metrics import evaluate_joint_dataset
from alignmodel.melody import micro_note_wise,load_bundle_notes,labels_with_canonical_locations
from datacreate.melody import match_note_wise_labels_detail
from precision_harness_v4 import decode_rows,remap,rms_db
from realistic92_aligner_common import load_clip,metric_sample
from audit_stack_v3_val_errors import _type_samples
from eval_datacreate_vs_human import human_population,labels_from_stack_alignment

STATE={}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def init(candidate):
    torch.set_num_threads(1); STATE['candidate']=candidate; STATE['presence']=load_verifier(Path(candidate['verifier']),'cpu')
def worker(job):
    variant,dataset,name,path,lineage=job
    try:
        c=STATE['candidate']; cache_name='baseline' if variant in ('baseline','aligner-only') else variant
        with np.load(OUT/'eval-cache'/cache_name/dataset/f'{name}.npz') as cache: outputs={k:cache[k].astype(np.float32) for k in cache.files}
        mel=outputs.pop('mel'); presence=lambda queries:score_presence(STATE['presence'],mel,queries,'cpu')
        score_path=Path(path)/'verified_score.musicxml'
        if variant=='baseline':
            index=ScoreEventIndex.from_musicxml(score_path)
            rows=decode_rows(outputs['ctc'],52,c['decoder'])
            alignment=align_v3(rows,index.events,score_path,AlignerV3Config(costs=RobustDPCosts(**c['aligner_costs']),**c['aligner_v3']))
            events,deletions,info=gate_alignment(alignment,index.events,{**outputs,'presence':presence,'hop':HOP,'midi_min':52},GateConfig(**c['gate']))
            info.update(status='ok')
        else:
            index,alignment,events,deletions,info=align_outputs(outputs,score_path,c,presence=presence)
        payload={'sample':name,'variant':variant,'diagnostics':info,'missed_score_event_indices':sorted(deletions),
                 'repeat_hypothesis':{'source_span':alignment.source_span,'extra_copies':alignment.copies},
                 'events':[{'note_index':e.rendered_index,'pitch':e.pitch,'start':e.start,'end':e.end,'score_span':e.score_span,
                            'relationship':e.relationship,'copy_pass':e.copy_pass,'confidence':e.confidence} for e in events]}
        if variant!='baseline': payload['feedback']=feedback(index,alignment,events,deletions,info)
        write(OUT/'predictions'/variant/dataset/f'{name}.json',payload)
        if lineage:
            clip=load_clip(Path(path).parent,name,Path(lineage))
            return variant,dataset,name,metric_sample(clip,remap(events,clip),deletions),None
        return variant,dataset,name,None,None
    except Exception:
        import traceback
        return variant,dataset,name,None,traceback.format_exc()

def aggregate(samples):
    combined=evaluate_joint_dataset(samples,tolerances_sec=())['aggregate']['official_note_wise']
    kinds={k:evaluate_joint_dataset(_type_samples(samples,k),tolerances_sec=())['aggregate']['official_note_wise'] for k in ('match','copy','substitute','extra','missed_note')}
    subset=[kinds[k] for k in ('substitute','extra','missed_note')]
    p=sum(x['predicted'] for x in subset); g=sum(x['gold'] for x in subset); credit=sum(x['credit'] for x in subset)
    return {'combined':combined,'per_type':kinds,'error_f1':2*credit/(p+g) if p+g else 1}

def core(label):
    part=label.get('score_part') or {}; a=part.get('core_start_note_index'); b=part.get('core_end_note_index')
    if a is None:
        if 'start_note_index' not in part:return None
        a=part['start_note_index']+part.get('pad_notes',0); b=part['end_note_index']-part.get('pad_notes',0)
    return list(range(a,max(a,b)+1))
def real_metrics(variant):
    kinds=('extra_note','missed_note','wrong_note','repetition'); reports=collections.defaultdict(list); details=[]
    for sample,gold in human_population(ROOT/'DataCreate/samples'):
        if not sample.name.isdigit() or not 1<=int(sample.name)<=94: continue
        prediction=read(OUT/'predictions'/variant/'datacreate'/f'{sample.name}.json')
        notes=load_bundle_notes(sample)
        pred=labels_from_stack_alignment(prediction,notes) if variant=='baseline' else prediction['feedback']['labels']
        # Actual affected score notes, excluding decorative/context padding.
        # Keep existing explicit human core indices; never infer identity from time.
        def canonical(labels):
            result=[]
            for l in labels:
                if l['type'] not in kinds:continue
                indices=l.get('score_event_indices') or core(l)
                v=dict(l)
                if indices:v['score_event_indices']=indices
                if v['type']=='repetition' and v.get('extra_copies') is None:v['extra_copies']=1
                result.append(v)
            return result
        g,p=canonical(gold),canonical(pred)
        row={'sample':sample.name,'status':prediction['diagnostics']['status'],'gold':g,'predicted':p,'metrics':{}}
        for k in kinds:
            metric=match_note_wise_labels_detail([x for x in g if x['type']==k],[x for x in p if x['type']==k],score_event_count=len(notes))
            reports[k].append(metric); row['metrics'][k]=metric
        details.append(row)
    result={'population':len(details),'identity':'exact affected score-event set plus type; excludes context padding; no timestamp matching',
        'summary':{k:micro_note_wise(reports[k]) for k in kinds},'per_sample':details}
    write(OUT/f'datacreate-{variant}.json',result)
    print('DATACREATE',variant,json.dumps(result['summary']),flush=True)

def cache(model_path,variant,jobs):
    todo=[j for j in jobs if not(OUT/'eval-cache'/variant/j[0]/f'{j[1]}.npz').exists()]
    if not todo:return
    torch.set_num_threads(4); device='cuda'
    model,_=load_dual_checkpoint(model_path,device); model.eval()
    for pos,(dataset,name,path,lineage) in enumerate(todo,1):
        audio=load_audio_mono(Path(path)/'performance_audio.wav',22050)
        with torch.inference_mode():
            mel,_=extract_dual_mel(audio,device); result=infer_dual_outputs(model,np.asarray(mel,np.float32),device)
        dest=OUT/'eval-cache'/variant/dataset/f'{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**{k:v.astype(np.float16) for k,v in result.items()},mel=np.asarray(mel,np.float32),rms_db=rms_db(audio,len(result['ctc'])))
        if pos%15==0 or pos==len(todo):print('transcribed',variant,pos,len(todo),flush=True)
    del model;torch.cuda.empty_cache()

def main():
    c=read(ROOT/'align-model/runs/precision-v4/CANDIDATE_STACK_V6.json')
    audit=read(ROOT/'reports/current_pipeline_20261003/manifest.json')
    synth=[j for j in audit['jobs'] if j[0]!='datacreate']
    dc=[j for j in audit['jobs'] if j[0]=='datacreate']
    halves={}
    for d in c['datasets']:
        names=[j[1] for j in synth if j[0]==d]
        halves[d]={'selection':names[::2],'check':names[1::2]}
    training=read(OUT/'training_manifest.json')
    for d in c['datasets']:
        assert not(set(training['families'][d]['selected']) & set(halves[d]['selection']+halves[d]['check']))
    checkpoints={'baseline':c['checkpoint'],**{p.stem:str(p) for p in sorted(OUT.glob('epoch-*.pt'))}}
    if len(checkpoints)<3:raise RuntimeError('Wait for both training epochs')
    write(OUT/'validation_protocol.json',{'halves':halves,'checkpoints':{k:{'path':v,'sha256':sha(v)} for k,v in checkpoints.items()},
          'selection':'macro over three families of exact canonical substitute/extra/missed micro F1','test_read':False,'datacreate_selection':False})
    for variant,path in checkpoints.items():cache(path,variant,synth)
    variants=['baseline','aligner-only',*[v for v in checkpoints if v!='baseline']]
    jobs=[(v,*j) for v in variants for j in synth]
    results=collections.defaultdict(list); errors={}
    with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(c,)) as pool:
        for i,(v,d,n,s,error) in enumerate(pool.map(worker,jobs),1):
            if error: errors[f'{v}/{d}/{n}']=error
            else: results[(v,d,'selection' if n in halves[d]['selection'] else 'check')].append(s)
            if i%30==0:print('aligned',i,len(jobs),flush=True)
    if errors:
        write(OUT/'evaluation_errors.json',errors);raise RuntimeError(f'{len(errors)} evaluation failures')
    metrics={v:{h:{d:aggregate(results[(v,d,h)]) for d in c['datasets']} for h in ('selection','check')} for v in variants}
    scores={v:sum(metrics[v]['selection'][d]['error_f1'] for d in c['datasets'])/3 for v in variants}
    selected=max((v for v in variants if v!='baseline'),key=lambda v:scores[v])
    outcome={'metrics':metrics,'selection_scores':scores,'selected_v7':selected,
             'beats_baseline_selection':scores[selected]>scores['baseline']}
    write(OUT/'validation.json',outcome);print('VALIDATION',json.dumps(scores),'selected',selected,flush=True)
    path=checkpoints['baseline'] if selected=='aligner-only' else checkpoints[selected]
    write(OUT/'CANDIDATE_STACK_V7.json',{'schema_version':'align-stack-v7-candidate-v1','checkpoint':str(Path(path).resolve()),
        'checkpoint_sha256':sha(path),'base_candidate':str(ROOT/'align-model/runs/precision-v4/CANDIDATE_STACK_V6.json'),
        'selected_variant':selected,'selection_metric':'exact score-note content error F1','decoder':c['decoder'],
        'minimum_match_fraction':.45,'promoted':False,'validation':str(OUT/'validation.json')})
    cache(checkpoints['baseline'],'baseline',dc)
    if selected!='aligner-only':cache(path,selected,dc)
    with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(c,)) as pool:
        for i,(v,d,n,s,error) in enumerate(pool.map(worker,[(v,*j) for v in ('baseline',selected) for j in dc]),1):
            if error:errors[f'{v}/{d}/{n}']=error
            if i%30==0:print('real aligned',i,188,flush=True)
    if errors:write(OUT/'evaluation_errors.json',errors);raise RuntimeError('DataCreate failure')
    real_metrics('baseline');real_metrics(selected)
    print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__': main()
