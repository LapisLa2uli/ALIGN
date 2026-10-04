"""Calibrate missed-note operating point on selection clips only, then fixed check.

Uses v7 architecture and all saved transcriber candidates. Threshold selection
never reads DataCreate labels or test sets. This is a second validation-stage
experiment; the check half was already inspected by the initial comparison.
"""
from __future__ import annotations
import sys,os,json,collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'align-model/runs/stack-v7'
sys.path.insert(0,str(ROOT/'align-model/scripts'))
import evaluate_stack_v7 as ev
from alignmodel.joint.stack_v7 import align_outputs,feedback
from alignmodel.joint.presence_verifier_v1 import score_presence
from precision_harness_v4 import remap
from realistic92_aligner_common import load_clip,metric_sample
import numpy as np

THRESHOLDS=(0.5,0.8,0.9,0.96)
def job_worker(job):
    model,dataset,name,path,lineage=job
    c=ev.STATE['candidate']
    with np.load(OUT/'eval-cache'/model/dataset/f'{name}.npz') as cache:outputs={k:cache[k].astype(np.float32) for k in cache.files}
    mel=outputs.pop('mel');presence=lambda queries:score_presence(ev.STATE['presence'],mel,queries,'cpu')
    # Run DP once; gates alone vary.
    index,alignment,_,_,_=align_outputs(outputs,Path(path)/'verified_score.musicxml',c,presence=presence)
    from alignmodel.joint.robust_dp_aligner_v4 import gate_v4
    from alignmodel.joint.robust_dp_aligner_v3 import GateConfig
    evidence={**outputs,'presence':presence,'hop':ev.HOP,'midi_min':52}
    clip=load_clip(Path(path).parent,name,Path(lineage)) if lineage else None
    results=[]
    for threshold in THRESHOLDS:
        events,deletions,info=gate_v4(alignment,index.events,evidence,GateConfig(**(c['gate']|{'missed_threshold':threshold})))
        variant=f'{model}-m{threshold:.2f}'
        payload={'sample':name,'variant':variant,'diagnostics':info,'missed_score_event_indices':sorted(deletions),
                 'repeat_hypothesis':{'source_span':alignment.source_span,'extra_copies':alignment.copies},
                 'events':[{'note_index':e.rendered_index,'pitch':e.pitch,'start':e.start,'end':e.end,'score_span':e.score_span,
                            'relationship':e.relationship,'copy_pass':e.copy_pass,'confidence':e.confidence} for e in events],
                 'feedback':feedback(index,alignment,events,deletions,info)}
        ev.write(OUT/'predictions'/variant/dataset/f'{name}.json',payload)
        if clip:results.append((variant,metric_sample(clip,remap(events,clip),deletions)))
    return dataset,name,results

def main():
    c=ev.read(ROOT/'align-model/runs/precision-v4/CANDIDATE_STACK_V6.json');protocol=ev.read(OUT/'validation_protocol.json')
    jobs=ev.read(ROOT/'reports/current_pipeline_20261003/manifest.json')['jobs']
    synth=[j for j in jobs if j[0]!='datacreate'];dc=[j for j in jobs if j[0]=='datacreate']
    # Both halves stay separately named; all choices below depend ONLY on selection.
    results=collections.defaultdict(list)
    with ProcessPoolExecutor(max_workers=4,initializer=ev.init,initargs=(c,)) as pool:
        for count,(d,n,rows) in enumerate(pool.map(job_worker,[(v,*j) for v in ('baseline','epoch-01','epoch-02') for j in synth]),1):
            half='selection' if n in protocol['halves'][d]['selection'] else 'check'
            for variant,sample in rows: results[(variant,d,half)].append(sample)
            if count%30==0: print('calibration',count,270,flush=True)
    variants=sorted(set(v for v,_,_ in results))
    metrics={v:{h:{d:ev.aggregate(results[(v,d,h)]) for d in c['datasets']} for h in ('selection','check')} for v in variants}
    scores={v:sum(metrics[v]['selection'][d]['error_f1'] for d in c['datasets'])/3 for v in variants}
    selected=max(variants,key=lambda v:(scores[v],float(v.rsplit('-m',1)[1])));model,threshold=selected.rsplit('-m',1)
    report={'metrics':metrics,'selection_scores':scores,'selected':selected,'thresholds':THRESHOLDS,
            'selection_uses':'synthetic selection half only; exact canonical note errors','check_status':'previously inspected validation, not sealed test'}
    ev.write(OUT/'calibration.json',report);print('SELECTED',selected,json.dumps(scores),flush=True)
    checkpoint=protocol['checkpoints'][model]['path']
    new_sources=['src/alignmodel/transcription/transition_v7.py','src/alignmodel/joint/restarts_v4.py',
                 'src/alignmodel/joint/robust_dp_aligner_v4.py','src/alignmodel/joint/stack_v7.py']
    ev.write(OUT/'CANDIDATE_STACK_V7.json',{'schema_version':'align-stack-v7-candidate-v1','checkpoint':checkpoint,
        'checkpoint_sha256':ev.sha(checkpoint),'base_candidate':str(ROOT/'align-model/runs/precision-v4/CANDIDATE_STACK_V6.json'),
        'selected_variant':selected,'selection_metric':'exact canonical score-note content-error F1',
        'decoder':c['decoder'],'gate':{'missed_threshold':float(threshold)},'minimum_match_fraction':.45,
        'code_sha256':{p:ev.sha(ROOT/'align-model'/p) for p in new_sources},'promoted':False,'validation':str(OUT/'calibration.json')})
    ev.cache(checkpoint,model,dc)
    with ProcessPoolExecutor(max_workers=4,initializer=ev.init,initargs=(c,)) as pool:
        for i,result in enumerate(pool.map(job_worker,[(model,*j) for j in dc]),1):
            if i%20==0:print('real',i,94,flush=True)
    ev.real_metrics(selected)
    print('CALIBRATION_COMPLETE',flush=True)
if __name__=='__main__': main()

