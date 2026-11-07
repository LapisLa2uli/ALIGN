"""Sequential GPU campaign; selection finishes before any test generation."""
import os
import subprocess
import sys
from common import ROOT,RUN,SIZES,read,write,sha

def run(arguments,name):
    log=RUN/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    write(RUN/'progress.json',{'stage':name,'command':arguments})
    print('START',name,flush=True)
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,*arguments],cwd=ROOT,env={**os.environ,'PYTHONNOUSERSITE':'1'},stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)

def main():
    p=read(RUN/'protocol.json')
    for rel,digest in p['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    engine=str(RUN/'engine.py')
    for model in ['polytune','laddersym']:
        run([engine,'infer','--model',model,'--count','0','--partition','validation'],model+'_validation_control')
        for n in SIZES:
            run([engine,'train','--model',model,'--count',str(n)],f'{model}_train_{n}')
            for step in [100,200,300]:
                run([engine,'infer','--model',model,'--count',str(n),'--step',str(step),'--partition','validation'],f'{model}_validate_{n}_{step}')
    selected={}
    for model in ['polytune','laddersym']:
        selected[model]={};base=read(RUN/'predictions/validation'/model/'n00/step000/metrics.json')
        for n in SIZES:
            candidates=[{'step':0,'f1':base['strict']['f1'],'capped_rate':base['capped_rate'],'eligible':True}]
            for step in [100,200,300]:
                r=read(RUN/'predictions/validation'/model/f'n{n:02}'/f'step{step:03}'/'metrics.json')
                candidates.append({'step':step,'f1':r['strict']['f1'],'capped_rate':r['capped_rate'],'eligible':r['capped_rate']<=base['capped_rate']+.005})
            best=max([c for c in candidates if c['eligible']],key=lambda c:(c['f1'],-c['step']))
            selected[model][str(n)]={'step':best['step'],'validation_f1':best['f1'],'candidates':candidates,'test_used':False}
    write(RUN/'selection.json',selected);print('SELECTED',selected,flush=True)
    for model in ['polytune','laddersym']:
        run([engine,'infer','--model',model,'--count','0','--partition','test'],model+'_test_control')
        for n in SIZES:
            step=selected[model][str(n)]['step']
            if step:run([engine,'infer','--model',model,'--count',str(n),'--step',str(step),'--partition','test'],f'{model}_test_{n}')
    run([str(RUN/'report_results.py')],'score_and_report')
    run([str(RUN/'audit_final.py')],'final_integrity')
    write(RUN/'progress.json',{'stage':'complete','selection_sha256':sha(RUN/'selection.json')})
    print('CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':main()
