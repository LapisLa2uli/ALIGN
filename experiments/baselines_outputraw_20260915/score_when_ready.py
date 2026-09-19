"""Freeze and score each completed model independently; never partial coverage."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'baselines/scripts'))
from evaluate_validation import merge_predictions
RUN=ROOT/'baselines/runs/evaluate_20260915/outputraw358_existing_best'
OUT=ROOT/'baselines/runs/evaluate_20260915/outputraw358_canonical'
OUT.mkdir(exist_ok=True)
state={"state":"waiting_for_inference","models":{}}
processes={};handles={}
while True:
    native=json.loads((RUN/'evaluation_status.json').read_text())
    for name in ('polytune','laddersym'):
        jobs=[j for j in native['jobs'].values() if j['model']==name]
        if any(j['state']=='failed' for j in jobs):
            raise RuntimeError(name+' inference failed')
        if name not in state['models'] and all(j['state']=='complete' for j in jobs):
            expected=sorted(tid for j in jobs for tid in j['expected_ids'])
            assert len(expected)==len(set(expected))==358
            merged=OUT/(name+'_midis')
            merge_predictions(expected,[Path(j['predictions']) for j in jobs],merged)
            command=[str(ROOT/'align-model/runs/env/bin/python'),str(ROOT/'baselines/scripts/evaluate_outputraw_canonical.py'),
                     'freeze','--pred-dir',str(merged),'--scores',str(ROOT/'baselines/data/outputraw_358_recovered'),
                     '--out',str(OUT/(name+'_frozen'))]
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            handles[name]=open(OUT/(name+'_freeze.log'),'x')
            processes[name]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handles[name],stderr=subprocess.STDOUT)
            state['models'][name]={"state":"freezing","pid":processes[name].pid}
        if name in processes and processes[name].poll() is not None:
            code=processes[name].returncode
            handles[name].close()
            if code:
                state['models'][name].update(state='failed',exit_code=code)
                (OUT/'status.json').write_text(json.dumps(state,indent=2))
                raise RuntimeError(name+' canonical step failed')
            if state['models'][name]['state']=='freezing':
                command=[str(ROOT/'align-model/runs/env/bin/python'),str(ROOT/'baselines/scripts/evaluate_outputraw_canonical.py'),
                         'score','--frozen',str(OUT/(name+'_frozen')),'--out',str(OUT/(name+'_results.json'))]
                handles[name]=open(OUT/(name+'_score.log'),'x')
                processes[name]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handles[name],stderr=subprocess.STDOUT)
                state['models'][name].update(state='scoring',pid=processes[name].pid)
            else:
                state['models'][name].update(state='complete')
                del processes[name]
    state['updated_at']=time.time()
    state['state']='complete' if len(state['models'])==2 and all(r['state']=='complete' for r in state['models'].values()) else 'running'
    (OUT/'status.json').write_text(json.dumps(state,indent=2))
    if state['state']=='complete':break
    time.sleep(10)
