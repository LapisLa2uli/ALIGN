"""Detached supervisor: launch two training jobs, monitor disk, score after both complete."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
lock=(RUN/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def write(obj):
 temp=RUN/'status.tmp.json';temp.write_text(json.dumps(obj,indent=2));temp.replace(RUN/'status.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proto=json.loads((RUN/'protocol.json').read_text())
assert sha(Path(proto['train_data'])/'split.json')==proto['training_split_sha256']
for sid,digest in proto['test_gold_sha256'].items():assert sha(Path(proto['test_gold'])/'bundles'/sid/'labels.json')==digest
for rel,digest in json.loads((RUN/'source_hashes.json').read_text()).items():assert sha(ROOT/rel)==digest,rel
for m in ['polytune','laddersym']:
 assert json.loads((RUN/'smoke'/m/'status.json').read_text())['state']=='complete'
 assert json.loads((RUN/f'{m}_fix_checks.json').read_text())['status']=='PASS'
 assert json.loads((RUN/f'{m}_overfit.json').read_text())['note_f1']>=.85
state={'state':'starting','supervisor_pid':os.getpid(),'started_at':time.time(),'jobs':{}}
children={};logs={}
try:
 while True:
  free=shutil.disk_usage(RUN).free/1024**3
  for gpu,m in enumerate(['polytune','laddersym']):
   if m not in children:
    mem=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(mem.strip())<11000 or free<20:
     state['jobs'][m]={'state':'waiting_for_resources','gpu':gpu};continue
    if (RUN/'training'/m).exists():raise FileExistsError('New training directory already exists: '+m)
    cmd=[str(ROOT/f'baselines/envs/{m}/bin/python'),str(RUN/'run_model.py'),'--model',m]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    logs[m]=(RUN/f'{m}_training.log').open('x');children[m]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=logs[m],stderr=subprocess.STDOUT,start_new_session=True)
    state['jobs'][m]={'state':'running','pid':children[m].pid,'gpu':gpu,'command':cmd,'log':str(RUN/f'{m}_training.log'),'paused':False}
   proc=children[m];job=state['jobs'][m]
   status=RUN/'training'/m/'status.json'
   if status.exists():job['progress']=json.loads(status.read_text())
   code=proc.poll()
   if code is not None:
    job['exit_code']=code;job['state']='complete' if code==0 and job.get('progress',{}).get('state')=='complete' else 'failed'
   elif free<12 and not job['paused']:
    os.killpg(proc.pid,signal.SIGSTOP);job['paused']=True;job['state']='paused_low_disk'
   elif free>=20 and job['paused']:
    os.killpg(proc.pid,signal.SIGCONT);job['paused']=False;job['state']='running'
  state.update(updated_at=time.time(),free_GiB=free,state='training_and_evaluation');write(state)
  if len(children)==2 and all(p.poll() is not None for p in children.values()):break
  time.sleep(20)
 if not all(j['state']=='complete' for j in state['jobs'].values()):
  state['state']='failed';write(state);sys.exit(1)
 state['state']='scoring_test';write(state)
 subprocess.run([str(ROOT/'align-model/runs/env/bin/python'),str(RUN/'finalize.py')],cwd=ROOT,check=True)
 state.update(state='complete',finished_at=time.time(),results=str(RUN/'test_results/RESULTS.md'));write(state)
except Exception:
 state.update(state='supervisor_failed',traceback=traceback.format_exc(),updated_at=time.time());write(state);raise
