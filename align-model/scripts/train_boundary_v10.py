"""Fit a v10 boundary veto from archived waveform-derived synthetic controls.

E: and raw acoustic caches are unavailable. This is a small development-data
experiment, not new E: training or a sealed/independent accuracy benchmark.
"""
import os, sys, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'align-model/runs/stack-v10'
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'align-model/runs/stack-v9/numba-cache'))
sys.path.insert(0,str(ROOT/'align-model/src'))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from alignmodel.transcription.same_pitch_v3 import FEATURE_NAMES,boundary_features,repeat_score

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 source=ROOT/'align-model/runs/stack-v9/control_features.json'
 cases=read(source);parts={};manifest={'source':str(source),'sha256':sha(source),'raw_E_available':False,
  'status':'reused historical development controls, not an unseen test','datacreate_selection':False,'groups':{}}
 for dataset in sorted({r['dataset'] for r in cases}):
  names=sorted({r['sample'] for r in cases if r['dataset']==dataset})
  for i,name in enumerate(names):parts[dataset,name]=('fit','fit','selection','check')[i%4]
  manifest['groups'][dataset]={s:[n for n in names if parts[dataset,n]==s] for s in ('fit','selection','check')}
 fit=[r for r in cases if parts[r['dataset'],r['sample']]=='fit' and boundary_features(r.get('evidence')) is not None]
 x=np.array([boundary_features(r['evidence']) for r in fit]);y=np.array([r['kind']=='genuine_repeat' for r in fit])
 clf=RandomForestClassifier(n_estimators=96,max_depth=5,min_samples_leaf=3,class_weight='balanced',random_state=20261005,n_jobs=4).fit(x,y)
 model={'schema_version':'same-pitch-boundary-forest-v1','feature_names':list(FEATURE_NAMES),'trees':[]}
 for estimator in clf.estimators_:
  t=estimator.tree_;v=t.value[:,0,:]
  model['trees'].append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),
   'threshold':t.threshold.tolist(),'repeat_fraction':(v[:,1]/v.sum(1)).tolist()})
 assert np.allclose([repeat_score(v,model) for v in x],clf.predict_proba(x)[:,1],atol=1e-12)
 write(OUT/'boundary_model.json',model)
 # The classifier is auxiliary evidence. Score-supported and non-regression
 # checks in stack_v10 prevent it from becoming a global split/merge switch.
 threshold=.5
 report={}
 for split in ('fit','selection','check'):
  report[split]={}
  for kind in ('genuine_repeat','injected_split'):
   rows=[r for r in cases if parts[r['dataset'],r['sample']]==split and r['kind']==kind]
   known=[r for r in rows if boundary_features(r.get('evidence')) is not None]
   report[split][kind]={'count':len(rows),'with_features':len(known),
    'acoustic_repeat_votes':sum(repeat_score(boundary_features(r['evidence']),model)>=threshold for r in known)}
 manifest['fit_controls']=len(fit);manifest['threshold']=threshold
 manifest['threshold_policy']='fixed 0.5 class vote; guarded by score-supported region and outside-output equality'
 write(OUT/'training_manifest.json',manifest)
 write(OUT/'boundary_control_results.json',{'metrics':report,'feature_importance':dict(zip(FEATURE_NAMES,clf.feature_importances_.tolist())),
  'warning':'Acoustic-only votes are not deployed decisions. Controls are reused development data.'})
 old=read(ROOT/'align-model/runs/stack-v9/CANDIDATE_STACK_V9.json')
 candidate={**old,'schema_version':'align-stack-v10-candidate-v1','selected_variant':'score-supported-boundary-v1',
  'boundary_model':str(OUT/'boundary_model.json'),'boundary_model_sha256':sha(OUT/'boundary_model.json'),
  'boundary':{'repeat_threshold':threshold},'pipeline_revision':'v10-boundary-v1','promoted':False,
  'calibration':'Archived synthetic control features, sample-separated fit/check; E: unavailable. DataCreate not used.',
  'validation':str(OUT/'evaluation.json')}
 candidate['code_sha256']=dict(old['code_sha256'])
 for relative in ('src/alignmodel/transcription/same_pitch_v3.py','src/alignmodel/joint/stack_v10.py',
  'src/alignmodel/joint/stack_v9_passage.py','src/alignmodel/joint/passage_v1.py','src/alignmodel/joint/robust_dp_aligner_v5.py'):
  candidate['code_sha256'][relative]=sha(ROOT/'align-model'/relative)
 write(OUT/'CANDIDATE_STACK_V10.json',candidate)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
