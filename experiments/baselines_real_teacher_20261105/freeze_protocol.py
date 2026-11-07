"""Freeze training and evaluation choices before model experiments begin."""
import platform
import sys
from common import ROOT,RUN,TEST,SIZES,SEED,read,write,sha

def main():
    if (RUN/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    caches={m:read(RUN/'cache'/m/'manifest.json') for m in ['polytune','laddersym']}
    order=read(RUN/'eligible.json')['ordered_ids']
    eligible=[sid for sid in order if all(caches[m][sid]['windows']>0 for m in caches)]
    if len(eligible)<50:raise ValueError(f'Only {len(eligible)} recordings have assessed windows')
    selected=eligible[:50];assert not set(selected)&set(TEST)
    source=ROOT/'experiments/baselines_retrain_20260918/source'
    files=list(source.rglob('*.py'))+list((ROOT/'baselines/common').glob('*.py'))+[ROOT/'baselines/scripts/label_datacreate_baselines.py']
    p={'schema':'teacher-real-learning-curve-v1','seed':SEED,'sizes':SIZES,'train_order':selected,
       'train_subsets':{str(n):selected[:n] for n in SIZES},'test_ids':TEST,
       'initial_checkpoints':read(RUN/'inputs.json')['initial_checkpoints'],
       'optimizer':{'name':'AdamW','lr':1e-5,'weight_decay':.01,'steps':200,'warmup_steps':20,'gradient_accumulation':4,'clip_grad_norm':1.0},
       'training_precision':'bf16 autocast; float32 weights/optimizer','selection':'final step, fixed before test scoring; no test-based early stopping',
       'finetune_scope':'All upstream trainable parameters; retain authors fixed positional embeddings','sampling':'uniform recordings, then uniform cached assessed windows; fixed total updates across sizes',
       'target_conversion':'published teacher labels only; wrong=extra+missing; repetition=extra; matched=correct; exclude windows with withheld/unassessed notes',
       'inference':{'precision':'float32','batch_size':1,'max_length':1024,'prompt_length':1024,'laddersym_prompted':True},
       'metric':'strict canonical affected-score-note error F1; official partial-type credit secondary; timestamps not primary',
       'reference':'AudioEval+MuScriptor model-generated pseudo-gold; measures teacher agreement, not independent human accuracy',
       'test_abstentions':[sid for sid in TEST if read(RUN/'teacher_gold'/f'{sid}.json')['status']!='ok'],
       'source_sha256':{str(path.relative_to(ROOT)):sha(path) for path in files},
       'experiment_code_sha256':{p.name:sha(p) for p in RUN.glob('*.py')},
       'cache_manifest_sha256':{m:sha(RUN/'cache'/m/'manifest.json') for m in caches},
       'teacher_gold_sha256':{sid:sha(RUN/'teacher_gold'/f'{sid}.json') for sid in TEST},
       'audio_bounds_audit_sha256':sha(RUN/'teacher_bounds_audit.json'),
       'inputs_sha256':sha(RUN/'inputs.json'),'environment':{'python':sys.executable,'platform':platform.platform()},
       'replicates':1,'limitations':['One fixed nested subset order and training seed; no seed-variance estimate','Sparse teacher labels, estimated missing-note timing','Existing pretrained models and baseline label adapter retained']}
    write(RUN/'protocol.json',p)
    print('FROZEN',selected,'test abstentions',p['test_abstentions'],flush=True)

if __name__=='__main__':main()
