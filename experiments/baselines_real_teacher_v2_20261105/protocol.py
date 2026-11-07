"""Freeze all selections and inputs before the test follow-up."""
from common import ROOT,RUN,PREVIOUS,SIZES,SEED,read,write,sha,split
from reference_alignment import PARAMETERS

SETTINGS={'lr':1e-4,'weight_decay':.01,'steps':300,'warmup_steps':20,'accumulation':4,
          'clip_grad_norm':1.,'kl_weight':.2,'candidates':[0,100,200,300],
          'lora_rank':4,'lora_alpha':8,'lora_dropout':.05}

def configuration():
    old=read(PREVIOUS/'protocol.json');parts=split()
    return {'schema':'teacher-real-v2','seed':SEED,'sizes':SIZES,'split':parts,
       'initial_checkpoints':old['initial_checkpoints'],'optimizer':SETTINGS,
       'alignment':PARAMETERS,'selection':'Highest strict reference-note F1 on 12 validation recordings among 0/100/200/300; require capped-window rate <= baseline + .005; ties prefer earlier step. Step 0 is an explicit no-adaptation fallback.',
       'sampling':'Half error-containing assessed windows and half clean, alternating microsteps; uniform eligible recording within stratum, then uniform window. Fixed update budget across dataset sizes.',
       'inference':{'precision':'float32','max_length':1024,'batch_size':1,'prompt_length':1024,'clip_to_audio_duration':True},
       'reference':'AudioEval v9 + MuScriptor medium pseudo-gold, not human ground truth',
       'metric':'strict canonical reference-score-note identity and error type; timestamps are not the main metric',
       'test_abstentions':old['test_abstentions'],'test_is_previous_benchmark':True,
       'validation_compositions_may_overlap_training':True,'replicates':1}

def main():
    assert not (RUN/'protocol.json').exists(),'Already frozen'
    p=configuration()
    files=list(RUN.glob('*.py'))+list((ROOT/'experiments/baselines_retrain_20260918/source').rglob('*.py'))
    files+=list((ROOT/'baselines/common').glob('*.py'))
    files += [ROOT/'baselines/scripts'/n for n in ['label_datacreate_baselines.py','label_datacreate_baselines_v2.py']]
    files += [PREVIOUS/n for n in ['run_model.py','supervision.py','protocol.json']]
    p['source_sha256']={str(f.relative_to(ROOT)):sha(f) for f in files}
    p['cache_manifest_sha256']={m:sha(RUN/'cache'/m/'manifest.json') for m in ['polytune','laddersym']}
    p['gold_sha256']={s:sha(PREVIOUS/'teacher_gold'/f'{s}.json') for s in p['split']['validation']+p['split']['test']}
    p['context_audit_sha256']=sha(RUN/'context_audit.json')
    write(RUN/'protocol.json',p);print('FROZEN',sha(RUN/'protocol.json'),flush=True)

if __name__=='__main__':main()
