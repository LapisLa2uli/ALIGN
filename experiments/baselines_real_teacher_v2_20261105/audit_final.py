"""Verify completed training, inference, and protected experiment inputs."""
from common import ROOT,RUN,PREVIOUS,SIZES,read,write,sha

def main():
    from pathlib import Path
    p=read(RUN/'protocol.json');selection=read(RUN/'selection.json');result=read(RUN/'results.json');checks=0
    for rel,digest in p['source_sha256'].items():assert sha(ROOT/rel)==digest,rel;checks+=1
    for s,digest in p['gold_sha256'].items():assert sha(PREVIOUS/'teacher_gold'/f'{s}.json')==digest;checks+=1
    for m,digest in p['cache_manifest_sha256'].items():assert sha(RUN/'cache'/m/'manifest.json')==digest;checks+=1
    for init in p['initial_checkpoints'].values():assert sha(init['path'])==init['sha256'];checks+=1
    protected=read(PREVIOUS/'inputs.json')['protected_live_files']
    for s,files in protected.items():
        for name,digest in files.items():assert sha(ROOT/'DataCreate/samples'/s/name)==digest,(s,name);checks+=1
    for m in ['polytune','laddersym']:
        for n in SIZES:
            train=RUN/'training'/m/f'n{n:02}';done=read(train/'done.json')
            assert done['steps']==300 and done['frozen_base_unchanged'] and done['adapter_changed'] and done['adapter_reload_identical']
            for name,digest in done['checkpoints'].items():assert sha(train/name)==digest;checks+=1
            assert len(selection[m][str(n)]['candidates'])==4 and not selection[m][str(n)]['test_used']
        for n in [0,*SIZES]:
            r=result['results'][m][str(n)];assert r['inferred_recordings']==21 and r['scored_recordings']==17 and r['strict']['gold']==76
    for path in (RUN/'predictions').rglob('mix.done.json'):
        done=read(path);assert done['sha256']==sha(path.with_name('mix.mid'));assert done['labels_read'] is False
        assert done['protocol_sha256']==sha(RUN/'protocol.json');checks+=1
    # Previous artifact hashes still match the previous frozen audit references.
    old=read(PREVIOUS/'protocol.json')
    for rel,digest in old['source_sha256'].items():assert sha(ROOT/rel)==digest;checks+=1
    for rel,digest in old['experiment_code_sha256'].items():assert sha(PREVIOUS/rel)==digest;checks+=1
    for path in (PREVIOUS/'predictions').rglob('mix.done.json'):
        done=read(path);assert done['sha256']==sha(path.with_name('mix.mid'));checks+=1
    write(RUN/'final_audit.json',{'status':'passed','hash_checks':checks,'six_trials_completed':True,'sizes':SIZES,
        'original_checkpoints_and_predictions_unchanged':True,'human_labels_unchanged':True,'test_gold_unchanged':True})
    print('FINAL_AUDIT_PASSED',checks,flush=True)

if __name__=='__main__':main()
