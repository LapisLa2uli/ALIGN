"""Snapshot inputs and reuse verified teacher predictions; never read human labels as targets."""
import shutil
from common import ROOT,RUN,OLD,TRAIN_POOL,TEST,read,write,sha,checkpoints

def main():
    if (RUN/'inputs.json').exists():raise FileExistsError('Inputs are already frozen')
    rows={};protected={};fresh=[]
    for sid in TRAIN_POOL+TEST:
        source=ROOT/'DataCreate/samples'/sid;dest=RUN/'bundles'/sid;dest.mkdir(parents=True,exist_ok=True)
        hashes={}
        for filename in ['performance_audio.wav','reference_audio.wav','reference_audio.mid','verified_score.musicxml','metadata.json']:
            file=source/filename
            if not file.is_file():raise FileNotFoundError(file)
            hashes[filename]=sha(file);shutil.copy2(file,dest/filename)
        protected[sid]={f:sha(source/f) for f in ['labels.json','performance_audio.wav','verified_score.musicxml'] if (source/f).exists()}
        metadata=read(dest/'metadata.json')
        rows[sid]={'role':'test' if sid in TEST else 'train_pool','hashes':hashes,
                   'source_score':metadata.get('source_score'),'source_filename':metadata.get('source_filename')}
        cached=OLD/'publish/staged'/sid/'note_alignment_v2.json'
        if cached.exists():
            doc=read(cached);p=doc['provenance']
            assert p['transcriber']=='muscriptor' and p['audio_sha256']==hashes['performance_audio.wav'] and p['score_sha256']==hashes['verified_score.musicxml']
            assert p['candidate_sha256']==sha(ROOT/'align-model/runs/stack-v9/CANDIDATE_STACK_V9.json')
            for filename in ['note_alignment_v2.json','labels_agent.json','feedback_v9.json']:
                shutil.copy2(cached.parent/filename,dest/filename)
            (RUN/'teacher_transcriptions').mkdir(exist_ok=True)
            shutil.copy2(OLD/'transcriptions'/f'{sid}.json',RUN/'teacher_transcriptions'/f'{sid}.json')
            rows[sid]['cached_teacher_sha256']=sha(cached)
        else:fresh.append(sid)
    assert not {rows[s]['hashes']['performance_audio.wav'] for s in TRAIN_POOL}&{rows[s]['hashes']['performance_audio.wav'] for s in TEST}
    overlap=sorted({rows[s]['source_score'] for s in TRAIN_POOL if rows[s]['source_score']}&{rows[s]['source_score'] for s in TEST if rows[s]['source_score']})
    write(RUN/'inputs.json',{'samples':rows,'fresh_teacher_ids':fresh,'test_ids':TEST,'protected_live_files':protected,'shared_source_scores':overlap,
          'initial_checkpoints':{k:{'path':str(v),'sha256':sha(v)} for k,v in checkpoints().items()},
          'teacher_labels':'model-generated experimental reference, not human gold','test_labels_ever_used_for_training':False})
    print('SNAPSHOT_COMPLETE',len(rows),'samples;',len(fresh),'new teacher transcriptions; shared pieces:',overlap,flush=True)

if __name__=='__main__':main()
