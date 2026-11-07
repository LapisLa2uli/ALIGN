"""Install the requested 100–120 teacher labels in the GUI, backing up agent files."""
import shutil
from common import ROOT,RUN,TEST,read,write,sha

FILES=('note_alignment_v2.json','transcription_notes.json','labels_agent.json','alignment.npz','candidates.json',
       'note_alignment_v9.json','transcription_v9.json','feedback_v9.json')

def main():
    inputs=read(RUN/'inputs.json');out=RUN/'test_teacher_ui'
    if out.exists():raise FileExistsError(out)
    before={};after={}
    for sid in TEST:
        live=ROOT/'DataCreate/samples'/sid;source=RUN/'bundles'/sid
        for file,digest in inputs['protected_live_files'][sid].items():assert sha(live/file)==digest,(sid,file)
        for file in FILES:
            if not (source/file).is_file():raise FileNotFoundError(source/file)
        before[sid]={f:sha(live/f) if (live/f).exists() else None for f in FILES}
        after[sid]={f:sha(source/f) for f in FILES}
    for sid in TEST:
        backup=out/'backup'/sid;backup.mkdir(parents=True)
        for file,digest in before[sid].items():
            if digest is not None:shutil.copy2(ROOT/'DataCreate/samples'/sid/file,backup/file)
    write(out/'manifest.json',{'status':'backed_up','before':before,'after':after,'label_source':'AudioEval+MuScriptor pseudo-gold for this experiment'})
    for sid in TEST:
        live=ROOT/'DataCreate/samples'/sid
        for file,digest in before[sid].items():assert (sha(live/file) if (live/file).exists() else None)==digest
        for file in FILES:
            temp=live/(file+'.teacher-experiment.tmp');shutil.copy2(RUN/'bundles'/sid/file,temp);temp.replace(live/file)
        for file,digest in after[sid].items():assert sha(live/file)==digest
        for file,digest in inputs['protected_live_files'][sid].items():assert sha(live/file)==digest
    write(out/'manifest.json',{'status':'complete','before':before,'after':after,'label_source':'AudioEval+MuScriptor pseudo-gold for this experiment',
                              'human_labels_unchanged':True,'test_ids':TEST})
    print('Published teacher agent labels for all',len(TEST),'test cases; human labels preserved',flush=True)

if __name__=='__main__':main()
