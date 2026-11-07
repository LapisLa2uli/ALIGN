"""Batch teacher inference inside the existing MuScriptor environment."""
import sys
from types import SimpleNamespace
from common import ROOT,RUN,read,write,sha,verify_inputs
sys.path[:0]=[str(ROOT/'DataCreate/scripts'),str(ROOT/'DataCreate/src')]
from datacreate.muscriptor_bridge import written_shift
from transcribe_muscriptor import load_local_model,transcribe_file

def main():
    manifest=verify_inputs();model,dtype,digest=load_local_model('cuda')
    for sid in manifest['fresh_teacher_ids']:
        sample=RUN/'bundles'/sid;dest=RUN/'new_transcriptions'/f'{sid}.json'
        shift=written_shift(sample,SimpleNamespace(alignment={}))
        p={'written_shift':shift,'model_sha256':digest,'audio_sha256':sha(sample/'performance_audio.wav'),'score_sha256':sha(sample/'verified_score.musicxml')}
        if dest.exists():
            assert all(read(dest)['provenance'][k]==v for k,v in p.items());continue
        result=transcribe_file(model,dtype,digest,sample/'performance_audio.wav',shift)
        result['provenance'].update(p)
        write(dest,result);write(RUN/'teacher_transcriptions'/f'{sid}.json',result)
        print('TEACHER_TRANSCRIBED',sid,len(result['transcribed_notes']),flush=True)

if __name__=='__main__':main()
