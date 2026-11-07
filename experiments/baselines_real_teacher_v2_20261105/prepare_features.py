"""Build corrected training inputs and target-free generation inputs."""
import argparse
import math
import sys
from common import RUN, PREVIOUS, read, write, sha, split
from features import Features

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);args=parser.parse_args()
    import torch
    import soundfile as sf
    import note_seq
    torch.set_num_threads(3);f=Features(args.model);part=split();records=read(PREVIOUS/'data/manifest.json')['tracks']
    def guard(event,values):
        if event=='open' and values:
            name=str(values[0]).replace('\\','/')
            if any(x in name for x in ['/teacher_gold/','/supervision/','/teacher_transcriptions/']) or name.endswith(('/labels.json','/labels_agent.json','/note_alignment_v2.json','/feedback_v9.json')):
                raise PermissionError('Feature preparation must not read gold: '+name)
    sys.addaudithook(guard)
    manifest={}
    for sid in part['train']+part['validation']+part['test']:
        r=records[sid];paths=r['paths'];mapping=read(RUN/'alignment'/f'{sid}.json')
        perf,sr=sf.read(paths['mistake_wav'],dtype='float32');ref,rr=sf.read(paths['score_wav'],dtype='float32');assert sr==rr==16000
        perf=perf[:round(r['duration']*sr)];ns=note_seq.midi_file_to_note_sequence(paths['score_mid'])
        for mode in (['train'] if sid in part['train'] else [])+(['infer'] if sid in part['validation']+part['test'] else []):
            path=RUN/'cache'/args.model/mode/f'{sid}.pt'
            if path.exists() and path.with_suffix('.json').exists():
                meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256'];manifest[mode+'/'+sid]=meta;continue
            outputs=[];starts=[];errors=[]
            if mode=='train':
                old_path=PREVIOUS/'cache'/args.model/f'{sid}.pt';old_meta=read(old_path.with_suffix('.json'));assert sha(old_path)==old_meta['sha256']
                old=torch.load(old_path,weights_only=True,map_location='cpu');bounds=old_meta['bounds'];parity=[]
                for entry,(a,b) in zip(old,bounds):
                    inp,start=f.inputs(perf,ref,ns,mapping,a,b)
                    parity.append(float((entry[0]-inp[0]).abs().max()))
                    outputs.append((inp[0],inp[1],entry[2],*inp[2:]));starts.append(start)
                    errors.append(bool(((entry[2]>=1135)&(entry[2]<=1136)).any()))
            else:
                bounds=[[i*2.048,(i+1)*2.048] for i in range(math.ceil(r['duration']/2.048))];parity=[]
                for a,b in bounds:
                    inp,start=f.inputs(perf,ref,ns,mapping,a,b);outputs.append(inp);starts.append(start)
            path.parent.mkdir(parents=True,exist_ok=True);torch.save(outputs,path)
            meta={'sample':sid,'mode':mode,'windows':len(outputs),'bounds':bounds,'reference_starts':starts,'error_windows':errors,
                'sha256':sha(path),'performance_parity_max_abs':max(parity,default=0.),'alignment_sha256':sha(RUN/'alignment'/f'{sid}.json')}
            write(path.with_suffix('.json'),meta);manifest[mode+'/'+sid]=meta
            print('FEATURES',args.model,mode,sid,len(outputs),'parity',meta['performance_parity_max_abs'],flush=True)
    write(RUN/'cache'/args.model/'manifest.json',manifest)

if __name__=='__main__':main()
