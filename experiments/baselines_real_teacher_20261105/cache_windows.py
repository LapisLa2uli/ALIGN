"""Cache native upstream training tensors for fully assessed teacher windows."""
import argparse
import math
import os
import sys
from pathlib import Path
from common import ROOT,RUN,read,write,sha
from supervision import valid_window

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['polytune','laddersym'],required=True)
    args=parser.parse_args();flavor=args.model
    source=ROOT/'experiments/baselines_retrain_20260918/source'/('Polytune' if flavor=='polytune' else 'LadderSym')
    sys.path.insert(0,str(source))
    os.environ['LADDERSYM_DETERMINISTIC_PROMPT']='1'
    import numpy as np
    import torch
    from dataset.dataset_2_random import Dataset
    torch.set_num_threads(3)
    class Cached(Dataset):
        def _preprocess_inputs(self,row):
            if getattr(self,'cached_path',None)!=row['mistake_audio']:
                self.cached_path=row['mistake_audio'];self.cached_inputs=super()._preprocess_inputs(row);self.cached_tokens=None
            return self.cached_inputs
        def _tokenize(self,*args,**kwargs):
            if self.cached_tokens is None:self.cached_tokens=super()._tokenize(*args,**kwargs)
            return self.cached_tokens
        def _random_chunk(self,row):
            frames=len(row['mistake_inputs']);start=round(self._validation_fraction*max(0,frames-self.mel_length))
            self.actual_start=start/self.spectrogram_config.frames_per_second
            self.actual_end=self.actual_start+self.mel_length/self.spectrogram_config.frames_per_second
            return super()._random_chunk(row)
        def _pad_length(self,row):
            self.truncated=len(row['targets'])>=self.event_length
            return super()._pad_length(row)
    eligible=read(RUN/'eligible.json')['ordered_ids']
    split={'midi_filename':{str(i):s+'.midi' for i,s in enumerate(eligible)},'split':{str(i):'train' for i in range(len(eligible))}}
    write(RUN/'data/split-cache.json',split)
    kwargs=dict(root_dir=str(RUN/'data'),split_json_path=str(RUN/'data/split-cache.json'),split='train',
        mel_length=256,event_length=1024,num_rows_per_batch=1,full_recording=True,is_deterministic=True,
        is_randomize_tokens=False,is_random_alignment_shift_augmentation=False,shuffle=False,audio_filename='mix.wav')
    if flavor=='laddersym':kwargs.update(use_prompt=True,prompt_length=1024)
    ds=Cached(**kwargs);manifest=read(RUN/'data/manifest.json')['tracks'];summary={}
    for ix,row in enumerate(ds.df):
        sid=Path(row['mistake_audio']).parent.name;out=RUN/'cache'/flavor/f'{sid}.pt'
        if out.exists() and out.with_suffix('.json').exists():
            summary[sid]=read(out.with_suffix('.json'));continue
        target=read(RUN/'supervision'/f'{sid}.json')
        # Roughly one-second spacing covers the entire take with overlapping 2.048s windows.
        count=max(2,math.ceil(manifest[sid]['padded_duration'])+1);ds.validation_windows=count
        windows=[];bounds=[];rejected=0
        for slot in range(count):
            tensors=ds[ix*count+slot];a,b=ds.actual_start,ds.actual_end
            if ds.truncated or a>=manifest[sid]['duration'] or not valid_window(a,b,target['unknown']):
                rejected+=1;continue
            if not bool((tensors[2]>=1004).any()):continue
            windows.append(tuple(t.contiguous().cpu() for t in tensors));bounds.append([a,b])
        out.parent.mkdir(parents=True,exist_ok=True);torch.save(windows,out)
        summary[sid]={'sample':sid,'windows':len(windows),'rejected_windows':rejected,'bounds':bounds,
                      'sha256':sha(out),'supervision_sha256':sha(RUN/'supervision'/f'{sid}.json')}
        write(out.with_suffix('.json'),summary[sid]);print('CACHED',flavor,sid,len(windows),'windows',flush=True)
    write(RUN/'cache'/flavor/'manifest.json',summary)

if __name__=='__main__':main()
