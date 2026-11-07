"""One input-only reference crop and prompt implementation for train/inference."""
import sys
import numpy as np
from common import ROOT
from reference_alignment import context_start, crop

def setup(flavor):
    source=ROOT/'experiments/baselines_retrain_20260918/source'/('Polytune' if flavor=='polytune' else 'LadderSym')
    sys.path.insert(0,str(source))

class Features:
    def __init__(self,flavor):
        setup(flavor)
        from contrib import spectrograms
        self.spec=spectrograms;self.cfg=spectrograms.SpectrogramConfig();self.flavor=flavor
        if flavor=='laddersym':
            from dataset.dataset_2_random import Dataset
            self.ds=Dataset(skip_build=True,use_prompt=True,is_randomize_tokens=False,is_train=False,prompt_length=1024)

    def mel(self,audio,multiplier):
        import torch
        s=self.spec.compute_spectrogram(audio,self.cfg,context_multiplier=multiplier)
        return torch.from_numpy(((np.clip(s[:256],-12,5)+12)/17).astype('float32')).unsqueeze(0)

    def prompt(self,reference,start):
        import note_seq
        ns=note_seq.NoteSequence(ticks_per_quarter=220)
        for note in reference.notes:
            if note.end_time<=start or note.start_time>=start+4.096:continue
            item=ns.notes.add();item.CopyFrom(note)
            item.start_time=max(0.,note.start_time-start);item.end_time=min(4.096,note.end_time-start)
        ns.total_time=max([n.end_time for n in ns.notes]+[0.])
        empty=note_seq.NoteSequence()
        row=self.ds._tokenize([empty,empty,empty,ns],np.zeros(32768,dtype='float32'),np.zeros(65536,dtype='float32'))
        # Extract only the prompt: empty target tracks have no state events.
        lo=int(row['prompt_event_start_indices'][0]);hi=int(row['prompt_event_end_indices'][-1])
        events=row['prompt_events'][lo:hi]
        state=row['prompt_state_events'];at=int(row['prompt_state_event_indices'][0])
        if len(state):
            end=at+1
            while end<len(state) and state[end-1]!=self.ds.tie_token:end+=1
            prefix=state[at:end]
        else:prefix=np.asarray([self.ds.tie_token],dtype='int64')
        row['prompt_events']=np.concatenate([prefix,events])
        row=self.ds.run_length_encode_shifts(row,feature_key='prompt_events')
        if len(row['prompt_events'])>=1024:raise ValueError('Truncated reference prompt')
        result=self.ds._pad_length({'prompt_events':row['prompt_events']})
        return result['prompts'].unsqueeze(0),result['prompts_attention_mask'].unsqueeze(0)

    def reference(self,reference_audio,reference_ns,mapping,a,b):
        start=context_start(mapping,a,b)
        result=(self.mel(crop(reference_audio,start,4.096),2),)
        if self.flavor=='laddersym':result+=self.prompt(reference_ns,start)
        return result,start

    def inputs(self,performance,reference_audio,reference_ns,mapping,a,b):
        ref,start=self.reference(reference_audio,reference_ns,mapping,a,b)
        return (self.mel(crop(performance,a,2.048),1),*ref),start
