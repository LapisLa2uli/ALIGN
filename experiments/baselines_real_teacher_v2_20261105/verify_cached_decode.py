"""Compare optimized generation to every saved validation token, without gold."""
import os
import time
from common import RUN,PREVIOUS,read,write

def main():
    os.environ['PYTHONNOUSERSITE']='1'
    import torch
    import lora
    from engine import build,no_gold
    from cached_decode import install
    torch.set_num_threads(3);p=read(RUN/'protocol.json');model=build('polytune',p['initial_checkpoints']['polytune']['path']).cuda().eval()
    install(model);rows=[];mismatches=[];began=time.monotonic();adapter_installed=False
    with no_gold(),torch.inference_mode():
        for count,step in [(0,0),(10,100)]:
            if step:
                if not adapter_installed:lora.install(model);adapter_installed=True
                lora.load(model,torch.load(RUN/f'training/polytune/n{count:02}/step{step:03}.pt',weights_only=True,map_location='cpu'));lora.mode(model,False)
            for done in sorted((RUN/f'predictions/validation/polytune/n{count:02}/step{step:03}').glob('*/mix.done.json')):
                sid=done.parent.name;cache=torch.load(RUN/f'cache/polytune/infer/{sid}.pt',weights_only=True,map_location='cpu')
                expected=torch.load(done.parent/'tokens.pt',weights_only=True,map_location='cpu');seconds=time.monotonic()
                for i,(entry,wanted) in enumerate(zip(cache,expected)):
                    got=model.generate(mistake_inputs=entry[0].cuda(),score_inputs=entry[1].cuda(),max_length=1024).cpu()
                    if not torch.equal(got,wanted):
                        common=min(got.shape[1],wanted.shape[1]);diff=(got[0,:common]!=wanted[0,:common]).nonzero()
                        mismatches.append({'count':count,'step':step,'sample':sid,'window':i,'old_length':wanted.shape[1],'new_length':got.shape[1],
                           'first_difference':int(diff[0]) if len(diff) else common})
                rows.append({'count':count,'step':step,'sample':sid,'windows':len(cache),'new_seconds':time.monotonic()-seconds,'old_seconds':read(done)['seconds']})
                print('CACHE_PARITY',count,sid,len(cache),'mismatches',len(mismatches),'seconds',round(time.monotonic()-seconds,2),flush=True)
    write(RUN/'cache_parity.json',{'rows':rows,'mismatches':mismatches,'all_tokens_identical':not mismatches,'seconds':time.monotonic()-began,'gold_read':False})
    assert not mismatches,mismatches

if __name__=='__main__':main()
