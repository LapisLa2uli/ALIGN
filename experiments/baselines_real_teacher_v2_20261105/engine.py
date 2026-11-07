"""Reproducible adapter training and target-free native decoding."""
import argparse
import contextlib
import importlib.util
import math
import os
import random
import sys
import time
from pathlib import Path
from common import ROOT,RUN,PREVIOUS,SEED,read,write,sha
from protocol import configuration

def build(flavor,path):
    spec=importlib.util.spec_from_file_location('previous_model',PREVIOUS/'run_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build(flavor,path)

def logits_for(model,batch,flavor):
    target=batch[2];length=int((target!=-100).sum(dim=1).max());target=target[:,:length]
    kw={'mistake_inputs':batch[0],'score_inputs':batch[1],'labels':target,'use_cache':False}
    if flavor=='laddersym':kw.update(decoder_input_ids=batch[3],decoder_attention_mask=batch[4])
    logits=model(**kw)
    if flavor=='laddersym':logits=logits[:,batch[3].shape[1]:,:]
    assert logits.shape[:2]==target.shape
    return logits.float(),target

def ce_loss(logits,target,flavor):
    import torch
    targets=target.reshape(-1);raw=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets,reduction='none')
    errors=(targets>=1135)&(targets<=1136);valid=targets!=-100
    if flavor=='polytune':return (raw[valid].sum()+8*raw[errors].sum())/(valid.sum()+errors.sum()).clamp_min(1)
    return (raw[valid&~errors].sum()+8*raw[errors].sum())/valid.sum().clamp_min(1)

def train(flavor,count,p,smoke=False):
    import numpy as np
    import torch
    import lora
    out=RUN/('smoke' if smoke else 'training')/flavor/f'n{count:02}';out.mkdir(parents=True,exist_ok=True)
    if not smoke and (out/'done.json').exists():return
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    init=p['initial_checkpoints'][flavor];assert sha(init['path'])==init['sha256']
    model=build(flavor,init['path']).cuda().eval();names=lora.install(model);lora.mode(model,False)
    base_hash=lora.frozen_hash(model);cfg=p['optimizer'];model.config.use_cache=False
    ids=p['split']['subsets'][str(count)];cache={};strata={True:{},False:{}}
    for sid in ids:
        path=RUN/'cache'/flavor/'train'/f'{sid}.pt';meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256']
        cache[sid]=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        for flag in strata:
            slots=[i for i,e in enumerate(meta['error_windows']) if e==flag]
            if slots:strata[flag][sid]=slots
    assert all(strata.values())
    params=[v for v in model.parameters() if v.requires_grad];initial=lora.state(model)
    check=tuple(x.cuda() for x in cache[ids[0]][0])
    with torch.no_grad():
        with lora.disabled(model):before,_=logits_for(model,check,flavor)
        after,_=logits_for(model,check,flavor)
        assert torch.equal(before,after),'Zero adapter changed initial logits'
    del before,after,check
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'],foreach=False)
    rng=random.Random(SEED);history=[];steps=2 if smoke else cfg['steps'];began=time.monotonic();first=0
    resume=out/'resume.pt'
    if not smoke and resume.exists():
        saved=torch.load(resume,weights_only=False,map_location='cpu');assert saved['protocol_sha256']==sha(RUN/'protocol.json')
        lora.load(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer']);first=saved['step'];history=saved['history']
        rng.setstate(saved['rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    write(out/'config.json',{'train_ids':ids,'initial_checkpoint':init,'modules':names,'trainable_parameters':sum(v.numel() for v in params),
          'total_parameters':sum(v.numel() for v in model.parameters()),'optimizer':cfg,'base_hash':base_hash,
          'error_window_count':sum(len(x) for x in strata[True].values()),'clean_window_count':sum(len(x) for x in strata[False].values()),'zero_adapter_logits_identical':True})
    for step in range(first,steps):
        optimizer.zero_grad(set_to_none=True);lora.mode(model,True);losses=[];ces=[];kls=[]
        factor=min(1.,(step+1)/cfg['warmup_steps'])*max(.1,.5*(1+math.cos(math.pi*max(0,step-cfg['warmup_steps'])/max(1,cfg['steps']-cfg['warmup_steps']))))
        for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
        for micro in range(cfg['accumulation']):
            eligible=strata[micro%2==0];sid=rng.choice(list(eligible));slot=rng.choice(eligible[sid]);batch=tuple(x.cuda() for x in cache[sid][slot])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.no_grad(),lora.disabled(model):base,target=logits_for(model,batch,flavor)
                logits,target=logits_for(model,batch,flavor);ce=ce_loss(logits,target,flavor);valid=target!=-100
                kl=torch.nn.functional.kl_div(torch.log_softmax(logits[valid],-1),torch.softmax(base[valid],-1),reduction='batchmean')
                loss=ce+cfg['kl_weight']*kl
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            (loss/cfg['accumulation']).backward();losses.append(float(loss.detach()));ces.append(float(ce.detach()));kls.append(float(kl.detach()))
            del base,logits,loss,ce,kl,batch
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip_grad_norm']);assert torch.isfinite(norm)
        optimizer.step();row={'step':step+1,'loss':sum(losses)/len(losses),'ce':sum(ces)/len(ces),'kl':sum(kls)/len(kls),
          'grad_norm':float(norm),'seconds':time.monotonic()-began,'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2}
        history.append(row)
        if step==0 or (step+1)%10==0:write(out/'progress.json',row);print('TRAIN',flavor,count,row,flush=True)
        if step+1 in cfg['candidates'] or step+1==steps:
            state=lora.state(model);torch.save(state,out/f'step{step+1:03}.pt');write(out/'history.json',history)
            if not smoke:
                torch.save({'adapter':state,'optimizer':optimizer.state_dict(),'step':step+1,'history':history,'rng':rng.getstate(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'protocol_sha256':sha(RUN/'protocol.json')},resume)
    state=lora.state(model);assert any(not torch.equal(initial[k],state[k]) for k in state)
    assert lora.frozen_hash(model)==base_hash,'Frozen author weights changed'
    restored=torch.load(out/f'step{steps:03}.pt',weights_only=True);lora.load(model,restored)
    assert all(torch.equal(state[k],lora.state(model)[k]) for k in state)
    write(out/'done.json',{'steps':steps,'train_ids':ids,'frozen_base_unchanged':True,'adapter_changed':True,'adapter_reload_identical':True,
          'seconds':time.monotonic()-began,'checkpoints':{f.name:sha(f) for f in out.glob('step*.pt')},'test_used':False})

@contextlib.contextmanager
def no_gold():
    # Python audit hooks cannot be removed; gate access only during generation.
    active=[True]
    def hook(event,values):
        if active[0] and event=='open' and values:
            name=str(values[0]).replace('\\','/')
            if any(x in name for x in ['/teacher_gold/','/supervision/','/teacher_transcriptions/']) or name.endswith(('/labels.json','/labels_agent.json','/note_alignment_v2.json','/feedback_v9.json')):
                raise PermissionError('Generation cannot read labels: '+name)
    sys.addaudithook(hook)
    try:yield
    finally:active[0]=False

def infer(flavor,count,step,partition,p,smoke=False):
    import numpy as np
    import torch
    import note_seq
    import lora
    init=p['initial_checkpoints'][flavor];assert sha(init['path'])==init['sha256']
    model=build(flavor,init['path']).cuda().eval()
    checkpoint=None
    if step:
        lora.install(model);checkpoint=RUN/('smoke' if smoke else 'training')/flavor/f'n{count:02}'/f'step{step:03}.pt'
        lora.load(model,torch.load(checkpoint,weights_only=True,map_location='cpu'));lora.mode(model,False)
    from inference_error import InferenceHandler
    handler=InferenceHandler(model=model,device='cuda',mel_norm=True)
    out=RUN/('smoke_predictions' if smoke else 'predictions')/partition/flavor/f'n{count:02}'/f'step{step:03}'
    ids=p['split'][partition];ids=ids[:1] if smoke else ids
    digest=sha(checkpoint) if checkpoint else init['sha256'];records=read(PREVIOUS/'data/manifest.json')['tracks']
    with no_gold(),torch.inference_mode():
        for i,sid in enumerate(ids,1):
            dest=out/sid/'mix.mid';done=dest.with_suffix('.done.json')
            if done.exists():
                old=read(done);assert old['sha256']==sha(dest) and old['checkpoint_sha256']==digest;continue
            path=RUN/'cache'/flavor/'infer'/f'{sid}.pt';meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256']
            features=torch.load(path,map_location='cpu',weights_only=True);results=[];frame_times=[];raw=[];lengths=[];capped=0
            dest.parent.mkdir(parents=True,exist_ok=True);began=time.monotonic()
            with (dest.parent/'inference.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                for entry,(a,b) in zip(features,meta['bounds']):
                    batch=tuple(x.cuda() for x in entry);kw={'mistake_inputs':batch[0],'score_inputs':batch[1],'max_length':1024,
                        'num_beams':1,'do_sample':False,'length_penalty':.4,'eos_token_id':model.config.eos_token_id,'early_stopping':False,'use_cache':True}
                    if flavor=='laddersym':
                        # Native training pads prompts with -100; native
                        # generation replaces these with the embedding pad ID.
                        prompt=batch[2].masked_fill(batch[2]==-100,model.config.pad_token_id)
                        kw.update(decoder_input_ids=prompt,decoder_attention_mask=batch[3])
                    result=model.generate(**kw);raw.append(result.cpu());lengths.append(result.shape[-1]);capped+=int(not bool((result==model.config.eos_token_id).any()))
                    results.append(handler._postprocess_batch(result));frame_times.append(np.asarray([a+np.arange(256)*.008]))
                event=handler._to_event(results,np.asarray(frame_times))
                kept=[];removed=clipped=0;duration=records[sid]['duration']
                for n in event.notes:
                    if n.start_time>=duration or n.end_time<=n.start_time:removed+=1;continue
                    if n.end_time>duration:n.end_time=duration;clipped+=1
                    kept.append(n)
                del event.notes[:]
                for n in kept:event.notes.add().CopyFrom(n)
                event.total_time=max([n.end_time for n in event.notes]+[0.])
                event.ClearField('instrument_infos')
                for cls,name in [(1,'extra'),(2,'missing'),(3,'correct')]:
                    info=event.instrument_infos.add();info.instrument=cls;info.name=name
                note_seq.note_sequence_to_midi_file(event,str(dest))
            torch.save(raw,dest.parent/'tokens.pt')
            write(done,{'sha256':sha(dest),'checkpoint_sha256':digest,'windows':len(features),'token_lengths':lengths,'capped_windows':capped,
               'notes_removed_outside_audio':removed,'notes_clipped_to_audio':clipped,'seconds':time.monotonic()-began,'labels_read':False,
               'protocol_sha256':sha(RUN/'protocol.json') if (RUN/'protocol.json').exists() else 'smoke-unfrozen'})
            print('INFER',partition,flavor,count,step,i,len(ids),sid,round(time.monotonic()-began,1),'caps',capped,flush=True)
    return out,ids

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['train','infer','smoke']);parser.add_argument('--model',required=True,choices=['polytune','laddersym']);parser.add_argument('--count',type=int,choices=[0,10,20,40],required=True)
    parser.add_argument('--step',type=int,default=0);parser.add_argument('--partition',choices=['validation','test'],default='validation');args=parser.parse_args()
    os.environ.update(PYTHONNOUSERSITE='1',LADDERSYM_DETERMINISTIC_PROMPT='1',LADDERSYM_PROMPT_LENGTH='1024',LADDERSYM_USE_CACHE='1')
    import torch
    torch.set_num_threads(3)
    p=read(RUN/'protocol.json') if (RUN/'protocol.json').exists() else configuration()
    if args.action=='train':train(args.model,args.count,p)
    elif args.action=='smoke':
        train(args.model,args.count,p,True);torch.cuda.empty_cache();out,ids=infer(args.model,args.count,2,'validation',p,True)
        from scoring import score_directory
        print('SMOKE_SCORE',score_directory(out,ids,args.model)['strict'],flush=True)
    else:
        out,ids=infer(args.model,args.count,args.step,args.partition,p)
        if args.partition=='validation':
            from scoring import score_directory
            print('VALIDATION',args.model,args.count,args.step,score_directory(out,ids,args.model)['strict'],flush=True)

if __name__=='__main__':main()
