"""Independent warm-start fine-tunes and gold-free inference for a frozen protocol."""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from common import ROOT,RUN,read,write,sha

def build(flavor,path):
    sys.path[:0]=[str(ROOT/'experiments/baselines_retrain_20260918'),str(ROOT/'baselines/common')]
    import torch
    import align_runtime
    def strict(model,checkpoint):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=payload.get('state_dict',payload)
        for prefix in ['module.','model.']:
            if all(k.startswith(prefix) for k in state):state={k[len(prefix):]:v for k,v in state.items()}
        model.load_state_dict(state,strict=True)
    align_runtime.load_model_weights=strict
    from infer_datacreate import build_model
    return build_model(flavor,Path(path))

def loss_for(model,batch,flavor):
    import torch
    target=batch[2]
    # Trimming only ignored trailing target padding leaves all supervised tokens.
    length=int((target!=-100).sum(dim=1).max());target=target[:,:length]
    kwargs={'mistake_inputs':batch[0],'score_inputs':batch[1],'labels':target}
    if flavor=='laddersym':kwargs.update(decoder_input_ids=batch[3],decoder_attention_mask=batch[4])
    logits=model(**kwargs)
    if flavor=='laddersym':logits=logits[:,batch[3].shape[1]:,:]
    targets=target.reshape(-1)
    raw=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),targets,reduction='none')
    errors=(targets>=1135)&(targets<=1136);valid=targets!=-100
    # Retain each author's loss formula, including Polytune's overlap in weights.
    if flavor=='polytune':return (raw[valid].sum()+8*raw[errors].sum())/(valid.sum()+errors.sum()).clamp_min(1)
    return (raw[valid&~errors].sum()+8*raw[errors].sum())/valid.sum().clamp_min(1)

def train(args,p):
    import numpy as np
    import torch
    flavor=args.model;count=args.count
    if count==0:raise ValueError('Zero examples means pretrained inference only')
    out=RUN/('smoke' if args.action=='smoke' else 'training')/flavor/f'n{count:02}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'done.json').exists():
        done=read(out/'done.json');assert sha(out/'model.pt')==done['checkpoint_sha256'];return
    init=p['initial_checkpoints'][flavor];assert sha(init['path'])==init['sha256']
    seed=p['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=build(flavor,init['path']).cuda().train()
    for module in model.modules():
        if hasattr(module,'gradient_checkpointing'):module.gradient_checkpointing=True
    model.config.use_cache=False
    ids=p['train_subsets'][str(count)];assert not set(ids)&set(p['test_ids'])
    cache={}
    for sid in ids:
        path=RUN/'cache'/flavor/f'{sid}.pt'
        assert sha(path)==read(path.with_suffix('.json'))['sha256']
        cache[sid]=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert cache[sid]
    cfg=p['optimizer'];steps=2 if args.action=='smoke' else cfg['steps'];accum=cfg['gradient_accumulation']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],foreach=False)
    rng=random.Random(seed);history=[];first=0;began=time.monotonic()
    resume=out/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==sha(RUN/'protocol.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        first=saved['step'];history=saved['history'];rng.setstate(saved['sampling_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    trainable=sum(v.numel() for v in model.parameters() if v.requires_grad)
    write(out/'config.json',{'train_ids':ids,'initial_checkpoint':init,'protocol_sha256':sha(RUN/'protocol.json'),
                           'steps':steps,'trainable_parameters':trainable,'total_parameters':sum(v.numel() for v in model.parameters()),'optimizer':cfg})
    for step in range(first,steps):
        optimizer.zero_grad(set_to_none=True)
        warmup=min(1.,(step+1)/cfg['warmup_steps']);decay=.5*(1+math.cos(math.pi*max(0,step-cfg['warmup_steps'])/max(1,steps-cfg['warmup_steps'])))
        lr=cfg['lr']*warmup*max(.1,decay)
        for group in optimizer.param_groups:group['lr']=lr
        losses=[]
        for micro in range(accum):
            sid=rng.choice(ids);entry=rng.choice(cache[sid]);batch=tuple(x.cuda(non_blocking=True) for x in entry)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_for(model,batch,flavor)
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss {flavor} {count} {step}')
            (loss/accum).backward();losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'])
        if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        row={'step':step+1,'loss':sum(losses)/len(losses),'lr':lr,'grad_norm':float(norm),'seconds':time.monotonic()-began,
             'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2}
        history.append(row)
        if (step+1)%10==0 or step==0 or step+1==steps:
            write(out/'progress.json',row);print('TRAIN',flavor,count,json.dumps(row),flush=True)
        if (step+1)%50==0 and step+1<steps:
            temp=resume.with_suffix('.tmp')
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'history':history,
                        'sampling_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                        'protocol_sha256':sha(RUN/'protocol.json')},temp);temp.replace(resume)
    path=out/'model.pt';temp=path.with_suffix('.tmp');torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},temp);temp.replace(path)
    final_sha=sha(path);assert final_sha!=init['sha256']
    write(out/'history.json',history)
    write(out/'done.json',{'checkpoint_sha256':final_sha,'initial_checkpoint_sha256':init['sha256'],
                         'steps':steps,'examples':count,'train_ids':ids,'seconds':time.monotonic()-began,
                         'test_used':False,'protocol_sha256':sha(RUN/'protocol.json')})
    print('TRAIN_COMPLETE',flavor,count,flush=True)

def infer(args,p):
    import numpy as np
    import soundfile as sf
    import torch
    flavor=args.model;count=args.count
    init=p['initial_checkpoints'][flavor]
    path=Path(init['path']) if count==0 else RUN/'training'/flavor/f'n{count:02}'/'model.pt'
    digest=init['sha256'] if count==0 else read(path.parent/'done.json')['checkpoint_sha256']
    assert sha(path)==digest
    model=build(flavor,path).cuda().eval()
    from inference_error import InferenceHandler
    handler=InferenceHandler(model=model,device='cuda',mel_norm=True)
    # Read only the frozen input manifest. No teacher decisions enter inference.
    records=read(RUN/'data/manifest.json')['tracks']
    def deny_gold(event,values):
        if event=='open' and values:
            name=str(values[0]).replace('\\','/')
            if '/teacher_gold/' in name or '/supervision/' in name or name.endswith(('/labels.json','/labels_agent.json','/note_alignment_v2.json','/feedback_v9.json')):
                raise PermissionError('Inference must not read labels: '+name)
    sys.addaudithook(deny_gold)
    for i,sid in enumerate(p['test_ids'],1):
        dest=RUN/'predictions'/flavor/f'n{count:02}'/sid/'mix.mid';done=dest.with_suffix('.done.json')
        if done.exists():
            old=read(done);assert old['checkpoint_sha256']==digest and old['sha256']==sha(dest);continue
        paths=records[sid]['paths']
        for key in ['mistake_wav','score_wav','score_mid']:assert sha(paths[key])==records[sid]['hashes'][key]
        perf,sr=sf.read(paths['mistake_wav'],dtype='float32');ref,rr=sf.read(paths['score_wav'],dtype='float32');assert sr==rr==16000
        dest.parent.mkdir(parents=True,exist_ok=True);began=time.monotonic()
        kwargs=dict(mistake_audio=perf,score_audio=ref,audio_path=sid,outpath=dest.as_posix(),batch_size=1,max_length=1024)
        if flavor=='laddersym':kwargs['prompt_path']=paths['score_mid']
        with (dest.parent/'inference.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log),torch.inference_mode():
            handler.inference(**kwargs)
        write(done,{'checkpoint_sha256':digest,'sha256':sha(dest),'seconds':time.monotonic()-began,
                    'protocol_sha256':sha(RUN/'protocol.json'),'test_labels_read':False})
        print('INFER',flavor,count,i,len(p['test_ids']),sid,round(time.monotonic()-began,1),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['train','infer','smoke']);parser.add_argument('--model',choices=['polytune','laddersym'],required=True)
    parser.add_argument('--count',type=int,choices=[0,10,20,30,40,50],required=True);args=parser.parse_args()
    os.environ.update(LADDERSYM_DETERMINISTIC_PROMPT='1',LADDERSYM_PROMPT_LENGTH='1024',LADDERSYM_USE_CACHE='1',PYTHONNOUSERSITE='1')
    import torch
    torch.set_num_threads(3);p=read(RUN/'protocol.json')
    if args.action=='infer':infer(args,p)
    else:train(args,p)

if __name__=='__main__':main()
