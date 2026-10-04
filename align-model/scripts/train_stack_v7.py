"""Bounded reproducible v7 fine tuning; E: datasets remain read-only.

No checkpoint is promoted by a pitch-sequence metric. Epoch candidates are
saved for downstream canonical score-note validation by evaluate_stack_v7.
"""
from __future__ import annotations
import argparse, json, random, sys, time, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in ('align-model/src','align-model/scripts','DataCreate/src','synth-pipeline/src'):
    sys.path.insert(0,str(ROOT/p))
import os
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "align-model/runs/stack-v7/numba-cache"))
print("loading training dependencies",flush=True)
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from alignmodel.transcription.mel_v1_data import MelPackedCache
from alignmodel.transcription.mel_v1 import mel_transcriber_loss
from alignmodel.transcription.mel_ctc_v1 import ctc_loss
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint, save_dual_checkpoint, augment_dual_batch
from alignmodel.transcription.transition_v7 import augment_v7, consistency_loss
from alignmodel.training_resources import resource_lease
from train_mel_ctc_realistic92 import CTCCropDataset

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'align-model/runs/stack-v7')
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--train-per-family',type=int,default=512)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--crop-frames',type=int,default=1024)
    parser.add_argument('--learning-rate',type=float,default=5e-5)
    parser.add_argument('--seed',type=int,default=20261004)
    args=parser.parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this training run')
    device=torch.device('cuda')
    c=json.loads((ROOT/'align-model/runs/precision-v4/CANDIDATE_STACK_V6.json').read_text())
    model,_=load_dual_checkpoint(c['checkpoint'],device)
    teacher,_=load_dual_checkpoint(c['checkpoint'],device)
    teacher.eval(); teacher.requires_grad_(False)
    folders=['realistic92-stack-v2','fast102-v1','dclike11-v1']
    sources=[]; manifest={'args':vars(args)|{'output':str(args.output)},'init_checkpoint':c['checkpoint'],
        'selection_metric':'downstream canonical score-note error F1; no timestamps','families':{},'test_read':False}
    for (name,spec),folder in zip(c['datasets'].items(),folders):
        freeze=json.loads(Path(spec['aligner_freeze']).read_text())
        cache=MelPackedCache(ROOT/'align-model/runs'/folder/'cache-dual-r3',deep=False)
        allowed=set(freeze['eligible']['train'])
        forbidden=set(freeze['eligible']['val'])|set(freeze['eligible']['test'])
        records=[r for r in cache.records('train') if Path(r.sample).name in allowed]
        unique=sorted(set(Path(r.sample).name for r in records))
        selected=set(random.Random(args.seed).sample(unique,min(args.train_per_family,len(unique))))
        records=[r for r in records if Path(r.sample).name in selected]
        assert selected and not(selected & forbidden)
        missing=[n for n in selected if not(Path(spec['root'])/n/'performance_audio.wav').is_file()]
        if missing: raise FileNotFoundError(missing[:4])
        sources.append((cache,records))
        manifest['families'][name]={'root':spec['root'],'cache':str(cache.root),'pack_id':cache.pack_id,
             'eligible_unique':len(unique),'selected':sorted(selected),'records':len(records),
             'freeze_sha256':hashlib.sha256(Path(spec['aligner_freeze']).read_bytes()).hexdigest()}
        print(name,'unique',len(selected),'records',len(records),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-3)
    history=[]
    with resource_lease(ROOT/'align-model/runs/TRAINING_RESOURCE_STATUS.json','gpu',track='stack-v7',command=[sys.executable,*sys.argv]):
        for epoch in range(1,args.epochs+1):
            datasets=[CTCCropDataset(cache.root,records,crop_frames=args.crop_frames,epoch=epoch,seed=args.seed,
                      midi_min=model.config.midi_min,n_pitches=model.config.n_pitches) for cache,records in sources]
            loader=DataLoader(ConcatDataset(datasets),batch_size=args.batch_size,shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed+epoch),num_workers=0,pin_memory=True)
            model.train(); totals=np.zeros(4); start=time.perf_counter()
            for step,batch in enumerate(loader,1):
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                clean=batch['mel']
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    target=teacher(clean)['ctc_logits']
                batch['mel']=augment_v7(augment_dual_batch(clean,batch['onset'],long_mels=128,probability=0.5),batch['onset'])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=model(batch['mel']); frame,_=mel_transcriber_loss(outputs,batch)
                sequence=ctc_loss(outputs['ctc_logits'],batch['frame_mask'],batch['ctc_target'],batch['ctc_length'])
                consistency=consistency_loss(outputs['ctc_logits'],target,batch['frame_mask'])
                loss=sequence+0.5*frame.float()+0.15*consistency
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step()
                totals+=np.array([float(loss),float(sequence),float(frame),float(consistency)])
                if step%25==0 or step==len(loader):
                    print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'loss':(totals/step).tolist(),'seconds':time.perf_counter()-start}),flush=True)
            row={'epoch':epoch,'loss':(totals/step).tolist(),'seconds':time.perf_counter()-start}
            history.append(row)
            save_dual_checkpoint(args.output/f'epoch-{epoch:02d}.pt',model,{'stack_version':7,'epoch':epoch,'history':history,
                 'training_manifest':str(args.output/'training_manifest.json'),'locked_test_materialized':False})
            torch.save({'optimizer':optimizer.state_dict(),'epoch':epoch,'history':history},args.output/'optimizer-last.pt')
            (args.output/'history.json').write_text(json.dumps(history,indent=2))
    print('TRAINING_COMPLETE',flush=True)
if __name__=='__main__': main()



