"""Prepare, benchmark, and train one shared model on the solo instrument corpus."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'align-model/src'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'align-model/runs/solo-v1/numba-cache'))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,save_dual_checkpoint,extract_dual_mel
from alignmodel.transcription.mel_v1 import load_audio_mono,make_mel_targets,mel_transcriber_loss
from alignmodel.transcription.mel_v1_data import augment_mel_batch
from alignmodel.transcription.mel_ctc_v1 import ctc_loss
from alignmodel.transcription.solo_v1 import initialize_solo,predict_notes,metrics,aggregate,HOP,MIDI_MIN,MIDI_MAX


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def prepare(args):
    source=json.loads((args.data/'manifest.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,row in enumerate(source['rows']):
        sample=Path(row['path'])
        notes=json.loads((sample/'notes.json').read_text())
        path=args.output/'mel'/f"{row['id']}.npy"
        if not path.exists() or path.stat().st_mtime < (sample/'performance_audio.wav').stat().st_mtime:
            mel,_=extract_dual_mel(load_audio_mono(sample/'performance_audio.wav',22050),args.device)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,mel.astype(np.float16))
        rows.append({**row,'mel':str(path.resolve()),'notes':notes})
        if (i+1)%200==0:print(json.dumps({'cached':i+1,'total':len(source['rows'])}),flush=True)
    if args.rehearsal:
        specs=[('realistic','outputRaw_realistic_10k','realistic92-transcriber-v1'),
               ('fast','outputRaw_fast16_focus_5k','fast102-v1'),
               ('dclike','outputRaw_dclike_11','dclike11-v1')]
        for family,folder,split_dir in specs:
            split=json.loads((ROOT/'align-model/runs'/split_dir/'split.json').read_text())['splits']
            for role,limit in [('train',200),('val',24)]:
                names=random.Random(args.seed).sample(sorted(split[role]),min(limit,len(split[role])))
                for name in names:
                    sample=Path('E:/ALIGN_Data')/folder/name
                    if not (sample/'performance_audio.wav').is_file():
                        raise FileNotFoundError(sample)
                    raw=json.loads((sample/'note_map.json').read_text())['rendered_notes']
                    notes=[{'pitch':int(n['pitch_midi_sounding']),'start':float(n['start_sec']),
                            'end':float(n['end_sec'])} for n in raw]
                    row_id=f'legacy-{family}-{name}'
                    path=args.output/'mel'/f'{row_id}.npy'
                    if not path.exists():
                        mel,_=extract_dual_mel(load_audio_mono(sample/'performance_audio.wav',22050),args.device)
                        np.save(path,mel.astype(np.float16))
                    rows.append({'id':row_id,'path':str(sample),'mel':str(path.resolve()),'notes':notes,
                                 'split':role,'instrument':'legacy_clarinet','bank':'musesounds',
                                 'style':family,'melody_seed':f'legacy-{family}-{name}'})
                print(f'rehearsal {family} {role}: {len(names)}',flush=True)
    write(args.output/'manifest.json',{'source_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
        'source':str(args.data.resolve()),'seed':args.seed,'rows':rows,'pitch_space':'sounding'})
    for split in ['train','val','test']:
        write(args.output/f'{split}.json',[r for r in rows if r['split']==split])


def load_rows(cache,splits):
    if all((cache/f'{split}.json').exists() for split in splits):
        return [r for split in splits for r in json.loads((cache/f'{split}.json').read_text())]
    return [r for r in json.loads((cache/'manifest.json').read_text())['rows'] if r['split'] in splits]


def select_rows(rows,split,limit,seed):
    groups={}
    for r in rows:
        if r['split']==split:groups.setdefault(r['instrument'],[]).append(r)
    selected=[]
    for name,items in sorted(groups.items()):
        selected.extend(random.Random(seed).sample(sorted(items,key=lambda r:r['id']),min(limit,len(items))))
    return selected


def evaluate(model,rows,device,*,legacy=False,blank_scale=.5):
    measured=[]
    for i,row in enumerate(rows):
        mel=np.load(row['mel'])
        predicted,_=predict_notes(model,mel,device,blank_scale=blank_scale,pitch_shift=-2 if legacy else 0)
        measured.append({**{k:row[k] for k in ('id','instrument','bank','style','melody_seed')},
                         **metrics(predicted,row['notes'])})
        if (i+1)%100==0:print(f'evaluated {i+1}/{len(rows)}',flush=True)
    return summarize_measurements(measured,blank_scale)


def summarize_measurements(measured,blank_scale=.5):
    groups={k:aggregate([r for r in measured if r['instrument']==k]) for k in sorted({r['instrument'] for r in measured})}
    return {'overall':aggregate(measured),'macro_instrument_f1':float(np.mean([r['f1'] for r in groups.values()])),
            'instruments':groups,'rows':measured,'blank_scale':blank_scale,
            'styles':{k:aggregate([r for r in measured if r['style']==k]) for k in sorted({r['style'] for r in measured})},
            'banks':{k:aggregate([r for r in measured if r['bank']==k]) for k in sorted({r['bank'] for r in measured})},
            'metric_version':'ordered-notes-v2-repeat-requires-both-attacks',
            'metric':'one-to-one ordered pitch-note F1; onset50 is a secondary timing diagnostic'}


class Crops(Dataset):
    def __init__(self,rows,frames,seed,epoch):
        self.rows=rows;self.frames=frames;self.seed=seed;self.epoch=epoch
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];mel=np.load(row['mel'],mmap_mode='r')
        rng=random.Random(self.seed+self.epoch*1000003+index)
        start=rng.randint(0,max(0,mel.shape[1]-self.frames))
        length=min(self.frames,mel.shape[1]-start)
        value=np.zeros((mel.shape[0],self.frames),np.float32)
        value[:,:length]=mel[:,start:start+length]
        targets=make_mel_targets(row['notes'],frames=self.frames,hop_sec=HOP,
                                 midi_min=MIDI_MIN,midi_max=MIDI_MAX,crop_start=start)
        tokens=[n['pitch']-MIDI_MIN+1 for n in row['notes'] if start*HOP <= (n['start']+n['end'])/2 < (start+length)*HOP]
        if any(not 1 <= token <= MIDI_MAX-MIDI_MIN+1 for token in tokens):
            raise ValueError('Out-of-range target; never clip note identity')
        padded=np.zeros(400,np.int64)
        if len(tokens)>len(padded):raise ValueError('Too many CTC tokens')
        padded[:len(tokens)]=tokens
        return {**{k:torch.from_numpy(v) for k,v in targets.items()},'mel':torch.from_numpy(value),
                'frame_mask':torch.arange(self.frames)<length,'ctc_target':torch.from_numpy(padded),
                'ctc_length':torch.tensor(len(tokens)),
                'legacy':torch.tensor(row['instrument']=='legacy_clarinet')}


def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows=load_rows(args.cache,['train','val'])
    training=[r for r in rows if r['split']=='train']
    if args.supplement_cache:
        supplement=load_rows(args.supplement_cache,['train'])
        main_seeds={r['melody_seed'] for r in rows}
        assert not main_seeds & {r['melody_seed'] for r in supplement}
        # Distinct IDs prevent a cache collision with the musical corpus.
        training += [{**r,'id':'coverage-'+r['id']} for r in supplement]
    if args.train_limit:
        training=select_rows(rows,'train',args.train_limit,args.seed)
    # Rehearsal preserves the original fast-note/realistic clarinet domain.
    training += [r for r in training if r['instrument']=='legacy_clarinet'] * (args.rehearsal_weight-1)
    validation=select_rows(rows,'val',args.limit,args.seed)
    assert not {r['melody_seed'] for r in training} & {r['melody_seed'] for r in validation}
    model,_=initialize_solo(args.checkpoint,args.device)
    teacher=None
    if args.distill:
        teacher,_=load_dual_checkpoint(ROOT/'align-model/runs/precision-v4/v5-dclike/best.pt',args.device)
        teacher.requires_grad_(False)
        teacher_columns=torch.tensor([0]+[p-2-MIDI_MIN+1 for p in range(teacher.config.midi_min,teacher.config.midi_max+1)],device=args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
    history=[];best=-1
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'args':{k:str(v) for k,v in vars(args).items()},'train_ids':[r['id'] for r in training],
              'validation_ids':[r['id'] for r in validation],
              'init_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'cache_manifest_sha256':hashlib.sha256((args.cache/'manifest.json').read_bytes()).hexdigest(),
              'training_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'supplement_manifest_sha256':hashlib.sha256((args.supplement_cache/'manifest.json').read_bytes()).hexdigest() if args.supplement_cache else None,
              'selection':'validation macro instrument ordered-note F1','test_used':False}
    write(args.output/'training_manifest.json',manifest)
    for epoch in range(1,args.epochs+1):
        loader=DataLoader(Crops(training,args.frames,args.seed,epoch),batch_size=args.batch,shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed+epoch),num_workers=0,pin_memory=True)
        model.train();start=time.monotonic();sums=np.zeros(3)
        for step,batch in enumerate(loader,1):
            batch={k:v.to(args.device,non_blocking=True) for k,v in batch.items()}
            teacher_logits=None
            legacy=batch['legacy'].bool()
            if teacher is not None and bool(legacy.any()):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    teacher_logits=teacher(batch['mel'][legacy])['ctc_logits'].float()
            if args.augment:
                batch['mel'][:,:128]=augment_mel_batch(batch['mel'][:,:128],probability=.7)
                batch['mel'][:,128:]=augment_mel_batch(batch['mel'][:,128:],probability=.7)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(args.device).startswith('cuda')):
                output=model(batch['mel'])
                frame,_=mel_transcriber_loss(output,batch)
            seq=ctc_loss(output['ctc_logits'],batch['frame_mask'],batch['ctc_target'],batch['ctc_length'])
            loss=seq+.35*frame.float()
            if teacher_logits is not None:
                probability=teacher_logits.softmax(-1)
                student_log=output['ctc_logits'][legacy].float().log_softmax(-1)[:,:,teacher_columns]
                mask=batch['frame_mask'][legacy] & (probability.max(-1).values>.8)
                kl=torch.nn.functional.kl_div(student_log,probability,reduction='none').sum(-1)
                loss=loss+.2*(kl*mask).sum()/mask.sum().clamp_min(1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
            sums+=np.array([float(loss),float(seq),float(frame)])
            if step%50==0 or step==len(loader):
                status={'epoch':epoch,'step':step,'steps':len(loader),'loss':(sums/step).tolist(),
                        'seconds':time.monotonic()-start}
                print(json.dumps(status),flush=True);write(args.output/'progress.json',status)
        result=evaluate(model,validation,args.device,blank_scale=args.blank_scale)
        row={'epoch':epoch,'loss':(sums/step).tolist(),'seconds':time.monotonic()-start,
             'macro_f1':result['macro_instrument_f1'],'overall':result['overall'],'instruments':result['instruments'],
             'styles':result['styles'],'banks':result['banks']}
        history.append(row)
        extra={'pitch_space':'sounding','instrument_conditioning':False,'epoch':epoch,'history':history,
               'training_manifest':str(args.output/'training_manifest.json'),'locked_test_materialized':False,
               'decoder':{'blank_scale':args.blank_scale},'supported_synthetic_instruments':sorted(result['instruments'])}
        save_dual_checkpoint(args.output/'last.pt',model,extra)
        if row['macro_f1']>best:
            best=row['macro_f1'];save_dual_checkpoint(args.output/'best.pt',model,extra)
            write(args.output/'best_validation.json',result)
        write(args.output/'history.json',history)
        print(json.dumps({'epoch':epoch,'macro_f1':row['macro_f1'],'best':best}),flush=True)
        if len(history)>=4 and max(x['macro_f1'] for x in history[-3:]) < best-1e-5:break
    print('TRAINING_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','eval','train'])
    p.add_argument('--data',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'align-model/runs/precision-v4/v5-dclike/best.pt')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--rehearsal',action='store_true')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--limit',type=int,default=16)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--frames',type=int,default=1024)
    p.add_argument('--lr',type=float,default=.00015)
    p.add_argument('--augment',action='store_true')
    p.add_argument('--distill',action='store_true')
    p.add_argument('--train-limit',type=int)
    p.add_argument('--rehearsal-weight',type=int,default=3)
    p.add_argument('--supplement-cache',type=Path)
    p.add_argument('--blank-scale',type=float,default=.5)
    args=p.parse_args();torch.set_num_threads(3)
    if args.action=='prepare':prepare(args)
    elif args.action=='train':
        from alignmodel.training_resources import resource_lease
        with resource_lease(ROOT/'align-model/runs/TRAINING_RESOURCE_STATUS.json','gpu',
                            track='solo-multi-instrument-v1',command=[sys.executable,*sys.argv]):
            train(args)
    else:
        rows=select_rows(load_rows(args.cache,[args.split]),args.split,args.limit,args.seed)
        model,payload=load_dual_checkpoint(args.checkpoint,args.device)
        result=evaluate(model,rows,args.device,legacy=payload.get('pitch_space')!='sounding',blank_scale=args.blank_scale)
        result.update(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),split=args.split)
        write(args.output,result)
        print(json.dumps(result['instruments'],indent=2),flush=True)


if __name__=='__main__':main()
