"""Evaluate global blank scaling and expanded-range class priors on validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from solo_experiment import load_rows,select_rows,write,summarize_measurements
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,infer_dual_outputs
from alignmodel.transcription.ctc_decode_v2 import greedy_decode_notes
from alignmodel.transcription.solo_v1 import metrics,HOP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--range-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-tag',help='Export an already evaluated grid setting without rerunning the grid')
    p.add_argument('--export-checkpoint',type=Path)
    args=p.parse_args();torch.set_num_threads(3);args.output.mkdir(parents=True,exist_ok=True)
    if args.export_tag:
        if not args.export_checkpoint or args.export_checkpoint.exists():
            raise ValueError('Specify a new --export-checkpoint path')
        reports=[args.output/f'{population}-{args.export_tag}.json' for population in ['musical','range']]
        reviewed=[json.loads(path.read_text()) for path in reports]
        digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if any(report['checkpoint_sha256']!=digest for report in reviewed):
            raise ValueError('Calibration evidence does not match the source checkpoint')
        calibration=reviewed[0]['calibration']
        if reviewed[1]['calibration']!=calibration:
            raise ValueError('Calibration settings differ between validation populations')
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if payload.get('pitch_space')!='sounding':
            raise ValueError('Expected sounding-pitch checkpoint')
        bias=payload['model_state_dict']['ctc_head.2.bias']
        from alignmodel.transcription.solo_v1 import MIDI_MIN,MIDI_MAX
        if len(bias)!=MIDI_MAX-MIDI_MIN+2:
            raise ValueError('Unexpected checkpoint pitch range')
        mask=torch.tensor([False]+[pitch<50 or pitch>98 for pitch in range(MIDI_MIN,MIDI_MAX+1)])
        bias[mask]+=calibration['extended_pitch_logit_boost']
        payload['decoder']={**payload.get('decoder',{}),'blank_scale':calibration['blank_scale']}
        payload['calibration']={**calibration,'source_checkpoint':str(args.checkpoint.resolve()),
            'source_sha256':digest,'validation_reports':{str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in reports},
            'test_used':False}
        args.export_checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,args.export_checkpoint)
        print(json.dumps({'checkpoint':str(args.export_checkpoint),'calibration':calibration}))
        return
    model,payload=load_dual_checkpoint(args.checkpoint,'cuda')
    assert payload.get('pitch_space')=='sounding'
    configs=[(.5,0),(.35,0),(.7,0),(.5,.75),(.5,1.5),(.5,3),(.7,1.5)]
    extended=np.array([False]+[pitch<50 or pitch>98 for pitch in range(model.config.midi_min,model.config.midi_max+1)])
    sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest();summaries={}
    for population,cache in [('musical',args.cache),('range',args.range_cache)]:
        rows=select_rows(load_rows(cache,['val']),'val',72,20261017)
        results={pair:[] for pair in configs}
        for i,row in enumerate(rows,1):
            outputs=infer_dual_outputs(model,np.load(row['mel']).astype(np.float32),'cuda')
            for blank,boost in configs:
                probability=outputs['ctc'].copy()
                probability[:,extended]*=np.exp(boost)
                probability/=probability.sum(axis=1,keepdims=True).clip(1e-12)
                predicted=[{'pitch':int(pitch),'start':frame*HOP} for pitch,frame in
                           greedy_decode_notes(probability,model.config.midi_min,blank)]
                results[(blank,boost)].append({**{k:row[k] for k in ['id','instrument','bank','style','melody_seed']},
                                               **metrics(predicted,row['notes'])})
            if i%100==0:print(population,i,len(rows),flush=True)
        for (blank,boost),measured in results.items():
            tag=f'b{blank:g}-p{boost:g}'
            report=summarize_measurements(measured,blank)
            report.update(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha,split='val',
                          calibration={'blank_scale':blank,'extended_pitch_logit_boost':boost})
            write(args.output/f'{population}-{tag}.json',report)
            instruments={k:v for k,v in report['instruments'].items() if k!='legacy_clarinet'}
            summaries.setdefault(tag,{})[population]={
                'macro_f1':float(np.mean([v['f1'] for v in instruments.values()])),
                'legacy':report['instruments'].get('legacy_clarinet',{}).get('f1'),
                'styles':{k:v['f1'] for k,v in report['styles'].items()},
                'precision':report['overall']['precision'],'recall':report['overall']['recall']}
        write(args.output/'grid.json',summaries)
    print('CALIBRATION_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':main()
