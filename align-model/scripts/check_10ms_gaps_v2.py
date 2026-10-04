"""Validate gap veto on the SAME 316 positive-control recordings with 10 ms gaps.

This is an acoustic intervention check, not an unseen model benchmark. Keep
cached model/mel evidence fixed to isolate waveform gap detection. No data is
written to source recordings. The two gap types are silence and 26 dB attenuation.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'align-model/runs/stack-v9'
os.environ.setdefault('NUMBA_CACHE_DIR',str(OUT/'numba-cache'))
import evaluate_stack_v7 as ev
import numpy as np
import torch
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.transcription.same_pitch_v2 import SamePitchConfig,repair_same_pitch


def main():
    torch.set_num_threads(1)
    c=SamePitchConfig(**ev.read(OUT/'CANDIDATE_STACK_V9.json')['same_pitch'])
    features=ev.read(OUT/'control_features.json')
    jobs=ev.read(ROOT/'reports/current_pipeline_20261003/manifest.json')['jobs']
    results=[]
    for dataset,name,path,_ in jobs:
        controls=[r for r in features if r['kind']=='injected_split' and r['dataset']==dataset and r['sample']==name]
        if not controls:continue
        with np.load(ROOT/'align-model/runs/stack-v7/eval-cache/baseline'/dataset/f'{name}.npz') as cache:
            outputs={k:cache[k].astype(np.float32) for k in cache.files}
        mel=outputs.pop('mel');original=load_audio_mono(Path(path)/'performance_audio.wav',22050)
        for row in controls:
            t=row['boundary_sec'];p=row['pitch']
            # 0.25s margins preserve the same +/-90ms analysis context.
            notes=[[p,t-.25,t,1.],[p,t,t+.25,1.],[p+1,t+.25,t+.35,1.]]
            for kind,scale in [('silence',0.),('attenuated_26db',.05)]:
                audio=original.copy();a=round((t-.005)*22050);b=a+round(.01*22050)
                audio[a:b]*=scale
                _,audit=repair_same_pitch(notes,outputs,mel,audio=audio,config=c)
                d=audit['decisions'][0]
                results.append({'dataset':dataset,'sample':name,'boundary_sec':t,'gap_type':kind,
                                'decision':d['decision'],'reason':d['reason'],'evidence':d.get('evidence')})
    summary={kind:{'count':sum(r['gap_type']==kind for r in results),
                   'gap_detected':sum(r['gap_type']==kind and r['reason']=='acoustic_gap' for r in results),
                   'incorrectly_merged':sum(r['gap_type']==kind and r['decision']=='merge' for r in results)}
             for kind in ('silence','attenuated_26db')}
    ev.write(OUT/'gap_10ms_validation.json',{'protocol':__doc__,'summary':summary,'cases':results})
    print('GAP_CHECK',summary,flush=True)


if __name__=='__main__':main()
