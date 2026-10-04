"""Fit relaxed bounds to all 316 old positive controls, then audit negative controls.

Positive controls are calibration data, NOT a held-out accuracy claim. No
DataCreate inputs/labels select parameters. Keep a separate 18 dB gap guard.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'align-model/runs/stack-v9'
os.environ.setdefault('NUMBA_CACHE_DIR',str(OUT/'numba-cache'))
import evaluate_stack_v7 as ev
import numpy as np
import torch
import math
from dataclasses import asdict,replace
from realistic92_aligner_common import load_clip
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.transcription.same_pitch_v2 import SamePitchConfig,acoustic_evidence,repair_same_pitch


def classify(e,c):
    if e['gap']['depth_db']>=c.gap_depth_db:return 'acoustic_gap'
    if e['energy_range_db']>c.energy_range_db:return 'energy_boundary'
    if e['spectral_step']>c.spectral_step_max or e['spectral_drift']>c.spectral_drift_max:return 'spectral_boundary'
    if e['attack_max']>c.merge_attack_max:return 'separate_attack'
    if e['voiced_min']<c.voiced_min or e['energy_min_db']<c.silence_db:return 'silence_or_unvoiced'
    return 'merge'


def main():
    torch.set_num_threads(1)
    feature_file=OUT/'control_features.json'
    if feature_file.exists():results=ev.read(feature_file)
    else:
        results=[]
        jobs=[j for j in ev.read(ROOT/'reports/current_pipeline_20261003/manifest.json')['jobs'] if j[0]!='datacreate']
        for pos,(dataset,name,path,lineage) in enumerate(jobs,1):
            with np.load(ROOT/'align-model/runs/stack-v7/eval-cache/baseline'/dataset/f'{name}.npz') as cache:
                outputs={k:cache[k].astype(np.float32) for k in cache.files}
            mel=outputs.pop('mel');audio=load_audio_mono(Path(path)/'performance_audio.wav',22050)
            prepared=acoustic_evidence(audio,mel.shape[1])
            events=sorted(load_clip(Path(path).parent,name,Path(lineage)).rendered,key=lambda e:e.start)
            rows=[[e.pitch,e.start,e.end,1.,0,-1,0.] for e in events]
            _,audit=repair_same_pitch(rows,outputs,mel,prepared=prepared)
            for d in audit['decisions']:results.append({'dataset':dataset,'sample':name,'kind':'genuine_repeat',**d})
            for i,e in enumerate(events):
                if e.end-e.start<.5 or any(j!=i and x.start<e.end and x.end>e.start for j,x in enumerate(events)):continue
                mid=(e.start+e.end)/2;a,b=rows[i].copy(),rows[i].copy();a[2]=mid;b[1]=mid
                _,audit=repair_same_pitch([a,b,[e.pitch+1,e.end,e.end+.1,1.]],outputs,mel,prepared=prepared)
                results.append({'dataset':dataset,'sample':name,'kind':'injected_split',**audit['decisions'][0]})
            if pos%10==0:print('controls',pos,len(jobs),flush=True)
        ev.write(feature_file,results)
    positive=[r['evidence'] for r in results if r['kind']=='injected_split']
    assert len(positive)==316,'Keep the previous control population identical'
    # Ceiling bounds + a margin; all ordinary model probabilities now qualify.
    relaxed=replace(SamePitchConfig(),
        energy_range_db=math.ceil(max(e['energy_range_db'] for e in positive)+1),
        spectral_step_max=math.ceil(max(e['spectral_step'] for e in positive)*100+1)/100,
        spectral_drift_max=math.ceil(max(e['spectral_drift'] for e in positive)*100+1)/100)
    # If a positive control conflicts with the gap veto, expose the required bound.
    all_pass=replace(relaxed,gap_depth_db=max(relaxed.gap_depth_db,math.ceil(max(e['gap']['depth_db'] for e in positive)+1)))
    configs={'guarded':relaxed,'all-pass':all_pass};summary={};decisions={}
    for variant,c in configs.items():
        decisions[variant]=[]
        for r in results:
            reason=classify(r['evidence'],c) if 'evidence' in r else r['reason']
            decisions[variant].append({**r,'decision':'merge' if reason=='merge' else 'retain','reason':reason})
        summary[variant]={}
        for kind in ('injected_split','genuine_repeat'):
            subset=[r for r in results if r['kind']==kind]
            summary[variant][kind]={'count':len(subset),'merged':sum('evidence' in r and classify(r['evidence'],c)=='merge' for r in subset)}
        candidate=ev.read(ROOT/'align-model/runs/stack-v7/CANDIDATE_STACK_V7.json')
        candidate.update(schema_version='align-stack-v9-candidate-v1',selected_variant=variant,same_pitch=asdict(c),
            validation=str(OUT/'COMPLETED.json'),promoted=False,
            calibration='All 316 injected midpoint splits; not held out. DataCreate not used.')
        candidate.pop('tie_break',None)
        for relative in ('src/alignmodel/transcription/same_pitch_v1.py','src/alignmodel/transcription/same_pitch_v2.py','src/alignmodel/joint/stack_v9.py'):
            candidate['code_sha256'][relative]=ev.sha(ROOT/'align-model'/relative)
        ev.write(OUT/f'CANDIDATE_STACK_V9_{variant.upper().replace("-","_")}.json',candidate)
        if variant=='all-pass':ev.write(OUT/'CANDIDATE_STACK_V9.json',candidate)
    assert summary['all-pass']['injected_split']['merged']==316
    ev.write(OUT/'calibration.json',{'protocol':__doc__,'configs':{v:asdict(c) for v,c in configs.items()},'summary':summary,
             'positive_max_gap_depth_db':max(e['gap']['depth_db'] for e in positive)})
    ev.write(OUT/'control_decisions.json',decisions)
    print('CALIBRATION',summary,'configs',{v:asdict(c) for v,c in configs.items()},flush=True)


if __name__=='__main__':main()
