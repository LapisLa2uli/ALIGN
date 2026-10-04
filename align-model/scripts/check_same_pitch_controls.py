"""Diagnostic positive/negative controls using synthetic rendered-note lineage.

Artificial splits are inserted at the midpoint of individual rendered notes
lasting >=0.5s. Genuine adjacent equal rendered notes are negative controls.
This is a repair sensitivity check, NOT end-to-end model accuracy or training.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'align-model/runs/stack-v8'
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT/'numba-cache'))
import evaluate_stack_v7 as ev
import numpy as np
import torch
from realistic92_aligner_common import load_clip
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.transcription.same_pitch_v1 import repair_same_pitch, waveform_levels


def main():
    torch.set_num_threads(1)
    results = []
    for dataset,name,path,lineage in ev.read(ROOT/'reports/current_pipeline_20261003/manifest.json')['jobs']:
        if dataset == 'datacreate': continue
        with np.load(ROOT/'align-model/runs/stack-v7/eval-cache/baseline'/dataset/f'{name}.npz') as cache:
            outputs = {k:cache[k].astype(np.float32) for k in cache.files}
        mel = outputs.pop('mel')
        audio = load_audio_mono(Path(path)/'performance_audio.wav', 22050)
        outputs['rms_db'] = waveform_levels(audio, mel.shape[1])
        clip = load_clip(Path(path).parent,name,Path(lineage))
        events = sorted(clip.rendered, key=lambda e:e.start)
        rows = [[e.pitch,e.start,e.end,1.,0,-1,0.] for e in events]
        # Every genuine equal-pitch boundary, assessed with original neighbours.
        _, audit = repair_same_pitch(rows, outputs, mel)
        for decision in audit['decisions']:
            results.append({'dataset':dataset, 'sample':name, 'kind':'genuine_repeat', **decision})
        for i,e in enumerate(events):
            if e.end-e.start < .5: continue
            # Stay clear of other rendered voices/events and lineage overlaps.
            if any(j!=i and x.start<e.end and x.end>e.start for j,x in enumerate(events)): continue
            midpoint = (e.start+e.end)/2
            a,b = rows[i].copy(),rows[i].copy();a[2]=midpoint;b[1]=midpoint
            # A different-pitch sentinel bounds context at the known rendered end.
            control = [a,b,[e.pitch+1,e.end,e.end+.1,1.]]
            _, audit = repair_same_pitch(control, outputs, mel)
            results.append({'dataset':dataset, 'sample':name, 'kind':'injected_split', **audit['decisions'][0]})
    summary = {kind:{'count':sum(x['kind']==kind for x in results),
                     'merged':sum(x['kind']==kind and x['decision']=='merge' for x in results)}
               for kind in ('injected_split','genuine_repeat')}
    ev.write(OUT/'controls.json', {'description':__doc__, 'energy_source':'waveform_12ms',
                                  'summary':summary, 'cases':results})
    print(summary,flush=True)


if __name__=='__main__':main()
