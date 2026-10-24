"""Wide-range, sounding-pitch solo transcriber; reuses the dual-mel encoder."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .mel_ctc_v3 import DualMelCTCTranscriber, load_dual_checkpoint, extract_dual_mel, infer_dual_outputs
from .ctc_decode_v2 import greedy_decode_notes
from .mel_v1 import load_audio_mono

HOP = 256 / 22050
MIDI_MIN, MIDI_MAX = 28, 108
PITCH_SPACE = 'sounding'


def initialize_solo(checkpoint: Path, device='cpu'):
    """Transfer every shared weight and shift old written-clarinet heads by -2."""
    old, payload = load_dual_checkpoint(checkpoint, 'cpu')
    if payload.get('pitch_space') == PITCH_SPACE:
        return old.to(device), payload
    new = DualMelCTCTranscriber(replace(old.config, midi_min=MIDI_MIN, midi_max=MIDI_MAX))
    source, target = old.state_dict(), new.state_dict()
    for key in target:
        if key.startswith('pitch_head.') or key.startswith('ctc_head.2.'):
            continue
        target[key].copy_(source[key])
    for name, blank in [('pitch_head', 0), ('ctc_head.2', 1)]:
        if blank:
            target[name+'.weight'][0].copy_(source[name+'.weight'][0])
            target[name+'.bias'][0].copy_(source[name+'.bias'][0])
        for pitch in range(MIDI_MIN, MIDI_MAX+1):
            written = pitch + 2
            nearest = written
            while nearest < old.config.midi_min:
                nearest += 12
            while nearest > old.config.midi_max:
                nearest -= 12
            at, before = pitch-MIDI_MIN+blank, nearest-old.config.midi_min+blank
            target[name+'.weight'][at].copy_(source[name+'.weight'][before])
            target[name+'.bias'][at].copy_(source[name+'.bias'][before])
            if nearest != written:
                target[name+'.bias'][at] -= 6  # suppress untrained extended classes initially
    new.load_state_dict(target)
    return new.to(device), payload


def predict_notes(model, mel, device, *, blank_scale=.5, pitch_shift=0):
    outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
    notes = greedy_decode_notes(outputs['ctc'], model.config.midi_min + pitch_shift, blank_scale)
    return [{'pitch': int(p), 'start': float(f*HOP), 'frame': int(f)} for p,f in notes], outputs


def transcribe(audio_path, checkpoint, *, device='cpu', written_to_sounding=0, blank_scale=.5):
    model, payload = load_dual_checkpoint(checkpoint, device)
    if payload.get('pitch_space') != PITCH_SPACE:
        raise ValueError('Expected a sounding-pitch solo checkpoint')
    audio = load_audio_mono(audio_path, 22050)
    mel, _ = extract_dual_mel(audio, device)
    notes, _ = predict_notes(model, mel, device, blank_scale=blank_scale)
    for i, note in enumerate(notes):
        note['sounding_pitch'] = note['pitch']
        note['written_pitch'] = note['pitch'] - written_to_sounding
        note['end'] = notes[i+1]['start'] if i+1 < len(notes) else min(len(audio)/22050, note['start']+.1)
    return {'schema_version':'align-solo-transcription-v1', 'pitch_space':PITCH_SPACE,
            'written_to_sounding':written_to_sounding, 'notes':notes,
            'note_end_policy':'next onset; final onset + 0.1s (not acoustic offsets)',
            'checkpoint':str(Path(checkpoint).resolve())}


def ordered_matches(predicted, target):
    """One-to-one monotone pitch matches; each repeated attack is a separate note."""
    a, b = list(predicted), list(target)
    table = np.zeros((len(a)+1, len(b)+1), np.int32)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            table[i,j] = table[i-1,j-1]+1 if a[i-1] == b[j-1] else max(table[i-1,j],table[i,j-1])
    pairs=[]
    i,j=len(a),len(b)
    while i and j:
        if a[i-1] == b[j-1]:
            pairs.append((i-1,j-1)); i-=1; j-=1
        elif table[i-1,j] >= table[i,j-1]:
            i-=1
        else:
            j-=1
    return list(reversed(pairs))


def metrics(predicted, target):
    pairs = ordered_matches([n['pitch'] for n in predicted], [n['pitch'] for n in target])
    hit = {j for _,j in pairs}
    repeat = {i for i in range(1,len(target)) if target[i]['pitch']==target[i-1]['pitch']}
    fast = {i for i,n in enumerate(target) if n['end']-n['start'] < .15}
    single = set(range(len(target))) if len(target)==1 else set()
    # Independent maximum bipartite onset/pitch matching for the timing diagnostic.
    from scipy.optimize import linear_sum_assignment
    cost = np.ones((len(predicted), len(target)))
    for i,p in enumerate(predicted):
        for j,t in enumerate(target):
            if p['pitch']==t['pitch'] and abs(p['start']-t['start']) <= .05:
                cost[i,j]=0
    ii,jj=linear_sum_assignment(cost)
    onset_hits=int((cost[ii,jj]==0).sum())
    return {'matched':len(pairs),'predicted':len(predicted),'target':len(target),
            'f1':2*len(pairs)/max(1,len(predicted)+len(target)),
            'onset50_matched':onset_hits,'onset50_f1':2*onset_hits/max(1,len(predicted)+len(target)),
            **{f'{name}_{key}':value for name,indices in [('repeat',repeat),('fast',fast),('single',single)]
               for key,value in [('matched',sum(i in hit and (name != 'repeat' or i-1 in hit) for i in indices)),
                                 ('target',len(indices))]}}


def aggregate(rows):
    keys=['matched','predicted','target','onset50_matched','repeat_matched','repeat_target',
          'fast_matched','fast_target','single_matched','single_target']
    result={key:sum(r[key] for r in rows) for key in keys}
    result.update(clips=len(rows), f1=2*result['matched']/max(1,result['predicted']+result['target']),
                  onset50_f1=2*result['onset50_matched']/max(1,result['predicted']+result['target']),
                  precision=result['matched']/max(1,result['predicted']),
                  recall=result['matched']/max(1,result['target']))
    for name in ['repeat','fast','single']:
        result[name+'_recall']=result[name+'_matched']/max(1,result[name+'_target'])
    return result
