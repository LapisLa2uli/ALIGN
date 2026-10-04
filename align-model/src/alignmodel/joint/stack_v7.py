"""Inference and score-note feedback for stack v7 (v6 remains unchanged)."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .index import ScoreEventIndex
from .robust_dp_aligner_v2 import RobustDPCosts
from .robust_dp_aligner_v3 import GateConfig
from .robust_dp_aligner_v4 import AlignerV4Config, align_v4, gate_v4
from ..transcription.transition_v7 import DecodeV7Config, decode_v7

HOP=256/22050

def align_outputs(outputs, score_path, candidate, *, presence=None, config=None):
    config=dict(config or {})
    index=ScoreEventIndex.from_musicxml(score_path)
    rows,decode_info=decode_v7(outputs,config=DecodeV7Config(**config.get('decoder',candidate['decoder'])))
    alignment=align_v4(rows,index.events,score_path,AlignerV4Config(
        costs=RobustDPCosts(**candidate['aligner_costs']),**candidate['aligner_v3']))
    evidence={**outputs,'hop':HOP,'midi_min':52}
    if presence is not None: evidence['presence']=presence
    gate=GateConfig(**(candidate['gate']|config.get('gate',{})))
    events,deletions,info=gate_v4(alignment,index.events,evidence,gate,
                               minimum_match_fraction=config.get('minimum_match_fraction',0.45))
    info.update(decode_info)
    return index,alignment,events,deletions,info

def feedback(index, alignment, events, deletions, info):
    """Canonical affected-note IDs are primary; context and playback are separate."""
    labels=[]
    score=index.events
    if info['status']!='ok':
        return {'schema_version':'align-score-feedback-v7','status':info['status'],'labels':[],
                'diagnostics':info,'unassessed_score_event_indices':list(range(len(score)))}
    def add(kind,indices,start,end,**extra):
        if not indices: return
        a,b=min(indices),max(indices)
        labels.append({'id':f'v7_{len(labels):04d}','type':kind,'score_event_indices':list(indices),
            'note_ids':[f'note_{i:04d}' for i in indices],
            'score_part':{'start_note_index':a,'end_note_index':b,'core_start_note_index':a,'core_end_note_index':b,'pad_notes':0},
            'context_score_event_indices':list(range(max(0,a-2),min(len(score),b+3))),
            'start_time':start,'end_time':end,**extra})
    for e in events:
        if e.relationship=='substitute' and e.copy_pass==0:
            add('wrong_note',range(*e.score_span),e.start,e.end,heard_pitch=e.pitch)
    grouped={}
    for at,e in enumerate(events):
        if e.relationship!='extra' or e.copy_pass: continue
        before=next((x.score_span[1]-1 for x in reversed(events[:at]) if x.score_span and not x.copy_pass),None)
        after=next((x.score_span[0] for x in events[at+1:] if x.score_span and not x.copy_pass),None)
        if before is None and after is None: continue
        anchor=before if before is not None else max(0,after-1)
        indices=tuple(range(anchor,min(anchor+2,len(score))))
        grouped.setdefault(indices,[]).append(e)
    for indices,extras in grouped.items():
        add('extra_note',indices,min(e.start for e in extras),max(e.end for e in extras),heard_pitches=[e.pitch for e in extras])
    missed={m.score_index:m for m in alignment.missed}
    for i in sorted(deletions):
        unit=missed.get(i)
        time=unit.interpolated_time if unit and np.isfinite(unit.interpolated_time) else unit.expected_time if unit else float('nan')
        valid=np.isfinite(time)
        add('missed_note',[i],max(0,float(time)) if valid else None,
            max(0,float(time))+max(.01,unit.expected_duration) if valid else None,
            timing_status='estimated' if valid else 'unavailable')
    for a,b,copies in alignment.restart_regions:
        replay=[e for e in events if e.is_copy and e.score_span and a<=e.score_span[0]<b]
        first=[e for e in events if not e.is_copy and e.score_span and a<=e.score_span[0]<b]
        add('repetition',range(a,b),min((e.start for e in replay),default=None),max((e.end for e in replay),default=None),
            extra_copies=copies,repeats_label_range={'start_time':min((e.start for e in first),default=None),
                                                  'end_time':max((e.end for e in first),default=None)})
    covered={i for e in events if e.score_span for i in range(*e.score_span)}|set(deletions)
    return {'schema_version':'align-score-feedback-v7','status':'ok','labels':labels,'diagnostics':info,
            'unassessed_score_event_indices':sorted(set(range(len(score)))-covered)}
