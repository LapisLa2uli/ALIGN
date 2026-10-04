"""Versioned restart proposals: local note boundaries and two independent restarts.

This deliberately bounded search augments, rather than replaces, the v6
measure/ornament hypotheses. Only adjacent phrases observed twice in the
transcription generate a proposal. Natural written repeats compete with the
unchanged no-restart hypothesis, so a repeated pitch motif alone is not an error.
"""
from dataclasses import dataclass
from itertools import combinations
from .grammar_mapper_v2 import GrammarHypothesis, grammar_hypotheses

@dataclass(frozen=True)
class LocalHypothesis(GrammarHypothesis):
    restart_regions: tuple[tuple[int,int,int], ...] = ()

def schedule_restarts(score, regions):
    regions=tuple(sorted(regions))
    for k,(start,end,copies) in enumerate(regions):
        if not 0 <= start < end <= len(score) or copies not in (1,2):
            raise ValueError('Invalid restart range')
        if k and regions[k-1][1]>start:
            raise ValueError('Overlapping restart regions require a more general grammar')
    units=[]; times=[]; shift=0.0; endings={end:(start,copies) for start,end,copies in regions}
    for i,event in enumerate(score):
        units.append((i,0)); times.append(float(event.ql_start)+shift)
        if i+1 in endings:
            start,copies=endings[i+1]
            duration=max(float(event.ql_end-score[start].ql_start),.001)
            for copy in range(1,copies+1):
                for j in range(start,i+1):
                    units.append((j,copy)); times.append(float(score[j].ql_start)+shift+duration*copy)
            shift+=duration*copies
    return LocalHypothesis(regions[0][:2] if len(regions)==1 else None,
                           sum(r[2] for r in regions),tuple(units),tuple(times),regions)

def local_hypotheses(score,notes,*,max_proposals=12,max_notes=24):
    pitch=[n.pitch for n in notes if not n.optional or n.confidence >= 0.85]
    score_pitch=[e.pitch for e in score]
    spans={}
    for length in range(2,min(max_notes,len(pitch)//2)+1):
        for at in range(length,len(pitch)-length+1):
            source=tuple(pitch[at-length:at]); replay=tuple(pitch[at:at+length])
            if len(set(source))<2:
                continue
            differences=sum(a!=b for a,b in zip(source,replay))
            if differences>(1 if length>=6 else 0):
                continue
            for start in range(len(score)-length+1):
                segment=tuple(score_pitch[start:start+length])
                if segment != source and segment != replay:
                    continue
                copies=2 if tuple(pitch[at+length:at+2*length])==source else 1
                key=(start,start+length,copies)
                spans[key]=max(spans.get(key,0),length-differences)
    # Keep diverse starts: prevents many nested variants crowding out a second restart.
    ordered=sorted(spans,key=lambda r:(-spans[r],r))
    selected=[]
    for region in ordered:
        if any(region[0]==r[0] for r in selected): continue
        selected.append(region)
        if len(selected)>=max_proposals: break
    output=[schedule_restarts(score,(r,)) for r in selected]
    for left,right in combinations(sorted(selected),2):
        if left[1]<=right[0]: output.append(schedule_restarts(score,(left,right)))
    return output

def hypotheses_v4(score, notes, primary_count, max_proposals=12):
    base=list(grammar_hypotheses(score,primary_count,edit_slack=100000))
    seen={h.units for h in base}
    for h in local_hypotheses(score,notes,max_proposals=max_proposals):
        if h.units not in seen:
            base.append(h); seen.add(h.units)
    return tuple(base)

