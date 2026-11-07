"""Convert published teacher decisions to the baselines' three native note classes.

Unassessed/withheld regions are excluded from training windows, never declared
correct. Wrong notes are Extra(heard)+Missing(expected); replay notes are Extra.
"""
import math

KINDS=('wrong_note','missed_note','extra_note','repetition')

def core(label):
    part=label.get('score_part') or {}
    if part:
        a=part.get('core_start_note_index',part['start_note_index']+part.get('pad_notes',0))
        b=part.get('core_end_note_index',part['end_note_index']-part.get('pad_notes',0))
        return list(range(a,b+1))
    return list(label.get('score_event_indices') or [])

def canonical(labels):
    return [{**row,'score_event_indices':core(row),**({'extra_copies':row.get('extra_copies') or 1} if row['type']=='repetition' else {})}
            for row in labels if row['type'] in KINDS]

def targets(gui,pitches,shift):
    if gui['diagnostics']['status']!='ok':
        return {'usable':False,'reason':'teacher_abstained','classes':{},'unknown':[]}
    labels=canonical(gui['labels']);classes={k:[] for k in ['extra','removed','correct']}
    unknown=[];used=set();mapped=[];removed=set();reasons={}
    def add(kind,pitch,start,end,**meta):
        if not all(math.isfinite(float(x)) for x in [start,end]) or end<=start:
            raise ValueError('Invalid teacher note timing')
        pitch=int(pitch)-shift
        if not 0<=pitch<=127:raise ValueError('Pitch outside MIDI range')
        classes[kind].append(dict(pitch=pitch,start=float(start),end=float(end),**meta))
    for i,n in enumerate(gui['transcribed_notes']):
        a,b=float(n['start']),float(n['end']);span=n.get('score_span')
        indices=set(range(*span)) if span else set();relation=n.get('relationship')
        if span:mapped.append((min(indices),max(indices),a,b))
        kind={'substitute':'wrong_note','copy':'repetition','extra':'extra_note'}.get(relation)
        applicable=[(j,l) for j,l in enumerate(labels) if l['type']==kind
                    and l.get('start_time') is not None and l.get('end_time') is not None
                    and a<float(l['end_time'])-1e-5 and b>float(l['start_time'])+1e-5
                    and (not indices or bool(indices&set(core(l))))]
        if n.get('ignored'):
            unknown.append([a,b,'dropped_candidate']);continue
        if relation=='match':add('correct',n['pitch'],a,b,source_note=i)
        elif applicable:
            add('extra',n['pitch'],a,b,source_note=i)
            for j,l in applicable:
                used.add(j)
                if kind=='wrong_note':
                    for index in sorted(indices&set(core(l))):
                        key=(index,a,b)
                        if key not in removed:
                            add('removed',pitches[index],a,b,score_index=index);removed.add(key)
        else:unknown.append([a,b,'withheld_or_unmapped'])
    for j,label in enumerate(labels):
        if label['type']=='missed_note':
            a,b=label.get('start_time'),label.get('end_time')
            if a is None or b is None:
                return {'usable':False,'reason':'missing_label_has_no_training_timing','classes':{},'unknown':[]}
            for index in core(label):add('removed',pitches[index],a,b,score_index=index)
            used.add(j)
    # A withheld deletion is an unknown gap. Exclude the whole enclosing region.
    for index in gui.get('unassessed_score_event_indices',[]):
        before=[(last,a) for first,last,a,b in mapped if last<index]
        after=[(first,b) for first,last,a,b in mapped if first>index]
        left=max(before,key=lambda v:v[0])[1] if before else 0.
        right=min(after,key=lambda v:v[0])[1] if after else max([n['end'] for n in gui['transcribed_notes']],default=left+.1)
        unknown.append([left,max(right,left+.01),'unassessed_score_gap'])
    unmapped=[labels[j] for j in range(len(labels)) if j not in used]
    if unmapped:
        return {'usable':False,'reason':'published_error_has_no_native_target','unconverted_labels':unmapped,'classes':classes,'unknown':unknown}
    for kind in classes:classes[kind].sort(key=lambda n:(n['start'],n['pitch']))
    return {'usable':bool(classes['correct'] or classes['extra']),'classes':classes,'unknown':unknown,
            'label_count':len(labels),'converted_label_count':len(used),
            'class_counts':{k:len(v) for k,v in classes.items()},'written_to_sounding':-shift}

def valid_window(start,end,unknown):
    return not any(start<b+.02 and end>a-.02 for a,b,*_ in unknown)
