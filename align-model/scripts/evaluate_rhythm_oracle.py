"""Small reproducible oracle-alignment rhythm study using existing synth injectors.

Creates symbolic synthetic bundles with exact tempo-integrated timings; no
acoustic model is evaluated. Selection/check seeds never overlap. Does not open
existing sealed corpora. Gold is changed canonical inter-onset spacing before
nuisance articulation/jitter, not audible sustain duration or padded labels.
"""
import argparse, copy, json, random, sys, hashlib
from pathlib import Path
from dataclasses import asdict
ROOT = Path(__file__).resolve().parents[2]
for p in ('align-model/src', 'DataCreate/src', 'synth-pipeline/src'):
    sys.path.insert(0, str(ROOT/p))
import numpy as np
from music21 import note, stream, tempo, meter
from synthpipeline import errors
from synthpipeline.note_map import tag_clean_notes, clean_note_index
from datacreate.melody import parse_sounding_notes
from alignmodel.joint.index import ScoreEvent, JointEvent
from alignmodel.joint.rhythm_v1 import RhythmConfig, classify_rhythm

KINDS = ('clean', 'late_start', 'early_start', 'early_end', 'late_end', 'tempo_change', 'uneven', 'smooth_tempo', 'short_releases', 'legato_overlap', 'final_release')

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def make_case(seed, kind):
    rng = random.Random(seed)
    bpm = rng.choice((72, 96, 120, 152))
    score = stream.Score()
    part = stream.Part()
    for bar in range(12):
        measure = stream.Measure(number=bar+1)
        if bar == 0:
            measure.insert(0, tempo.MetronomeMark(number=bpm))
            measure.insert(0, meter.TimeSignature('4/4'))
        for beat in range(4):
            duration = rng.choice((.5,.75)) if bar%2 else 1.
            n = note.Note(rng.randint(60,85), quarterLength=duration)
            measure.insert(beat, n)
            if duration < 1: measure.insert(beat+duration, note.Rest(quarterLength=1-duration))
        part.append(measure)
    score.insert(0,part)
    tag_clean_notes(score)
    original = list(score.recurse().getElementsByClass(note.Note))
    refs = [ScoreEvent(i,int(n.pitch.midi),float(n.getOffsetInHierarchy(score)),
        float(n.getOffsetInHierarchy(score)+n.quarterLength),(i,),i//4+1)
        for i,n in enumerate(original)]
    perf = copy.deepcopy(score)
    comment = ''
    if kind not in {'clean', 'smooth_tempo', 'short_releases', 'legato_overlap', 'final_release'}:
        for attempt in range(100):
            try:
                result = getattr(errors, '_rhythm_'+kind)(copy.deepcopy(score),rng,bpm,parse_sounding_notes(score),[])
                break
            except errors.InjectionError:
                if attempt == 99: raise
        perf = result.score
        comment = result.labels[0].comment
    # music21 integrates all tempo marks; identities survive the injector deepcopy.
    exact = []
    gold = set()
    for row in perf.flatten().secondsMap:
        n = row['element']
        if not isinstance(n,note.Note): continue
        i = clean_note_index(n)
        start,end = float(row['offsetSeconds']),float(row['endTimeSeconds'])
        exact.append((i,start,end))
    exact.sort(key=lambda row:row[1])
    # Release-only changes are negative controls under the onset-based policy.
    # Compare exact injected onsets before adding tempo scaling and jitter.
    for (i,start,_), (j,next_start,_) in zip(exact,exact[1:]):
        if j != i+1: continue
        expected = (refs[j].ql_start-refs[i].ql_start)*60/bpm
        if abs((next_start-start)-expected) > 1e-5: gold.add(i)
    scale = rng.uniform(.8,1.25)
    articulation = rng.uniform(.9,1.)
    events = []
    # Slow 15% tempo drift is a clean control, not a planted mistake.
    length = max(e for _,_,e in exact)
    def warp(t):
        return scale*(t + (.15*t*t/(2*length) if kind=='smooth_tempo' else 0))
    for i,start,end in exact:
        onset = warp(start)+2.+rng.uniform(-.004,.004)
        duration = (warp(end)-warp(start))*articulation
        if kind=='short_releases': duration *= rng.uniform(.15,.5)
        elif kind=='legato_overlap': duration = 1.3*scale*60/bpm
        elif kind=='final_release' and i==len(refs)-1: duration *= .15
        events.append(JointEvent(refs[i].pitch,onset,onset+duration,(i,i+1)))
    return refs,events,gold,dict(seed=seed,kind=kind,bpm=bpm,global_scale=scale,
        articulation=articulation,comment=comment),score,perf

def metrics(cases, cfg, measured):
    tp=fp=fn=0
    by_kind={}
    details=[]
    for refs,events,gold,meta,*_ in cases:
        labels,info=classify_rhythm(refs,events,measured_offsets=measured,config=cfg)
        predicted={i for l in labels for i in l['score_event_indices']}
        counts=(len(predicted&gold),len(predicted-gold),len(gold-predicted))
        tp+=counts[0]; fp+=counts[1]; fn+=counts[2]
        totals=by_kind.setdefault(meta['kind'],[0,0,0])
        for j,v in enumerate(counts): totals[j]+=v
        details.append(dict(seed=meta['seed'],kind=meta['kind'],gold=sorted(gold),
            predicted=sorted(predicted),counts=counts))
    def summary(c):
        t,p,n=c
        return dict(tp=t,fp=p,fn=n,precision=t/max(t+p,1),recall=t/max(t+n,1),
            f1=2*t/max(2*t+p+n,1))
    return {**summary((tp,fp,fn)), 'per_kind':{k:summary(v) for k,v in by_kind.items()},
            'cases':details}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'reports/rhythm_oracle_20261106')
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    splits={}
    for split,base in [('selection',61000),('check',71000)]:
        cases=[]
        for repeat in range(5):
            for j,kind in enumerate(KINDS):
                case=make_case(base+repeat*len(KINDS)+j,kind)
                cases.append(case)
                refs,events,gold,meta,score,perf=case
                folder=args.output/'bundles'/split/str(meta['seed'])
                folder.mkdir(parents=True)
                score.write('musicxml',fp=folder/'verified_score.musicxml')
                perf.write('musicxml',fp=folder/'performance_score.musicxml')
                write(folder/'oracle_alignment.json',dict(metadata=meta,
                    score=[asdict(e) for e in refs],events=[asdict(e) for e in events],
                    gold_rhythm_score_indices=sorted(gold)))
        splits[split]=cases
    baseline=RhythmConfig()
    initial=metrics(splits['selection'],baseline,True)
    # Predeclared pilot adequacy gate. Only selection data trigger alternatives.
    adequate=initial['precision']>=.85 and initial['recall']>=.8
    grid=[baseline]
    if not adequate:
        grid += [RhythmConfig(method=method,log_threshold=t,window=w)
            for method in ('ewma','median') for t in (.15,.2,.25,.3)
            for w in ((17,) if method=='ewma' else (17,33,65))]
    scored=[(metrics(splits['selection'],cfg,True),cfg) for cfg in grid]
    chosen=max(scored,key=lambda pair:(pair[0]['f1'],pair[0]['precision'],pair[1].log_threshold))
    write(args.output/'selection.json',dict(adequacy_gate=dict(precision=.85,recall=.8),
        ewma_adequate=adequate,comparisons=[dict(config=asdict(c),metrics=m) for m,c in scored],
        selected=asdict(chosen[1])))
    # Selection is complete before check labels are scored.
    report=dict(protocol=f'{2*5*len(KINDS)} newly generated symbolic clips: {5*len(KINDS)} selection, {5*len(KINDS)} check; exact oracle identity and tempo-integrated timings; no acoustic inference',
        gold='Consecutive canonical onset interval changed by injector before nuisance tempo/articulation/jitter; release-only changes are clean; no display padding',
        selected=asdict(chosen[1]),
        selection_baseline=initial,
        deployed=asdict(RhythmConfig(method='median',window=65,log_threshold=.20)),
        check_deployed=metrics(splits['check'],RhythmConfig(method='median',window=65,log_threshold=.20),True),
        check_ewma=metrics(splits['check'],baseline,True),
        check_selected=metrics(splits['check'],chosen[1],True),
        check_onset_only=metrics(splits['check'],chosen[1],False),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['align-model/src/alignmodel/joint/rhythm_v1.py','synth-pipeline/src/synthpipeline/errors.py']})
    write(args.output/'report.json',report)
    print(json.dumps({k:{n:v[n] for n in ('precision','recall','f1','tp','fp','fn')} for k,v in report.items() if k.startswith('check_')},indent=2))
    print('Selected',report['selected'])
if __name__=='__main__': main()

