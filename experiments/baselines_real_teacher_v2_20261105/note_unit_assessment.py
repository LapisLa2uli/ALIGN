"""Secondary reference-note/type diagnostic, never used for model selection."""
import argparse
from collections import Counter
from common import RUN,PREVIOUS,SIZES,SEED,read,write,sha
from scoring import canonical,KINDS

def units(labels,unclassified=0):
    result=Counter()
    for ordinal,label in enumerate(canonical(labels)):
        kind=label['type'];indices=tuple(sorted(set(label.get('score_event_indices') or [])))
        if not indices:
            result[('unlocated',kind,ordinal)]+=1;continue
        if kind=='extra_note':
            # An insertion is one gap, even when two canonical neighbors locate it.
            result[(kind,indices)]+=1
        else:
            copies=max(1,int(label.get('extra_copies') or 1)) if kind=='repetition' else 1
            for index in indices:result[(kind,index)]+=copies
    result[('unclassified',)]+=unclassified
    return +result

def counts(gold,pred):
    # Unlocated predictions never match; normal gold contains no unlocated units.
    assert not any(k[0] in ('unlocated','unclassified') for k in gold)
    return {'credit':sum((gold&pred).values()),'predicted':sum(pred.values()),'gold':sum(gold.values())}

def metrics(rows):
    total={k:sum(r[k] for r in rows) for k in ['credit','predicted','gold']}
    c,p,g=total['credit'],total['predicted'],total['gold']
    return {**total,'precision':c/max(1,p),'recall':c/max(1,g),'f1':2*c/max(1,p+g)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','selftest']);args=parser.parse_args()
    if args.action=='selftest':
        whole=[{'type':'wrong_note','score_event_indices':[2,3]}]
        split=[{'type':'wrong_note','score_event_indices':[2]},{'type':'wrong_note','score_event_indices':[3]}]
        assert counts(units(whole),units(split))=={'credit':2,'predicted':2,'gold':2}
        assert counts(units(whole),units(split+split))=={'credit':2,'predicted':4,'gold':2}
        assert sum(units([{'type':'extra_note','score_event_indices':[2,3]}]).values())==1
        assert sum(units([{'type':'repetition','score_event_indices':[2,3],'extra_copies':2}]).values())==4
        assert counts(units(whole),units([{'type':'wrong_note'}]))['credit']==0
        print('NOTE_UNIT_SELFTEST_PASSED');return
    path=RUN/'note_unit_protocol.json'
    if args.action=='freeze':
        assert not path.exists() and not (RUN/'predictions/test').exists()
        write(path,{'scope':'Secondary diagnostic added during validation; not used for selection. Original strict range metric unchanged.',
          'definition':'Multiset intersection of (error type, canonical reference note) for wrong/missed notes; repetitions count each reference note once per extra copy; each extra-note insertion is one exact canonical-neighbor tuple. Unlocated/unclassified predictions are false positives. No timestamp matching.',
          'source_sha256':sha(__file__),'declared_before_any_test_generation':True})
        print('FROZEN_NOTE_UNIT_DIAGNOSTIC');return
    import numpy as np
    assert read(path)['source_sha256']==sha(__file__)
    assert read(RUN/'trained_assessment_audit.json')['status']=='passed'
    p=read(RUN/'protocol.json');selection=read(RUN/'trained_selection.json');results={}
    ids=[s for s in p['split']['test'] if s not in p['test_abstentions']]
    for model in ['polytune','laddersym']:
        results[model]={}
        for n in [0,*SIZES]:
            step=0 if n==0 else selection[model][str(n)]['step'];rows=[]
            for sid in ids:
                doc=read(RUN/f'predictions/test/{model}/n{n:02}/step{step:03}/{sid}/labels.json')
                gold=units(read(PREVIOUS/f'teacher_gold/{sid}.json')['labels'])
                pred=units(doc['labels'],doc['baseline_labeling']['unclassified_count'])
                row={'sample':sid,**counts(gold,pred)}
                row['by_type']={k:counts(Counter({key:v for key,v in gold.items() if key[0]==k}),
                    Counter({key:v for key,v in pred.items() if key[0]==k or (key[0]=='unlocated' and key[1]==k)})) for k in KINDS}
                rows.append(row)
            results[model][str(n)]={**metrics(rows),'samples':rows,'by_type':{k:metrics([r['by_type'][k] for r in rows]) for k in KINDS},'step':step}
        draws=np.random.default_rng(SEED).integers(0,len(ids),(2000,len(ids)))
        def boot(row):
            values=np.asarray([[r[k] for k in ['credit','predicted','gold']] for r in row['samples']])[draws].sum(axis=1)
            return 2*values[:,0]/np.maximum(1,values[:,1]+values[:,2])
        base=boot(results[model]['0'])
        for row in results[model].values():row['paired_delta_f1_95pct_ci']=np.quantile(boot(row)-base,[.025,.975]).tolist()
    write(RUN/'note_unit_results.json',{'protocol_sha256':sha(path),'results':results})
    strict=read(RUN/'trained_results.json')['results']
    lines=['# Secondary reference-note/type assessment','',
      'This diagnostic was declared during validation, before test generation, and is not used to select checkpoints. It measures individual reference-note/type agreement while reducing the penalty for different label grouping. The primary exact-range score remains in TRAINED_RESULTS.md. Timestamps are not used in either metric.','',
      'Wrong and missed labels contribute one unit per affected canonical reference note. Repetitions contribute one unit per note per extra copy. An extra-note insertion contributes one unit identified by its exact canonical-neighbor tuple. Duplicate predictions cannot earn duplicate credit beyond gold multiplicity; unlocated and unclassified outputs count as false positives.','',
      '| Model | Training recordings | Exact-range F1 | Note/type precision | Note/type recall | Note/type F1 | Paired change vs control 95% interval |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for model,rows in results.items():
        for n,r in rows.items():
            lo,hi=r['paired_delta_f1_95pct_ci'];lines.append(f"| {model} | {n} | {strict[model][n]['strict']['f1']:.2%} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.2%} | {lo:+.2%} to {hi:+.2%} |")
    lines+=['','These scores still measure agreement with model-generated teacher labels, not human-verified accuracy. The unit weighting differs from the primary metric: long wrong/missed/repeated ranges contribute more units, and the metric does not require the same segmentation into labels. All six nonzero conditions use actual trained checkpoints chosen only by validation F1.','']
    (RUN/'NOTE_UNIT_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('NOTE_UNIT_ASSESSMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
