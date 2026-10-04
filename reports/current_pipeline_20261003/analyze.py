from pathlib import Path
import json, collections, sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_audit as a

def prf(c,p,g):
    return {'credit':c,'predicted':p,'gold':g,'precision':c/p if p else 0,'recall':c/g if g else 0,'f1':2*c/(p+g) if p+g else 1}

def main():
    synthetic=a.read(HERE/'synthetic.json'); dc=a.read(HERE/'datacreate.json')
    output={'synthetic':{},'datacreate':{'population':dc['population'],'summary':dc['summary'],'unsupported':dc['unsupported_gold_types']},'cases':[]}
    for name,d in synthetic['datasets'].items():
        selected=[r for r in synthetic['details'] if r['dataset']==name]
        error_only={}
        for stage in ('pre_gate','post_gate'):
            kinds=[d[stage]['per_type'][k] for k in ('extra','missed_note','substitute')]
            error_only[stage]=prf(sum(k['credit'] for k in kinds),sum(k['predicted'] for k in kinds),sum(k['gold'] for k in kinds))
        output['synthetic'][name]={'clips':d['clips'],'pre_gate':d['pre_gate']['per_type'],'post_gate':d['post_gate']['per_type'],
            'combined':d['post_gate']['combined'],'content_error_only':error_only,
            'missed_gold':sum(len(r['gold_missed']) for r in selected),
            'missed_removed_by_gate':sum(len(r['missed_removed_by_gate']) for r in selected),
            'missed_not_proposed':sum(len(r['missed_not_proposed']) for r in selected),
            'gate_totals':d['gate_totals']}
    from eval_datacreate_vs_human import _core
    for row in dc['per_sample']:
        for p in row['predicted']:
            pc=_core(p)
            same=[g for g in row['gold'] if g['type']==p['type']]
            def dist(g):
                gc=_core(g)
                return min(abs(pc[0]-gc[1]),abs(gc[0]-pc[1])) if pc and gc else 100000
            nearest=min(same,key=dist) if same else None
            output['cases'].append({'take':row['sample'],'type':p['type'],'pred_core':pc,'time':[p['start_time'],p['end_time']],
                'pred_span':p['score_part'],'nearest_gold_core':_core(nearest) if nearest else None,
                'nearest_gold_time':[nearest['start_time'],nearest['end_time']] if nearest else None})
    output['synthetic_gate_examples']=[r for r in synthetic['details'] if r.get('missed_removed_by_gate')][:12]
    # Analyze the final event output for implausibly brief notes and confidence.
    extras=[]
    coverage=[]
    for file in sorted((HERE/'alignments/datacreate').glob('*.json')):
        al=a.read(file)
        coverage.append({'take':file.stem,'match_fraction':al['match_fraction'],'transcribed':al['transcription_count'],'score_events':al['score_event_count']})
        for e in al['events']:
            if e['relationship']=='extra':
                extras.append({'take':file.stem,**e,'duration':e['end']-e['start']})
    output['short_extra_events']=sorted(extras,key=lambda e:e['duration'])[:30]
    output['low_coverage']=sorted(coverage,key=lambda e:e['match_fraction'])[:10]
    output['extra_event_stats']={'total':len(extras),'under_50ms':sum(e['duration']<.05 for e in extras),'under_80ms':sum(e['duration']<.08 for e in extras)}
    a.write(HERE/'diagnostics.json',output)
    for name,d in output['synthetic'].items():
        print(name,'combined',d['combined']['f1'],'content_error_only',d['content_error_only'])
        for k in ('extra','missed_note','substitute'):
            print(k,{s:{m:d[s][k][m] for m in ('precision','recall','f1','predicted','gold')} for s in ('pre_gate','post_gate')})
        print('missed gold / removed / not proposed',d['missed_gold'],d['missed_removed_by_gate'],d['missed_not_proposed'])
    print('DC',json.dumps(output['datacreate']))
    print('SHORT EXTRAS',json.dumps(output['short_extra_events'][:8]))
    print('COVERAGE',json.dumps(output['low_coverage']))
    for case in output['cases']:
        if case['take'] in ('005','006','020','033','034','003','012'):
            print('CASE',json.dumps(case))

if __name__=='__main__': main()
