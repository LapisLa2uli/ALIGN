"""Post-hoc output diagnostics; never changes checkpoint selection or metrics."""
import argparse
from collections import Counter
from statistics import mean
from common import RUN,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['validation','test','both'],default='validation');args=parser.parse_args()
    phases=['validation','test'] if args.phase=='both' else [args.phase];results={};training={}
    for phase in phases:
        results[phase]={}
        for metrics in sorted((RUN/'predictions'/phase).glob('*/*/*/metrics.json')):
            run='/'.join(metrics.parent.parts[-3:]);values=read(metrics);counts=Counter();by_sample=[]
            for sample in values['samples']:
                if not sample['scored']:continue
                doc=read(metrics.parent/sample['sample']/'labels.json');native=doc['baseline_labeling']
                counts.update(native['native_counts']);counts['unlocated_labels']+=native['unlocated_events'];counts['unclassified_notes']+=native['unclassified_count']
                by_sample.append({'sample':sample['sample'],'predicted':sample['strict']['predicted'],'gold':sample['strict']['gold'],
                                  'exact_matches':sample['strict']['credit'],'unlocated_labels':native['unlocated_events'],'native':native['native_counts']})
            results[phase][run]={'native_totals':dict(counts),'strict':values['strict'],'capped_windows':values['capped_windows'],
                'windows':values['windows'],'largest_prediction_counts':sorted(by_sample,key=lambda v:-v['predicted'])[:5]}
    for history in (RUN/'training').glob('*/*/history.json'):
        rows=read(history);name='/'.join(history.parent.parts[-2:])
        training[name]={'steps':len(rows),'first50_mean_ce':mean(r['ce'] for r in rows[:50]),'last50_mean_ce':mean(r['ce'] for r in rows[-50:]),
                        'last50_mean_kl':mean(r['kl'] for r in rows[-50:])}
    write(RUN/f'output_diagnostics_{args.phase}.json',{'scope':'Descriptive only; not used to choose checkpoints','outputs':results,'training':training})
    for phase,runs in results.items():
        print(phase,flush=True)
        for name,row in runs.items():print(name,row['native_totals'],'matches',row['strict']['credit'],'predicted',row['strict']['predicted'],flush=True)

if __name__=='__main__':main()
