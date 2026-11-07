"""Assess an actual trained checkpoint for every requested N, including rejects.

This supplemental rule is frozen before test generation. It reports accuracy
of the trained models separately from the stability-gated deployment selection.
"""
import argparse
import os
import subprocess
import sys
from common import ROOT,RUN,PREVIOUS,SIZES,SEED,read,write,sha
from scoring import score_directory,KINDS

def freeze():
    assert not (RUN/'predictions/test').exists(),'Declare rule before test generation'
    path=RUN/'trained_assessment_protocol.json';assert not path.exists()
    write(path,{'rule':'For each model and N=10/20/40, select highest validation strict F1 among trained steps100/200/300; ties earlier. Report stability-guard eligibility but never replace the trained row with step0. No test results select the checkpoint.',
        'purpose':'Ensure all six requested trained trials are actually tested, including candidates rejected for deployment by the original protocol.',
        'source_sha256':sha(__file__),'original_protocol_sha256':sha(RUN/'protocol.json'),
        'runtime_amendment_sha256':sha(RUN/'runtime_amendment.json'),'declared_before_any_test_generation':True})
    print('FROZEN_TRAINED_ASSESSMENT',sha(path),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    if args.action=='freeze':freeze();return
    import numpy as np
    protocol=read(RUN/'trained_assessment_protocol.json');assert protocol['source_sha256']==sha(__file__)
    assert protocol['original_protocol_sha256']==sha(RUN/'protocol.json')
    assert read(RUN/'final_audit.json')['status']=='passed'
    p=read(RUN/'protocol.json');guarded=read(RUN/'selection.json');choices={}
    for model in ['polytune','laddersym']:
        choices[model]={}
        for n in SIZES:
            candidates=[c for c in guarded[model][str(n)]['candidates'] if c['step']>0]
            best=max(candidates,key=lambda x:(x['f1'],-x['step']))
            choices[model][str(n)]={**best,'guarded_selected_step':guarded[model][str(n)]['step'],'test_used_for_selection':False}
    write(RUN/'trained_selection.json',choices)
    for model in choices:
        for n,row in choices[model].items():
            step=row['step'];stage=f'{model}_trained_test_{n}_{step}'
            write(RUN/'trained_progress.json',{'stage':stage})
            script=RUN/('engine_cached.py' if model=='polytune' else 'engine.py')
            log=RUN/'logs'/f'{stage}.log'
            print('START',stage,flush=True)
            with log.open('a',encoding='utf-8') as handle:
                subprocess.run([sys.executable,str(script),'infer','--model',model,'--count',n,'--step',str(step),'--partition','test'],
                    cwd=ROOT,env={**os.environ,'PYTHONNOUSERSITE':'1'},stdout=handle,stderr=subprocess.STDOUT,check=True)
            print('DONE',stage,flush=True)
    results={}
    for model in choices:
        results[model]={'0':score_directory(RUN/f'predictions/test/{model}/n00/step000',p['split']['test'],model)}
        for n,row in choices[model].items():
            metrics=score_directory(RUN/f'predictions/test/{model}/n{int(n):02}/step{row["step"]:03}',p['split']['test'],model)
            metrics.update(trained_step=row['step'],validation_f1=row['f1'],passed_validation_stability_guard=row['eligible'],guarded_selected_step=row['guarded_selected_step'])
            results[model][n]=metrics
        draws=np.random.default_rng(SEED).integers(0,17,(2000,17))
        def boot(result):
            counts=np.array([[r['strict'][k] for k in ['credit','predicted','gold']] for r in result['samples'] if r['scored']])
            totals=counts[draws].sum(axis=1);return 2*totals[:,0]/np.maximum(1,totals[:,1]+totals[:,2])
        base=boot(results[model]['0'])
        for n,r in results[model].items():r['paired_delta_f1_95pct_ci']=np.quantile(boot(r)-base,[.025,.975]).tolist()
    write(RUN/'trained_results.json',{'protocol_sha256':sha(RUN/'trained_assessment_protocol.json'),'choices':choices,'results':results,
      'reference':'AudioEval v9 + MuScriptor medium pseudo-gold','training_replicates':1,'bootstrap_replicates':2000})
    lines=['# Actual trained-model assessment: 10, 20, and 40 recordings','',
      'Every row for N=10/20/40 below uses a genuinely fine-tuned checkpoint. The checkpoint is selected by validation reference-note F1 among updates100/200/300, with earlier-update ties. This supplemental rule was frozen before any test generation. Rejected candidates are included for scientific assessment, not promoted for use. The separate guarded-selection report is RESULTS.md.','',
      'Primary metric: exact canonical reference-score note set and error type; timestamps are not the matching standard. All21 DataCreate recordings100–120 are inferred, and17 with76 usable AudioEval v9 + MuScriptor medium pseudo-gold labels are scored. Abstentions101/105/114/117 are excluded from scoring. This is the same previously examined benchmark, not a new holdout.','',
      '| Model | Training recordings | Trained update | Precision | Recall | Strict F1 | Change vs corrected control | Paired 95% interval | Validation stability guard |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for model,rows in results.items():
        base=rows['0']['strict']['f1']
        for n,r in rows.items():
            v=r['strict'];lo,hi=r['paired_delta_f1_95pct_ci'];flag='control' if n=='0' else ('passed' if r['passed_validation_stability_guard'] else 'failed')
            lines.append(f"| {model} | {n} | {r.get('trained_step',0)} | {v['precision']:.2%} | {v['recall']:.2%} | {v['f1']:.2%} | {v['f1']-base:+.2%} | {lo:+.2%} to {hi:+.2%} | {flag} |")
    lines+=['','## Comparison with previous fine-tunes','',
      '| Model | Recordings | Previous published F1 | Previous outputs + v2 adapter | Revised actual trained F1 |',
      '|---|---:|---:|---:|---:|']
    old=read(PREVIOUS/'results.json');same_adapter=read(RUN/'results.json')['previous_outputs_rescored_v2']
    for model,rows in results.items():
        for n,r in rows.items():lines.append(f"| {model} | {n} | {old['results'][model][n]['strict']['f1']:.2%} | {same_adapter[model][n]['strict']['f1']:.2%} | {r['strict']['f1']:.2%} |")
    lines+=['','## Error types and output stability','',
      '| Model | Recordings | Wrong F1 | Missed F1 | Extra F1 | Repetition F1 | Exact / predicted / teacher labels | Test capped windows |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,rows in results.items():
        for n,r in rows.items():
            v=r['strict'];scores=' | '.join(f"{r['by_type'][k]['f1']:.2%}" for k in KINDS)
            lines.append(f"| {model} | {n} | {scores} | {v['credit']:g} / {v['predicted']} / {v['gold']} | {r['capped_windows']} / {r['windows']} |")
    lines+=['','A failed validation stability guard means this candidate produced more length-capped outputs than allowed, even if its F1 happened to rise. The guard and unchanged-model fallback remain in force in the original selection report; these diagnostic results do not override them.','',
      'The teacher labels are not human ground truth. Exact range matching is sensitive to grouping, and native Extra/Missing/Correct conversion still limits wrong-note and repetition reconstruction. One seed and one nested subset order were used. Confidence intervals are paired recording bootstraps, not estimates of seed variance. Original checkpoints, human labels, previous results, and application defaults are preserved.','',
      'See RESULTS.md for preprocessing, training settings, context audit, and runtime validation; ANALYSIS.md for all validation candidates; trained_results.json for all per-recording trained-test results.','']
    (RUN/'TRAINED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    for model,rows in results.items():
        for n,r in rows.items():
            assert r['inferred_recordings']==21 and r['scored_recordings']==17 and r['strict']['gold']==76
            if n!='0':assert r['trained_step']>0
    assert protocol['source_sha256']==sha(__file__)
    write(RUN/'trained_assessment_audit.json',{'status':'passed','six_actual_trained_trials_tested':True,'sizes':SIZES,
        'selection_uses_validation_only':True,'protocol_sha256':sha(RUN/'trained_assessment_protocol.json'),'results_sha256':sha(RUN/'trained_results.json')})
    write(RUN/'trained_progress.json',{'stage':'complete'})
    print('TRAINED_ASSESSMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
