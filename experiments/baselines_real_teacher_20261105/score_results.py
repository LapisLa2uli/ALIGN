"""Score frozen predictions against teacher pseudo-gold by affected score-note identity."""
import collections
import sys
from common import ROOT,RUN,read,write,sha
from supervision import canonical,KINDS
sys.path[:0]=[str(ROOT/'DataCreate/src'),str(ROOT/'align-model/src'),str(ROOT/'baselines/scripts'),str(ROOT/'baselines/common')]

def main():
    import numpy as np
    from datacreate.melody import match_note_wise_labels_detail
    from alignmodel.melody import micro_note_wise
    from label_datacreate_baselines import document_from_midi
    p=read(RUN/'protocol.json');results={};all_rows={}
    usable=[sid for sid in p['test_ids'] if sid not in p['test_abstentions']]
    for flavor in ['polytune','laddersym']:
        results[flavor]={};all_rows[flavor]={}
        for size in p['sizes']:
            rows=[]
            for sid in p['test_ids']:
                path=RUN/'predictions'/flavor/f'n{size:02}'/sid/'mix.mid';done=read(path.with_suffix('.done.json'))
                assert sha(path)==done['sha256'] and done['protocol_sha256']==sha(RUN/'protocol.json')
                doc=document_from_midi(path,RUN/'bundles'/sid/'verified_score.musicxml',model=flavor,sample_id=sid)
                write(RUN/'converted'/flavor/f'n{size:02}'/f'{sid}.json',doc)
                gold_path=RUN/'teacher_gold'/f'{sid}.json';assert sha(gold_path)==p['teacher_gold_sha256'][sid]
                gold=read(gold_path);pred=canonical(doc['labels'])
                pred += [{'type':'unclassified'} for _ in range(doc['baseline_labeling']['unclassified_count'])]
                row={'sample':sid,'scored':sid in usable,'predicted_labels':len(pred),'teacher_status':gold['status']}
                if sid in usable:
                    for metric,credit in [('strict',0.),('official',.5)]:
                        value=match_note_wise_labels_detail(gold['labels'],pred,score_event_count=gold['score_event_count'],type_mismatch_credit=credit)
                        assert value['status']=='available';row[metric]=value
                    row['by_type']={}
                    for kind in KINDS:
                        row['by_type'][kind]=match_note_wise_labels_detail([l for l in gold['labels'] if l['type']==kind],
                           [l for l in pred if l['type']==kind],score_event_count=gold['score_event_count'],type_mismatch_credit=0.)
                rows.append(row)
            scored=[r for r in rows if r['scored']]
            result={metric:micro_note_wise([r[metric] for r in scored]) for metric in ['strict','official']}
            result['by_type']={k:micro_note_wise([r['by_type'][k] for r in scored]) for k in KINDS}
            result['scored_recordings']=len(scored);result['inferred_recordings']=len(rows)
            results[flavor][str(size)]=result;all_rows[flavor][str(size)]=rows
        # Paired recording bootstrap estimates test-sampling uncertainty, not seed variance.
        rng=np.random.default_rng(p['seed']);draws=rng.integers(0,len(usable),(2000,len(usable)))
        def bootstrap_f1(size):
            rows=[r for r in all_rows[flavor][str(size)] if r['scored']]
            counts=np.array([[r['strict'][k] for k in ['credit','predicted','gold']] for r in rows])
            summed=counts[draws].sum(axis=1)
            return 2*summed[:,0]/np.maximum(1,summed[:,1]+summed[:,2])
        baseline=bootstrap_f1(0)
        for size in p['sizes']:
            values=bootstrap_f1(size)
            results[flavor][str(size)]['paired_delta_f1_95pct_ci']=np.quantile(values-baseline,[.025,.975]).tolist()
    report={'protocol_sha256':sha(RUN/'protocol.json'),'reference_kind':'AudioEval+MuScriptor pseudo-gold',
            'test_ids':p['test_ids'],'scored_test_ids':usable,'teacher_abstentions':p['test_abstentions'],
            'results':results,'samples':all_rows,'bootstrap_replicates':2000,'training_replicates':1}
    write(RUN/'results.json',report)
    lines=['# Real DataCreate fine-tuning learning curve','',
      'Both author-pretrained models were independently fine-tuned on nested sets of 0, 10, 20, 30, 40, and 50 real recordings. '
      'Targets and evaluation references come from AudioEval v9 + MuScriptor medium. These results measure agreement with that teacher, not independently verified performance-error accuracy.','',
      f"All 21 recordings 100–120 were decoded. The primary score uses {len(usable)} recordings with usable teacher annotations. "
      f"Teacher abstentions ({', '.join(p['test_abstentions']) or 'none'}) are listed without treating their empty labels as clean gold. Their predictions remain available.",'',
      '| Model | Training recordings | Strict precision | Strict recall | Strict F1 | Change vs 0 | Paired change 95% interval |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for model in results:
        base=results[model]['0']['strict']['f1']
        for size in p['sizes']:
            r=results[model][str(size)];v=r['strict'];lo,hi=r['paired_delta_f1_95pct_ci']
            lines.append(f"| {model} | {size} | {v['precision']:.2%} | {v['recall']:.2%} | {v['f1']:.2%} | {v['f1']-base:+.2%} | {lo:+.2%} to {hi:+.2%} |")
    lines+=['','![Learning curves](learning_curve.png)','','## Protocol','',
      'Each nonzero run starts from the same corresponding author checkpoint, with a fresh optimizer. All models receive 200 optimizer updates with accumulation of 4 windows, AdamW at peak learning rate 1e-5, and the original weighted classification loss. Sampling is uniform over recordings, then over assessed cached windows. All upstream trainable parameters are fine-tuned; fixed positional embeddings stay fixed. Training uses bf16 autocast; decoding uses float32, batch 1, and 1024 generated-token/prompt budgets. Final-step weights are evaluated; test scores never select epochs, thresholds, or hyperparameters.','',
      'The zero-data condition is the unchanged author checkpoint. Nested training IDs, checkpoint hashes, source hashes, and settings are frozen in `protocol.json`. The split has no shared source recordings or source pieces between training and test. One ordering and one training seed were used; bootstrap intervals describe test-recording variation only.','',
      'Native supervision uses Extra / Missing / Correct note sequences. Published wrong notes become Extra(heard pitch) + Missing(expected pitch); repetition notes become Extra. Correct targets require an aligned match. Windows overlapping withheld notes or unassessed score regions are excluded. Missing-note timestamps are teacher estimates. Samples with teacher abstention or unconvertible error labels are excluded from training before selecting the nested subsets.','',
      'MuScriptor can emit events beyond the recording duration. A uniform input-boundary correction discards notes starting after the waveform ends and clips offsets to the waveform. Changed cases were re-aligned; `teacher_bounds_audit.json` records this. No teacher decision was adjusted using either baseline output.','',
      'The primary metric requires the exact canonical affected-score-note location and error type, with one-to-one matching and repetition copy counts. GUI context padding is excluded. The existing baseline adapter converts native MIDI classes to error labels without teacher access. Unlocated and unclassified predictions remain false positives. The repository metric allowing half credit for a wrong type at the correct location is also saved in `results.json`; timestamps are not the primary score.','',
      'For each model, label type, and training size, counts and metrics are in `results.json`; every test prediction and its converted label JSON is retained. Teacher annotations for 100–120 were installed as agent labels with backups under `test_teacher_ui/`. Human label files and original checkpoints were preserved.','',
      'This experiment cannot establish that either fine-tuned model is more accurate than the teacher or a human annotator. Sparse or incorrect teacher labels, the three-class-to-error adapter, and only 17 usable test recordings limit interpretation.','']
    (RUN/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8))
    for model,color in [('polytune','#3478b8'),('laddersym','#d16c24')]:
        ax.plot(p['sizes'],[100*results[model][str(n)]['strict']['f1'] for n in p['sizes']],marker='o',label=model,color=color)
    ax.set(xlabel='Real training recordings',ylabel='Strict reference-note error F1 (%)',xticks=p['sizes'],
           title='Agreement with AudioEval + MuScriptor teacher labels')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(RUN/'learning_curve.png',dpi=170);plt.close(fig)
    print(json.dumps({m:{n:r['strict']['f1'] for n,r in v.items()} for m,v in results.items()},indent=2))

if __name__=='__main__':main()
