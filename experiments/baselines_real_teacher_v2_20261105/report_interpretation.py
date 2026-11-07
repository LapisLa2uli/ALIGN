"""Explain completed outcomes without changing decisions, predictions, or scores."""
from common import RUN,PREVIOUS,SIZES,read,write,sha
from scoring import KINDS

def main():
    assert read(RUN/'final_audit.json')['status']=='passed'
    result=read(RUN/'results.json');selection=read(RUN/'selection.json');previous=read(PREVIOUS/'results.json')
    lines=['# Assessment interpretation','',
      'Primary scores require exact reference-score note sets and error types, including repetition copy counts. The teacher is AudioEval v9 + MuScriptor medium; these are not independently human-verified labels.','',
      '## Selected test outcomes','',
      '| Model | Training recordings | Selected update | Test exact matches | Test predictions | Test strict F1 | Change from corrected control |',
      '|---|---:|---:|---:|---:|---:|---:|']
    selection_notes=[]
    for model,rows in result['results'].items():
        base=rows['0']['strict']['f1']
        for n,row in rows.items():
            v=row['strict'];lines.append(f"| {model} | {n} | {row['selected_step']} | {v['credit']:g} / {v['gold']} | {v['predicted']} | {v['f1']:.2%} | {v['f1']-base:+.2%} |")
        fallbacks=[n for n in SIZES if selection[model][str(n)]['step']==0]
        trained=[n for n in SIZES if selection[model][str(n)]['step']>0]
        if fallbacks:selection_notes+=['',f"{model}: training with {', '.join(map(str,fallbacks))} recordings selected the unchanged-model fallback. Those test rows reuse the control predictions and do not demonstrate learning from those recordings."]
        if trained:
            selection_notes+=['',f"{model}: the validation-selected trained checkpoints were evaluated for {', '.join(map(str,trained))} recordings. Their test changes, including regressions if present, are shown above."]
    lines+=selection_notes
    lines+=['','## Every validation candidate','',
      'All six fine-tunes ran 300 updates. Free generation, not teacher-forced loss, evaluated every checkpoint. The unchanged model is eligible; trained candidates must not exceed its capped-window rate by more than 0.005. Earlier updates win F1 ties. No test score selects checkpoints.','',
      '| Model | Recordings | Update | Exact matches | Predictions | Strict F1 | Capped windows | Eligible | Selected |',
      '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for model in ['polytune','laddersym']:
        base=read(RUN/f'predictions/validation/{model}/n00/step000/metrics.json');v=base['strict']
        lines.append(f"| {model} | 0 | 0 | {v['credit']:g} / {v['gold']} | {v['predicted']} | {v['f1']:.2%} | {base['capped_windows']} / {base['windows']} | yes | control |")
        for n in SIZES:
            selected=selection[model][str(n)]
            for candidate in selected['candidates']:
                step=candidate['step']
                if not step:continue
                row=read(RUN/f'predictions/validation/{model}/n{n:02}/step{step:03}/metrics.json');v=row['strict']
                lines.append(f"| {model} | {n} | {step} | {v['credit']:g} / {v['gold']} | {v['predicted']} | {v['f1']:.2%} | {row['capped_windows']} / {row['windows']} | {'yes' if candidate['eligible'] else 'no'} | {'yes' if selected['step']==step else 'no'} |")
    lines+=['','## Remaining output problems','']
    for model,rows in result['results'].items():
        # Describe every evaluated nonzero condition, not a test-selected winner.
        for n in map(str,SIZES):
            row=rows[n];v=row['strict'];no_recall=[k for k in KINDS if row['by_type'][k]['gold'] and not row['by_type'][k]['credit']]
            worst=sorted([r for r in row['samples'] if r['scored']],key=lambda x:-(x['strict']['predicted']-x['strict']['credit']))[:3]
            examples=', '.join(f"{r['sample']} ({r['strict']['credit']:g} exact / {r['strict']['predicted']} predictions)" for r in worst)
            lines.append(f"- {model}, N={n}: {v['predicted']-v['credit']:g} unmatched predicted labels and {v['gold']-v['credit']:g} unmatched teacher labels. Largest false-positive contributors: {examples}. No exact recovery for: {', '.join(no_recall) if no_recall else 'none of the four types'}. Capped generation: {row['capped_windows']} / {row['windows']} windows.")
    lines+=['',
      'Capped generation is evidence of decoder instability; it is not a physical repeated-note diagnosis. Legitimate adjacent equal-pitch notes are not suppressed. Mislocated ranges, incorrect error types, extra/missing pairing, and repetition reconstruction can all produce strict-score failures. An error range that overlaps the right notes but has different boundaries receives no exact match under this metric.','',
      'The context audit established a preprocessing improvement, not an end-to-end accuracy guarantee. On auditable N40 training note occurrences, the corresponding reference note was outside the supplied crop 50.08% before retrieval and 8.92% afterward. The remaining mismatch and excluded ambiguous reference renders limit that audit.','',
      'Conservative decoder adapters prevent changes to the original weights, but do not guarantee stable free generation. The KL term uses teacher-forced prefixes; it does not directly penalize loops reached only during autoregressive decoding. This is a plausible limitation of the training objective, not a proven sole cause. The experiment does not isolate the effects of LoRA, balancing, retrieval, and regularization from one another.','',
      '## Interpretation limits','',
      'The same DataCreate 100–120 benchmark was examined previously. Four teacher abstentions are excluded rather than treated as clean recordings. One training seed and one nested subset order were used. Paired confidence intervals in RESULTS.md describe variation across the 17 scored recordings, not training-seed uncertainty.','',
      'Test scores compare agreement with the teacher and cannot establish human-verified error accuracy. The selected checkpoints remain experimental; application defaults were preserved. Stronger claims need manually checked score-note labels, additional recordings, and a new evaluation split.','']
    (RUN/'ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('WROTE',RUN/'ANALYSIS.md',flush=True)

if __name__=='__main__':main()
