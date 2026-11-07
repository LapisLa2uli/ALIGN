"""Report selected tests, same-adapter comparisons, and paired uncertainty."""
import json
from common import RUN,PREVIOUS,SIZES,SEED,read,write,sha
from scoring import score_directory,KINDS

def main():
    import numpy as np
    p=read(RUN/'protocol.json');selection=read(RUN/'selection.json');ids=p['split']['test'];results={};old_adapter={}
    for model in ['polytune','laddersym']:
        results[model]={};old_adapter[model]={}
        for n in [0,*SIZES]:
            step=0 if n==0 else selection[model][str(n)]['step'];effective_n=n if step else 0
            directory=RUN/'predictions/test'/model/f'n{effective_n:02}'/f'step{step:03}'
            result=score_directory(directory,ids,model);result['selected_step']=step
            results[model][str(n)]=result
            # Re-score original outputs read-only; never change old files.
            old_adapter[model][str(n)]=score_directory(PREVIOUS/'predictions'/model/f'n{n:02}',ids,model,save=False)
        scored=[r for r in results[model]['0']['samples'] if r['scored']];draws=np.random.default_rng(SEED).integers(0,len(scored),(2000,len(scored)))
        def boot(r):
            counts=np.array([[v['strict'][k] for k in ['credit','predicted','gold']] for v in r['samples'] if v['scored']])
            totals=counts[draws].sum(axis=1);return 2*totals[:,0]/np.maximum(1,totals[:,1]+totals[:,2])
        baseline=boot(results[model]['0'])
        for n,r in results[model].items():
            v=boot(r);r['f1_95pct_ci']=np.quantile(v,[.025,.975]).tolist();r['paired_delta_f1_95pct_ci']=np.quantile(v-baseline,[.025,.975]).tolist()
            r['paired_delta_vs_old_same_adapter_95pct_ci']=np.quantile(v-boot(old_adapter[model][n]),[.025,.975]).tolist()
    previous=read(PREVIOUS/'results.json');audit=read(RUN/'context_audit.json')
    write(RUN/'results.json',{'protocol_sha256':sha(RUN/'protocol.json'),'selection_sha256':sha(RUN/'selection.json'),
       'results':results,'previous_outputs_rescored_v2':old_adapter,'bootstrap_replicates':2000,'training_replicates':1})
    lines=['# Corrected real-data fine-tuning assessment','',
      'Six independent fine-tunes were completed: PolyTune and LadderSym with 10, 20, and 40 recordings. Each starts from the author checkpoint. Unchanged-checkpoint controls use the same corrected inputs and v2 adapter.','',
      'The primary metric requires reference-score note identity and error type. AudioEval v9 + MuScriptor medium provides pseudo-gold; this measures agreement with that teacher, not human-verified accuracy. The test set is the previously examined DataCreate 100–120 benchmark. All 21 recordings are inferred; 17 with 76 teacher labels are scored. Teacher abstentions 101, 105, 114, and 117 are excluded from scoring.','',
      '| Model | Training recordings | Selected update | Precision | Recall | Strict F1 | Change vs corrected control | Paired 95% interval |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,values in results.items():
        base=values['0']['strict']['f1']
        for n,r in values.items():
            v=r['strict'];lo,hi=r['paired_delta_f1_95pct_ci']
            lines.append(f"| {m} | {n} | {r['selected_step']} | {v['precision']:.2%} | {v['recall']:.2%} | {v['f1']:.2%} | {v['f1']-base:+.2%} | {lo:+.2%} to {hi:+.2%} |")
    lines+=['','Update 0 means validation selected the unchanged model after actually training and evaluating all three candidate adapters. Such a row is a fallback, not an improvement due to training.','',
      '## Comparison with the previous experiment','',
      '| Model | Recordings | Previous published F1 | Previous outputs + v2 adapter F1 | Corrected selected F1 |',
      '|---|---:|---:|---:|---:|']
    for m,values in results.items():
        for n,r in values.items():lines.append(f"| {m} | {n} | {previous['results'][m][n]['strict']['f1']:.2%} | {old_adapter[m][n]['strict']['f1']:.2%} | {r['strict']['f1']:.2%} |")
    lines+=['','The middle column isolates the extra-note location adapter on unchanged old MIDI outputs. Differences from it also include reference retrieval, consistent input processing, performance-duration bounds, and (where selected) training. This is not a factorial causal ablation.','',
      '## What changed','',
      '- A label-free CQT/chroma path retrieves the corresponding 4.096-second reference passage for each 2.048-second performance window. It permits expensive forward/backward jumps. Reference audio and LadderSym MIDI prompts use the same crop. Pitch is not shifted to alter tempo.','- Upstream mel compression by a factor of two is preserved. Training and inference share one frontend; performance tensors match the old native training tensors within 1e-5.','- Decoder attention q/v projections receive rank-4 LoRA adapters (alpha 8, dropout 0.05). All author weights remain frozen. Training uses AdamW, peak LR 1e-4, 300 updates, accumulation 4, original weighted CE plus 0.2 KL regularization against the frozen model.','- Sampling alternates error-containing and clean assessed windows, balancing recordings within each stratum. With 1,200 sampled windows per trial, larger datasets receive fewer repetitions per recording; these are fixed-compute trials, not equal-epoch trials.','- Checkpoints at 0, 100, 200, and 300 updates are selected using full free-generation strict F1 on 12 disjoint validation recordings, with a capped-generation-rate guard and earlier-step tie-break. Test labels do not select checkpoints.','- The v2 adapter gives ordinary extras canonical neighboring score-note locations when supported by performed anchors. It refuses to bridge backward or copy-pass discontinuities. Other error conversions remain unchanged.','',
      '## Reference-context audit','',
      '| Subset | Audited note occurrences | Outside old context | Outside retrieved context |',
      '|---|---:|---:|---:|']
    for n,r in audit['summary'].items():lines.append(f"| {n} | {r['assessed']} | {r['old_absent_fraction']:.2%} | {r['new_absent_fraction']:.2%} |")
    lines+=['','This audit uses teacher-assigned correct-note occurrences only where XML and rendered MIDI have unambiguous identical pitch sequences. Ambiguous scores are excluded, and test labels are not used. Remaining mismatch includes repeated or acoustically ambiguous passages and windows crossing jumps.','',
      '## Error-type results','', '| Model | Recordings | Wrong F1 | Missed F1 | Extra F1 | Repetition F1 | Predicted / teacher labels | Capped windows |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,values in results.items():
        for n,r in values.items():
            v=r['strict'];scores=' | '.join(f"{r['by_type'][k]['f1']:.2%}" for k in KINDS)
            lines.append(f"| {m} | {n} | {scores} | {v['predicted']} / {v['gold']} | {r['capped_windows']} / {r['windows']} |")
    lines+=['','## Limitations and retained artifacts','',
      'One seed and one nested subset order were used. Paired bootstrap intervals (2,000 recording resamples) measure test-recording uncertainty, not training-seed variance. Validation recordings may share musical material with training; identical audio/XML hashes and available source filenames do not overlap, but older recordings have incomplete source metadata.','',
      'Wrong notes still require reconstructing paired native Extra+Missing events, and repetitions are reconstructed from Extra runs. Sparse/incomplete teacher labels, estimated missing-note times, ambiguous rendered score structure, and adapter range grouping can still prevent exact reference-note matches. No decoder heuristic suppresses adjacent identical notes.','',
      'Original checkpoints, old experiment files, human labels, and application defaults are preserved. This is an isolated experimental pipeline, not a promoted Studio default. `selection.json` records every validation candidate; `results.json` contains per-recording counts, per-type results, confidence intervals, and adapter-only comparisons. `predictions/` retains native MIDI, tokens, label JSON, and decode diagnostics. `training/` retains adapter weights, loss history, and frozen-weight verification.','',
      '![Strict F1 comparison](learning_curve.png)','']
    (RUN/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,m in zip(axes,['polytune','laddersym']):
        sizes=[0,*SIZES]
        ax.plot(sizes,[100*results[m][str(n)]['strict']['f1'] for n in sizes],marker='o',label='Corrected + selected')
        ax.plot(sizes,[100*old_adapter[m][str(n)]['strict']['f1'] for n in sizes],marker='s',linestyle='--',label='Previous + same adapter')
        ax.set(title=m,xlabel='Training recordings',xticks=sizes);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('Strict reference-note error F1 (%)');fig.tight_layout();fig.savefig(RUN/'learning_curve.png',dpi=180);plt.close(fig)
    print(json.dumps({m:{n:{'f1':r['strict']['f1'],'step':r['selected_step']} for n,r in v.items()} for m,v in results.items()},indent=2),flush=True)

if __name__=='__main__':main()
