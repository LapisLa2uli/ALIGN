"""Build the final held-out transcription and real-feedback evaluation report."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parents[1]/'runs/solo-v1')
    args = parser.parse_args()
    run = args.run
    old, new = [read(run/name) for name in ['baseline_test.json', 'candidate_test.json']]
    baseline_val, candidate_val = [read(run/name) for name in ['validation_baseline_full.json', 'validation_candidate_full.json']]
    corpus = read(run/'corpus_summary.json')
    candidate = read(run/'CANDIDATE_SOLO_V1.json')
    real = read(run/'datacreate/report.json')
    instruments = sorted(new['instruments'])
    assert instruments == sorted(old['instruments'])
    assert {r['id'] for r in old['rows']} == {r['id'] for r in new['rows']}
    seeds = sorted({r['melody_seed'] for r in new['rows']})
    counts = np.zeros((2, len(instruments), len(seeds), 3), np.int64)
    for version, data in enumerate([old, new]):
        for row in data['rows']:
            counts[version, instruments.index(row['instrument']), seeds.index(row['melody_seed'])] += [row[k] for k in ['matched', 'predicted', 'target']]
    # Resample melody families together across timbres, preserving their dependence.
    bootstrap = np.random.default_rng(20261017).integers(0, len(seeds), (2000, len(seeds)))
    totals = counts[:, :, bootstrap].sum(axis=3)
    scores = 2*totals[..., 0]/np.maximum(1, totals[..., 1]+totals[..., 2])
    changes = scores[1].mean(axis=0)-scores[0].mean(axis=0)
    interval = np.quantile(changes, [.025, .975]).tolist()
    macro_old = float(np.mean([old['instruments'][k]['f1'] for k in instruments]))
    macro_new = float(np.mean([new['instruments'][k]['f1'] for k in instruments]))
    stats = {'macro_f1': {'baseline':macro_old, 'candidate':macro_new},
             'paired_macro_improvement_95pct_interval':interval,
             'bootstrap_unit':'held-out melody seed, grouped across instruments',
             'bootstrap_replicates':2000, 'test_clips':len(new['rows'])}
    (run/'comparison_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
    lines = ['# Solo multi-instrument v1: measured results', '',
        f"One shared sounding-pitch model, MIDI 28–108. Checkpoint: `{candidate['checkpoint']}`.", '',
        f"Dataset: {corpus['clips']:,} clips, {corpus['notes']:,} notes, {corpus['hours']:.2f} hours; "
        '500 clips per instrument, split 400/50/50 by melody family. Two SoundFonts are represented in training and evaluation.', '',
        f"An independent training-only chromatic supplement adds {corpus['chromatic_supplement']['clips']:,} clips "
        f"and {corpus['chromatic_supplement']['hours']:.2f} hours, covering every configured instrument pitch. "
        f"A separate {corpus['range_validation']['clips']}-clip register validation set uses independent seeds and rhythms. "
        f"Total generated audio: {corpus['generated_total_clips']:,} clips / {corpus['generated_total_hours']:.2f} hours.", '',
        f"Selected variant: **{candidate['variant']}**. {candidate['selection']['reason']}", '',
        '## Held-out synthetic transcription', '',
        f"Macro instrument F1: **{macro_old:.3f} → {macro_new:.3f}**. "
        f"Paired 95% bootstrap interval for the improvement: {interval[0]:+.3f} to {interval[1]:+.3f}.", '',
        'Primary metric is one-to-one ordered pitch-note F1, counting repeated attacks separately. '
        'It measures transcription sequences; it does not directly measure final error-label correctness. '
        'The checkpoint was selected using validation data before opening this test split.', '',
        '| Instrument | Original F1 | Solo F1 | Change |', '|---|---:|---:|---:|']
    for name in instruments:
        a, b = old['instruments'][name]['f1'], new['instruments'][name]['f1']
        lines.append(f'| {name.replace("_", " ")} | {a:.3f} | {b:.3f} | {b-a:+.3f} |')
    lines += ['', '![Per-instrument comparison](instrument_f1.png)', '',
              '## Note types and timing', '',
              '| Diagnostic | Original | Solo |', '|---|---:|---:|']
    for label, field in [('All-note micro F1','f1'), ('Onset/pitch F1 at 50 ms','onset50_f1'),
                         ('Repeated-attack pair recall','repeat_recall'), ('Fast-note recall (<150 ms)','fast_recall'),
                         ('Single-note recall','single_recall')]:
        lines.append(f"| {label} | {old['overall'][field]:.3f} | {new['overall'][field]:.3f} |")
    for style in ['fast','repeated','single','mixed']:
        lines.append(f"| {style.capitalize()} passage F1 | {old['styles'][style]['f1']:.3f} | {new['styles'][style]['f1']:.3f} |")
    a, b = baseline_val['instruments']['legacy_clarinet'], candidate_val['instruments']['legacy_clarinet']
    lines += ['', 'Repeated-attack recall requires both members of a reference repeated pair to match distinct predictions. '
              'Passage F1 also penalizes extra notes; recall alone does not.', '', '## Existing clarinet behavior', '',
              f"Legacy synthetic validation F1: **{a['f1']:.3f} → {b['f1']:.3f}** over {b['clips']} clips.", '',
              f"Real DataCreate regression: {real['gold_population']} manually labeled takes from 001–094. "
              'Labels from 095 onwards are excluded from gold. Sample 095 is retained only for visual inspection. '
              'The metric below matches affected reference-score note identities, without timestamp matching.', '',
              '| Feedback type | v10 precision / recall / F1 | Solo precision / recall / F1 |', '|---|---:|---:|']
    for kind in ['wrong_note','missed_note','extra_note','repetition']:
        formatted = []
        for version in ['v10','solo']:
            value = real['summary'][version][kind]
            formatted.append(' / '.join(f"{value[k]:.3f}" if value.get(k) is not None else 'n/a' for k in ['precision','recall','f1']))
        lines.append(f'| {kind} | {formatted[0]} | {formatted[1]} |')
    lines += ['', 'These real-feedback results do **not** support promoting the shared model over v10. '
              'The solo path has no independently calibrated multi-instrument presence verifier or error gates; '
              'it inherits clarinet-trained alignment/gating behavior and uses its learned CTC boundaries. '
              'No complete human note transcriptions are available for these recordings, so error-label F1 '
              'must not be interpreted as the acoustic transcription F1. Both the old and new systems perform '
              'poorly on this small, strict affected-note benchmark.', '',
              'Inspection-only examples (no gold accuracy claims):', '',
              '| Sample | Model | Transcribed/alignment notes | Suggested labels | Withheld extra / missed |',
              '|---|---|---:|---:|---:|']
    for sample in ['007','095']:
        for version in ['v10','solo']:
            value=read(run/f'datacreate/{sample}-{version}.json')
            diag=value['feedback']['diagnostics'];summary=value['summary']
            lines.append(f"| {sample} | {version} | {summary['notes']} | {summary['labels']} | "
                         f"{diag.get('extras_withheld',0)} / {diag.get('missed_withheld',0)} |")
    lines += ['', 'Sample 095 still produces repetition labels. In 007, the solo path emits wrong-note and '
              'repetition suggestions, while its extra/missing candidates are withheld by inherited gates. '
              'All regression outputs are saved separately under `datacreate/`; live sample labels were not overwritten.']
    lines += ['', '## Development comparisons', '',
              '| Experiment | Musical validation macro F1 | Legacy clarinet F1 | Register diagnostic macro F1 |',
              '|---|---:|---:|---:|']
    for name in ['baseline','transfer','augmented','chromatic','calibrated']:
        if not (run/f'validation_{name}_full.json').exists():continue
        value=read(run/f'validation_{name}_full.json')
        musical=float(np.mean([v['f1'] for k,v in value['instruments'].items() if k!='legacy_clarinet']))
        range_path=run/f'validation_range_{name}.json'
        register=f"{read(range_path)['macro_instrument_f1']:.3f}" if range_path.exists() else '—'
        lines.append(f"| {name} | {musical:.3f} | {value['instruments']['legacy_clarinet']['f1']:.3f} | {register} |")
    lines += ['', 'Register diagnostics deliberately exercise six low/mid/high pitches per instrument and bank, '
              'including pitches sparse or absent in the initial musical training distribution. '
              'They are development validation, not the final held-out musical test.']
    lines += ['', '## Audio quality and scope', '',
              f"All 12,000 clips passed finite-sample, clipping, and per-note audibility checks. "
              f"{len(corpus['bank_fallbacks'])} clips needed an alternate sound bank because of missing or silent note regions. "
              'The chromatic supplement and register validation clips also passed full audio QA. '
              '192 randomly sampled clips received an additional harmonic pitch check. Audition WAVs are in `audio-review/`. '
              'No perceptual listening was performed because this environment has no reliable listening tool.', '',
              'The evaluation covers these 24 synthetic instrument programs. It does not establish universal accuracy '
              'on every real instrument, microphone, playing technique, or notes outside MIDI 28–108. Both test banks '
              'also appear in training. Only clarinet real recordings were available for paired feedback evaluation. '
              'Inherited error-confidence gates remain experimental outside clarinet. Note offsets are estimates from '
              'the next onset, not measured acoustic endings. v9/v10 defaults remain unchanged.', '',
              '## Use', '',
              'See `../../SOLO_MULTI_INSTRUMENT_V1.md` for commands and transposing-score handling. '
              'The opt-in DataCreate configuration is `DataCreate/config/solo.yaml`; '
              'legacy clarinet XML without transposition metadata requires `written_to_sounding: -2`.', '']
    (run/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y = np.arange(len(instruments))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(y+.18, [old['instruments'][k]['f1'] for k in instruments], .34, color='#aeb7c2', label='Original acoustic model')
    ax.barh(y-.18, [new['instruments'][k]['f1'] for k in instruments], .34, color='#2176ae', label='Shared solo model')
    ax.set(yticks=y, yticklabels=[k.replace('_',' ').title() for k in instruments], xlim=(0,1), xlabel='Ordered note F1')
    ax.invert_yaxis();ax.set_title('Held-out solo transcription: 50 clips per instrument', pad=15)
    ax.grid(axis='x', alpha=.2);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.07),ncol=2,frameon=False)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
    fig.tight_layout();fig.savefig(run/'instrument_f1.png', dpi=160);plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__=='__main__':
    main()
