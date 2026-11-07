"""Describe completed results, including output failure modes and training exposure."""
from collections import Counter
from common import RUN, read, write


def main():
    r = read(RUN / 'results.json')
    p = read(RUN / 'protocol.json')
    exposure = read(RUN / 'training_diagnostics.json')
    diagnostics = {}
    lines = ['# Interpretation and diagnostics', '',
        'These scores measure agreement with AudioEval v9 + MuScriptor medium, using exact canonical reference-note sets and error types. They are not an independent estimate of human-verified accuracy.', '',
        '## Observed outcome', '',
        '**Every fine-tuned condition scored below its unchanged pretrained baseline.** Under this fixed 200-update, single-seed protocol, the real-data fine-tunes did not improve the requested reference-score labeling metric. This is a negative result for this training setup, not proof that these architectures cannot benefit from real data.', '',
        '- PolyTune: baseline F1 **6.78%**; highest fine-tuned F1 **6.29%** at 10 recordings. Its exact matches fell from 8 to 5, while predicted error labels fell from 160 to 83. The precision gain was outweighed by lower recall. The paired change interval at 10 recordings is -5.63 to +3.43 percentage points, so that small observed decline is inconclusive.',
        '- LadderSym: baseline F1 **6.86%**; highest fine-tuned F1 **2.35%** at 30 recordings. The 20/30/40-recording models predict only 18/9/23 error labels across all 17 scored recordings. At 50 recordings it instead produces 487 labels, 432 of which cannot be located in the reference score. Samples 102 and 104 account for 245 and 226 predictions respectively, with no exact matches in either sample.',
        '- None of the 12 conditions exactly recovered any of the 5 extra-note or 13 repetition labels. Missed-note recall was also very low. Most exact matches came from wrong-note labels.',
        '- Allowing half credit for a wrong error type at the correct score location does not reverse the result: every fine-tuned condition also scores below its corresponding baseline under that secondary metric.',
        '- Average training loss decreased in every run, but held-out error-label F1 did not. The observations are consistent with poor transfer and unstable output classification; they do not by themselves identify a single causal mechanism. Sparse teacher errors, fixed training budget, input/target representation differences, and the unchanged score-localization adapter are possible contributors.', '',
        'The checkpoints remain experimental artifacts. No result here supports replacing the existing baseline with one of these fine-tunes. See `RESULTS.md` for the complete precision/recall table and paired intervals.', '',
        '## Native output and localization', '',
        'Counts below use only the 17 scored recordings. Native note counts precede conversion into error-label ranges. Unlocated label ranges and unclassified notes remain false positives in the primary score.', '',
        '| Model | N | Native Correct | Native Extra | Native Missing | Predicted error labels | Unlocated labels | Exact matches / 76 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in ['polytune', 'laddersym']:
        diagnostics[model] = {}
        for n in p['sizes']:
            totals = Counter()
            for sid in r['scored_test_ids']:
                doc = read(RUN / 'converted' / model / f'n{n:02}' / f'{sid}.json')
                b = doc['baseline_labeling']
                totals.update(b['native_counts'])
                totals['unlocated_labels'] += b['unlocated_events']
                totals['unclassified_notes'] += b['unclassified_count']
            score = r['results'][model][str(n)]['strict']
            diagnostics[model][str(n)] = dict(totals)
            lines.append(f"| {model} | {n} | {totals['correct']} | {totals['extra']} | {totals['missing']} | {score['predicted']} | {totals['unlocated_labels']} | {score['credit']:g} / {score['gold']} |")
    lines += ['', '## Error-type recall', '',
        'Recall is exact label-location/type recall. A partial overlap with a longer error range receives no strict credit.', '',
        '| Model | N | Wrong (49) | Missed (9) | Extra (5) | Repetition (13) |',
        '|---|---:|---:|---:|---:|---:|']
    for model in ['polytune', 'laddersym']:
        for n in p['sizes']:
            row = r['results'][model][str(n)]['by_type']
            values = ' | '.join(f"{row[k]['recall']:.1%}" for k in ['wrong_note', 'missed_note', 'extra_note', 'repetition'])
            lines.append(f'| {model} | {n} | {values} |')
    lines += ['', '## Fixed training budget', '',
        '| Recordings | Available windows | Unique windows sampled | Sampled windows with native errors / 800 |',
        '|---:|---:|---:|---:|']
    for n in p['sizes'][1:]:
        x = exposure['polytune'][str(n)]
        lines.append(f"| {n} | {x['available_windows']} | {x['unique_sampled_windows']} | {x['sampled_windows_overlapping_native_errors']} / 800 |")
    lines += ['',
        'Both models use the same window draws. More recordings provide greater diversity but fewer revisits per recording under this budget. A non-monotonic learning curve does not establish that more data is intrinsically harmful, or that 200 updates is sufficient for every dataset size.', '',
        '## Coverage and limitations', '',
        '- All 21 samples were decoded for all 12 conditions. Four teacher abstentions (101, 105, 114, 117) are excluded from F1, not relabeled as clean.',
        '- The 17 scored samples contain 677 canonical score notes. Seven are marked unassessed by the teacher (100: three; 119: four). Unpublished teacher decisions are not independent negative labels; disagreement can reflect a teacher omission.',
        '- Exact range matching is sensitive to grouping: finding some correct affected notes inside a different range earns no strict match. Native transcription, error classification, and score localization all contribute to the final score.',
        '- Original reference renders can contain expanded repeats, ornaments, or chord notes outside the existing canonical parser\'s note list. This is particularly relevant to 100, 104, and 109; see `data_audit.json` and `README.md`.',
        '- There is no detected identical training/test audio or full canonical pitch sequence, and available source-score names are disjoint. Missing metadata prevents a complete proof of piece-level independence.',
        '- The bootstrap intervals in `RESULTS.md` use paired resampling of recordings; there is only one training seed and one nested subset ordering. The highest score on this test set should be treated as exploratory, not a deployment selection validated on another holdout.', '',
        'See `results.json` for every per-recording match and the secondary type-mismatch-credit metric, `training_diagnostics.json` for training loss and exposure, and `final_audit.json` for preservation and integrity checks.', '']
    write(RUN / 'output_diagnostics.json', diagnostics)
    (RUN / 'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
