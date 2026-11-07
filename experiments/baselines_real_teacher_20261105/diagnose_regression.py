"""Post-experiment diagnostics; never changes predictions, targets, or metrics."""
from collections import Counter
import random
import sys
from common import ROOT, RUN, read, write


def main():
    import pretty_midi
    sys.path.insert(0, str(ROOT / 'DataCreate/src'))
    from datacreate.melody import parse_sounding_notes
    p = read(RUN / 'protocol.json')
    cache = read(RUN / 'cache/polytune/manifest.json')
    records = read(RUN / 'data/manifest.json')['tracks']
    known = {}
    rejected = []
    for sid in p['train_order']:
        score = parse_sounding_notes(RUN / 'bundles' / sid / 'verified_score.musicxml')
        midi = pretty_midi.PrettyMIDI(records[sid]['paths']['score_mid'])
        reference = sorted([n for ins in midi.instruments for n in ins.notes], key=lambda n: (n.start, n.pitch))
        if [n.pitch for n in reference] != [n.pitch - 2 for n in score]:
            rejected.append(sid)
            continue
        teacher = read(RUN / 'bundles' / sid / 'note_alignment_v2.json')['transcribed_notes']
        targets = read(RUN / 'supervision' / f'{sid}.json')['classes']['correct']
        windows = []
        for a, b in cache[sid]['bounds']:
            assessed = 0
            absent = []
            for note in targets:
                if note['start'] >= b or note['end'] <= a:
                    continue
                span = teacher[note['source_note']].get('score_span')
                if not span or span[1] != span[0] + 1:
                    continue
                expected = reference[span[0]]
                assessed += 1
                # Both native loaders use a 2.048s performance window plus
                # 1.024s reference context on each side (compressed to 256 mels).
                # Allow another 100ms for renderer/STFT boundary uncertainty.
                if expected.end <= a - 1.024 - .1 or expected.start >= b + 1.024 + .1:
                    absent.append({'score_index': span[0], 'performance_note_start': note['start'],
                        'reference_note_start': expected.start, 'reference_note_end': expected.end})
            windows.append({'assessed_correct_note_occurrences': assessed,
                'corresponding_reference_note_outside_context': len(absent),
                'example': absent[0] if absent else None, 'performance_window': [a, b],
                'reference_window': [max(0., a - 1.024), b + 1.024]})
        known[sid] = windows
    sizes = {}
    for size in p['sizes'][1:]:
        rng = random.Random(p['seed'])
        counts = Counter()
        examples = []
        for _ in range(p['optimizer']['steps'] * p['optimizer']['gradient_accumulation']):
            sid = rng.choice(p['train_subsets'][str(size)])
            slot = rng.choice(range(cache[sid]['windows']))
            if sid not in known:
                continue
            row = known[sid][slot]
            counts['audited_sampled_windows'] += 1
            counts['assessed_correct_note_occurrences'] += row['assessed_correct_note_occurrences']
            counts['reference_note_outside_context'] += row['corresponding_reference_note_outside_context']
            if row['example'] and len(examples) < 4:
                examples.append({'sample': sid, **row})
        sizes[str(size)] = {**counts,
            'outside_fraction': counts['reference_note_outside_context'] / max(1, counts['assessed_correct_note_occurrences']),
            'examples': examples}
    outliers = []
    for count, sid in [(10, '106'), (50, '102'), (50, '104')]:
        for n in [0, count]:
            midi = pretty_midi.PrettyMIDI(RUN / 'predictions/laddersym' / f'n{n:02}' / sid / 'mix.mid')
            tracks = {}
            for ins in midi.instruments:
                tracks[ins.name] = {'notes': len(ins.notes),
                    'top_pitches': Counter(n.pitch for n in ins.notes).most_common(5),
                    'top_2s_bins': Counter(int(n.start / 2.048) for n in ins.notes).most_common(4),
                    'notes_after_actual_audio_end': sum(int(n.start >= records[sid]['duration']) for n in ins.notes)}
            outliers.append({'sample': sid, 'training_count': n, 'tracks': tracks})
    report = {'scope': 'Post-hoc diagnosis; frozen experiment outputs unchanged',
        'reference_timing_method': 'Only recordings with identical canonical XML and sounding MIDI pitch sequences; teacher-assigned correct-note identities; 100ms extra tolerance',
        'auditable_training_recordings': list(known), 'excluded_ambiguous_reference_recordings': rejected,
        'training_context_coverage': sizes, 'laddersym_outlier_native_notes': outliers}
    write(RUN / 'regression_diagnostics.json', report)
    print('Unambiguous reference recordings:', len(known), 'of 50')
    for n, row in sizes.items():
        print(n, {k: v for k, v in row.items() if k != 'examples'}, flush=True)
    print('Example', sizes['50']['examples'][0])
    print('Native outliers', outliers)


if __name__ == '__main__':
    main()
