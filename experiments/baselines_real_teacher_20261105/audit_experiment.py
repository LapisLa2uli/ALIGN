"""Read-only integrity and supervision checks, before and after the campaign."""
import argparse
from collections import Counter
import sys
from common import ROOT, RUN, read, write, sha, verify_inputs
from supervision import valid_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'DataCreate/src'))
    import pretty_midi
    from datacreate.melody import parse_sounding_notes, canonical_note_location
    p = read(RUN / 'protocol.json')
    inputs = verify_inputs()
    assert sha(RUN / 'inputs.json') == p['inputs_sha256']
    for rel, digest in p['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in p['experiment_code_sha256'].items():
        assert sha(RUN / rel) == digest, rel
    if (RUN / 'runtime_sources.json').exists():
        for rel, digest in read(RUN / 'runtime_sources.json')['files'].items():
            assert sha(ROOT / rel) == digest, rel
    for entry in p['initial_checkpoints'].values():
        assert sha(entry['path']) == entry['sha256']
    for sid, files in inputs['protected_live_files'].items():
        for name, digest in files.items():
            assert sha(ROOT / 'DataCreate/samples' / sid / name) == digest, (sid, name)
    publication = read(RUN / 'test_teacher_ui/manifest.json')
    assert publication['status'] == 'complete'
    for sid, files in publication['after'].items():
        for name, digest in files.items():
            assert sha(ROOT / 'DataCreate/samples' / sid / name) == digest, (sid, name)
    records = read(RUN / 'data/manifest.json')['tracks']
    scores = {}
    reference_pitch_differences = {}
    roundtrips = 0
    for sid in p['train_order'] + p['test_ids']:
        notes = parse_sounding_notes(RUN / 'bundles' / sid / 'verified_score.musicxml')
        scores[sid] = tuple(n.pitch for n in notes)
        native = pretty_midi.PrettyMIDI(records[sid]['paths']['score_mid'])
        midi_pitches = [n.pitch for ins in native.instruments for n in ins.notes]
        expected = Counter(n.pitch - 2 for n in notes)
        if expected != Counter(midi_pitches):
            reference_pitch_differences[sid] = {'xml_count': sum(expected.values()), 'midi_count': len(midi_pitches),
                'xml_only': dict(expected - Counter(midi_pitches)), 'midi_only': dict(Counter(midi_pitches) - expected)}
        if sid in p['test_ids']:
            for name in ['extra', 'removed', 'correct']:
                from pathlib import Path
                assert not Path(records[sid]['paths'][name]).exists(), sid
            gold = read(RUN / 'teacher_gold' / f'{sid}.json')
            assert sha(RUN / 'teacher_gold' / f'{sid}.json') == p['teacher_gold_sha256'][sid]
            assert all(canonical_note_location(l, score_event_count=len(notes)) is not None for l in gold['labels'])
            continue
        target = read(RUN / 'supervision' / f'{sid}.json')
        assert target['usable']
        assert read(RUN / 'cache/polytune' / f'{sid}.json')['bounds'] == read(RUN / 'cache/laddersym' / f'{sid}.json')['bounds'], sid
        for name, expected_notes in target['classes'].items():
            native = pretty_midi.PrettyMIDI(records[sid]['paths'][name])
            observed = sorted((n.pitch, n.start, n.end) for ins in native.instruments for n in ins.notes)
            expected_notes = sorted((n['pitch'], n['start'], n['end']) for n in expected_notes)
            assert len(observed) == len(expected_notes), (sid, name)
            for actual, wanted in zip(observed, expected_notes):
                assert actual[0] == wanted[0] and max(abs(actual[i] - wanted[i]) for i in [1, 2]) < .003, (sid, name, actual, wanted)
            roundtrips += len(observed)
        for model in ['polytune', 'laddersym']:
            cache = read(RUN / 'cache' / model / f'{sid}.json')
            assert sha(RUN / 'cache' / model / f'{sid}.pt') == cache['sha256']
            assert all(valid_window(a, b, target['unknown']) for a, b in cache['bounds'])
    shared_sequences = [(a, b) for a in p['train_order'] for b in p['test_ids'] if scores[a] == scores[b]]
    assert not shared_sequences, shared_sequences
    if args.final:
        for model in ['polytune', 'laddersym']:
            for n in p['sizes']:
                if n:
                    folder = RUN / 'training' / model / f'n{n:02}'
                    done = read(folder / 'done.json')
                    assert done['steps'] == 200 and done['train_ids'] == p['train_subsets'][str(n)]
                    assert done['protocol_sha256'] == sha(RUN / 'protocol.json')
                    checkpoint_hash = sha(folder / 'model.pt')
                    assert checkpoint_hash == done['checkpoint_sha256']
                else:
                    checkpoint_hash = p['initial_checkpoints'][model]['sha256']
                for sid in p['test_ids']:
                    midi = RUN / 'predictions' / model / f'n{n:02}' / sid / 'mix.mid'
                    done = read(midi.with_suffix('.done.json'))
                    assert sha(midi) == done['sha256'] and done['checkpoint_sha256'] == checkpoint_hash
                    assert done['protocol_sha256'] == sha(RUN / 'protocol.json')
                    assert done['test_labels_read'] is False
        report = read(RUN / 'results.json')
        assert report['protocol_sha256'] == sha(RUN / 'protocol.json')
        assert report['test_ids'] == p['test_ids'] and len(report['scored_test_ids']) == 17
        for model in ['polytune', 'laddersym']:
            for n in p['sizes']:
                rows = report['samples'][model][str(n)]
                assert [row['sample'] for row in rows] == p['test_ids']
                totals = report['results'][model][str(n)]
                assert totals['inferred_recordings'] == 21 and totals['scored_recordings'] == 17
                for metric in ['strict', 'official']:
                    value = totals[metric]
                    assert value['gold'] == 76
                    for key in ['credit', 'predicted', 'gold']:
                        assert value[key] == sum(row[metric][key] for row in rows if row['scored'])
                    expected_f1 = 2 * value['credit'] / (value['predicted'] + value['gold'])
                    assert abs(value['f1'] - expected_f1) < 1e-12
    result = {'phase': 'final' if args.final else 'data', 'status': 'passed',
        'source_and_original_checkpoint_hashes_unchanged': True, 'human_labels_unchanged': True,
        'published_test_labels_verified': True, 'test_native_targets_absent': True,
        'native_target_notes_roundtripped': roundtrips, 'shared_full_pitch_sequences': shared_sequences,
        'reference_midi_pitch_count_differences': reference_pitch_differences,
        'protocol_sha256': sha(RUN / 'protocol.json')}
    write(RUN / ('final_audit.json' if args.final else 'data_audit.json'), result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
