"""Audit MIDI/reference consistency and rescore legacy SoundFont clips.

The soundfont_v1 renderer consumes performance_audio.mid directly. Some old
note_map files are stale; preserve the initial report, and write an audited one.
No inference, sample selection, or prediction modification occurs here.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import mido

from benchmark_muscriptor_local import aggregate, bootstrap, lcs, onset_matches, prf, read, save, sha256


def midi_onsets(path, shift):
    seconds = 0.0
    notes = []
    for event in mido.MidiFile(path):
        seconds += event.time  # mido applies the complete MIDI tempo map.
        if event.type == 'note_on' and event.velocity > 0:
            notes.append(dict(pitch=int(event.note) + shift, start=seconds))
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.run / 'manifest.json')
    original = read(args.run / 'results.json')
    report = copy.deepcopy(original)
    audit = dict(original_report_sha256=sha256(args.run / 'results.json'),
                 reason='Legacy FreePats note_map rendered_notes disagree with the actual rendering MIDI.',
                 method='Score every original FreePats clip against performance_audio.mid, the soundfont_v1 renderer input. Keep all cached predictions unchanged.',
                 realistic92_matching_pitch_sequences=0, per_clip=[])
    rows = {r['sample']: r for r in report['per_clip']}
    for row in manifest['samples']:
        sample = Path(row['path'])
        metadata = read(sample / 'metadata.json')
        for filename, expected in row['hashes'].items():
            assert sha256(sample / filename) == expected
        midi_space = metadata.get('midi_pitch_space', 'written' if metadata.get('audio_render')=='soundfont_rerender' else 'sounding')
        shift = 0 if midi_space == 'written' else -int(metadata.get('sounding_transpose', -2))
        gold = midi_onsets(sample / 'performance_audio.mid', shift)
        old_gold = read(sample / 'note_map.json')['rendered_notes']
        same = [n['pitch'] for n in gold] == [n['pitch_midi_written'] for n in old_gold]
        if row['corpus'] == 'realistic92':
            assert same, f'Realistic gold/MIDI discrepancy: {row["sample"]}'
            audit['realistic92_matching_pitch_sequences'] += 1
            continue
        assert metadata['audio_render'] == 'soundfont_v1'
        prediction = read(args.run / 'predictions' / row['corpus'] / (row['sample'] + '.json'))
        assert prediction['audio_sha256'] == row['hashes']['performance_audio.wav']
        predicted = [dict(n, pitch=n['pitch'] + row['written_shift']) for n in prediction['notes']]
        result = rows[row['sample']]
        audit['per_clip'].append(dict(sample=row['sample'], midi_sha256=sha256(sample / 'performance_audio.mid'),
                                     midi_pitch_space=midi_space, midi_written_shift=shift,
                                     note_map_notes=len(old_gold), midi_notes=len(gold), pitch_sequences_match=same,
                                     original_sequence=result['sequence'], original_onset_50ms=result['onset_50ms']))
        result['sequence'] = prf(lcs([n['pitch'] for n in predicted], [n['pitch'] for n in gold]), len(predicted), len(gold))
        result['onset_50ms'] = prf(onset_matches(predicted, gold), len(predicted), len(gold))
        result['gold_source'] = 'performance_audio.mid (soundfont_v1 rendering input)'
    subset = [r for r in report['per_clip'] if r['corpus']=='freepats6']
    report['results']['freepats6'].update(sequence=aggregate(subset), onset_50ms=aggregate(subset, 'onset_50ms'),
                                        sequence_clip_bootstrap_95ci=bootstrap(subset, manifest['seed']),
                                        gold_source='performance_audio.mid (soundfont_v1 rendering input)')
    assert report['results']['realistic92'] == original['results']['realistic92']
    report['gold_audit'] = audit
    save(args.run / 'GOLD_AUDIT.json', audit)
    save(args.run / 'results_audited.json', report)
    print(report['results']['freepats6'])
    print('Verified all 100 realistic pitch sequences against performance MIDI. Main result unchanged.')


if __name__ == '__main__':
    main()
