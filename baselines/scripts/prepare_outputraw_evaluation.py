"""Prepare all frozen outputRaw validation inputs and audited native targets."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import sqlite3
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'baselines/common'))
sys.path.insert(0, str(ROOT / 'align-model/src'))
from prepare_dataset import output_paths, load_audio_16k, write_wav16, write_label_midi, copy_reference_midi
from synth_supervision import capture_missing_rests, _MISSING, MidiTimeline
from restore_outputraw_validation import sha, SPLIT_HASH, TARGET_HASH


def prepare(job):
    row, target, root, out = job
    from music21 import note
    from synthpipeline.config import SynthConfig
    from synthpipeline.errors import inject_error
    from synthpipeline.note_map import build_note_map, tag_clean_notes
    from synthpipeline.scoregen import load_score, snippet_score, write_musicxml
    from alignmodel.joint.index import ScoreEventIndex
    bundle = Path(root) / row['sample']
    recovered = json.loads((bundle / 'recovery.json').read_text())
    for name, digest in recovered['output_sha256'].items():
        assert sha(bundle / name) == digest
    index = ScoreEventIndex.from_musicxml(bundle / 'verified_score.musicxml', target)
    classes = {'correct': [], 'extra': [], 'removed': []}
    def event(start, end, pitch):
        return dict(start=float(start), end=float(end), pitch=int(pitch))
    for item in index.rendered_events:
        cls = 'correct' if item.relationship == 'match' and not item.is_copy else 'extra'
        classes[cls].append(event(item.start, item.end, item.pitch - 2))
        if item.relationship == 'substitute' and not item.is_copy:
            assert item.score_span is not None
            for score_id in range(*item.score_span):
                classes['removed'].append(event(item.start, item.end, index.events[score_id].pitch - 2))
    # Reproduce only the symbolic edits to carry deleted-note identity through
    # later timing changes; no timestamp is inferred from padded error labels.
    config = SynthConfig.load(ROOT / 'synth-pipeline/config/rawdata_sf_10k.yaml')
    score_path = next((ROOT / 'RawData/Score').glob(row['source'] + '.*'))
    rng = random.Random(int(row['sample'].rsplit('_', 1)[1]))
    clean, _ = snippet_score(load_score(score_path, config), rng, config)
    with tempfile.TemporaryDirectory(prefix='outputraw-supervision-') as temp:
        write_musicxml(clean, Path(temp) / 'clean.musicxml')
        tag_clean_notes(clean)
        with capture_missing_rests():
            result = inject_error(copy.deepcopy(clean), rng, config)
        write_musicxml(result.score, Path(temp) / 'performed.musicxml')
    lineage = build_note_map(clean, result.score)
    for key in ('clean_notes', 'performed_notes', 'deleted_clean_notes'):
        assert lineage[key] == target[key], (row['sample'], key)
    rests = {}
    for rest in result.score.recurse().getElementsByClass(note.Rest):
        clean_id = getattr(rest, _MISSING, None)
        if clean_id is not None:
            start = float(rest.getOffsetInHierarchy(result.score))
            interval = (start, start + float(rest.duration.quarterLength))
            rests[clean_id] = min(rests.get(clean_id, interval), interval)
    timeline = MidiTimeline(bundle / 'performance_audio.mid')
    for score_id in sorted(index.deleted_event_indices):
        score_event = index.events[score_id]
        spans = [rests[source] for source in score_event.source_indices]
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        classes['removed'].append(event(timeline.seconds_at(start), timeline.seconds_at(end), score_event.pitch - 2))
    # Verify that every performed target is backed by the exact input MIDI.
    assert len(timeline.events) == len(index.rendered_events)
    max_midi_time_delta = 0.0
    for midi, actual in zip(timeline.events, index.rendered_events):
        assert midi['pitch'] == actual.pitch - 2
        max_midi_time_delta = max(max_midi_time_delta, abs(midi['start']-actual.start), abs(midi['end']-actual.end))
    paths = output_paths(Path(out), row['sample'])
    perf, ref = [load_audio_16k(bundle / (name + '_audio.wav')) for name in ('performance', 'reference')]
    length = max(len(perf), len(ref))
    write_wav16(paths['mistake_wav'], np.pad(perf, (0, length-len(perf))))
    write_wav16(paths['score_wav'], np.pad(ref, (0, length-len(ref))))
    copy_reference_midi(bundle / 'reference_audio.mid', paths['score_mid'])
    for cls, notes in classes.items():
        write_label_midi(paths[cls], notes)
    gold = dict(classes=classes, canonical_score_events=len(index.events),
                deleted_event_indices=sorted(index.deleted_event_indices),
                maximum_midi_vs_canonical_time_delta=max_midi_time_delta)
    gold_path = Path(out) / 'native_targets' / (row['sample'] + '.json')
    gold_path.parent.mkdir(exist_ok=True)
    gold_path.write_text(json.dumps(gold, indent=2) + '\n')
    return dict(track_id=row['sample'], bundle=str(bundle.resolve()), real_test=False,
                source=row['source'], written_duration_s=length/16000,
                original_performance_sha256=row['source_hashes']['performance_audio.wav'],
                recovery_mode=recovered['mode'], n_correct=len(classes['correct']),
                n_extra=len(classes['extra']), n_removed=len(classes['removed']),
                maximum_midi_vs_canonical_time_delta=max_midi_time_delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovered', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    assert sha(ROOT / 'upload/split.json') == SPLIT_HASH
    assert sha(ROOT / 'upload/canonical_dev_targets.sqlite') == TARGET_HASH
    split = json.loads((ROOT / 'upload/split.json').read_text())
    rows = split['val']; assert len(rows) == 358
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    with sqlite3.connect(f'file:{ROOT / "upload/canonical_dev_targets.sqlite"}?mode=ro', uri=True) as db:
        for row in rows:
            saved = db.execute('SELECT split,source_hashes,payload FROM targets WHERE ordinal=?', (row['target_record'],)).fetchone()
            assert saved[0] == 'val' and json.loads(saved[1]) == row['source_hashes']
            jobs.append((row, json.loads(zlib.decompress(saved[2])), str(args.recovered), str(args.out)))
    records, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, job): job[0]['sample'] for job in jobs}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                records[sample] = future.result()
                print(f'{len(records)}/{len(rows)} {sample} prepared', flush=True)
            except Exception as exc:
                errors.append(dict(sample=sample, error=repr(exc)))
                print(f'{sample} FAILED {exc!r}', flush=True)
    (args.out / 'preparation_audit.json').write_text(json.dumps(dict(records=records, errors=errors), indent=2)+'\n')
    if errors:
        raise SystemExit('Preparation failed; no evaluation split published')
    assert len(records) == 358
    manifest = dict(tracks=records, frozen_outputraw_split_sha256=SPLIT_HASH,
                    canonical_targets_sha256=TARGET_HASH)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    derived = dict(midi_filename={str(i): sample+'.midi' for i,sample in enumerate(sorted(records))},
                   split={str(i): 'validation' for i in range(len(records))})
    (args.out / 'split.json').write_text(json.dumps(derived, indent=2)+'\n')


if __name__ == '__main__':
    main()
