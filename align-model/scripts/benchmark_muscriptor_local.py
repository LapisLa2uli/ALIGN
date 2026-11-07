"""Reproducible, score-free evaluation of the local MuScriptor HTTP server.

Dataset 9.2 has renderer/MIDI timing drift, so pitch-sequence LCS is primary.
Exact-pitch/50-ms-onset matching is reported separately, including on FreePats.
Only performance audio is sent to the model. Gold and metadata stay in scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import httpx
import numpy as np
import soundfile as sf


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def lcs(left, right):
    row = [0] * (len(right) + 1)
    for pitch in left:
        previous = 0
        for col, other in enumerate(right, 1):
            old = row[col]
            row[col] = previous + 1 if pitch == other else max(row[col], row[col - 1])
            previous = old
    return row[-1]


def onset_matches(predicted, gold, tolerance=0.05):
    # Same maximum one-to-one per-pitch matcher as transcription/evaluate.py.
    pp, gg = defaultdict(list), defaultdict(list)
    for n in predicted:
        pp[n['pitch']].append(n['start'])
    for n in gold:
        gg[n['pitch']].append(n['start'])
    matched = 0
    for pitch in pp.keys() & gg.keys():
        a, b = sorted(pp[pitch]), sorted(gg[pitch])
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] < b[j] - tolerance - 1e-9:
                i += 1
            elif b[j] < a[i] - tolerance - 1e-9:
                j += 1
            else:
                matched += 1
                i += 1
                j += 1
    return matched


def prf(matched, predicted, gold):
    return dict(matched=int(matched), predicted=int(predicted), gold=int(gold),
                precision=matched / max(predicted, 1), recall=matched / max(gold, 1),
                f1=2 * matched / max(predicted + gold, 1))


def aggregate(rows, key='sequence'):
    return prf(*(sum(r[key][field] for r in rows) for field in ('matched', 'predicted', 'gold')))


def bootstrap(rows, seed):
    counts = np.array([[r['sequence'][k] for k in ('matched', 'predicted', 'gold')] for r in rows])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        m, p, g = counts[rng.integers(len(rows), size=len(rows))].sum(axis=0)
        values.append(2 * m / max(p + g, 1))
    return list(map(float, np.quantile(values, [0.025, 0.975])))


def self_check():
    assert lcs([60, 60, 62], [60, 62]) == 2
    assert lcs([60, 62], [62, 60]) == 1
    assert lcs([], [60]) == 0
    gold = [dict(pitch=60, start=0), dict(pitch=60, start=0.1)]
    assert onset_matches([dict(pitch=60, start=0.05)], gold) == 1
    assert onset_matches([dict(pitch=61, start=0)], gold) == 0
    assert onset_matches([dict(pitch=60, start=0.05001)], gold[:1]) == 0
    assert prf(2, 3, 4)['f1'] == 4 / 7


def transcribe(client, endpoint, wav):
    starts, ends = {}, {}
    complete = False
    began = time.perf_counter()
    with wav.open('rb') as stream:
        with client.stream('POST', endpoint + '/transcribe',
                           files={'file': (wav.name, stream, 'audio/wav')},
                           data={'detect_tempo': 'false'}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                event = json.loads(line[6:])
                if event['type'] == 'start':
                    starts[event['index']] = event
                elif event['type'] == 'end':
                    ends[event['start_event_index']] = event['end_time']
                elif event['type'] == 'transcription_complete':
                    complete = True
    if not complete or starts.keys() != ends.keys():
        raise RuntimeError('Incomplete transcription event stream')
    notes = [dict(index=index, pitch=event['pitch'], start=event['start_time'],
                  end=ends[index], instrument=event['instrument']) for index, event in starts.items()]
    return sorted(notes, key=lambda n: (n['start'], n['index'])), time.perf_counter() - began


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('E:/ALIGN_Data'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8222')
    args = parser.parse_args()
    self_check()
    workspace = Path(__file__).resolve().parents[2]
    split_dir = workspace / 'align-model/runs/realistic92-transcriber-v1'
    freeze = read(split_dir / 'FREEZE.json')
    assert sha256(split_dir / 'split.json') == freeze['split_sha256']
    assert sha256(split_dir / 'test_hashes.json') == freeze['test_hashes_sha256']
    names = sorted(random.Random(args.seed).sample(sorted(read(split_dir / 'split.json')['splits']['test']), args.count))
    expected = read(split_dir / 'test_hashes.json')
    samples = [('realistic92', args.data_root / 'outputRaw_realistic_10k' / name) for name in names]
    samples += [('freepats6', p) for p in sorted((args.data_root / 'outputRaw_sf_10k').glob('synth_*')) if p.is_dir()]
    model_files = list((Path.home() / '.cache/huggingface/hub/models--MuScriptor--muscriptor-medium/snapshots').glob('*/model.safetensors'))
    assert len(model_files) == 1, 'Specify the model revision explicitly if multiple revisions are cached'
    manifest = dict(schema='muscriptor-synthetic-eval-v1', seed=args.seed,
                    selection='random.Random(seed).sample(sorted(frozen_test_names), count)',
                    model='medium', model_sha256=sha256(model_files[0]), model_revision=model_files[0].parent.name,
                    muscriptor_commit=subprocess.check_output(['git', '-C', str(workspace / 'muscriptor'), 'rev-parse', 'HEAD'], text=True).strip(),
                    device='cuda', dtype='float16', instrument_hint=None, tempo_detection=False,
                    postprocessing='none; raw streamed note events; all instruments retained',
                    split_sha256=freeze['split_sha256'], samples=[])
    for corpus, sample in samples:
        hashes = {f: sha256(sample / f) for f in ('performance_audio.wav', 'note_map.json', 'metadata.json')}
        if corpus == 'realistic92':
            for f in ('performance_audio.wav', 'note_map.json'):
                assert hashes[f] == expected[sample.name][f], f'Frozen input changed: {sample.name}/{f}'
        metadata = read(sample / 'metadata.json')
        shift = metadata.get('effective_audio_transpose')
        if shift is None:
            shift = 0 if metadata.get('audio_pitch_space') == 'written' else -int(metadata.get('sounding_transpose', -2))
        manifest['samples'].append(dict(corpus=corpus, sample=sample.name, path=str(sample),
                                        group='procedural' if sample.name.startswith('synth_gen_') else 'rawdata',
                                        written_shift=int(shift), hashes=hashes,
                                        audio_seconds=sf.info(sample / 'performance_audio.wav').duration))
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest, 'Frozen run configuration or inputs changed'
    else:
        save(manifest_path, manifest)
    print(json.dumps({'frozen_clips':len(samples), 'audio_minutes':sum(r['audio_seconds'] for r in manifest['samples']) / 60,
                      'realistic_groups':dict(Counter(r['group'] for r in manifest['samples'] if r['corpus']=='realistic92'))}), flush=True)
    rows = []
    with httpx.Client(timeout=httpx.Timeout(600, connect=15), trust_env=False) as client:
        client.get(args.endpoint + '/health').raise_for_status()
        for index, row in enumerate(manifest['samples'], 1):
            sample = Path(row['path'])
            cache = args.out / 'predictions' / row['corpus'] / (row['sample'] + '.json')
            if cache.exists():
                prediction = read(cache)
                assert prediction['audio_sha256'] == row['hashes']['performance_audio.wav']
                assert prediction['model_sha256'] == manifest['model_sha256']
            else:
                notes, elapsed = transcribe(client, args.endpoint, sample / 'performance_audio.wav')
                prediction = dict(notes=notes, inference_seconds=elapsed,
                                  audio_sha256=row['hashes']['performance_audio.wav'], model_sha256=manifest['model_sha256'])
                save(cache, prediction)
            predicted = [dict(n, pitch=int(n['pitch']) + row['written_shift']) for n in prediction['notes']]
            gold = [dict(pitch=int(n['pitch_midi_written']), start=float(n['start_sec']), end=float(n['end_sec']))
                    for n in read(sample / 'note_map.json')['rendered_notes']]
            result = dict(sample=row['sample'], corpus=row['corpus'], group=row['group'],
                          audio_seconds=row['audio_seconds'], inference_seconds=prediction['inference_seconds'],
                          instruments=dict(Counter(n['instrument'] for n in predicted)),
                          sequence=prf(lcs([n['pitch'] for n in predicted], [n['pitch'] for n in gold]), len(predicted), len(gold)),
                          onset_50ms=prf(onset_matches(predicted, gold), len(predicted), len(gold)))
            rows.append(result)
            save(args.out / 'progress.json', dict(completed=len(rows), total=len(samples), per_clip=rows))
            print(f"{index}/{len(samples)} {row['sample']} seq_f1={result['sequence']['f1']:.4f} onset_f1={result['onset_50ms']['f1']:.4f} sec={result['inference_seconds']:.1f}", flush=True)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), model=manifest['model'], clips=len(rows),
                  manifest_sha256=sha256(manifest_path),
                  primary_metric='Micro pitch-sequence LCS F1; exact written MIDI pitch; timing/instrument identity ignored',
                  caveat='Not official ALIGN score-event/error-detection F1. 9.2 onset times have renderer/MIDI drift.',
                  results={}, per_clip=rows)
    for corpus in ('realistic92', 'freepats6'):
        subset = [r for r in rows if r['corpus'] == corpus]
        report['results'][corpus] = dict(clips=len(subset), sequence=aggregate(subset), onset_50ms=aggregate(subset, 'onset_50ms'),
                                       sequence_clip_bootstrap_95ci=bootstrap(subset, args.seed),
                                       audio_seconds=sum(r['audio_seconds'] for r in subset),
                                       inference_seconds=sum(r['inference_seconds'] for r in subset))
        if corpus == 'realistic92':
            report['results'][corpus]['by_group'] = {group:dict(clips=len(s), **aggregate(s))
                    for group in ('procedural', 'rawdata') if (s := [r for r in subset if r['group']==group])}
    baseline_path = split_dir / 'TEST_CTC_R2A.json'
    baseline = {r['sample']: r for r in read(baseline_path)['per_clip']}
    paired = [baseline[name] for name in names]
    for row in rows:
        if row['corpus']=='realistic92':
            assert row['sequence']['gold'] == baseline[row['sample']]['gold']
    report['saved_ctc_same_100_clips'] = dict(source=str(baseline_path), source_sha256=sha256(baseline_path),
            **prf(*(sum(r[k] for r in paired) for k in ('matched', 'predicted', 'gold'))))
    save(args.out / 'results.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_clip'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
