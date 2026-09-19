"""Recover the frozen outputRaw validation inputs, requiring exact audio hashes.

Only validation rows are opened. Original bundles remain unchanged. Missing
inputs are deterministically rendered from the source score and sample seed;
their performance WAV and MIDI must match the frozen manifest byte for byte.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import sqlite3
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_HASH = '40f4900c142862fd06a69b1adcb2049701655db5ccb0266b925ff8cdc04bce6d'
TARGET_HASH = '137d6069c2909e111ef663f87bb193801e79f056916c695d343b1a641bf61584'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore(job):
    row, target, output, existing = job
    from synthpipeline.config import SynthConfig
    from synthpipeline.errors import inject_error
    from synthpipeline.note_map import build_note_map, tag_clean_notes
    from synthpipeline.render import render_score_as_clarinet
    from synthpipeline.scoregen import load_score, snippet_score, write_musicxml

    dest = Path(output) / row['sample']
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / 'recovery.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        for name, digest in saved['output_sha256'].items():
            if sha(dest / name) != digest:
                raise ValueError(f'Changed recovered input: {dest / name}')
        return saved
    source = next((Path(root) / row['sample'] for root in existing
                   if (Path(root) / row['sample'] / 'performance_audio.wav').exists()
                   and sha(Path(root) / row['sample'] / 'performance_audio.wav')
                   == row['source_hashes']['performance_audio.wav']), None)
    names = ('performance_audio.wav', 'reference_audio.wav',
             'performance_audio.mid', 'reference_audio.mid',
             'verified_score.musicxml', 'performance_score.musicxml')
    if source is not None:
        for name in names:
            link = dest / name
            if not link.exists():
                link.symlink_to((source / name).resolve())
        lineage = json.loads((source / 'note_map.json').read_text())
        mode = 'existing_exact_audio'
    else:
        config = SynthConfig.load(ROOT / 'synth-pipeline/config/rawdata_sf_10k.yaml')
        config.paths['score_root'] = str(ROOT / 'RawData/Score')
        score_path = ROOT / 'RawData/Score' / (row['source'] + '.musicxml')
        if not score_path.exists():
            choices = list((ROOT / 'RawData/Score').glob(row['source'] + '.*'))
            if len(choices) != 1:
                raise ValueError(f'Cannot resolve score {row["source"]}')
            score_path = choices[0]
        rng = random.Random(int(row['sample'].rsplit('_', 1)[1]))
        clean, snippet = snippet_score(load_score(score_path, config), rng, config)
        write_musicxml(clean, dest / 'verified_score.musicxml')
        tag_clean_notes(clean)
        result = inject_error(copy.deepcopy(clean), rng, config)
        write_musicxml(result.score, dest / 'performance_score.musicxml')
        lineage = build_note_map(clean, result.score)
        dc_config = config.to_datacreate_config()
        log = logging.getLogger(row['sample'])
        for name, score in [('reference', clean), ('performance', result.score)]:
            render_score_as_clarinet(
                dc_config, dest / ('verified_score.musicxml' if name == 'reference'
                                   else 'performance_score.musicxml'),
                dest / f'{name}_audio.wav', log, config.clarinet_program(),
                midi_backend='music21', score=score, sounding_transpose=-2,
                pitch_bends=result.extra.get('pitch_bends') if name == 'performance' else None,
                bpm=result.bpm if name == 'performance' else None)
        mode = 'deterministic_reconstruction'
    for name in ('performance_audio.wav', 'performance_audio.mid'):
        if sha(dest / name) != row['source_hashes'][name]:
            raise ValueError(f'Frozen hash mismatch: {row["sample"]}/{name}')
    for key in ('clean_notes', 'performed_notes', 'deleted_clean_notes'):
        if lineage[key] != target[key]:
            raise ValueError(f'Canonical lineage differs: {row["sample"]}/{key}')
    saved = dict(sample=row['sample'], mode=mode, original=row['sample_dir'],
                 source=str(source) if source else None,
                 canonical_target_record=row['target_record'],
                 exact_performance_audio=True, exact_performance_midi=True,
                 exact_symbolic_lineage=True,
                 output_sha256={name: sha(dest / name) for name in names})
    marker.write_text(json.dumps(saved, indent=2) + '\n')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', type=Path, default=ROOT / 'upload')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--first-n', type=int)
    args = parser.parse_args()
    assert sha(args.upload / 'split.json') == SPLIT_HASH, 'Different frozen split'
    assert sha(args.upload / 'canonical_dev_targets.sqlite') == TARGET_HASH, 'Different targets'
    rows = json.loads((args.upload / 'split.json').read_text())['val']
    assert len(rows) == len({r['sample'] for r in rows}) == 358
    if args.first_n:
        rows = rows[:args.first_n]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    with sqlite3.connect(f'file:{(args.upload / "canonical_dev_targets.sqlite").resolve().as_posix()}?mode=ro', uri=True) as db:
        for row in rows:
            record = db.execute('SELECT split,source_hashes,payload FROM targets WHERE ordinal=?',
                                (row['target_record'],)).fetchone()
            assert record[0] == 'val' and json.loads(record[1]) == row['source_hashes']
            target = json.loads(zlib.decompress(record[2]))
            jobs.append((row, target, str(args.out.resolve()),
                         [str(ROOT / 'synth-pipeline/output_2k_rawdata')]))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(restore, job): job[0]['sample'] for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
                print(f'{len(results)}/{len(jobs)} {futures[future]} verified', flush=True)
            except Exception as exc:
                results.append(dict(sample=futures[future], error=str(exc)))
                print(f'{futures[future]} FAILED: {exc}', flush=True)
            (args.out / 'recovery_progress.json').write_text(json.dumps(results, indent=2) + '\n')
    if any('error' in row for row in results):
        raise SystemExit('Some inputs could not be verified; see recovery_progress.json')


if __name__ == '__main__':
    main()
