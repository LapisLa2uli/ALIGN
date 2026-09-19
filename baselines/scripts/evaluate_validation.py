"""Run native validation inference in disjoint shards and audit full coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASELINES = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def balanced_shards(ids, durations, count):
    if not 1 <= count <= len(ids) or len(ids) != len(set(ids)):
        raise ValueError('Require unique IDs and nonempty shards')
    shards, seconds = [[] for _ in range(count)], [0.0] * count
    for tid in sorted(ids, key=lambda x: (-durations[x], x)):
        index = min(range(count), key=lambda i: (seconds[i], i))
        shards[index].append(tid)
        seconds[index] += durations[tid]
    return [sorted(shard) for shard in shards]


def merge_predictions(expected, directories, destination):
    selections, paths = [], {}
    for directory in directories:
        ids = json.loads((directory / 'evaluated_ids.json').read_text())
        actual = {p.parent.name: p for p in directory.glob('*/mix.mid')}
        if len(ids) != len(set(ids)) or set(ids) != set(actual):
            raise ValueError(f'Incomplete or duplicate shard: {directory}')
        selections.extend(ids)
        paths.update(actual)
    if len(selections) != len(set(selections)) or set(selections) != set(expected):
        raise ValueError('Shards overlap or do not cover the original validation set')
    destination.mkdir()
    for tid in sorted(expected):
        (destination / tid).symlink_to(paths[tid].parent, target_is_directory=True)
    write_json(destination / 'evaluated_ids.json', sorted(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--polytune-gpus', default='3,6')
    parser.add_argument('--laddersym-gpus', default='4,5')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--models', nargs='+', choices=('polytune', 'laddersym'),
                        default=['polytune', 'laddersym'])
    parser.add_argument('--laddersym-use-cache', action='store_true')
    parser.add_argument('--reuse-polytune-run', type=Path)
    parser.add_argument('--reuse-laddersym-cache', type=Path)
    args = parser.parse_args()
    data, run = args.data.resolve(), args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.read_text())
    split_path = data / 'split.json'
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != selection['split_sha256']:
        raise ValueError('Original validation split changed')
    split = json.loads(split_path.read_text())
    keys = {Path(path).stem: key for key, path in split['midi_filename'].items()
            if split['split'][key] == 'validation'}
    if len(keys) != sum(v == 'validation' for v in split['split'].values()):
        raise ValueError('Duplicate validation IDs')
    manifest = json.loads((data / 'manifest.json').read_text())
    durations = {tid: manifest['tracks'][tid]['written_duration_s'] for tid in keys}
    jobs, processes, handles = {}, {}, {}
    state = dict(state='preparing', supervisor_pid=os.getpid(), started_at=time.time(),
                 split_sha256=split_hash, expected_validation_count=len(keys), jobs=jobs,
                 checkpoint_selection=selection, batch_size=args.batch_size,
                 event_length=1024, prompt_length=1024, precision='float32',
                 inference='official inference handler; greedy decoding; deterministic prompts',
                 kv_cache={'polytune': False, 'laddersym': args.laddersym_use_cache})
    reused_jobs = []
    reused_ladder_ids = set()
    if args.reuse_laddersym_cache:
        cache = args.reuse_laddersym_cache
        cached = json.loads((cache / 'cache_manifest.json').read_text())
        if cached['split_sha256'] != split_hash or cached['checkpoint_sha256'] != selection['models']['laddersym']['sha256']:
            raise ValueError('Cached LadderSym data or checkpoint mismatch')
        for path, digest in cached['inference_code_sha256'].items():
            if selection['code_sha256'][path] != digest:
                raise ValueError(f'Cached inference code differs: {path}')
        if cached['kv_cache'] != args.laddersym_use_cache or cached['batch_size'] != args.batch_size:
            raise ValueError('Cached inference settings differ')
        reused_ladder_ids = set(json.loads((cache / 'evaluated_ids.json').read_text()))
        if not reused_ladder_ids <= set(keys):
            raise ValueError('Cached predictions contain non-validation IDs')
        state['reused_laddersym_predictions'] = len(reused_ladder_ids)
        state['reused_laddersym_cache'] = str(cache.resolve())
    if args.reuse_polytune_run:
        if 'polytune' in args.models:
            raise ValueError('Do not both launch and reuse Polytune')
        previous = json.loads((args.reuse_polytune_run / 'evaluation_manifest.json').read_text())
        if previous['split_sha256'] != split_hash or previous['checkpoint_selection']['models']['polytune']['sha256'] != selection['models']['polytune']['sha256']:
            raise ValueError('Reused Polytune run has different data or weights')
        reused_jobs = [j for j in previous['jobs'].values() if j['model'] == 'polytune']
        state['reused_polytune_run'] = str(args.reuse_polytune_run.resolve())
    write_json(run / 'original_split.json', split)
    for name, gpu_arg in [('polytune', args.polytune_gpus), ('laddersym', args.laddersym_gpus)]:
        if name not in args.models:
            continue
        # Bash reads the script again after child processes return. Keep a
        # private copy so edits to workspace documentation cannot shift the
        # active shell's file offsets during a long inference run.
        wrapper = run / f'{name}_eval.sh'
        wrapper.write_bytes((BASELINES / 'scripts' / f'{name}_eval.sh').read_bytes())
        gpus = [int(x) for x in gpu_arg.split(',')]
        ckpt = selection['models'][name]
        with open(ckpt['path'], 'rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != ckpt['sha256']:
                raise ValueError(f'{name} checkpoint changed')
        ids_to_run = list(set(keys) - reused_ladder_ids) if name == 'laddersym' else list(keys)
        shards = balanced_shards(ids_to_run, durations, len(gpus))
        for index, (gpu, ids) in enumerate(zip(gpus, shards)):
            label = f'{name}_{index}'
            tag = f'{run.name}_shard{index}'
            derived = dict(split)
            wanted = set(ids)
            derived['split'] = {key: 'validation' if Path(path).stem in wanted else 'train'
                                for key, path in split['midi_filename'].items()}
            shard_split = run / f'{label}_split.json'
            write_json(shard_split, derived)
            command = ['bash', str(wrapper),
                       '--ckpt', ckpt['path'], '--data', str(data),
                       '--split-json', str(shard_split), '--split', 'validation',
                       '--profile', 'cuda', '--tag', tag, '--max-length', '1024',
                       '--native-only', '--', f'eval.batch_size={args.batch_size}']
            output = BASELINES / 'runs' / name / f'eval_{tag}'
            if output.exists():
                raise FileExistsError(output)
            jobs[label] = dict(model=name, gpu=gpu, expected_ids=ids, state='pending',
                               command=command, log=str(run / f'{label}.log'),
                               predictions=str(output / tag), expected_count=len(ids))
    write_json(run / 'evaluation_manifest.json', state)
    try:
        for label, job in jobs.items():
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       BASELINE_HOME=str(BASELINES),
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
            env['LADDERSYM_USE_CACHE'] = '1' if args.laddersym_use_cache else '0'
            handles[label] = open(job['log'], 'x')
            child = subprocess.Popen(job['command'], cwd=BASELINES.parent, env=env,
                                     stdin=subprocess.DEVNULL, stdout=handles[label],
                                     stderr=subprocess.STDOUT, start_new_session=True)
            processes[label] = child
            job.update(pid=child.pid, state='running')
        state['state'] = 'inference'
        while True:
            for label, child in processes.items():
                job = jobs[label]
                job['predicted_count'] = len(list(Path(job['predictions']).glob('*/mix.mid')))
                code = child.poll()
                if code is not None:
                    job.update(state='complete' if code == 0 else 'failed', exit_code=code)
            state['updated_at'] = time.time()
            write_json(run / 'evaluation_status.json', state)
            if all(p.poll() is not None for p in processes.values()):
                break
            time.sleep(15)
        if any(p.returncode != 0 for p in processes.values()):
            raise RuntimeError('Inference shard failed; retained outputs and logs for diagnosis')
        while reused_jobs and not all((Path(j['predictions']).parent / 'note_metrics.json').is_file()
                                      for j in reused_jobs):
            previous = json.loads((args.reuse_polytune_run / 'evaluation_status.json').read_text())
            if any(j['state'] == 'failed' for j in previous['jobs'].values() if j['model'] == 'polytune'):
                raise RuntimeError('Reused Polytune inference failed')
            state.update(state='waiting_for_polytune', updated_at=time.time())
            write_json(run / 'evaluation_status.json', state)
            time.sleep(15)
        state['state'] = 'scoring'
        write_json(run / 'evaluation_status.json', state)
        summaries = {}
        for name in (['polytune'] if reused_jobs else []) + args.models:
            combined = run / f'{name}_predictions'
            source_jobs = reused_jobs if name == 'polytune' and reused_jobs else jobs.values()
            directories = [Path(j['predictions']) for j in source_jobs if j['model'] == name]
            if name == 'laddersym' and args.reuse_laddersym_cache:
                directories.append(args.reuse_laddersym_cache.resolve())
            merge_predictions(list(keys), directories, combined)
            out = run / f'{name}_note_metrics.json'
            command = [str(BASELINES / 'envs' / name / 'bin/python'),
                       str(BASELINES / 'common/evaluate_notes.py'), '--data', str(data),
                       '--pred-dir', str(combined), '--allow-unclassified', '--out', str(out)]
            with (run / f'{name}_metrics.log').open('x') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads(out.read_text())
            assert report['n_pieces'] == len(keys)
            summaries[name] = {k: v for k, v in report.items()
                               if k not in ('per_piece', 'prediction_counts')}
            summaries[name]['checkpoint'] = selection['models'][name]
        write_json(run / 'results.json', dict(split_sha256=split_hash, validation_count=len(keys),
                   epoch_budget_note=selection['epoch_budget_note'], models=summaries))
        state.update(state='complete', finished_at=time.time(), updated_at=time.time())
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), updated_at=time.time())
        raise
    finally:
        write_json(run / 'evaluation_status.json', state)
        for handle in handles.values():
            handle.close()


if __name__ == '__main__':
    main()
