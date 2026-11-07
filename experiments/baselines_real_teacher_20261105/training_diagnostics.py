"""Report fixed-budget training exposure without inspecting test predictions."""
from collections import Counter
import random
from statistics import mean
from common import RUN, read, write


def main():
    p = read(RUN / 'protocol.json')
    report = {}
    for model in ['polytune', 'laddersym']:
        cache = read(RUN / 'cache' / model / 'manifest.json')
        report[model] = {}
        for size in p['sizes'][1:]:
            ids = p['train_subsets'][str(size)]
            rng = random.Random(p['seed'])
            sampled = Counter()
            for _ in range(p['optimizer']['steps'] * p['optimizer']['gradient_accumulation']):
                sid = rng.choice(ids)
                slot = rng.choice(range(cache[sid]['windows']))
                sampled[sid, slot] += 1
            record_counts = Counter()
            error_windows = 0
            for (sid, slot), times in sampled.items():
                record_counts[sid] += times
                a, b = cache[sid]['bounds'][slot]
                classes = read(RUN / 'supervision' / f'{sid}.json')['classes']
                if any(n['start'] < b and n['end'] > a for cls in ['extra', 'removed'] for n in classes[cls]):
                    error_windows += times
            row = {'available_windows': sum(cache[s]['windows'] for s in ids),
                'sampled_windows': sum(sampled.values()), 'unique_sampled_windows': len(sampled),
                'recordings_sampled': len(record_counts), 'windows_per_recording': dict(sorted(record_counts.items())),
                'sampled_windows_overlapping_native_errors': error_windows}
            path = RUN / 'training' / model / f'n{size:02}'
            if (path / 'done.json').exists():
                history = read(path / 'history.json')
                row.update(mean_first_20_loss=mean(r['loss'] for r in history[:20]),
                    mean_last_20_loss=mean(r['loss'] for r in history[-20:]),
                    peak_gpu_mib=max(r['peak_gpu_mib'] for r in history),
                    train_seconds=read(path / 'done.json')['seconds'])
            report[model][str(size)] = row
    write(RUN / 'training_diagnostics.json', report)
    for model, rows in report.items():
        for size, row in rows.items():
            print(model, size, {k: v for k, v in row.items() if k != 'windows_per_recording'}, flush=True)


if __name__ == '__main__':
    main()
