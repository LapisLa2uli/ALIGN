"""Compare cached first/middle/last windows with uncached upstream dataset reads."""
import argparse
import math
import os
from pathlib import Path
import sys
from common import ROOT, RUN, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['polytune', 'laddersym'], required=True)
    args = parser.parse_args()
    source = ROOT / 'experiments/baselines_retrain_20260918/source' / ('Polytune' if args.model == 'polytune' else 'LadderSym')
    sys.path.insert(0, str(source))
    os.environ['LADDERSYM_DETERMINISTIC_PROMPT'] = '1'
    import torch
    from dataset.dataset_2_random import Dataset
    torch.set_num_threads(2)
    kwargs = dict(root_dir=str(RUN / 'data'), split_json_path=str(RUN / 'data/split-cache.json'), split='train',
        mel_length=256, event_length=1024, num_rows_per_batch=1, full_recording=True, is_deterministic=True,
        is_randomize_tokens=False, is_random_alignment_shift_augmentation=False, shuffle=False, audio_filename='mix.wav')
    if args.model == 'laddersym':
        kwargs.update(use_prompt=True, prompt_length=1024)
    ds = Dataset(**kwargs)
    manifest = read(RUN / 'data/manifest.json')['tracks']
    rows = []
    for sid in ['046', '038']:
        ix = next(i for i, row in enumerate(ds.df) if Path(row['mistake_audio']).parent.name == sid)
        meta = read(RUN / 'cache' / args.model / f'{sid}.json')
        cached = torch.load(RUN / 'cache' / args.model / f'{sid}.pt', weights_only=True, mmap=True)
        ds.validation_windows = max(2, math.ceil(manifest[sid]['padded_duration']) + 1)
        frames = round(manifest[sid]['padded_duration'] * 16000) // 128 + 1
        for index in sorted({0, len(cached) // 2, len(cached) - 1}):
            start = round(meta['bounds'][index][0] * 125)
            slots = [s for s in range(ds.validation_windows) if round(s / (ds.validation_windows - 1) * max(0, frames - 256)) == start]
            assert slots, (sid, start, frames)
            actual = ds[ix * ds.validation_windows + slots[0]]
            expected = cached[index]
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                assert a.shape == b.shape
                assert torch.allclose(a, b, atol=1e-6, rtol=1e-6) if a.is_floating_point() else torch.equal(a, b), (sid, index)
            rows.append({'sample': sid, 'cached_index': index, 'dataset_slot': slots[0], 'status': 'equal'})
    write(RUN / f'cache_verification_{args.model}.json', rows)
    print(args.model, len(rows), 'cached windows equal uncached upstream reads', flush=True)


if __name__ == '__main__':
    main()
