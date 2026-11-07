"""Verify actual tensor changes, rather than comparing different file formats."""
from common import RUN, read, write, sha


def main():
    import torch
    torch.set_num_threads(2)
    p = read(RUN / 'protocol.json')
    results = {}
    for model, entry in p['initial_checkpoints'].items():
        assert sha(entry['path']) == entry['sha256']
        payload = torch.load(entry['path'], map_location='cpu', weights_only=False, mmap=True)
        original = payload.get('state_dict', payload)
        for prefix in ['module.', 'model.']:
            if all(k.startswith(prefix) for k in original):
                original = {k[len(prefix):]: v for k, v in original.items()}
        results[model] = {}
        for n in p['sizes'][1:]:
            folder = RUN / 'training' / model / f'n{n:02}'
            if not (folder / 'done.json').exists():
                continue
            done = read(folder / 'done.json')
            assert sha(folder / 'model.pt') == done['checkpoint_sha256']
            current = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
            assert current.keys() == original.keys()
            changed = 0
            largest = 0.
            for name, value in current.items():
                assert value.shape == original[name].shape
                assert torch.isfinite(value).all(), name
                if not torch.equal(value, original[name]):
                    changed += 1
                    largest = max(largest, float((value - original[name]).abs().max()))
            assert changed > 0 and largest > 0, (model, n)
            row = {'changed_state_tensors': changed, 'total_state_tensors': len(current),
                'maximum_absolute_change': largest, 'all_weights_finite': True,
                'checkpoint_sha256': done['checkpoint_sha256']}
            results[model][str(n)] = row
            print(model, n, row, flush=True)
            del current
        del original, payload
    write(RUN / 'weights_audit.json', results)


if __name__ == '__main__':
    main()
