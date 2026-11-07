"""Paths and bookkeeping for the corrected real-data experiment."""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'experiments/baselines_real_teacher_20261105'
SIZES = [10, 20, 40]
SEED = 20261105

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def split():
    old = read(PREVIOUS / 'protocol.json')
    order = read(PREVIOUS / 'eligible.json')['ordered_ids']
    return {'train': order[:40], 'validation': order[40:], 'test': old['test_ids'],
        'subsets': {str(n): order[:n] for n in SIZES}}
