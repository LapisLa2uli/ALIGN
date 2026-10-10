"""Time the current publisher and warm aligner on isolated input copies.

Nested timings are inclusive and must not be added together.
"""
import argparse
from collections import defaultdict
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'align-model/runs/stack-v9/numba-cache'))
for relative in ('align-model/src', 'align-model/scripts', 'DataCreate/src', 'synth-pipeline/src'):
    sys.path.insert(0, str(ROOT / relative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sample = out / 'sample'
    sample.mkdir()
    protected = {}
    for name in ('performance_audio.wav', 'verified_score.musicxml', 'labels.json'):
        source = args.sample / name
        if source.exists():
            protected[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            shutil.copy2(source, sample / name)
    started = time.perf_counter()
    import publish_datacreate_v9 as publisher
    from alignmodel.joint import stack_v9_passage as stack, robust_dp_aligner_v5 as aligner
    from alignmodel.joint import passage_v1 as passage
    from music21 import converter
    import_seconds = time.perf_counter() - started
    stats = defaultdict(lambda: {'calls': 0, 'seconds': 0., 'first_seconds': None})
    dp_calls = []
    captured = {}

    def wrap(module, name, label=None):
        original = getattr(module, name)
        label = label or name
        @functools.wraps(original)
        def timed(*a, **kw):
            begin = time.perf_counter()
            value = original(*a, **kw)
            seconds = time.perf_counter() - begin
            row = stats[label]
            row['calls'] += 1
            row['seconds'] += seconds
            if row['first_seconds'] is None:
                row['first_seconds'] = seconds
            if name == '_dp':
                dp_calls.append({'n': len(a[0]), 'm': len(a[2]), 'seconds': seconds})
            if name == 'align_notes':
                captured.update(args=a, kwargs=kw, snapshot=repr(value))
            return value
        setattr(module, name, timed)

    for name in ('load_dual_checkpoint', 'load_verifier', 'load_audio_mono', 'extract_dual_mel',
                 'infer_dual_outputs', 'score_presence', 'align_outputs', 'gui_documents',
                 'feedback', 'build_note_alignment', 'validate_labels_file', 'sha'):
        wrap(publisher, name)
    for name in ('decode_v7', 'repair_same_pitch', 'align_notes', 'locate_passages',
                 'score_ornament_patterns', 'align_v5', 'gate_v4'):
        wrap(stack, name)
    for name in ('_dp', '_dp_timed', 'expand_ornament_hypothesis', '_mergeable', '_backtrace'):
        wrap(aligner, name)
    for name in ('local_hypotheses', 'schedule_restarts'):
        wrap(passage, name)
    wrap(converter, 'parse', 'musicxml_parse')
    sys.argv = ['publish_datacreate_v9.py', '--sample', str(sample), '--output', str(out / 'publish'), '--device', 'cuda']
    begin = time.perf_counter()
    publisher.main()
    publisher_seconds = time.perf_counter() - begin
    cold = dict(stats)
    first_dp = list(dp_calls)
    stats.clear()
    dp_calls.clear()
    captured['kwargs']['progress'] = None
    previous = captured['snapshot']
    begin = time.perf_counter()
    repeated = stack.align_notes(*captured['args'], **captured['kwargs'])
    warm_seconds = time.perf_counter() - begin
    same = repr(repeated) == previous
    doc = json.loads((sample / 'feedback_v9.json').read_text())
    assert all(hashlib.sha256((args.sample / name).read_bytes()).hexdigest() == digest
               for name, digest in protected.items())
    result = {'source': str(args.sample), 'import_seconds': import_seconds,
              'publisher_seconds': publisher_seconds, 'initial_total_seconds': import_seconds + publisher_seconds,
              'initial_timings': cold, 'warm_alignment_seconds': warm_seconds, 'warm_timings': dict(stats),
              'warm_output_identical': same, 'dp_calls': first_dp,
              'dp_cells': sum((r['n']+1)*(r['m']+1) for r in first_dp),
              'status': doc['status'], 'label_count': len(doc['labels']),
              'location': doc['diagnostics']['passage_location'], 'source_unchanged': True}
    (out / 'timings.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('source','initial_total_seconds','warm_alignment_seconds',
                     'warm_output_identical','dp_cells')}), flush=True)
    assert same, 'Warm alignment changed the output'


if __name__ == '__main__':
    main()
