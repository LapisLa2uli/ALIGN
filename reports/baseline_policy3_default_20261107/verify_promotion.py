"""Verify the promoted exporter on frozen controls without publishing live labels."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path[:0] = [str(ROOT / 'baselines/scripts'), str(ROOT / 'DataCreate/src'),
               str(ROOT / 'experiments/polytune_alignment_policies_20261106')]
import label_datacreate_baselines_v3 as current
import label_datacreate_baselines as legacy
from datacreate.validation import validate_labels_file
from datacreate.melody import match_note_wise_labels_detail
from policy_lib import core_labels, aggregate

read = current._read
sha = legacy.sha256
EXPERIMENT = ROOT / 'experiments/polytune_alignment_policies_20261106'
PREVIOUS = ROOT / 'experiments/baselines_real_teacher_20261105'
LATEST = ROOT / 'experiments/baselines_real_teacher_v2_20261105'


def signature(labels):
    return [(l.get('id'), l['type'], l.get('start_time'), l.get('end_time'),
             l.get('extra_copies')) for l in labels]


def main():
    protocol = read(EXPERIMENT / 'protocol.json')
    checkpoints = read(LATEST / 'protocol.json')['initial_checkpoints']
    ids = protocol['test_ids']
    protected = {ROOT / name for name in protocol['protected_sha256']}
    protected.update(p for p in EXPERIMENT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.update((ROOT / 'DataCreate/samples').glob('*/labels*.json'))
    # Only read checkpoint bytes; model inference/training is unnecessary.
    for checkpoint in checkpoints.values():
        path = Path(checkpoint['path'])
        if not path.is_file():
            path = ROOT / 'experiments' / checkpoint['path'].replace('\\', '/').split('/experiments/', 1)[1]
        assert sha(path) == checkpoint['sha256'], path
        protected.add(path)
    for model in checkpoints:
        for path in (LATEST / f'predictions/test/{model}/n00/step000').rglob('*'):
            if path.is_file():
                protected.add(path)
    before = {str(p): sha(p) for p in sorted(protected)}
    rows = []
    scores = []
    for model, checkpoint in checkpoints.items():
        pred = LATEST / f'predictions/test/{model}/n00/step000'
        command = [sys.executable, str(ROOT / 'baselines/scripts/label_datacreate_baselines.py'),
                   '--model', model, '--pred-dir', str(pred), '--samples', str(PREVIOUS / 'bundles'),
                   '--labels-root', str(RUN / 'labels'), '--output', str(RUN / 'exports' / model),
                   '--checkpoint-sha256', checkpoint['sha256']]
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONNOUSERSITE': '1'},
                                   capture_output=True, text=True)
        (RUN / f'{model}_export.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
        assert completed.returncode == 0, completed.stdout + completed.stderr
        manifest = read(RUN / f'exports/{model}/{model}_label_manifest.json')
        assert manifest['alignment_policy'] == 'transcription'
        assert manifest['n_samples'] == len(ids) == 21
        for sid in ids:
            path = RUN / 'labels' / sid / legacy.LABEL_FILES[model]
            assert not validate_labels_file(path), path
            doc = read(path)
            native = legacy.document_from_midi(pred / sid / 'mix.mid',
                PREVIOUS / f'bundles/{sid}/verified_score.musicxml', model=model, sample_id=sid)
            assert signature(doc['labels']) == signature(native['labels']), (model, sid)
            assert doc['baseline_labeling']['unclassified_count'] == native['baseline_labeling']['unclassified_count']
            assert doc['baseline_labeling']['alignment_policy'] == 3
            assert doc['baseline_labeling']['alignment_lookup']['error_label_fields_used'] is False
            done = read(pred / sid / 'mix.done.json')
            assert sha(pred / sid / 'mix.mid') == done['sha256']
            assert done['checkpoint_sha256'] == checkpoint['sha256']
            predictions = doc['labels'] + [dict(id=f'unclassified_{i}', type='unclassified', score_event_indices=[])
                for i in range(doc['baseline_labeling']['unclassified_count'])]
            if model == 'polytune':
                expected = read(EXPERIMENT / f'projected/{sid}.json')['labels']
                assert signature(predictions) == signature(expected), sid
                assert [l['score_event_indices'] for l in predictions] == [l['score_event_indices'] for l in expected], sid
                # Scoring is separate from export. Never canonicalize v3 predictions:
                # explicit identities must win over their padded GUI envelope.
                gold = read(PREVIOUS / f'teacher_gold/{sid}.json')
                if gold['status'] == 'ok':
                    scores.append(match_note_wise_labels_detail(core_labels(gold['labels']), predictions,
                        score_event_count=gold['score_event_count'], type_mismatch_credit=0.))
            rows.append(dict(model=model, sample=sid, labels=len(doc['labels']),
                unclassified=doc['baseline_labeling']['unclassified_count'],
                unlocated=doc['baseline_labeling']['unlocated_events'],
                types_times_counts_copies_unchanged=True, schema_valid=True,
                frozen_projection_identical=True if model == 'polytune' else None))
        print(f'{model}: {len(ids)} documents verified', flush=True)
    metric = aggregate(scores)
    expected_metric = read(EXPERIMENT / 'results.json')['results']['transcription_lookup']['strict']
    assert len(scores) == 17
    for key in ['credit', 'predicted', 'gold', 'precision', 'recall', 'f1']:
        assert abs(metric[key] - expected_metric[key]) < 1e-12, (key, metric, expected_metric)
    assert before == {str(p): sha(p) for p in sorted(protected)}, 'Protected inputs changed'
    archive = ROOT / 'baselines/legacy/policy12_20261106/label_datacreate_baselines.py'
    assert sha(archive) == '656113e0988a3ab98c8ac3304d552d9bf1711d224d4d3647f5ae03bdb21f10bb'
    sources = ['baselines/scripts/label_datacreate_baselines.py',
        'baselines/scripts/label_datacreate_baselines_v3.py', 'baselines/common/transcription_label_alignment.py',
        'DataCreate/src/datacreate/models.py', 'DataCreate/src/datacreate/web/static/annotate.js']
    result = dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
        alignment_policy=3, adapter_version=current.VERSION, model_inference_or_training=False,
        labels_published_to_live_samples=False, documents_verified=len(rows),
        protected_files_unchanged=len(protected), protected_sha256=before,
        source_sha256={name: sha(ROOT / name) for name in sources},
        archived_v1_sha256=sha(archive), polytune_frozen_projection_parity_samples=21,
        polytune_strict_metric=metric, scored_samples=17, rows=rows)
    (RUN / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['rows', 'protected_sha256', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
