"""Freeze gold-free baseline score locations, then score in a separate process."""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
import sys
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'baselines/common'))
sys.path.insert(0, str(ROOT / 'align-model/src'))
from canonical_adapter import Note, adapt, VERSION, PARAMETERS
from restore_outputraw_validation import sha, SPLIT_HASH, TARGET_HASH


def code_hashes():
    names = ['baselines/common/canonical_adapter.py',
             'baselines/scripts/evaluate_outputraw_canonical.py',
             'DataCreate/src/datacreate/melody.py',
             'align-model/src/alignmodel/joint/index.py',
             'align-model/src/alignmodel/joint/outputraw_metrics.py']
    return {name: sha(ROOT / name) for name in names}


def freeze(args):
    from eval_bridge import read_pred_midi_tracks, class_from_name, Warner
    from alignmodel.joint.index import ScoreEventIndex
    def deny_gold(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            name = str(values[0]).replace('\\', '/')
            forbidden = ('/upload/', '/native_targets/', '.sqlite', 'labels.json',
                         'note_map.json', 'preparation_audit.json')
            if any(part in name for part in forbidden):
                raise PermissionError(f'Gold access denied during conversion: {name}')
    sys.addaudithook(deny_gold)
    ids = json.loads((args.pred_dir / 'evaluated_ids.json').read_text())
    if len(ids) != len(set(ids)) or len(ids) != 358:
        raise ValueError('Require exactly the complete 358-clip inference')
    actual = {p.parent.name for p in args.pred_dir.glob('*/mix.mid')}
    if actual != set(ids):
        raise ValueError('Incomplete or extra MIDI predictions')
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in sorted(ids):
        midi = args.pred_dir / sample / 'mix.mid'
        score = args.scores / sample / 'verified_score.musicxml'
        tracks, _, _ = read_pred_midi_tracks(str(midi), Warner())
        notes = []
        for track in tracks:
            kind = class_from_name(track['name'])
            if kind is None and track['name'].strip() and track['notes']:
                raise ValueError(f'Unknown named MIDI class: {track["name"]}')
            for start, end, pitch in track['notes']:
                notes.append(Note(start, end, int(pitch)+2, kind or 'unclassified'))
        score_index = ScoreEventIndex.from_musicxml(score)
        converted = adapt(notes, [event.pitch for event in score_index.events])
        converted.update(sample=sample, score_event_count=len(score_index.events))
        output = args.out / (sample + '.json')
        output.write_text(json.dumps(converted, indent=2)+'\n')
        rows.append(dict(sample=sample, output_sha256=sha(output),
                         prediction_midi_sha256=sha(midi), score_sha256=sha(score)))
    manifest = dict(adapter_version=VERSION, parameters=PARAMETERS,
                    code_sha256=code_hashes(), gold_access_guard='enabled; no gold opened',
                    rows=rows, prediction_root=str(args.pred_dir.resolve()),
                    scores_root=str(args.scores.resolve()))
    (args.out / 'freeze_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(frozen=len(rows), directory=str(args.out)), indent=2))


def gold_labels(score_path, target):
    """Exact combined-pipeline target semantics from outputraw_metrics.py."""
    from alignmodel.joint.index import ScoreEventIndex
    from alignmodel.joint.outputraw_metrics import _event_label
    from alignmodel.joint.outputraw_train import _target_layer2
    index = ScoreEventIndex.from_musicxml(score_path, target)
    rhythm_labels = [row for row in target['valid_labels'] if row['type'] == 'rhythm_error']
    layer2 = _target_layer2(index.rendered_events)
    labels = []
    for event, kind in zip(index.rendered_events, layer2):
        rhythm = any(float(row['start_time']) < event.end
                     and event.start < float(row['end_time']) for row in rhythm_labels)
        label_type = '|'.join((event.relationship, kind, 'rhythm' if rhythm else 'no_rhythm',
                               'copy' if event.is_copy else 'ordinary'))
        labels.append(_event_label(event, label_type))
    labels.extend(dict(type='missed_note', score_event_indices=[i])
                  for i in sorted(index.deleted_event_indices))
    return labels, index


def pooled(rows):
    credit = sum(row['credit'] for row in rows)
    predicted, gold = sum(row['predicted'] for row in rows), sum(row['gold'] for row in rows)
    return dict(credit=credit, predicted=predicted, gold=gold,
                precision=credit/predicted if predicted else 0.0,
                recall=credit/gold if gold else 0.0,
                f1=2*credit/(predicted+gold) if predicted+gold else 1.0)


def score(args):
    from alignmodel.melody import match_note_wise_labels_detail
    manifest_path = args.frozen / 'freeze_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['code_sha256'] != code_hashes():
        raise ValueError('Frozen inference or scoring implementation changed')
    assert sha(ROOT / 'upload/split.json') == SPLIT_HASH
    assert sha(ROOT / 'upload/canonical_dev_targets.sqlite') == TARGET_HASH
    split = json.loads((ROOT / 'upload/split.json').read_text())
    targets = {row['sample']: row for row in split['val']}
    assert len(manifest['rows']) == 358 and {r['sample'] for r in manifest['rows']} == set(targets)
    rows = []
    db = sqlite3.connect(f'file:{ROOT / "upload/canonical_dev_targets.sqlite"}?mode=ro', uri=True)
    for row in manifest['rows']:
        sample = row['sample']; selected = targets[sample]
        path = args.frozen / (sample+'.json')
        assert sha(path) == row['output_sha256']
        prediction = json.loads(path.read_text())
        score_path = Path(manifest['scores_root']) / sample / 'verified_score.musicxml'
        assert sha(score_path) == row['score_sha256']
        original = db.execute('SELECT split,source_hashes,payload FROM targets WHERE ordinal=?',
                              (selected['target_record'],)).fetchone()
        assert original[0] == 'val' and json.loads(original[1]) == selected['source_hashes']
        target = json.loads(zlib.decompress(original[2]))
        gold, index = gold_labels(score_path, target)
        assert len(index.events) == prediction['score_event_count']
        detail = match_note_wise_labels_detail(gold, prediction['labels'], score_event_count=len(index.events))
        assert detail['status'] == 'available'
        rows.append(dict(sample=sample, credit=detail['credit'], predicted=detail['predicted'],
                         gold=detail['gold'], pair_counts=detail['pair_counts'],
                         unlocated_events=prediction['unlocated_events'],
                         gold_sha256=hashlib.sha256(json.dumps(gold,sort_keys=True).encode()).hexdigest()))
    db.close()
    rng = np.random.default_rng(365)
    values = [pooled([rows[int(i)] for i in rng.integers(0,len(rows),len(rows))])['f1'] for _ in range(1000)]
    report = dict(metric='align-note-wise-score-event-metric-v1', task='combined_pipeline',
                  validation_count=len(rows), split_sha256=SPLIT_HASH, target_sha256=TARGET_HASH,
                  freeze_manifest_sha256=sha(manifest_path), adapter_version=VERSION,
                  parameters=PARAMETERS, micro=pooled(rows),
                  bootstrap=dict(seed=365,replicates=1000,unit='clip',
                                 lower_95=float(np.quantile(values,.025)),upper_95=float(np.quantile(values,.975))),
                  per_clip=rows)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key != 'per_clip'},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    p = sub.add_parser('freeze');p.add_argument('--pred-dir',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('score');p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();(freeze if args.mode=='freeze' else score)(args)


if __name__=='__main__':
    main()
