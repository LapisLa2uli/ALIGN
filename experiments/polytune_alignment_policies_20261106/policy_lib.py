"""Frozen alignment-policy comparison helpers; projection never reads error gold."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'experiments/baselines_real_teacher_20261105'
LATEST = ROOT / 'experiments/baselines_real_teacher_v2_20261105'
PRED = LATEST / 'predictions/test/polytune/n00/step000'
KINDS = ('wrong_note', 'missed_note', 'extra_note', 'repetition')
SETTINGS = {
    'timestamp_primary_iou': .5,
    'timestamp_iou_sensitivity': [.25, .5, .75],
    'timestamp_onset_sensitivity_seconds': [.05, .1, .2, .5],
    'projection_min_shorter_interval_overlap': .5,
    'timestamp_assignment': 'maximum cardinality one-to-one, then maximum similarity; exact type and repetition copy count',
    'wrong_projection': 'best temporal IoU transcript event, requiring >=50% overlap of shorter interval; no error-class or pitch filter',
    'missing_projection': 'same lookup plus score-gap intervals interpolated between consecutive forward aligned anchors; no extrapolation',
    'extra_projection': 'previous completed and next starting aligned transcript notes; canonical gap is left score index and following index; reject crossing backward jumps',
    'repetition_projection': 'union of covered transcript score indices; preserve predicted extra_copies without consulting teacher repetition decisions',
    'projection_no_fallback': 'unmapped predictions remain false positives, never reuse old score location',
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def core_labels(labels):
    result = deepcopy(labels)
    for label in result:
        part = label.get('score_part') or {}
        if part:
            a = part.get('core_start_note_index', part['start_note_index'] + part.get('pad_notes', 0))
            b = part.get('core_end_note_index', part['end_note_index'] - part.get('pad_notes', 0))
            label['score_event_indices'] = list(range(a, b + 1))
        if label['type'] == 'repetition':
            label['extra_copies'] = label.get('extra_copies') or 1
    return result

def overlap(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))

def iou(a, b):
    inter = overlap(a, b)
    return inter / max(1e-12, max(a[1], b[1]) - min(a[0], b[0]))

def interval(label):
    a, b = label.get('start_time'), label.get('end_time')
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b <= a:
        return None
    return float(a), float(b)

def prf(credit, predicted, gold):
    return dict(credit=credit, predicted=predicted, gold=gold,
                precision=credit/max(1, predicted), recall=credit/max(1, gold),
                f1=2*credit/max(1, predicted+gold))

def aggregate(rows):
    return prf(*(sum(r[k] for r in rows) for k in ('credit', 'predicted', 'gold')))

def timestamp_score(gold, pred, mode='iou', threshold=.5):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    matrix = np.zeros((len(pred), len(gold)))
    qualities = np.zeros_like(matrix)
    for i, p in enumerate(pred):
        pi = interval(p)
        for j, g in enumerate(gold):
            gi = interval(g)
            if pi is None or gi is None or p['type'] != g['type']:
                continue
            if p['type'] == 'repetition' and (p.get('extra_copies') or 1) != (g.get('extra_copies') or 1):
                continue
            if mode == 'iou':
                quality = iou(pi, gi)
                valid = quality >= threshold - 1e-12
            else:
                distance = abs(pi[0] - gi[0])
                quality = max(0., 1-distance/max(threshold, 1e-12))
                valid = distance <= threshold + 1e-12
            if valid:
                # One extra match must outweigh every similarity tie-break.
                matrix[i, j] = min(len(pred), len(gold)) + 1 + quality
                qualities[i, j] = quality
    rows, cols = linear_sum_assignment(-matrix)
    pairs = [dict(prediction_index=int(i), gold_index=int(j), similarity=float(qualities[i, j]))
             for i, j in zip(rows, cols) if matrix[i, j] > 0]
    return {**prf(len(pairs), len(pred), len(gold)), 'pairs': pairs}

def note_units(labels):
    result = Counter()
    for ordinal, label in enumerate(labels):
        kind = label['type']
        indices = tuple(sorted(set(label.get('score_event_indices') or [])))
        if not indices:
            result[('unlocated', kind, ordinal)] += 1
        elif kind == 'extra_note':
            result[(kind, indices)] += 1
        else:
            copies = max(1, int(label.get('extra_copies') or 1)) if kind == 'repetition' else 1
            for index in indices:
                result[(kind, index)] += copies
    return result

def note_score(gold, pred):
    g, p = note_units(gold), note_units(pred)
    assert not any(k[0] == 'unlocated' for k in g)
    return prf(sum((g & p).values()), sum(p.values()), sum(g.values()))

class TimeScoreLookup:
    """Use only sanitized note times/pitches/locations and score rhythm.

    Never use the teacher's labels, relationships, repetitions, or error types.
    """
    def __init__(self, document):
        assert set(document) <= {'sample', 'notes', 'score', 'source_sha256', 'duration'}
        self.score = document['score']
        self.notes = sorted(document['notes'], key=lambda n: (n['start'], n['end'], n['ordinal']))
        self.virtual = []
        self.jumps = []
        previous = None
        for n in self.notes:
            indices = n['score_event_indices']
            if not indices:
                continue
            if previous is not None:
                left = max(previous['score_event_indices'])
                right = min(indices)
                if right < min(previous['score_event_indices']):
                    self.jumps.append(n['start'])
                if right > left + 1 and n['start'] > previous['start']:
                    q0 = self.score[left]['ql_start']
                    q1 = self.score[right]['ql_start']
                    if q1 > q0:
                        scale = (n['start']-previous['start'])/(q1-q0)
                        for index in range(left+1, right):
                            s = self.score[index]
                            a = previous['start']+(s['ql_start']-q0)*scale
                            b = min(n['start'], previous['start']+(s['ql_end']-q0)*scale)
                            if b > a:
                                self.virtual.append(dict(start=a, end=b, score_event_indices=[index],
                                                         ordinal=-1, source='interpolated_score_gap'))
            previous = n

    def covered(self, note, span):
        ni = (note['start'], note['end'])
        return overlap(ni, span)/max(1e-12, min(ni[1]-ni[0], span[1]-span[0])) >= SETTINGS['projection_min_shorter_interval_overlap']-1e-12

    def project(self, labels):
        outputs, traces = [], []
        for original in labels:
            label = deepcopy(original)
            for key in ('score_event_indices', 'score_part', 'note_ids', 'core_note_ids', 'note_id',
                        'measure_number', 'pitches', 'context_score_event_indices', 'extra_identity', 'performed_event_id', 'rendered_index', 'copy_pass'):
                label.pop(key, None)
            span = interval(label)
            indices = []
            evidence = dict(method='unmapped', transcript_ordinals=[])
            if span and label['type'] in ('wrong_note', 'missed_note'):
                pool = self.notes + (self.virtual if label['type'] == 'missed_note' else [])
                candidates = [n for n in pool if self.covered(n, span)]
                if candidates:
                    best = max(candidates, key=lambda n: (iou((n['start'], n['end']), span), -abs(n['start']-span[0]), -n['ordinal']))
                    indices = best['score_event_indices']
                    evidence = dict(method=best.get('source', 'transcript_event'), transcript_ordinals=[best['ordinal']],
                                    interval=[best['start'], best['end']], iou=iou((best['start'], best['end']), span))
            elif span and label['type'] == 'repetition':
                candidates = [n for n in self.notes if self.covered(n, span)]
                indices = sorted({index for n in candidates for index in n['score_event_indices']})
                evidence = dict(method='covered_transcript_events', transcript_ordinals=[n['ordinal'] for n in candidates])
            elif span and label['type'] == 'extra_note':
                left = [n for n in self.notes if n['score_event_indices'] and n['end'] <= span[0]+1e-4]
                right = [n for n in self.notes if n['score_event_indices'] and n['start'] >= span[1]-1e-4]
                l = max(left, key=lambda n:n['end']) if left else None
                r = min(right, key=lambda n:n['start']) if right else None
                if l and r:
                    li, ri = max(l['score_event_indices']), min(r['score_event_indices'])
                    if ri >= li and not any(l['start'] < t <= r['start'] for t in self.jumps):
                        indices = list(range(li, min(len(self.score), li+2)))
                        evidence = dict(method='transcript_gap_neighbors', transcript_ordinals=[l['ordinal'], r['ordinal']])
            label['score_event_indices'] = sorted(set(indices))
            if indices:
                label['core_note_ids'] = [f'note_{i:04d}' for i in label['score_event_indices']]
                label['note_ids'] = list(label['core_note_ids'])
                label['note_id'] = label['note_ids'][0]
                label['pitches'] = [self.score[i]['pitch'] for i in label['score_event_indices']]
            label['location_policy'] = 'stored_transcription_lookup_v1'
            outputs.append(label)
            traces.append({'id': label.get('id'), 'indices': label['score_event_indices'], **evidence})
        return outputs, traces
