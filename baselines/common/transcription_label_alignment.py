"""Policy 3: map baseline error intervals through stored transcript score locations.

Promoted from the tested 2026-11-06 policy comparison. This module has no
experiment-directory dependency and consumes sanitized note metadata only.
"""
from copy import deepcopy
import math

VERSION = "stored_transcription_lookup_v1"
MIN_SHORTER_INTERVAL_OVERLAP = 0.5


def overlap(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    return overlap(a, b) / max(1e-12, max(a[1], b[1]) - min(a[0], b[0]))


def interval(label):
    a, b = label.get("start_time"), label.get("end_time")
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b <= a:
        return None
    return float(a), float(b)


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
        return overlap(ni, span)/max(1e-12, min(ni[1]-ni[0], span[1]-span[0])) >= MIN_SHORTER_INTERVAL_OVERLAP-1e-12

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
            label['location_policy'] = VERSION
            outputs.append(label)
            traces.append({'id': label.get('id'), 'indices': label['score_event_indices'], **evidence})
        return outputs, traces
