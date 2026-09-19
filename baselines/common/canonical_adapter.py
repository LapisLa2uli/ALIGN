"""Fixed, gold-free conversion of native baseline notes to score locations.

Both baselines use exactly these constants. They are not fitted or calibrated
on outputRaw validation labels. Unlocatable events remain unmatched predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

VERSION = 'baseline-score-location-v1'
PARAMETERS = dict(wrong_pair_seconds=0.1, insertion_cost=1.0,
                  deletion_cost=1.0, pitch_mismatch_cost=1.5,
                  repeat_min_notes=3, repeat_min_pitch_agreement=0.8,
                  sounding_to_written=2, rhythm_prediction=False)


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int
    kind: str


def align_anchors(pitches, score_pitches):
    """Global monotonic edit alignment, with deterministic diagonal tie-breaks."""
    n, m = len(pitches), len(score_pitches)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    parent = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], parent[i][0] = i * PARAMETERS['insertion_cost'], 1
    for j in range(1, m + 1):
        cost[0][j], parent[0][j] = j * PARAMETERS['deletion_cost'], 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (cost[i-1][j-1] + (0 if pitches[i-1] == score_pitches[j-1]
                                   else PARAMETERS['pitch_mismatch_cost']), 0),
                (cost[i-1][j] + PARAMETERS['insertion_cost'], 1),
                (cost[i][j-1] + PARAMETERS['deletion_cost'], 2)]
            cost[i][j], parent[i][j] = min(choices)
    mapping = {}
    i, j = n, m
    while i or j:
        action = parent[i][j]
        if action == 0:
            mapping[i-1] = j-1
            i -= 1; j -= 1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    return mapping


def adapt(notes, score_pitches):
    """Return canonical combined-pipeline labels without using any gold data.

Input pitches are already written pitches. Pairing a substitution consumes
one Extra and one Missing token. Every other decoded event is retained,
including duplicates, unclassified notes, and unlocated Missing predictions.
Only native Missing tokens can emit deletions; edit-distance gaps do not.
"""
    ordered = sorted(enumerate(notes), key=lambda x: (x[1].start, x[1].pitch, x[0]))
    missing = [(i, n) for i, n in ordered if n.kind == 'missing']
    extra = [(i, n) for i, n in ordered if n.kind == 'extra']
    def nearest(item, options):
        candidates = [(abs(item[1].start-n.start), i) for i,n in options
                      if item[1].pitch != n.pitch
                      and abs(item[1].start-n.start) <= PARAMETERS['wrong_pair_seconds']]
        return min(candidates)[1] if candidates else None
    e_to_m = {i: nearest((i,n), missing) for i,n in extra}
    m_to_e = {i: nearest((i,n), extra) for i,n in missing}
    pairs = {i: j for i,j in e_to_m.items() if j is not None and m_to_e.get(j) == i}
    consumed_missing = set(pairs.values())
    anchors, events = [], []
    for original, note in ordered:
        if original in consumed_missing:
            continue
        kind = 'wrong' if original in pairs else note.kind
        expected_pitch = notes[pairs[original]].pitch if original in pairs else note.pitch
        item = dict(original_index=original, start=note.start, end=note.end,
                    pitch=note.pitch, kind=kind, score_index=None, copy_pass=0)
        events.append(item)
        if kind in ('correct', 'wrong', 'missing'):
            anchors.append((len(events)-1, expected_pitch))
    mapping = align_anchors([pitch for _,pitch in anchors], score_pitches)
    for anchor_id, score_id in mapping.items():
        events[anchors[anchor_id][0]]['score_index'] = score_id

    # Locate extra-note runs as repetitions of an already visited score prefix.
    # Longest acceptable contiguous match wins; ties prefer greater agreement,
    # then the most recent source position. No future or gold events are read.
    seen_copies = {}
    frontier = 0
    position = 0
    while position < len(events):
        item = events[position]
        if item['kind'] != 'extra':
            if item['score_index'] is not None:
                frontier = max(frontier, item['score_index'] + 1)
                seen_copies.setdefault(item['score_index'], 0)
            position += 1
            continue
        end = position
        while end < len(events) and events[end]['kind'] == 'extra':
            end += 1
        cursor = position
        while cursor < end:
            best = None
            for length in range(min(end-cursor, frontier), PARAMETERS['repeat_min_notes']-1, -1):
                candidates = []
                for start in range(frontier-length+1):
                    if not all(start+k in seen_copies for k in range(length)):
                        continue
                    matches = sum(events[cursor+k]['pitch'] == score_pitches[start+k]
                                  for k in range(length))
                    if matches / length >= PARAMETERS['repeat_min_pitch_agreement']:
                        candidates.append((matches, start))
                if candidates:
                    _, start = max(candidates)
                    best = (length, start)
                    break
            if best is None:
                cursor += 1
                continue
            length, start = best
            copy_pass = 1 + max(seen_copies[start+k] for k in range(length))
            for k in range(length):
                events[cursor+k].update(score_index=start+k, copy_pass=copy_pass, kind='copy')
                seen_copies[start+k] = copy_pass
            cursor += length
        position = end

    labels = []
    for event in events:
        kind = event['kind']
        if kind == 'missing':
            label = dict(type='missed_note')
        else:
            relationship = {'correct': 'match', 'wrong': 'substitute',
                            'extra': 'extra', 'copy': 'copy'}.get(kind, 'unclassified')
            layer2 = {'correct': 'match', 'wrong': 'wrong_note',
                      'extra': 'extra_note'}.get(kind, 'unclassified')
            if kind == 'copy':
                layer2 = 'match' if event['pitch'] == score_pitches[event['score_index']] else 'wrong_note'
            label = dict(type='|'.join((relationship, layer2, 'no_rhythm',
                                       'copy' if event['copy_pass'] else 'ordinary')))
        if event['score_index'] is not None:
            label['score_event_indices'] = [event['score_index']]
            if kind != 'missing':
                label['copy_pass'] = event['copy_pass']
        # Never guess a gold rendered_index from the predicted sequence ordinal.
        labels.append(label)
    return dict(labels=labels, events=events, raw_note_count=len(notes),
                paired_wrong_notes=len(pairs), converted_event_count=len(labels),
                unlocated_events=sum('score_event_indices' not in x for x in labels))
