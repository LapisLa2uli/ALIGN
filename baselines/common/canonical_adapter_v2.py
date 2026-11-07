"""Versioned baseline adapter adding canonical neighbors for ordinary extras."""
from canonical_adapter import Note, PARAMETERS as BASE_PARAMETERS, adapt as adapt_v1

VERSION = 'baseline-score-location-v2-extra-neighbors'
PARAMETERS = {**BASE_PARAMETERS, 'ordinary_extra_location': 'preceding-score-note-and-next-neighbor'}

def locate_extra_neighbors(events, score_count):
    for index, event in enumerate(events):
        if event['kind'] != 'extra' or event.get('score_index') is not None or not score_count:
            continue
        left = next((e for e in reversed(events[:index]) if e.get('score_index') is not None and e['kind'] in ('correct', 'wrong', 'copy')), None)
        right = next((e for e in events[index + 1:] if e.get('score_index') is not None and e['kind'] in ('correct', 'wrong', 'copy')), None)
        if left is None and right is None:
            continue
        if left and right and (right['score_index'] < left['score_index'] or right.get('copy_pass', 0) != left.get('copy_pass', 0)):
            continue
        anchor = int(left['score_index']) if left else max(0, int(right['score_index']) - 1)
        if not 0 <= anchor < score_count:
            continue
        event['score_index'] = anchor
        event['score_event_indices'] = list(range(anchor, min(score_count, anchor + 2)))
        event['location_evidence'] = 'adjacent_performed_anchors' if left and right else 'one_sided_performed_anchor'
    return events

def adapt(notes, score_pitches):
    result = adapt_v1(notes, score_pitches)
    locate_extra_neighbors(result['events'], len(score_pitches))
    for event, label in zip(result['events'], result['labels']):
        if event.get('score_event_indices'):
            label['score_event_indices'] = event['score_event_indices']
    result['unlocated_events'] = sum('score_event_indices' not in label for label in result['labels'])
    result['adapter_version'] = VERSION
    return result
