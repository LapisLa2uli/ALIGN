"""Tempo-normalized rhythm decisions on first-pass canonical alignment events."""
from dataclasses import dataclass, asdict
import math
import numpy as np

REVISION = 'aligned-rhythm-v2-onsets'

@dataclass(frozen=True)
class RhythmConfig:
    method: str = 'ewma'
    alpha: float = .3
    log_threshold: float = .25
    minimum_ms: float = 60.
    window: int = 17
    minimum_context: int = 6


def classify_rhythm(score, events, *, measured_offsets=False, config=RhythmConfig()):
    """Judge pulse from consecutive attacks, independently of note releases.

    Audible sustain need not fill the written duration: articulation gaps,
    legato overlap and phrase-final releases are not rhythm errors by themselves.
    ``measured_offsets`` records input capability only; it never enables release
    scoring. Extras, copies, merged/duplicate mappings, skipped score intervals
    and final notes without a following attack are excluded. Unassessed notes
    must not be interpreted as rhythm-correct. No gold inputs are used.
    """
    if config.method not in {'ewma', 'median'}:
        raise ValueError('Unknown rhythm method')
    if not 0 < config.alpha <= 1 or config.window < 3 or config.minimum_context < 2:
        raise ValueError('Invalid rhythm configuration')
    ordered = sorted(events, key=lambda e: e.start)
    counts = {}
    for e in ordered:
        if e.score_span and not e.is_copy:
            for i in range(*e.score_span): counts[i] = counts.get(i, 0) + 1
    rows = []
    for pos, e in enumerate(ordered):
        if e.is_copy or e.relationship not in {'match', 'substitute'} or not e.score_span:
            continue
        i, end = e.score_span
        if end != i+1 or not 0 <= i < len(score) or counts.get(i) != 1:
            continue
        if pos+1 >= len(ordered): continue
        following = ordered[pos+1]
        if (following.is_copy or following.relationship not in {'match', 'substitute'}
                or following.score_span != (i+1, i+2) or counts.get(i+1) != 1): continue
        q = score[i+1].ql_start-score[i].ql_start
        seconds = following.start-e.start
        if q <= 0 or seconds <= 0 or not math.isfinite(seconds/q): continue
        rows.append((i, e, q, seconds, seconds/q))
    diagnostics = {'revision': REVISION, 'config': asdict(config),
        'timing_source': 'inter_onset_intervals',
        'measured_offsets_available': measured_offsets, 'release_policy': 'not_assessed',
        'eligible_notes': len(rows), 'assessed_score_event_indices': [],
        'limitation': 'Release lengths are not scored; IOI deviations belong to the preceding score interval. Final notes without a following attack are unassessed.'}
    if len(rows) < config.minimum_context:
        diagnostics['status'] = 'insufficient_context'
        return [], diagnostics
    ratios = np.array([r[4] for r in rows])
    baseline = float(np.median(ratios[:config.minimum_context]))
    hits = []
    for k, (i, e, q, seconds, ratio) in enumerate(rows):
        if config.method == 'median':
            half = config.window//2
            neighbors = np.concatenate((ratios[max(0,k-half):k], ratios[k+1:k+half+1]))
            if len(neighbors) < config.minimum_context: continue
            baseline = float(np.median(neighbors))
        expected = q*baseline
        deviation = seconds-expected
        diagnostics['assessed_score_event_indices'].append(i)
        if abs(math.log(ratio/baseline)) >= config.log_threshold and abs(deviation)*1000 >= config.minimum_ms:
            hits.append({'id': f'rhythm_{i:04d}', 'type': 'rhythm_error',
                'score_event_indices': [i], 'note_ids': [f'note_{i:04d}'],
                'score_part': {'start_note_index': i, 'end_note_index': i,
                    'core_start_note_index': i, 'core_end_note_index': i, 'pad_notes': 0},
                'context_score_event_indices': list(range(max(0,i-2),min(len(score),i+3))),
                'start_time': e.start, 'end_time': e.start+seconds,
                'deviation_ms': round(deviation*1000, 3),
                'rhythm_subtype': 'long' if deviation > 0 else 'short',
                'timing_source': diagnostics['timing_source'],
                'comment': f'Time to the next note attack is {abs(deviation)*1000:.0f} ms {"longer" if deviation > 0 else "shorter"} than expected from the surrounding pulse; note release length is not scored.'})
        if config.method == 'ewma':
            baseline = config.alpha*ratio+(1-config.alpha)*baseline
    diagnostics['status'] = 'ok'
    return hits, diagnostics

