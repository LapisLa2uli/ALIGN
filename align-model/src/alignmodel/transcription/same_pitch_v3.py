"""V10: conservative learned veto on v9 same-pitch merges.

The acoustic transcriber and all different-pitch candidates remain unchanged.
No score or reference labels enter the boundary classifier at inference.
"""
from dataclasses import asdict, dataclass
import numpy as np
from .same_pitch_v2 import SamePitchConfig as V9Config, acoustic_evidence, repair_same_pitch as repair_v9


FEATURE_NAMES = ('energy_range_db', 'spectral_step', 'spectral_drift',
                 'attack_max', 'voiced_min', 'gap_depth_db', 'gap_core_duration_sec')


@dataclass(frozen=True)
class BoundaryConfig:
    # Merge only below a selected repeat-score threshold; equality retains.
    repeat_threshold: float = .5


def boundary_features(evidence):
    """Joint local acoustic evidence; excludes score, duration and identity."""
    if not evidence:
        return None
    try:
        values = [evidence[k] for k in FEATURE_NAMES[:5]]
        values += [evidence['gap']['depth_db'], evidence['gap']['core_duration_sec']]
        result = np.asarray(values, np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return result if result.shape == (len(FEATURE_NAMES),) and np.isfinite(result).all() else None

def repeat_score(features, model):
    # sklearn trees evaluate float32 feature values even when fitted on float64.
    features = np.asarray(features, np.float32)
    if tuple(model['feature_names']) != FEATURE_NAMES:
        raise ValueError('Boundary model feature schema mismatch')
    values = []
    for tree in model['trees']:
        node = 0
        while tree['left'][node] >= 0:
            node = tree['left'][node] if float(features[tree['feature'][node]]) <= tree['threshold'][node] else tree['right'][node]
        values.append(tree['repeat_fraction'][node])
    if not values:
        raise ValueError('Empty boundary model')
    return float(np.mean(values))


def repair_same_pitch(rows, outputs, mel, *, model, audio=None, sample_rate=22050,
                      hop=256/22050, config=BoundaryConfig(), v9_config=V9Config(), prepared=None,
                      supported_boundaries=frozenset(), baseline_audit=None):
    if not 0 <= config.repeat_threshold <= 1:
        raise ValueError('repeat_threshold must be in [0, 1]')
    original = [list(r) for r in rows]
    if prepared is None and audio is not None:
        prepared = acoustic_evidence(audio, mel.shape[1], sample_rate=sample_rate, hop=hop, config=v9_config)
    from copy import deepcopy
    if baseline_audit is None:
        _, audit = repair_v9(original, outputs, mel, audio=audio, prepared=prepared,
                             sample_rate=sample_rate, hop=hop, config=v9_config)
    else:
        audit = deepcopy(baseline_audit)
    merge_at = set()
    for decision in audit['decisions']:
        decision['v9_decision'] = decision['decision']
        decision['v9_reason'] = decision['reason']
        if decision['decision'] != 'merge':
            continue
        boundary = decision['source_indices'][1]
        if boundary not in supported_boundaries:
            merge_at.add(boundary)
            decision['reason'] = 'v9_preserved_without_score_support'
            continue
        feature = boundary_features(decision.get('evidence'))
        if feature is None:
            merge_at.add(boundary)
            decision['reason'] = 'v9_preserved_without_boundary_evidence'
            continue
        score = repeat_score(feature, model)
        decision['repeat_score'] = score
        if score >= config.repeat_threshold:
            decision.update(decision='retain', reason='score_supported_rearticulation')
        else:
            decision['reason'] = 'confident_continuation'
            merge_at.add(decision['source_indices'][1])
    repaired, groups = [], []
    for i, row in enumerate(original):
        if i not in merge_at:
            repaired.append(row.copy()); groups.append([i])
            continue
        last = repaired[-1]
        last[2], last[3] = max(last[2], row[2]), max(last[3], row[3])
        if len(last) > 4 and len(row) > 4:
            last[4] = int(bool(last[4]) and bool(row[4]))
        if len(last) >= 7:
            last[5:7] = [-1, 0.]
        groups[-1].append(i)
    audit.update(schema_version='same-pitch-repair-v3', boundary_config=asdict(config),
                 boundary_model=model['schema_version'], merged_boundaries=len(merge_at),
                 output_note_count=len(repaired), source_groups=groups)
    return repaired, audit

