"""Conservative, score-free repair of repeated CTC emissions of a sustained note.

Input mel is the v3 clip-normalized dual mel (192 x frames). Its short-window
branch retains attacks hidden by the long-window branch. Times share the
centered mel frame origin. Decoder row ends are NOT evidence of an audio gap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class SamePitchConfig:
    context_sec: float = 0.09
    attack_threshold: float = 0.35
    merge_attack_max: float = 0.20
    voiced_min: float = 0.80
    energy_range_db: float = 2.5
    spectral_step_max: float = 0.06
    spectral_drift_max: float = 0.06
    silence_db: float = -65.0


def waveform_levels(audio, frames, *, sample_rate=22050, hop=256/22050):
    """Centered 12 ms RMS: do not smooth away a brief tongue/silence boundary."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('audio must be finite and mono')
    window = max(2, round(sample_rate * .012))
    positions = np.rint(np.arange(frames) * hop * sample_rate).astype(int)
    lo = np.maximum(0, positions - window // 2)
    hi = np.minimum(len(x), positions + window // 2)
    valid = (hi > lo) & (positions < len(x))
    result = np.full(frames, np.nan)
    integral = np.r_[0., np.cumsum(x * x)]
    result[valid] = 10 * np.log10(np.maximum(
        (integral[hi[valid]] - integral[lo[valid]]) / (hi[valid] - lo[valid]), 1e-12))
    return result


def repair_same_pitch(rows, outputs, mel, *, audio=None, sample_rate=22050,
                      hop=256/22050, config=SamePitchConfig()):
    """Return repaired rows and one diagnostic for EVERY original equal pair.

    Ambiguous/missing evidence retains both notes. Decisions use original
    boundaries, so merging a chain cannot change subsequent evidence windows.
    No score or labels enter this decision. Confidence remains the maximum
    source confidence, not a fabricated independent-probability combination.
    """
    if hop <= 0 or sample_rate <= 0 or config.context_sec <= 0:
        raise ValueError('hop, sample rate and context must be positive')
    original = [list(row) for row in rows]
    if any(len(r) < 4 or not np.isfinite(r[:4]).all() or r[2] < r[1] for r in original):
        raise ValueError('invalid note row')
    if any(b[1] < a[1] for a, b in zip(original, original[1:])):
        raise ValueError('note rows must be chronological')
    spectrogram = np.asarray(mel) if mel is not None else np.empty((0, 0))
    frames = spectrogram.shape[1] if spectrogram.ndim == 2 else 0
    valid_mel = spectrogram.ndim == 2 and spectrogram.shape[0] == 192
    levels = (waveform_levels(audio, frames, sample_rate=sample_rate, hop=hop)
              if audio is not None else np.asarray(outputs.get('rms_db', [])))
    heads = {k: np.asarray(outputs.get(k, [])) for k in ('onset', 'rearticulation', 'voiced')}
    width = max(3, int(np.ceil(config.context_sec / hop)))
    decisions, merge_at = [], set()
    for i in range(1, len(original)):
        left, right = original[i-1:i+1]
        if left[0] != right[0]:
            continue
        t = float(right[1]); frame = int(round(t / hop))
        lo, hi = frame-width, frame+width+1
        d = {'source_indices': [i-1, i], 'pitch': int(right[0]), 'boundary_sec': t,
             'decision': 'retain', 'reason': 'insufficient_context'}
        decisions.append(d)
        # Avoid the previous note's attack and the next note's transition.
        next_start = original[i+1][1] if i+1 < len(original) else frames*hop
        if lo < 0 or hi > frames or lo*hop <= left[1]+.025 or hi*hop >= next_start-.025:
            continue
        if not valid_mel or levels.ndim != 1 or len(levels) < hi or any(
                v.ndim != 1 or len(v) < hi for v in heads.values()):
            d['reason'] = 'missing_evidence'; continue
        short = spectrogram[128:, lo:hi].astype(np.float64)
        energy = levels[lo:hi]
        if not all(np.isfinite(v).all() for v in [short, energy, *[v[lo:hi] for v in heads.values()]]):
            d['reason'] = 'nonfinite_evidence'; continue
        # Brightest quarter of short-mel bands: suppress unstable noise-floor bins.
        strength = np.median(short, axis=1)
        active = strength >= np.quantile(strength, .75)
        spectrum = short[active]
        step = float(np.max(np.sqrt(np.mean(np.diff(spectrum, axis=1)**2, axis=0))))
        flank = max(2, width//2)
        drift = float(np.sqrt(np.mean((np.mean(spectrum[:, :flank], axis=1)
                                      - np.mean(spectrum[:, -flank:], axis=1))**2)))
        attack = max(float(np.max(heads[k][lo:hi])) for k in ('onset', 'rearticulation'))
        voiced = float(np.min(heads['voiced'][lo:hi]))
        spread = float(np.ptp(energy)); floor = float(np.min(energy))
        d['evidence'] = {'window_sec': [lo*hop, (hi-1)*hop], 'attack_max': attack,
                         'voiced_min': voiced, 'energy_range_db': spread, 'energy_min_db': floor,
                         'spectral_step': step, 'spectral_drift': drift}
        if attack >= config.attack_threshold:
            d['reason'] = 'separate_attack'
        elif floor < config.silence_db or voiced < config.voiced_min:
            d['reason'] = 'silence_or_unvoiced'
        elif spread > config.energy_range_db:
            d['reason'] = 'energy_boundary'
        elif step > config.spectral_step_max or drift > config.spectral_drift_max:
            d['reason'] = 'spectral_boundary'
        elif attack > config.merge_attack_max:
            d['reason'] = 'ambiguous_attack'
        else:
            d.update(decision='merge', reason='continuous_acoustic_evidence')
            merge_at.add(i)
    repaired, groups = [], []
    for i, row in enumerate(original):
        if i not in merge_at:
            repaired.append(row.copy()); groups.append([i]); continue
        previous = repaired[-1]
        previous[2] = max(previous[2], row[2])
        previous[3] = max(previous[3], row[3])
        if len(previous) > 4 and len(row) > 4:
            previous[4] = int(bool(previous[4]) and bool(row[4]))
        # Alternative hypotheses from two emissions need not agree; clear them.
        if len(previous) >= 7:
            previous[5:7] = [-1, 0.]
        groups[-1].append(i)
    return repaired, {'schema_version': 'same-pitch-repair-v1', 'config': asdict(config),
                      'energy_source': 'waveform_12ms' if audio is not None else 'cached_rms',
                      'pairs_checked': len(decisions), 'merged_boundaries': len(merge_at),
                      'input_note_count': len(original), 'output_note_count': len(repaired),
                      'source_groups': groups, 'decisions': decisions}
