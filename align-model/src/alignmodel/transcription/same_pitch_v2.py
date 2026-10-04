"""Sensitive same-pitch repair with a waveform gap detector at 1 ms resolution.

The mel hop is 11.61 ms: it cannot reliably resolve a 10 ms interruption.
Gap evidence therefore uses a separate 4 ms RMS window stepped every 1 ms.
The wide continuity limits are experimental; inspect repeat false merges.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from .same_pitch_v1 import waveform_levels


@dataclass(frozen=True)
class SamePitchConfig:
    context_sec: float = .09
    neighbor_margin_sec: float = .005
    energy_range_db: float = 60.
    spectral_step_max: float = .40
    spectral_drift_max: float = .70
    merge_attack_max: float = 1.
    voiced_min: float = 0.
    silence_db: float = -120.
    gap_depth_db: float = 18.
    gap_min_sec: float = .010
    envelope_window_sec: float = .004
    envelope_hop_sec: float = .001


def acoustic_evidence(audio, frames, *, sample_rate=22050, hop=256/22050,
                      config=SamePitchConfig()):
    """Prepare once per recording; frame times are sample-accurate and centered."""
    x = np.asarray(audio, np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or sample_rate <= 0:
        raise ValueError('audio must be finite mono at a positive sample rate')
    if not 0 < config.envelope_hop_sec <= .001 or not 0 < config.envelope_window_sec <= .004:
        raise ValueError('10 ms gap support requires hop <=1 ms and window <=4 ms')
    window = max(2, round(sample_rate*config.envelope_window_sec))
    times = np.arange(int(np.ceil(len(x)/sample_rate/config.envelope_hop_sec)))*config.envelope_hop_sec
    centers = np.rint(times*sample_rate).astype(int)
    lo = np.maximum(0, centers-window//2); hi = np.minimum(len(x), centers+window//2)
    sums = np.r_[0., np.cumsum(x*x)]
    power = (sums[hi]-sums[lo])/np.maximum(hi-lo, 1)
    return {'time': times, 'level_db': 10*np.log10(np.maximum(power, 1e-12)),
            'rms_db': waveform_levels(x, frames, sample_rate=sample_rate, hop=hop),
            'duration': len(x)/sample_rate, 'sample_rate': sample_rate,
            'envelope_hop_sec': config.envelope_hop_sec,
            'envelope_window_sec': config.envelope_window_sec}


def gap_evidence(prepared, start, end, config=SamePitchConfig()):
    """Find a low core bounded by higher energy on BOTH sides (release/recovery).

    A 10 ms rectangular gap contains about 6 ms of fully low 4 ms RMS windows.
    Allow one envelope hop for sample phase. Longer cores cover longer gaps.
    Smooth fades with no recovery do not qualify. Times describe the observed
    low core, not a claimed sample-exact acoustic onset/offset.
    """
    step = config.envelope_hop_sec
    a = int(np.searchsorted(prepared['time'], start)); b = int(np.searchsorted(prepared['time'], end, side='right'))
    levels = prepared['level_db'][a:b]
    shortest = max(2, int(np.ceil((config.gap_min_sec-config.envelope_window_sec)/step))-1)
    flank = max(3, int(round(.006/step)))
    result = {'depth_db': 0., 'core_start_sec': None, 'core_end_sec': None, 'core_duration_sec': 0.}
    # Windows slide at 1 ms; flank medians suppress isolated waveform zero crossings.
    for size in sorted(set([shortest, *range(shortest+5, len(levels)-2*flank+1, 5)])):
        if size+2*flank > len(levels): continue
        windows = np.lib.stride_tricks.sliding_window_view(levels, size+2*flank)
        depth = np.minimum(np.median(windows[:, :flank], axis=1), np.median(windows[:, -flank:], axis=1))
        depth -= np.max(windows[:, flank:flank+size], axis=1)
        j = int(np.argmax(depth))
        if depth[j] > result['depth_db']:
            t = float(prepared['time'][a+j+flank])
            result = {'depth_db': float(depth[j]), 'core_start_sec': t,
                      'core_end_sec': t+size*step, 'core_duration_sec': size*step}
    return result


def repair_same_pitch(rows, outputs, mel, *, audio=None, sample_rate=22050,
                      hop=256/22050, config=SamePitchConfig(), prepared=None):
    """Check every original adjacent equal pitch; merge only when all gates pass.

    Models' attack/voicing scores remain observable and configurable, but the
    permissive defaults do not let those same-model outputs veto a repair.
    Raw waveform is required for the gap claim; missing evidence abstains.
    """
    if hop <= 0 or config.context_sec <= 0 or config.gap_min_sec < .010 or config.gap_depth_db <= 0:
        raise ValueError('invalid timing or gap configuration')
    original = [list(r) for r in rows]
    if any(len(r)<4 or not np.isfinite(r[:4]).all() or r[1]<0 or r[2]<r[1] for r in original):
        raise ValueError('invalid note row')
    if any(b[1]<a[1] for a,b in zip(original, original[1:])):
        raise ValueError('notes must be chronological')
    spec = np.asarray(mel) if mel is not None else np.empty((0,0))
    valid_mel = spec.ndim==2 and spec.shape[0]==192
    frames = spec.shape[1] if valid_mel else 0
    if prepared is None and audio is not None:
        prepared = acoustic_evidence(audio, frames, sample_rate=sample_rate, hop=hop, config=config)
    if prepared is not None and any(prepared[k] != getattr(config,k) for k in ('envelope_hop_sec','envelope_window_sec')):
        raise ValueError('prepared evidence envelope configuration mismatch')
    heads = {k:np.asarray(outputs.get(k, [])) for k in ('onset','rearticulation','voiced')}
    decisions, merge_at = [], set()
    for i in range(1,len(original)):
        left,right=original[i-1:i+1]
        if left[0]!=right[0]: continue
        t=float(right[1])
        d={'source_indices':[i-1,i], 'pitch':int(right[0]), 'boundary_sec':t,
           'decision':'retain','reason':'missing_evidence'}
        decisions.append(d)
        if not valid_mel or prepared is None: continue
        next_start=original[i+1][1] if i+1<len(original) else prepared['duration']
        start=max(0.,t-config.context_sec,left[1]+config.neighbor_margin_sec)
        end=min(prepared['duration'],t+config.context_sec,next_start-config.neighbor_margin_sec)
        lo=max(0,int(np.ceil(start/hop))); hi=min(frames,int(np.floor(end/hop))+1)
        # Adaptive context allows much shorter candidates than v1.
        if start>t-.008 or end<t+.008 or hi-lo<3:
            d['reason']='insufficient_context';continue
        if len(prepared['rms_db'])<hi or any(v.ndim!=1 or len(v)<hi for v in heads.values()): continue
        short=spec[128:,lo:hi].astype(np.float64); energy=prepared['rms_db'][lo:hi]
        if not all(np.isfinite(v).all() for v in [short,energy,*[v[lo:hi] for v in heads.values()]]):
            d['reason']='nonfinite_evidence';continue
        active=np.median(short,axis=1)>=np.quantile(np.median(short,axis=1),.75)
        spectrum=short[active];flank=max(1,(hi-lo)//4)
        evidence={'window_sec':[start,end], 'energy_range_db':float(np.ptp(energy)),
                  'energy_min_db':float(np.min(energy)),
                  'spectral_step':float(np.max(np.sqrt(np.mean(np.diff(spectrum,axis=1)**2,axis=0)))),
                  'spectral_drift':float(np.sqrt(np.mean((np.mean(spectrum[:,:flank],axis=1)-np.mean(spectrum[:,-flank:],axis=1))**2))),
                  'attack_max':max(float(np.max(heads[k][lo:hi])) for k in ('onset','rearticulation')),
                  'voiced_min':float(np.min(heads['voiced'][lo:hi])),
                  'gap':gap_evidence(prepared,start,end,config)}
        d['evidence']=evidence
        if evidence['gap']['depth_db'] >= config.gap_depth_db: d['reason']='acoustic_gap'
        elif evidence['energy_range_db']>config.energy_range_db: d['reason']='energy_boundary'
        elif evidence['spectral_step']>config.spectral_step_max or evidence['spectral_drift']>config.spectral_drift_max: d['reason']='spectral_boundary'
        elif evidence['attack_max']>config.merge_attack_max: d['reason']='separate_attack'
        elif evidence['voiced_min']<config.voiced_min or evidence['energy_min_db']<config.silence_db: d['reason']='silence_or_unvoiced'
        else:
            d.update(decision='merge',reason='continuous_within_relaxed_limits');merge_at.add(i)
    repaired,groups=[],[]
    for i,r in enumerate(original):
        if i not in merge_at: repaired.append(r.copy());groups.append([i]);continue
        last=repaired[-1];last[2]=max(last[2],r[2]);last[3]=max(last[3],r[3])
        if len(last)>4 and len(r)>4:last[4]=int(bool(last[4]) and bool(r[4]))
        if len(last)>=7:last[5:7]=[-1,0.]
        groups[-1].append(i)
    return repaired,{'schema_version':'same-pitch-repair-v2','config':asdict(config),
        'energy_source':'waveform_4ms_at_1ms' if prepared is not None else 'missing_waveform',
        'pairs_checked':len(decisions),'merged_boundaries':len(merge_at),
        'input_note_count':len(original),'output_note_count':len(repaired),
        'source_groups':groups,'decisions':decisions}
