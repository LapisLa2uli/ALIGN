"""Stack v7: transition-robust training and score-preserving candidate decoding.

Keep the v3 network/checkpoint format. New training uses supervised CTC plus
clean-teacher consistency under stronger recording/transition corruption.
Brief unsupported transition pitches remain optional alignment candidates:
we never delete a genuine short score note solely because of its duration.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch.nn import functional as F
from .ctc_decode_v2 import rich_decode
from .realism_augment_v4 import augment_realism_v4

@dataclass(frozen=True)
class DecodeV7Config:
    blank_scale: float = 0.5
    candidate_threshold: float = 0.08
    transition_max_sec: float = 0.045
    attack_threshold: float = 0.35
    attack_window: int = 1

    def to_dict(self):
        return asdict(self)

def augment_v7(mel, onset, *, long_mels=128):
    value = augment_realism_v4(mel, onset, long_mels=long_mels,
                              blip_probability=0.5, scoop_probability=0.35)
    # Raise quiet-band energy continuously rather than adding isolated attacks.
    # This operates in normalized log-mel space; it is an augmentation, not
    # a calibrated physical noise simulator.
    b, bands, frames = value.shape
    noise = torch.randn(b, bands, 1, device=value.device) * 0.10
    floor = torch.empty(b, 1, 1, device=value.device).uniform_(-2.8, -1.5) + noise
    active = torch.rand(b, 1, 1, device=value.device) < 0.5
    raised = torch.maximum(value, floor)
    return torch.where(active, raised, value).clamp(-4.5, 3.5)

def consistency_loss(student_logits, teacher_logits, valid):
    teacher = teacher_logits.detach().float().softmax(-1)
    confidence = teacher.max(-1).values
    mask = valid.bool() & (confidence > 0.8)
    divergence = F.kl_div(student_logits.float().log_softmax(-1), teacher, reduction='none').sum(-1)
    return (divergence * mask).sum() / mask.sum().clamp_min(1)

def decode_v7(outputs, midi_min=52, config=DecodeV7Config(), hop=256/22050):
    ctc = np.asarray(outputs['ctc'], np.float32)
    if not len(ctc):
        return [], {'transition_candidates': 0}
    notes = rich_decode(ctc, midi_min, config.blank_scale, config.candidate_threshold)
    primary = [i for i, note in enumerate(notes) if not note['optional']]
    changed = 0
    for position in range(1, len(primary)-1):
        left, at, right = primary[position-1:position+2]
        previous, note, following = notes[left], notes[at], notes[right]
        gap = (following['frame'] - note['frame']) * hop
        p, q, r = previous['pitch'], note['pitch'], following['pitch']
        transitional = (p == r and q != p) or (min(p,r) < q < max(p,r))
        if not transitional or gap > config.transition_max_sec:
            continue
        lo, hi = max(0,note['frame']-config.attack_window), min(len(ctc),note['frame']+config.attack_window+1)
        # Pitch boundaries alone are not evidence of a separate attack.
        attack = max(float(np.max(outputs[k][lo:hi])) for k in ('onset','rearticulation'))
        if attack < config.attack_threshold:
            note['optional'] = True
            changed += 1
    starts = [n['frame']*hop for n in notes]
    rows = [[n['pitch'],t,max(t+0.01,starts[k+1] if k+1<len(starts) else t+0.1),
             n['confidence'],int(n['optional']),n['alternative_pitch'],n['alternative_confidence']]
            for k,(n,t) in enumerate(zip(notes,starts))]
    return rows, {'transition_candidates': changed}
