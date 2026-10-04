"""Label-preserving mel augmentations for real-recording transcription artifacts (v4).

Real clarinet takes contain events the Muse Sounds training audio never has:

* transition blips: 10-40 ms of an in-between pitch where a slurred interval
  changes fingering;
* attack scoops: a note that starts flat and settles within 20-60 ms;
* level dips inside a sustained note (breath, embouchure) with no new attack.

A human does not hear any of them as a separate note, so targets are left
unchanged. Pitch changes are applied as shifts along the mel axis (the long
branch shift is halved for the 64-band short branch), the same mechanism the
v3 vibrato augmentation uses.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _shift_bands(mel: Tensor, shift: Tensor, long_mels: int) -> Tensor:
    result = mel.clone()
    bands = mel.shape[1]
    for lo, hi, scale in ((0, long_mels, 1.0), (long_mels, bands, 0.5)):
        segment = mel[:, lo:hi]
        count = hi - lo
        positions = torch.arange(count, device=mel.device, dtype=torch.float32)[None, :, None]
        source = (positions - scale * shift[:, None, :]).clamp(0, count - 1)
        low = source.floor().long()
        high = (low + 1).clamp(max=count - 1)
        frac = source - low.float()
        result[:, lo:hi] = torch.gather(segment, 1, low) * (1 - frac) + torch.gather(segment, 1, high) * frac
    return result


def augment_realism_v4(
    mel: Tensor,
    onset: Tensor,
    *,
    long_mels: int,
    blip_probability: float = 0.35,
    scoop_probability: float = 0.25,
    dip_rate: float = 0.4,
    clip_probability: float = 0.7,
) -> Tensor:
    """Add transition blips, attack scoops and mid-note dips to a [B, bands, T] batch."""

    batch, _bands, frames = mel.shape
    device = mel.device
    clip_on = (torch.rand(batch, 1, device=device) < clip_probability)
    marks = (onset > 0.5) & clip_on
    shift = torch.zeros(batch, frames, device=device)

    # Transition blips: 1-3 frames straddling a note boundary, shifted 1-4 long bands either way.
    chosen = marks & (torch.rand(batch, frames, device=device) < blip_probability)
    length = torch.randint(1, 4, (batch, frames), device=device)
    start = -torch.randint(0, 3, (batch, frames), device=device)
    size = torch.empty(batch, frames, device=device).uniform_(1.0, 4.0)
    size = size * torch.where(torch.rand(batch, frames, device=device) < 0.5, -1.0, 1.0)
    for offset in range(-2, 3):
        inside = chosen & (offset >= start) & (offset < start + length)
        target = torch.zeros_like(shift)
        if offset >= 0:
            target[:, offset:] = torch.where(inside[:, :frames - offset], size[:, :frames - offset], 0.0)
        else:
            target[:, :offset] = torch.where(inside[:, -offset:], size[:, -offset:], 0.0)
        shift = torch.where(target != 0, target, shift)

    # Attack scoops: the first 2-5 frames of a note start 0.5-2 bands flat and settle.
    chosen = marks & (torch.rand(batch, frames, device=device) < scoop_probability)
    span = torch.randint(2, 6, (batch, frames), device=device).float()
    depth = torch.empty(batch, frames, device=device).uniform_(0.5, 2.0)
    for offset in range(0, 5):
        value = torch.where(chosen & (offset < span), -depth * (1.0 - offset / span), torch.zeros_like(depth))
        target = torch.zeros_like(shift)
        target[:, offset:] = value[:, :frames - offset] if offset else value
        shift = torch.where((target != 0) & (shift == 0), target, shift)
    result = _shift_bands(mel, shift, long_mels)

    # Mid-note dips: 2-6 frames, 0.3-1.0 log units, at least 5 frames from any onset.
    near_onset = torch.zeros(batch, frames, dtype=torch.bool, device=device)
    raw = onset > 0.5
    for back in range(-5, 6):
        if back >= 0:
            near_onset[:, back:] |= raw[:, :frames - back] if back else raw
        else:
            near_onset[:, :back] |= raw[:, -back:]
    count = int(dip_rate * frames / 100)
    if count > 0:
        centers = torch.randint(3, max(frames - 3, 4), (batch, count), device=device)
        widths = torch.randint(1, 4, (batch, count), device=device)
        depths = torch.empty(batch, count, device=device).uniform_(0.3, 1.0)
        dip = torch.zeros(batch, frames, device=device)
        index = torch.arange(frames, device=device)[None, None, :]
        window = (index >= (centers - widths)[..., None]) & (index <= (centers + widths)[..., None])
        dip = torch.maximum(dip, (window.float() * depths[..., None]).amax(dim=1))
        dip = torch.where(near_onset | ~clip_on, torch.zeros_like(dip), dip)
        result = result - dip[:, None, :]
    return result.clamp(-4.5, 3.5)
