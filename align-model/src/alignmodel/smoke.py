from __future__ import annotations

import torch

from alignmodel.config import FRAME_HOP_SEC, ModelConfig
from alignmodel.model import RumaLite
from alignmodel.train import compute_loss, downsample_extra


def test_forward_shapes() -> None:
    cfg = ModelConfig(max_audio_frames=128, max_score_notes=16, audio_layers=1, score_layers=1, fusion_layers=1)
    model = RumaLite(cfg)
    b, t, n = 2, 128, 16
    mel = torch.randn(b, cfg.n_mels, t)
    mel_mask = torch.ones(b, t, dtype=torch.bool)
    pitch = torch.randint(40, 80, (b, n))
    onset = torch.linspace(0, 4, n).unsqueeze(0).expand(b, -1)
    duration = torch.full((b, n), 0.25)
    note_mask = torch.ones(b, n, dtype=torch.bool)
    note_mask[:, 12:] = False
    out = model(mel, mel_mask, pitch, onset, duration, note_mask, FRAME_HOP_SEC)
    assert out["score_logits"].shape == (b, n, cfg.num_score_classes)
    assert out["extra_logits"].ndim == 2
    assert out["repeat_logits"].shape == (b,)
    extra_y = torch.zeros(b, t)
    extra_y[:, 40:50] = 1
    batch = {
        "score_y": torch.zeros(b, n, dtype=torch.long),
        "note_mask": note_mask,
        "extra_y": extra_y,
        "repeat_y": torch.tensor([0.0, 1.0]),
    }
    loss, parts = compute_loss(out, batch, cfg)
    assert torch.isfinite(loss)
    assert "score" in parts
    pooled = downsample_extra(extra_y, out["audio_mask"], cfg.audio_stride)
    assert pooled.shape == out["extra_logits"].shape


if __name__ == "__main__":
    test_forward_shapes()
    print("forward ok")
