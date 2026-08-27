from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from alignmodel.device import resolve_device
from alignmodel.stage_train import CROP_FRAMES_RESTART, _mel_crop, _mel_span_resize, _rhythm_aux
from alignmodel.stages.models import EDIT_CLASSES, EditCropNet, RestartScorer, RhythmNet
from alignmodel.types import PipelineLabel, PipelineState, RepeatRange, next_label_id


@dataclass
class StageModels:
    restart: RestartScorer | None = None
    edits: EditCropNet | None = None
    rhythm: RhythmNet | None = None
    device: torch.device | None = None
    restart_threshold: float = 0.0
    rhythm_threshold: float = 0.0


def load_stage_models(weights_dir: Path | None, device: str = "cuda") -> StageModels:
    if weights_dir is None:
        return StageModels()
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        return StageModels()
    torch_device = resolve_device(device)
    out = StageModels(device=torch_device)
    s1 = weights_dir / "stage1.pt"
    if s1.exists():
        blob = torch.load(s1, map_location=torch_device, weights_only=False)
        model = RestartScorer()
        try:
            model.load_state_dict(blob["model"])
            out.restart = model.to(torch_device).eval()
            out.restart_threshold = float(blob.get("logit_threshold", 0.0))
        except RuntimeError:
            print(f"skip incompatible stage1 weights in {s1}")
    s2 = weights_dir / "stage2.pt"
    if s2.exists():
        model = EditCropNet()
        blob = torch.load(s2, map_location=torch_device, weights_only=False)
        model.load_state_dict(blob["model"])
        out.edits = model.to(torch_device).eval()
    s3 = weights_dir / "stage3.pt"
    if s3.exists():
        blob = torch.load(s3, map_location=torch_device, weights_only=False)
        model = RhythmNet()
        try:
            model.load_state_dict(blob["model"])
            out.rhythm = model.to(torch_device).eval()
            out.rhythm_threshold = float(blob.get("logit_threshold", 0.0))
        except RuntimeError:
            print(f"skip incompatible stage3 weights in {s3}")
    return out


COPY_SCAN_LENGTHS = (1.25, 1.65, 2.1)
COPY_SCAN_STEP = 0.35


def propose_copy_cuts(
    mel: np.ndarray, models: StageModels, duration: float
) -> list[float]:
    """Times t where [t, t+L] restates [t-L, t]; also return source/end edges."""
    if models.restart is None or mel is None or duration < 1.5:
        return []
    device = models.device
    model = models.restart
    thr = models.restart_threshold
    hits: list[tuple[float, float, float]] = []
    with torch.no_grad():
        for length in COPY_SCAN_LENGTHS:
            times: list[float] = []
            crops_a: list[np.ndarray] = []
            crops_b: list[np.ndarray] = []
            t = length
            while t + length <= duration + 1e-6:
                times.append(t)
                crops_a.append(
                    _mel_span_resize(mel, t, t + length, width=CROP_FRAMES_RESTART)
                )
                crops_b.append(
                    _mel_span_resize(mel, t - length, t, width=CROP_FRAMES_RESTART)
                )
                t += COPY_SCAN_STEP
            if not times:
                continue
            for i0 in range(0, len(times), 64):
                a = torch.from_numpy(np.stack(crops_a[i0 : i0 + 64])).to(device)
                b = torch.from_numpy(np.stack(crops_b[i0 : i0 + 64])).to(device)
                logits = model(a, b).detach().cpu().numpy()
                for t_cut, logit in zip(times[i0 : i0 + 64], logits):
                    if float(logit) > thr:
                        hits.append((float(logit), float(t_cut), float(length)))
    if not hits:
        return []
    hits.sort(key=lambda h: -h[0])
    kept: list[tuple[float, float, float]] = []
    for logit, t_cut, length in hits:
        if any(abs(t_cut - k[1]) < 0.35 for k in kept):
            continue
        kept.append((logit, t_cut, length))
        if len(kept) >= 1:
            break
    cuts: list[float] = []
    for _logit, t_cut, length in kept:
        cuts.extend([t_cut - length, t_cut, t_cut + length])
    return cuts


def apply_learned_restarts(state: PipelineState, mel: np.ndarray, models: StageModels) -> None:
    if models.restart is None or mel is None or len(state.segments) < 2:
        return
    device = models.device
    model = models.restart
    thr = models.restart_threshold

    def _score(t0: float, t1: float, s0: float, s1: float) -> float:
        a = torch.from_numpy(
            _mel_span_resize(mel, t0, t1, width=CROP_FRAMES_RESTART)
        ).unsqueeze(0).to(device)
        b = torch.from_numpy(
            _mel_span_resize(mel, s0, s1, width=CROP_FRAMES_RESTART)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(model(a, b).squeeze())

    for i, cur in enumerate(state.segments):
        if i == 0 or (cur.perf_end - cur.perf_start) < 0.45:
            continue
        prev = state.segments[i - 1]
        prev_len = prev.perf_end - prev.perf_start
        cur_len = cur.perf_end - cur.perf_start
        if prev_len < 0.45:
            continue
        ratio = cur_len / max(prev_len, 1e-3)
        if ratio < 0.8 or ratio > 1.25:
            continue
        logit = _score(cur.perf_start, cur.perf_end, prev.perf_start, prev.perf_end)
        if logit > thr:
            cur.is_repetition = True
            cur.repeats_label_range = RepeatRange(prev.perf_start, prev.perf_end)
            if abs(ratio - 1.0) <= 0.25:
                cur.score_i0 = prev.score_i0
                cur.score_i1 = prev.score_i1


def apply_learned_edits(state: PipelineState, mel: np.ndarray, models: StageModels) -> None:
    if models.edits is None or mel is None:
        return
    device = models.device
    model = models.edits
    editable = {"missed_note", "extra_note", "wrong_note", "intonation_error"}
    kept: list[PipelineLabel] = []
    for lab in state.labels:
        if lab.type not in editable:
            kept.append(lab)
            continue
        crop = torch.from_numpy(_mel_crop(mel, lab.start_time, lab.end_time)).unsqueeze(0).to(
            device
        )
        with torch.no_grad():
            pred = int(model(crop).argmax(-1).item())
        name = EDIT_CLASSES[pred]
        if name == "match":
            continue
        lab.type = name
        lab.comment = (lab.comment or "") + " | stage2-net"
        kept.append(lab)
    state.labels = kept


def apply_learned_rhythm(state: PipelineState, mel: np.ndarray, models: StageModels) -> None:
    if models.rhythm is None or mel is None:
        return
    pairs = [p for p in state.pairs if p.kind in {"match", "substitute"}]
    if not pairs:
        return
    pairs = sorted(pairs, key=lambda p: p.perf_start)
    windows: list[tuple[float, float]] = []
    cur_s, cur_e = pairs[0].perf_start, pairs[0].perf_end
    for pair in pairs[1:]:
        if pair.perf_start - cur_e > 0.25 or pair.perf_end - cur_s > 1.35:
            if cur_e - cur_s >= 0.28:
                windows.append((cur_s, cur_e))
            cur_s, cur_e = pair.perf_start, pair.perf_end
        else:
            cur_e = max(cur_e, pair.perf_end)
    if cur_e - cur_s >= 0.28:
        windows.append((cur_s, cur_e))
    device = models.device
    model = models.rhythm
    thr = models.rhythm_threshold
    for t0, t1 in windows:
        crop = torch.from_numpy(_mel_span_resize(mel, t0, t1)).unsqueeze(0).to(device)
        aux = torch.from_numpy(_rhythm_aux(mel, t0, t1)).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = float(model(crop, aux).squeeze())
        if logit > thr:
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="rhythm_error",
                    start_time=t0,
                    end_time=t1,
                    comment=f"stage3-net logit={logit:.2f}",
                    deviation_ms=(t1 - t0) * 1000.0,
                )
            )
