from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from alignmodel.device import resolve_device
from alignmodel.stage_train import MEL_HOP, _cosine, _mel_crop, _pool_span
from alignmodel.stages.models import EDIT_CLASSES, EditCropNet, RestartScorer, RhythmNet
from alignmodel.types import PairedEvent, PipelineLabel, PipelineState, RepeatRange, next_label_id


@dataclass
class StageModels:
    restart: RestartScorer | None = None
    edits: EditCropNet | None = None
    rhythm: RhythmNet | None = None
    device: torch.device | None = None


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
        model = RestartScorer()
        blob = torch.load(s1, map_location=torch_device, weights_only=False)
        model.load_state_dict(blob["model"])
        out.restart = model.to(torch_device).eval()
    s2 = weights_dir / "stage2.pt"
    if s2.exists():
        model = EditCropNet()
        blob = torch.load(s2, map_location=torch_device, weights_only=False)
        model.load_state_dict(blob["model"])
        out.edits = model.to(torch_device).eval()
    s3 = weights_dir / "stage3.pt"
    if s3.exists():
        model = RhythmNet()
        blob = torch.load(s3, map_location=torch_device, weights_only=False)
        model.load_state_dict(blob["model"])
        out.rhythm = model.to(torch_device).eval()
    return out


def restart_feature(mel: np.ndarray, t0: float, t1: float, s0: float, s1: float) -> torch.Tensor:
    cur = _pool_span(mel, t0, t1)
    src = _pool_span(mel, s0, s1)
    dur = max(t1 - t0, 1e-3)
    sdur = max(s1 - s0, 1e-3)
    feat = np.concatenate(
        [
            cur,
            src,
            np.array(
                [
                    np.log(dur),
                    np.log(sdur),
                    _cosine(cur, src),
                    t0 / max(mel.shape[1] * MEL_HOP, 1e-3),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    return torch.from_numpy(feat).unsqueeze(0)


def apply_learned_restarts(state: PipelineState, mel: np.ndarray, models: StageModels) -> None:
    if models.restart is None or mel is None or len(state.segments) < 2:
        return
    device = models.device
    model = models.restart
    for i in range(1, len(state.segments)):
        prev = state.segments[i - 1]
        cur = state.segments[i]
        feat = restart_feature(
            mel, cur.perf_start, cur.perf_end, prev.perf_start, prev.perf_end
        ).to(device)
        with torch.no_grad():
            logit = float(model(feat).squeeze())
        if logit > 0:
            cur.is_repetition = True
            cur.repeats_label_range = RepeatRange(prev.perf_start, prev.perf_end)


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


def rhythm_feature(pair: PairedEvent, idx: int, n: int, prev_ratio: float, ewma: float, far_med: float) -> np.ndarray:
    ref = max(pair.ref_end - pair.ref_start, 1e-3)
    perf = max(pair.perf_end - pair.perf_start, 1e-3)
    ratio = perf / ref
    log_jump = abs(float(np.log(ratio / max(ewma, 1e-6))))
    log_far = abs(float(np.log(ratio / max(far_med, 1e-6)))) if far_med > 0 else 0.0
    return np.array(
        [
            float(np.log(ratio)),
            abs(float(np.log(ratio))),
            float(np.log(ref)),
            float(np.log(perf)),
            log_jump,
            log_far,
            idx / max(n - 1, 1),
            float(np.log(max(prev_ratio, 1e-4))),
        ],
        dtype=np.float32,
    )


def apply_learned_rhythm(state: PipelineState, models: StageModels) -> None:
    if models.rhythm is None:
        return
    pairs = [p for p in state.pairs if p.kind in {"match", "substitute"}]
    if len(pairs) < 2:
        return
    pairs = sorted(pairs, key=lambda p: p.perf_start)
    device = models.device
    model = models.rhythm
    ewma = None
    ratios = []
    for i, pair in enumerate(pairs):
        ref = max(pair.ref_end - pair.ref_start, 1e-3)
        perf = max(pair.perf_end - pair.perf_start, 1e-3)
        ratio = perf / ref
        ratios.append(ratio)
        if ewma is None:
            ewma = ratio
            prev = ratio
        else:
            prev = ratios[i - 1]
        far = [r for r in ratios[max(0, i - 12) : max(0, i - 6)] if r]
        far_med = float(np.median(far)) if far else 0.0
        feat = torch.from_numpy(
            rhythm_feature(pair, i, len(pairs), prev, ewma, far_med)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = float(model(feat).squeeze())
        ewma = 0.3 * ratio + 0.7 * ewma
        if logit > 0:
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="rhythm_error",
                    start_time=pair.perf_start,
                    end_time=pair.perf_end,
                    comment=f"stage3-net logit={logit:.2f}",
                    deviation_ms=(pair.perf_end - pair.perf_start) * 1000.0,
                    measure_number=pair.measure,
                    note_id=f"note_{pair.score_index:04d}",
                )
            )
