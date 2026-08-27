from __future__ import annotations

import numpy as np

from alignmodel.types import PairedEvent, PipelineLabel, PipelineState, next_label_id


def run_stage3(state: PipelineState, *, learned=None, mel=None) -> None:
    if (
        learned is not None
        and getattr(learned, "rhythm", None) is not None
        and mel is not None
    ):
        from alignmodel.stages.learned import apply_learned_rhythm

        apply_learned_rhythm(state, mel, learned)
        state.stages_run.append(3)
        return
    cfg = state.config
    pairs = [p for p in state.pairs if p.kind in {"match", "substitute"}]
    if len(pairs) < 3:
        state.stages_run.append(3)
        return
    pairs = sorted(pairs, key=lambda p: p.perf_start)
    ratios: list[float | None] = [_ratio(p) for p in pairs]
    ewma: float | None = None
    ewma_flagged: set[int] = set()
    for i, (pair, ratio) in enumerate(zip(pairs, ratios)):
        if ratio is None:
            continue
        if ewma is None:
            ewma = ratio
            continue
        log_jump = abs(float(np.log(ratio / max(ewma, 1e-6))))
        dur_ms = abs((pair.perf_end - pair.perf_start) - (pair.ref_end - pair.ref_start)) * 1000.0
        if log_jump > cfg.ewma_log_threshold and dur_ms >= cfg.min_rhythm_ms:
            ewma_flagged.add(i)
            _rhythm_label(
                state,
                pair,
                f"ewma tempo jump (log={log_jump:.3f})",
                (pair.perf_end - pair.perf_start) * 1000.0,
            )
        ewma = cfg.ewma_alpha * ratio + (1.0 - cfg.ewma_alpha) * ewma

    far_window = cfg.far_window
    far_gap = cfg.far_gap
    for i, (pair, ratio) in enumerate(zip(pairs, ratios)):
        if ratio is None or i in ewma_flagged:
            continue
        end = i - far_gap
        if end <= 0:
            continue
        start = max(0, end - far_window)
        window_ratios = [r for r in ratios[start:end] if r is not None]
        if len(window_ratios) < max(3, far_window // 3):
            continue
        far_median = float(np.median(window_ratios))
        if far_median <= 0:
            continue
        log_drift = abs(float(np.log(ratio / far_median)))
        dur_ms = abs((pair.perf_end - pair.perf_start) - (pair.ref_end - pair.ref_start)) * 1000.0
        if log_drift > cfg.far_log_threshold and dur_ms >= cfg.min_rhythm_ms:
            _rhythm_label(
                state,
                pair,
                f"far-window tempo drift (log={log_drift:.3f})",
                (pair.perf_end - pair.perf_start) * 1000.0,
            )
    state.stages_run.append(3)


def _ratio(pair: PairedEvent, eps: float = 1e-4) -> float | None:
    ref = pair.ref_end - pair.ref_start
    perf = pair.perf_end - pair.perf_start
    if ref < eps or perf < eps:
        return None
    return perf / ref


def _rhythm_label(state: PipelineState, pair: PairedEvent, comment: str, ms: float) -> None:
    state.labels.append(
        PipelineLabel(
            id=next_label_id(state),
            type="rhythm_error",
            start_time=pair.perf_start,
            end_time=pair.perf_end,
            comment=comment,
            deviation_ms=ms,
            measure_number=pair.measure,
            note_id=f"note_{pair.score_index:04d}",
        )
    )
