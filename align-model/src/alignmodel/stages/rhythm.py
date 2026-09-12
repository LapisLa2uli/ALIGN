from __future__ import annotations

import numpy as np

from alignmodel.stages.dc_alignment import ensure_rhythm_pairs
from alignmodel.types import PairedEvent, PipelineLabel, PipelineState, next_label_id

RHYTHM_PAIR_KINDS = {"match", "substitute", "rest"}


def run_stage3(state: PipelineState, *, learned=None, mel=None) -> None:
    pairs = [
        p
        for p in ensure_rhythm_pairs(state, learned=learned, mel=mel)
        if p.kind in RHYTHM_PAIR_KINDS
    ]
    detector = str(getattr(state.config, "rhythm_detector", "gated_net") or "gated_net")
    has_net = (
        learned is not None
        and getattr(learned, "rhythm", None) is not None
        and mel is not None
    )
    override = getattr(state.config, "rhythm_logit_override", None)
    if has_net and override is not None:
        learned.rhythm_threshold = float(override)
    if has_net and detector in {"gated_net", "net"}:
        from alignmodel.stages.learned import apply_learned_rhythm

        gate = None
        if detector == "gated_net":
            gate = merge_time_spans(
                flagged_rhythm_spans(pairs, state.config),
                gap=float(state.config.rhythm_merge_gap_sec),
                min_dur=float(state.config.min_candidate_sec),
            )
        apply_learned_rhythm(state, mel, learned, pairs=pairs, gate_spans=gate)
    else:
        if len(pairs) >= 3:
            for pair, comment, ms in flagged_rhythm_hits(pairs, state.config):
                _rhythm_label(state, pair, comment, ms)
    merge_rhythm_labels(state, gap=float(state.config.rhythm_merge_gap_sec))
    state.stages_run.append(3)


def flagged_rhythm_spans(pairs: list[PairedEvent], cfg) -> list[tuple[float, float]]:
    return [
        (pair.perf_start, pair.perf_end)
        for pair, _comment, _ms in flagged_rhythm_hits(pairs, cfg)
    ]


def flagged_rhythm_hits(
    pairs: list[PairedEvent], cfg
) -> list[tuple[PairedEvent, str, float]]:
    """EWMA + far-window flags on DataCreate (or fallback) duration ratios."""
    if len(pairs) < 3:
        return []
    pairs = sorted(pairs, key=lambda p: (p.perf_start, p.ref_start))
    ratios: list[float | None] = [_ratio(p) for p in pairs]
    hits: list[tuple[PairedEvent, str, float]] = []
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
            hits.append(
                (
                    pair,
                    f"ewma tempo jump (log={log_jump:.3f})",
                    (pair.perf_end - pair.perf_start) * 1000.0,
                )
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
            hits.append(
                (
                    pair,
                    f"far-window tempo drift (log={log_drift:.3f})",
                    (pair.perf_end - pair.perf_start) * 1000.0,
                )
            )
    return hits


def merge_time_spans(
    spans: list[tuple[float, float]],
    *,
    gap: float = 0.05,
    min_dur: float = 0.15,
) -> list[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted((float(a), float(b)) for a, b in spans if b > a)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_s, last_e = merged[-1]
        if start <= last_e + gap:
            merged[-1] = (last_s, max(last_e, end))
        else:
            merged.append((start, end))
    out: list[tuple[float, float]] = []
    for start, end in merged:
        if end - start < min_dur:
            extra = 0.5 * (min_dur - (end - start))
            start = max(0.0, start - extra)
            end = start + min_dur
        out.append((start, end))
    return out


def merge_rhythm_labels(state: PipelineState, *, gap: float = 0.05) -> None:
    rhythm = [lab for lab in state.labels if lab.type == "rhythm_error"]
    other = [lab for lab in state.labels if lab.type != "rhythm_error"]
    if len(rhythm) <= 1:
        state.labels = other + rhythm
        return
    ordered = sorted(rhythm, key=lambda lab: (lab.start_time, lab.end_time))
    merged = [ordered[0]]
    for lab in ordered[1:]:
        last = merged[-1]
        if lab.start_time <= last.end_time + gap:
            last.end_time = max(last.end_time, lab.end_time)
            if lab.deviation_ms is not None:
                last.deviation_ms = max(float(last.deviation_ms or 0.0), float(lab.deviation_ms))
            if lab.comment and last.comment and lab.comment not in last.comment:
                last.comment = f"{last.comment}; {lab.comment}"
        else:
            merged.append(lab)
    min_dur = float(getattr(state.config, "min_candidate_sec", 0.15))
    for lab in merged:
        width = lab.end_time - lab.start_time
        if width < min_dur:
            extra = 0.5 * (min_dur - width)
            lab.start_time = max(0.0, lab.start_time - extra)
            lab.end_time = lab.start_time + min_dur
    state.labels = other + merged


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
            note_id=f"note_{pair.score_index:04d}" if pair.score_index >= 0 else None,
        )
    )
