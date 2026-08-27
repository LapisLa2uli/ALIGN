from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class RepeatRange:
    start_time: float
    end_time: float


@dataclass
class GraphNote:
    index: int
    pitch: int
    start: float
    end: float
    duration: float
    ql_start: float
    ql_end: float
    measure: int | None = None
    is_rest: bool = False


@dataclass
class LegalEdge:
    from_index: int
    to_index: int
    kind: str  # next | written_repeat | volta


@dataclass
class ScoreGraph:
    notes: list[GraphNote] = field(default_factory=list)
    legal_edges: list[LegalEdge] = field(default_factory=list)
    duration_sec: float = 0.0
    bpm: float = 120.0


@dataclass
class UnfoldedSegment:
    perf_start: float
    perf_end: float
    score_i0: int
    score_i1: int
    dtw_cost: float
    is_repetition: bool = False
    repeats_label_range: RepeatRange | None = None
    hypothesis_rank: int = 0


@dataclass
class RestartHypothesis:
    segments: list[UnfoldedSegment] = field(default_factory=list)
    total_cost: float = 0.0
    unexplained_sec: float = 0.0
    score: float = 0.0


@dataclass
class PairedEvent:
    score_index: int
    pitch: int
    ref_start: float
    ref_end: float
    perf_start: float
    perf_end: float
    kind: str  # match | substitute
    cents: float | None = None
    measure: int | None = None


@dataclass
class PipelineLabel:
    id: str
    type: str
    start_time: float
    end_time: float
    source: str = "pipeline"
    comment: str | None = None
    deviation_cents: float | None = None
    deviation_ms: float | None = None
    measure_number: int | None = None
    note_id: str | None = None
    repeats_label_range: RepeatRange | None = None


@dataclass
class PipelineConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    silence_db: float = -40.0
    min_silence_sec: float = 0.15
    min_hold_sec: float = 0.8
    copy_window_sec: float = 1.6
    copy_sim_threshold: float = 0.72
    min_window_sec: float = 0.55
    beam_k: int = 5
    span_dur_lo: float = 0.5
    span_dur_hi: float = 1.8
    cents_tolerance: float = 20.0
    min_candidate_sec: float = 0.15
    min_extra_sec: float = 0.08
    min_miss_sec: float = 0.08
    chroma_peak_min: float = 0.2
    ewma_alpha: float = 0.3
    ewma_log_threshold: float = 0.25
    far_window: int = 12
    far_gap: int = 6
    far_log_threshold: float = 0.35
    min_rhythm_ms: float = 80.0
    onset_lookback_sec: float = 0.15
    onset_max_shift_sec: float = 0.6
    onset_rise_db: float = 8.0
    bad_start_sec: float = 0.4
    squeak_max_sec: float = 0.28
    device: str = "cuda"
    n_fft: int = 2048
    weights_dir: str | None = "align-model/runs/stages"


@dataclass
class PipelineState:
    sample_id: str
    sample_dir: str
    sr: int
    duration_sec: float
    hop_sec: float
    config: PipelineConfig
    score: ScoreGraph
    device: str = "cpu"
    boundaries: list[float] = field(default_factory=list)
    beam: list[RestartHypothesis] = field(default_factory=list)
    segments: list[UnfoldedSegment] = field(default_factory=list)
    pairs: list[PairedEvent] = field(default_factory=list)
    labels: list[PipelineLabel] = field(default_factory=list)
    stages_run: list[int] = field(default_factory=list)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def state_to_dict(state: PipelineState) -> dict[str, Any]:
    return _to_plain(state)


def labels_document(state: PipelineState) -> dict[str, Any]:
    labels = []
    for lab in state.labels:
        item: dict[str, Any] = {
            "id": lab.id,
            "source": lab.source,
            "start_time": round(lab.start_time, 4),
            "end_time": round(lab.end_time, 4),
            "type": lab.type,
        }
        if lab.comment:
            item["comment"] = lab.comment
        if lab.deviation_cents is not None:
            item["deviation_cents"] = round(lab.deviation_cents, 2)
        if lab.deviation_ms is not None:
            item["deviation_ms"] = round(lab.deviation_ms, 2)
        if lab.measure_number is not None:
            item["measure_number"] = lab.measure_number
        if lab.note_id:
            item["note_id"] = lab.note_id
        if lab.repeats_label_range is not None:
            item["repeats_label_range"] = {
                "start_time": round(lab.repeats_label_range.start_time, 4),
                "end_time": round(lab.repeats_label_range.end_time, 4),
            }
        labels.append(item)
    return {
        "schema_version": "1.1",
        "audio_reference": "performance_audio.wav",
        "annotator_id": "align_pipeline",
        "labels": labels,
        "pipeline": {
            "sample_id": state.sample_id,
            "stages_run": state.stages_run,
            "boundaries": [round(t, 4) for t in state.boundaries],
            "n_segments": len(state.segments),
            "n_pairs": len(state.pairs),
        },
    }


def next_label_id(state: PipelineState) -> str:
    return f"pipe_{len(state.labels):03d}"
