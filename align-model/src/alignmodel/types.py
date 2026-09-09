from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class RepeatRange:
    start_time: float
    end_time: float


@dataclass
class ScorePart:
    start_note_index: int
    end_note_index: int
    pad_notes: int = 2
    start_measure: int | None = None
    end_measure: int | None = None
    core_start_note_index: int | None = None
    core_end_note_index: int | None = None


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
    score_part: ScorePart | None = None
    pitches: list[int] | None = None
    note_ids: list[str] | None = None
    extra_copies: int | None = None


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
    use_dc_rhythm_alignment: bool = True
    rhythm_detector: str = "gated_net"  # gated_net | net | heuristic
    rhythm_merge_gap_sec: float = 0.05
    # 2k holdout: checkpoint -0.6 over-fires; 1.0 with DC-gated spans matches gold count better.
    rhythm_logit_override: float | None = 1.0
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
    rhythm_pairs: list[PairedEvent] = field(default_factory=list)
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


def pipeline_label_to_dict(lab: PipelineLabel) -> dict[str, Any]:
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
    if lab.score_part is not None:
        part: dict[str, Any] = {
            "start_note_index": lab.score_part.start_note_index,
            "end_note_index": lab.score_part.end_note_index,
            "pad_notes": lab.score_part.pad_notes,
        }
        if lab.score_part.start_measure is not None:
            part["start_measure"] = lab.score_part.start_measure
        if lab.score_part.end_measure is not None:
            part["end_measure"] = lab.score_part.end_measure
        if lab.score_part.core_start_note_index is not None:
            part["core_start_note_index"] = lab.score_part.core_start_note_index
        if lab.score_part.core_end_note_index is not None:
            part["core_end_note_index"] = lab.score_part.core_end_note_index
        item["score_part"] = part
    if lab.pitches:
        item["pitches"] = [int(p) for p in lab.pitches]
    if lab.note_ids:
        item["note_ids"] = [str(x) for x in lab.note_ids]
    if lab.extra_copies is not None:
        item["extra_copies"] = int(lab.extra_copies)
    return item


def labels_document(state: PipelineState) -> dict[str, Any]:
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": "align_pipeline",
        "labels": [pipeline_label_to_dict(lab) for lab in state.labels],
        "pipeline": {
            "sample_id": state.sample_id,
            "stages_run": state.stages_run,
            "boundaries": [round(t, 4) for t in state.boundaries],
            "n_segments": len(state.segments),
            "n_pairs": len(state.pairs),
            "n_rhythm_pairs": len(state.rhythm_pairs),
        },
    }


def schema12_document(
    *,
    sample_id: str,
    labels: list[dict[str, Any]],
    annotator_id: str = "align_melody",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": annotator_id,
        "labels": labels,
    }
    if extra:
        doc.update(extra)
    doc.setdefault("pipeline", {})["sample_id"] = sample_id
    return doc


def next_label_id(state: PipelineState) -> str:
    return f"pipe_{len(state.labels):03d}"
