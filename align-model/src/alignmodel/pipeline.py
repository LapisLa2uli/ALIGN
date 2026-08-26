from __future__ import annotations

from pathlib import Path

from alignmodel.audio import extract_chroma, load_mono, set_audio_device
from alignmodel.device import device_label
from alignmodel.stage_train import _load_mel
from alignmodel.stages.boundaries import apply_boundaries
from alignmodel.stages.edits import run_stage2
from alignmodel.stages.learned import load_stage_models
from alignmodel.stages.restart import run_stage1
from alignmodel.stages.rhythm import run_stage3
from alignmodel.stages.score_graph import build_score_graph
from alignmodel.stages.timbre import run_stage4
from alignmodel.types import (
    PipelineConfig,
    PipelineState,
    UnfoldedSegment,
    labels_document,
    state_to_dict,
)


def load_bundle_audio(sample_dir: Path, cfg: PipelineConfig):
    wav = sample_dir / "performance_audio.wav"
    if not wav.exists():
        raise FileNotFoundError(f"Missing {wav}")
    audio = load_mono(wav, cfg.sample_rate)
    chroma, hop_sec = extract_chroma(
        audio, cfg.sample_rate, cfg.hop_length, n_fft=cfg.n_fft
    )
    duration = len(audio) / float(cfg.sample_rate)
    return audio, chroma, hop_sec, duration


def run_pipeline(
    sample_dir: Path,
    *,
    stages: set[int] | None = None,
    timbre: bool = False,
    config: PipelineConfig | None = None,
    device: str | None = None,
    weights_dir: Path | str | None = None,
) -> PipelineState:
    sample_dir = Path(sample_dir)
    cfg = config or PipelineConfig()
    if device:
        cfg.device = device
    if weights_dir is not None:
        cfg.weights_dir = str(weights_dir)
    torch_device = set_audio_device(cfg.device)
    learned = load_stage_models(Path(cfg.weights_dir) if cfg.weights_dir else None, cfg.device)
    wanted = set(stages or {1, 2, 3})
    if timbre:
        wanted.add(4)
    score_path = sample_dir / "verified_score.musicxml"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing {score_path}")

    audio, chroma, hop_sec, duration = load_bundle_audio(sample_dir, cfg)
    ref_chroma = None
    ref_wav = sample_dir / "reference_audio.wav"
    if ref_wav.exists():
        ref_audio = load_mono(ref_wav, cfg.sample_rate)
        ref_chroma, _ = extract_chroma(
            ref_audio, cfg.sample_rate, cfg.hop_length, n_fft=cfg.n_fft
        )
    graph = build_score_graph(score_path)
    state = PipelineState(
        sample_id=sample_dir.name,
        sample_dir=str(sample_dir),
        sr=cfg.sample_rate,
        duration_sec=duration,
        hop_sec=hop_sec,
        config=cfg,
        score=graph,
        device=device_label(torch_device),
    )
    mel = None
    if (sample_dir / "performance_mel.npy").exists():
        mel = _load_mel(sample_dir)

    if 1 in wanted:
        apply_boundaries(state, audio, chroma)
        run_stage1(state, chroma, ref_chroma, mel=mel, learned=learned)
    else:
        state.segments = [
            UnfoldedSegment(
                0.0,
                duration,
                0,
                len(graph.notes),
                0.0,
                False,
                None,
            )
        ]

    if 2 in wanted:
        run_stage2(state, audio, chroma, ref_chroma, mel=mel, learned=learned)
    if 3 in wanted:
        run_stage3(state, learned=learned)
    if 4 in wanted:
        run_stage4(state, audio)
    return state


def write_prediction(state: PipelineState, path: Path, *, include_state: bool = False) -> None:
    import json

    doc = labels_document(state)
    if include_state:
        payload = dict(doc)
        payload["state"] = _strip_config(state_to_dict(state))
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _strip_config(blob: dict) -> dict:
    # Config is useful; keep it. Drop huge nothing.
    return blob
