"""Bridge DataCreate alignment actions to ALIGN's note-first pipeline."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datacreate.config import PipelineConfig
from datacreate.models import Label

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class NoteFirstAlignmentResult:
    candidates: list[Label]
    alignment_path: Path
    warping_path: np.ndarray
    wp: np.ndarray


def _configured_path(
    config: PipelineConfig, key: str, fallback: Path
) -> Path:
    configured = config.resolved_path(key)
    return configured if configured is not None else fallback


def _joint_checkpoint(config: PipelineConfig, weights: Path) -> Path | None:
    explicit = config.resolved_path("note_alignment_checkpoint")
    if explicit is not None and explicit.is_file():
        return explicit
    if weights.is_file() and weights.suffix.lower() == ".pt":
        return weights
    nested = weights / "joint_decoder.pt"
    if nested.is_file():
        return nested
    return None


def _bridge_command_env(config: PipelineConfig) -> tuple[Path, dict[str, str]]:
    python = _configured_path(
        config,
        "note_alignment_python",
        ROOT / "align-model" / ".venv-amt-bench" / "Scripts" / "python.exe",
    )
    if not python.is_file():
        raise FileNotFoundError(f"Note alignment Python not found: {python}")
    env = os.environ.copy()
    source_paths = (
        ROOT / "align-model" / "src",
        ROOT / "synth-pipeline" / "src",
        ROOT / "DataCreate" / "src",
    )
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in source_paths), *([existing] if existing else [])]
    )
    return python, env


def dump_transcription(
    sample_dir: Path,
    config: PipelineConfig,
    logger,
    *,
    output: Path | None = None,
) -> Path:
    """Write frozen Basic Pitch notes for score-span location."""

    python, env = _bridge_command_env(config)
    output = output or (sample_dir / "transcription_notes.json")
    command = [
        str(python),
        str(ROOT / "align-model" / "scripts" / "datacreate_alignment_bridge.py"),
        "--sample",
        str(sample_dir),
        "--out",
        str(output),
        "--transcribe-only",
        "--device",
        str(config.alignment.get("note_alignment_device", "cuda")),
    ]
    logger.info("Transcribing %s", sample_dir.name)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=float(config.alignment.get("note_alignment_timeout_sec", 900)),
    )
    if completed.returncode:
        raise RuntimeError(
            "Transcription failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return output


def _compatibility_alignment(
    payload: dict,
    sample_dir: Path,
    config: PipelineConfig,
) -> tuple[Path, np.ndarray]:
    sample_rate = config.sample_rate()
    hop_length = int(config.mel.get("hop_length", 512))
    points = []
    for event in payload.get("events") or []:
        for ref_key, perf_key in (
            ("ref_start", "perf_start"),
            ("ref_end", "perf_end"),
        ):
            if event.get(ref_key) is None or event.get(perf_key) is None:
                continue
            points.append(
                (
                    int(round(float(event[ref_key]) * sample_rate / hop_length)),
                    int(round(float(event[perf_key]) * sample_rate / hop_length)),
                )
            )
    points = sorted(set(points))
    wp = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    ref_frames = max(
        2,
        1
        + max(
            (
                int(round(float(event.get("ref_end") or 0.0) * sample_rate / hop_length))
                for event in payload.get("events") or []
            ),
            default=0,
        ),
    )
    perf_frames = max(
        2,
        1
        + max(
            (
                int(round(float(event.get("perf_end") or 0.0) * sample_rate / hop_length))
                for event in payload.get("events") or []
            ),
            default=0,
        ),
    )
    path = sample_dir / "alignment.npz"
    np.savez_compressed(
        path,
        warping_path=wp,
        wp=wp,
        frame_residuals=np.zeros(len(wp), dtype=np.float32),
        ref_features=np.zeros((1, ref_frames), dtype=np.float32),
        perf_features=np.zeros((1, perf_frames), dtype=np.float32),
        hop_length=np.asarray(hop_length, dtype=np.int32),
        sample_rate=np.asarray(sample_rate, dtype=np.int32),
        engine=np.asarray(str(payload.get("engine") or "align-note-first")),
    )
    return path, wp


def run_preferred_alignment(
    performance_wav: Path,
    reference_wav: Path,
    sample_dir: Path,
    config: PipelineConfig,
    logger,
    *,
    detect_candidates: bool = True,
) -> NoteFirstAlignmentResult:
    """Run joint transcription/alignment; the reference WAV is retained for API parity."""

    if not performance_wav.is_file() or not reference_wav.is_file():
        raise FileNotFoundError("Both performance and reference audio required")
    python, env = _bridge_command_env(config)
    weights = _configured_path(
        config,
        "note_alignment_weights",
        ROOT
        / "align-model"
        / "runs"
        / "contextual-aligner-outputRaw_sf-1k"
        / "weights",
    )
    checkpoint = _joint_checkpoint(config, weights)
    if checkpoint is None and not weights.is_dir():
        raise FileNotFoundError(f"Note alignment weights not found: {weights}")
    output = sample_dir / "note_alignment_v2.json"
    command = [
        str(python),
        str(ROOT / "align-model" / "scripts" / "datacreate_alignment_bridge.py"),
        "--sample",
        str(sample_dir),
        "--out",
        str(output),
        "--device",
        str(config.alignment.get("note_alignment_device", "cuda")),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    else:
        command.extend(["--weights", str(weights)])
    logger.info("Running note alignment: %s", subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=float(config.alignment.get("note_alignment_timeout_sec", 900)),
    )
    if completed.returncode:
        raise RuntimeError(
            "Alignment failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    alignment_path, wp = _compatibility_alignment(payload, sample_dir, config)
    candidates = []
    if detect_candidates:
        for raw in payload.get("labels") or []:
            values = dict(raw)
            values["source"] = "auto"
            candidates.append(Label(**values))
    logger.info(
        "%s alignment wrote %d events and %d candidates",
        payload.get("engine") or "note",
        len(payload.get("events") or []),
        len(candidates),
    )
    return NoteFirstAlignmentResult(candidates, alignment_path, wp, wp)
