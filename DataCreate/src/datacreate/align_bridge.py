"""Bridge DataCreate alignment actions to ALIGN's note-first pipeline."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import uuid
from copy import deepcopy
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
    model_version: str = "legacy"


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


def _matching_transcribed_index(
    notes: list[dict],
    *,
    pitch: int,
    start: float,
    tolerance: float = 0.15,
) -> int | None:
    matches = [
        (abs(float(note.get("start") or 0.0) - start), index)
        for index, note in enumerate(notes)
        if int(note.get("pitch", note.get("midi", -1))) == pitch
    ]
    if not matches:
        return None
    distance, index = min(matches)
    return index if distance <= tolerance else None


def apply_alignment_overrides(payload: dict, sample_dir: Path) -> dict:
    """Apply persistent human mapping corrections after model inference."""

    path = sample_dir / "note_alignment_overrides.json"
    if not path.exists():
        return payload
    document = json.loads(path.read_text(encoding="utf-8"))
    notes = list(payload.get("transcribed_notes") or [])
    mapping = list(payload.get("note_mapping") or [])
    if len(mapping) < len(notes):
        mapping.extend([None] * (len(notes) - len(mapping)))
    events = list(payload.get("events") or [])
    repetitions = list(payload.get("repetitions") or [])
    for override in document.get("overrides") or []:
        pitch = int(override["pitch"])
        start = float(override["performance_start"])
        score_index = int(override["score_index"])
        transcribed_index = _matching_transcribed_index(
            notes,
            pitch=pitch,
            start=start,
            tolerance=float(override.get("onset_tolerance_sec", 0.15)),
        )
        if transcribed_index is None:
            raise ValueError(
                f"Could not find override note MIDI {pitch} near {start:.3f}s"
            )
        template = next(
            (
                event
                for event in events
                if int(event.get("score_index", -1)) == score_index
            ),
            None,
        )
        if template is None:
            raise ValueError(
                f"Could not find reference score event {score_index} for override"
            )
        mapping[transcribed_index] = score_index
        note = notes[transcribed_index]
        event = deepcopy(template)
        event["id"] = f"manual_override_{transcribed_index:05d}"
        event["perf_start"] = float(note["start"])
        event["perf_end"] = float(note["end"])
        relationship = str(override.get("relationship") or "match")
        event["alignment_kind"] = relationship
        event["is_repetition"] = relationship == "copy"
        events = [
            value
            for value in events
            if value.get("id") != event["id"]
            and not (
                abs(float(value.get("perf_start") or -1.0) - float(note["start"]))
                <= 1e-5
                and abs(float(value.get("perf_end") or -1.0) - float(note["end"]))
                <= 1e-5
            )
        ]
        events.append(event)
        if relationship == "copy":
            source_index = _matching_transcribed_index(
                notes,
                pitch=pitch,
                start=float(override["source_performance_start"]),
                tolerance=float(override.get("onset_tolerance_sec", 0.15)),
            )
            if source_index is None:
                raise ValueError("Could not find repetition source note")
            repetitions = [
                value
                for value in repetitions
                if int(value.get("repeat_i0", -1)) != transcribed_index
            ]
            repetitions.append(
                {
                    "source_i0": source_index,
                    "source_i1": source_index + 1,
                    "repeat_i0": transcribed_index,
                    "repeat_i1": transcribed_index + 1,
                    "source_start": float(notes[source_index]["start"]),
                    "source_end": float(notes[source_index]["end"]),
                    "repeat_start": float(note["start"]),
                    "repeat_end": float(note["end"]),
                    "confidence": float(note.get("confidence") or 1.0),
                    "source": "manual_override",
                }
            )
    events.sort(key=lambda event: float(event.get("perf_start") or 0.0))
    payload["events"] = events
    payload["note_mapping"] = mapping
    payload["repetitions"] = repetitions
    summary = dict(payload.get("summary") or {})
    summary["event_count"] = len(events)
    summary["mapped_note_count"] = sum(value is not None for value in mapping)
    summary["repetition_count"] = len(repetitions)
    summary["manual_override_count"] = len(document.get("overrides") or [])
    payload["summary"] = summary
    return payload


def _bridge_command_env(config: PipelineConfig) -> tuple[Path, dict[str, str]]:
    python = _configured_path(
        config,
        "note_alignment_python",
        Path(sys.executable),
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
    env.setdefault("NUMBA_CACHE_DIR", str(ROOT / "align-model/runs/stack-v9/numba-cache"))
    env["PYTHONNOUSERSITE"] = "1"
    return python, env


def _v9_candidate(config: PipelineConfig) -> Path:
    return _configured_path(
        config, "note_alignment_candidate",
        ROOT / "align-model/runs/stack-v9/CANDIDATE_STACK_V9.json",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_v9_feedback(sample_dir: Path, config: PipelineConfig) -> bool:
    """Only reuse feedback for this candidate and these exact score/audio inputs."""
    from datacreate.transcription_labeling import _has_model_feedback

    path = sample_dir / "note_alignment_v2.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _has_model_feedback(payload):
        return False
    provenance = payload.get("provenance") or {}
    inputs = {
        "candidate_sha256": _v9_candidate(config),
        "audio_sha256": sample_dir / "performance_audio.wav",
        "score_sha256": sample_dir / "verified_score.musicxml",
    }
    return all(path.is_file() and provenance.get(key) == _sha256(path)
               for key, path in inputs.items())


def ensure_current_model_feedback(sample_dir: Path, config: PipelineConfig, logger) -> bool:
    """Upgrade legacy/stale UI artifacts before relabeling; never silently downgrade."""
    if config.alignment.get("model_version", "legacy") != "stack-v9":
        return False
    if current_v9_feedback(sample_dir, config):
        return False
    run_preferred_alignment(
        sample_dir / "performance_audio.wav", sample_dir / "reference_audio.wav",
        sample_dir, config, logger,
    )
    if not current_v9_feedback(sample_dir, config):
        raise RuntimeError("V9 regeneration did not produce current model feedback")
    return True


def _run_v9_alignment(sample_dir: Path, config: PipelineConfig, logger, *, detect_candidates: bool):
    """Use the same validated publisher as the full v9 DataCreate inference run."""
    python, env = _bridge_command_env(config)
    work = config.resolved_path("work_dir") or ROOT / "DataCreate/work"
    run = work / "v9-ui" / f"{sample_dir.name}-{uuid.uuid4().hex}"
    command = [
        str(python), str(ROOT / "align-model/scripts/publish_datacreate_v9.py"),
        "--sample", str(sample_dir.resolve()), "--output", str(run),
        "--candidate", str(_v9_candidate(config)),
        "--device", str(config.alignment.get("note_alignment_device", "cuda")),
        "--sample-rate", str(config.sample_rate()),
        "--hop-length", str(config.mel.get("hop_length", 512)),
    ]
    logger.info("Running ALIGN v9: %s", subprocess.list2cmdline(command))
    completed = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True,
        timeout=float(config.alignment.get("note_alignment_timeout_sec", 900)),
    )
    if completed.returncode:
        raise RuntimeError("ALIGN v9 failed: " + (completed.stderr.strip() or completed.stdout.strip()))
    if not current_v9_feedback(sample_dir, config):
        raise RuntimeError("ALIGN v9 returned without matching score/audio/candidate provenance")
    payload = json.loads((sample_dir / "note_alignment_v2.json").read_text(encoding="utf-8"))
    alignment_path = sample_dir / "alignment.npz"
    with np.load(alignment_path) as archive:
        wp = archive["wp"].copy()
    candidates = [Label(**{**raw, "source": "auto"}) for raw in payload["labels"]] if detect_candidates else []
    logger.info("ALIGN v9 wrote %d feedback labels; backup/run: %s", len(payload["labels"]), run)
    return NoteFirstAlignmentResult(candidates, alignment_path, wp, wp, "stack-v9")


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
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
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
    """Run the configured version; v9 regenerates alignment and agent feedback together."""

    version = config.alignment.get("model_version", "legacy")
    if version == "stack-v9":
        return _run_v9_alignment(sample_dir, config, logger, detect_candidates=detect_candidates)
    if version != "legacy":
        raise ValueError(f"Unknown alignment model_version: {version}")
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
    payload = apply_alignment_overrides(payload, sample_dir)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
