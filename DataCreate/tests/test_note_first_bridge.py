from __future__ import annotations

import json
import logging
import subprocess
import sys
import hashlib
import pytest
from pathlib import Path
from unittest.mock import patch

from datacreate.align_bridge import (
    apply_alignment_overrides,
    dump_transcription,
    run_preferred_alignment,
)
from datacreate.config import PipelineConfig
from datacreate.note_alignment import build_note_alignment
from datacreate.sample_prep import _invalidate_alignment_artifacts


def _config(python: Path, weights: Path, checkpoint: Path | None = None) -> PipelineConfig:
    paths = {
        "note_alignment_python": str(python),
        "note_alignment_weights": str(weights),
    }
    if checkpoint is not None:
        paths["note_alignment_checkpoint"] = str(checkpoint)
    return PipelineConfig(
        paths=paths,
        audio={"sample_rate": 22050},
        mel={"hop_length": 512},
        alignment={"note_alignment_device": "cpu"},
    )


def test_bridge_writes_compatibility_and_gui_alignment(tmp_path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"x")
    weights = tmp_path / "weights"
    weights.mkdir()
    sample = tmp_path / "sample"
    sample.mkdir()
    perf = sample / "performance_audio.wav"
    ref = sample / "reference_audio.wav"
    perf.write_bytes(b"wav")
    ref.write_bytes(b"wav")

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--out") + 1])
        output.write_text(
            json.dumps(
                {
                    "engine": "align-note-first",
                    "events": [
                        {
                            "id": "aligned_00000",
                            "score_index": 0,
                            "is_rest": False,
                            "midi": 60,
                            "ref_start": 0.0,
                            "ref_end": 0.5,
                            "perf_start": 0.1,
                            "perf_end": 0.6,
                        }
                    ],
                    "labels": [],
                    "transcribed_notes": [
                        {"pitch": 60, "start": 0.1, "end": 0.6, "confidence": 0.9}
                    ],
                    "note_mapping": [0],
                    "summary": {"event_count": 1},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    with patch("datacreate.align_bridge.subprocess.run", side_effect=fake_run):
        result = run_preferred_alignment(
            perf,
            ref,
            sample,
            _config(python, weights),
            logging.getLogger("test"),
        )
    assert result.alignment_path.exists()
    assert (sample / "note_alignment_v2.json").exists()
    payload = build_note_alignment(sample)
    assert payload["summary"]["engine"] == "align-note-first"
    assert payload["events"][0]["perf_start"] == 0.1
    assert payload["transcribed_notes"][0]["midi"] == 60
    assert payload["transcribed_notes"][0]["pitch"] == "C4"
    assert payload["transcribed_notes"][0]["score_index"] == 0
    assert payload["note_mapping"] == [0]


def test_bridge_prefers_joint_checkpoint(tmp_path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"x")
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = tmp_path / "joint_decoder.pt"
    checkpoint.write_bytes(b"ckpt")
    sample = tmp_path / "sample"
    sample.mkdir()
    perf = sample / "performance_audio.wav"
    ref = sample / "reference_audio.wav"
    perf.write_bytes(b"wav")
    ref.write_bytes(b"wav")
    seen = {}

    def fake_run(command, **_kwargs):
        seen["command"] = [str(item) for item in command]
        output = Path(command[command.index("--out") + 1])
        output.write_text(
            json.dumps(
                {
                    "engine": "align-joint",
                    "events": [
                        {
                            "id": "aligned_00000",
                            "score_index": 0,
                            "is_rest": False,
                            "midi": 60,
                            "pitch": "C4",
                            "ref_start": 0.0,
                            "ref_end": 0.5,
                            "perf_start": 0.1,
                            "perf_end": 0.6,
                        }
                    ],
                    "labels": [],
                    "transcribed_notes": [
                        {"pitch": 60, "start": 0.1, "end": 0.6, "confidence": 0.9}
                    ],
                    "note_mapping": [0],
                    "summary": {
                        "engine": "align-joint",
                        "backend": "joint-path-crf",
                        "event_count": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    with patch("datacreate.align_bridge.subprocess.run", side_effect=fake_run):
        run_preferred_alignment(
            perf,
            ref,
            sample,
            _config(python, weights, checkpoint),
            logging.getLogger("test"),
        )
    assert "--checkpoint" in seen["command"]
    assert str(checkpoint) in seen["command"]
    payload = build_note_alignment(sample)
    assert payload["summary"]["engine"] == "align-joint"
    assert payload["summary"]["backend"] == "joint-path-crf"
    assert payload["transcribed_notes"][0]["score_index"] == 0


def test_transcription_dump_uses_configured_joint_candidate_version(tmp_path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"x")
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = tmp_path / "joint_decoder.pt"
    checkpoint.write_bytes(b"ckpt")
    sample = tmp_path / "sample"
    sample.mkdir()
    seen = {}

    def fake_run(command, **_kwargs):
        seen["command"] = [str(item) for item in command]
        output = Path(command[command.index("--out") + 1])
        output.write_text('{"transcribed_notes": []}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    with patch("datacreate.align_bridge.subprocess.run", side_effect=fake_run):
        dump_transcription(
            sample,
            _config(python, weights, checkpoint),
            logging.getLogger("test"),
        )
    assert "--transcribe-only" in seen["command"]
    assert "--checkpoint" in seen["command"]
    assert str(checkpoint) in seen["command"]


def test_manual_alignment_override_survives_model_rerun(tmp_path):
    (tmp_path / "note_alignment_overrides.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "overrides": [
                    {
                        "pitch": 90,
                        "performance_start": 7.4,
                        "score_index": 22,
                        "relationship": "copy",
                        "source_performance_start": 5.4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "transcribed_notes": [
            {"pitch": 90, "start": 5.4, "end": 5.8, "confidence": 0.9},
            {"pitch": 90, "start": 7.4, "end": 8.1, "confidence": 0.8},
        ],
        "note_mapping": [22, None],
        "events": [
            {
                "id": "aligned_0",
                "score_index": 22,
                "perf_start": 5.4,
                "perf_end": 5.8,
                "alignment_kind": "match",
            }
        ],
        "repetitions": [],
        "summary": {},
    }
    corrected = apply_alignment_overrides(payload, tmp_path)
    assert corrected["note_mapping"] == [22, 22]
    repeated = next(
        event
        for event in corrected["events"]
        if event["id"] == "manual_override_00001"
    )
    assert repeated["alignment_kind"] == "copy"
    assert repeated["is_repetition"] is True
    assert corrected["summary"]["manual_override_count"] == 1


def test_invalidation_removes_note_first_artifact(tmp_path):
    for name in ("alignment.npz", "note_alignment_v2.json", "candidates.json"):
        (tmp_path / name).write_bytes(b"x")
    _invalidate_alignment_artifacts(tmp_path, logging.getLogger("test"))
    assert not any(
        (tmp_path / name).exists()
        for name in ("alignment.npz", "note_alignment_v2.json", "candidates.json")
    )


@pytest.mark.parametrize('tag,revision', [('v9','v9-passage-v2'), ('v10','v10-boundary-v1')])
def test_v9_bridge_uses_serving_environment_and_never_legacy_checkpoint(tmp_path, tag, revision):
    candidate = tmp_path / "candidate.json"
    candidate.write_text("v9")
    sample = tmp_path / "095"
    sample.mkdir()
    (sample / "performance_audio.wav").write_bytes(b"audio")
    (sample / "verified_score.musicxml").write_bytes(b"score")
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["env"] = command, kwargs["env"]
        payload = {
            "engine": "align-joint",
            "label_generation": {"schema_version": "datacreate-model-feedback-v1"},
            "labels": [], "events": [],
            "provenance": {
                key: hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in {
                    "candidate_sha256": candidate,
                    "audio_sha256": sample / "performance_audio.wav",
                    "score_sha256": sample / "verified_score.musicxml",
                }.items()
            },
        }
        payload["provenance"]["pipeline_revision"] = revision
        (sample / "note_alignment_v2.json").write_text(json.dumps(payload))
        from datacreate.align_bridge import _compatibility_alignment
        _compatibility_alignment(payload, sample, config)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    config = PipelineConfig(
        paths={"note_alignment_candidate": str(candidate), "work_dir": str(tmp_path / "work"),
               "note_alignment_checkpoint": "unused-legacy.pt"},
        alignment={"model_version": f"stack-{tag}", "note_alignment_device": "cpu"},
    )
    with patch("datacreate.align_bridge.subprocess.run", side_effect=fake_run):
        result = run_preferred_alignment(
            sample / "performance_audio.wav", sample / "reference_audio.wav",
            sample, config, logging.getLogger("test"),
        )
    assert result.model_version == f"stack-{tag}"
    assert result.alignment_path.is_file()
    assert seen["command"][0] == sys.executable
    assert Path(seen["command"][1]).name == f"publish_datacreate_{tag}.py"
    assert "--sample" in seen["command"] and "--checkpoint" not in seen["command"]
    assert seen["env"]["PYTHONNOUSERSITE"] == "1"

    from datacreate.align_bridge import current_v9_feedback
    assert current_v9_feedback(sample, config)
    artifact = sample / "note_alignment_v2.json"
    payload = json.loads(artifact.read_text())
    payload["provenance"]["pipeline_revision"] = "v9-passage-v1"
    artifact.write_text(json.dumps(payload))
    assert not current_v9_feedback(sample, config)


def test_default_ui_configuration_selects_v9_in_serving_environment():
    config = PipelineConfig.load()
    assert config.alignment["model_version"] == "stack-v9"
    assert config.resolved_path("note_alignment_candidate").name == "CANDIDATE_STACK_V9.json"
    assert config.paths.get("note_alignment_python") is None
