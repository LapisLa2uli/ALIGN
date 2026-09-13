from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

from datacreate.align_bridge import run_preferred_alignment
from datacreate.config import PipelineConfig
from datacreate.note_alignment import build_note_alignment
from datacreate.sample_prep import _invalidate_alignment_artifacts


def _config(python: Path, weights: Path) -> PipelineConfig:
    return PipelineConfig(
        paths={
            "note_alignment_python": str(python),
            "note_alignment_weights": str(weights),
        },
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


def test_invalidation_removes_note_first_artifact(tmp_path):
    for name in ("alignment.npz", "note_alignment_v2.json", "candidates.json"):
        (tmp_path / name).write_bytes(b"x")
    _invalidate_alignment_artifacts(tmp_path, logging.getLogger("test"))
    assert not any(
        (tmp_path / name).exists()
        for name in ("alignment.npz", "note_alignment_v2.json", "candidates.json")
    )
