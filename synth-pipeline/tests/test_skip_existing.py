from pathlib import Path

import pytest

from synthpipeline.pipeline import _COMPLETE_FILES, complete_sample_ids, sample_is_complete


def _touch(path: Path, *names: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (path / name).write_text("ok", encoding="utf-8")


def test_complete_sample_ids_ignores_partial(tmp_path: Path) -> None:
    files = (
        "labels.json",
        "metadata.json",
        "performance_audio.wav",
        "reference_audio.wav",
        "verified_score.musicxml",
        "performance_score.musicxml",
        "note_map.json",
        "candidates.json",
        "alignment.npz",
        "performance_mel.npy",
        "reference_mel.npy",
        "performance_audio.mid",
        "reference_audio.mid",
    )
    _touch(tmp_path / "synth_Mozart_1209", *files)
    _touch(tmp_path / "synth_WeberITAV_11209", *files)
    _touch(tmp_path / "synth_001_1210", "labels.json", "performance_audio.wav")
    assert sample_is_complete(tmp_path / "synth_Mozart_1209")
    assert not sample_is_complete(tmp_path / "synth_001_1210")
    assert complete_sample_ids(tmp_path) == {1209, 11209}


@pytest.mark.parametrize("missing", _COMPLETE_FILES)
def test_missing_artifact_is_not_skipped(tmp_path: Path, missing: str) -> None:
    sample = tmp_path / "synth_gen_0042"
    _touch(sample, *_COMPLETE_FILES)
    (sample / missing).unlink()
    assert not sample_is_complete(sample)
    assert complete_sample_ids(tmp_path) == set()


def test_empty_artifact_is_not_complete(tmp_path: Path) -> None:
    sample = tmp_path / "synth_gen_0042"
    _touch(sample, *_COMPLETE_FILES)
    (sample / "alignment.npz").write_bytes(b"")
    assert not sample_is_complete(sample)
