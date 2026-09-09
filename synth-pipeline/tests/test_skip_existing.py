from pathlib import Path

from synthpipeline.pipeline import complete_sample_ids, sample_is_complete


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
    )
    _touch(tmp_path / "synth_Mozart_1209", *files)
    _touch(tmp_path / "synth_WeberITAV_11209", *files)
    _touch(tmp_path / "synth_001_1210", "labels.json", "performance_audio.wav")
    assert sample_is_complete(tmp_path / "synth_Mozart_1209")
    assert not sample_is_complete(tmp_path / "synth_001_1210")
    assert complete_sample_ids(tmp_path) == {1209, 11209}
