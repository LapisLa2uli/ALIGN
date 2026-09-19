from pathlib import Path
import zipfile

import numpy as np
import pytest

from datacreate.alignment_storage import alignment_storage_kind, omit_dtw_cost, save_alignment


def _arrays():
    return dict(
        ref_features=np.arange(18, dtype=np.float32).reshape(3, 6),
        perf_features=np.asfortranarray(np.ones((3, 8), dtype=np.float64)),
        warping_path=np.asarray([[0, 0], [0, 1], [1, 2], [5, 7]], dtype=np.int32),
        dtw_cost=np.full((6, 8), np.inf),
        frame_residuals=np.asarray([0, np.nan, 2], dtype=np.float64),
        hop_length=np.int64(512), sample_rate=np.int64(22050),
        silence_frames=np.asarray([0, 6, 0, 8], dtype=np.int32),
    )


def test_conversion_preserves_every_other_member_and_is_idempotent(tmp_path):
    path = tmp_path / "alignment.npz"
    np.savez(path, **_arrays(), extension=np.asarray([7], dtype=np.uint8))
    with zipfile.ZipFile(path) as archive:
        original = {name: archive.read(name) for name in archive.namelist() if name != "dtw_cost.npy"}
    row = omit_dtw_cost(path)
    assert row["changed"]
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == set(original) | {"dtw_cost_omitted.npy"}
        assert all(archive.read(name) == data for name, data in original.items())
    saved = path.read_bytes()
    assert not omit_dtw_cost(path)["changed"]
    assert path.read_bytes() == saved
    with np.load(path) as archive:
        assert alignment_storage_kind(archive) == "compact"


@pytest.mark.parametrize("keep", [True, False])
def test_new_archives_record_storage_policy(tmp_path, keep):
    path = tmp_path / "alignment.npz"
    save_alignment(path, save_dtw_cost=keep, **_arrays())
    with np.load(path) as archive:
        assert alignment_storage_kind(archive) == ("full" if keep else "compact")
        assert ("dtw_cost" in archive.files) == keep


def test_missing_matrix_requires_explicit_marker(tmp_path):
    path = tmp_path / "broken.npz"
    arrays = _arrays()
    arrays.pop("dtw_cost")
    np.savez(path, **arrays)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="explicit omission"):
        omit_dtw_cost(path)
    assert path.read_bytes() == original


def test_failed_verification_leaves_original_in_place(tmp_path, monkeypatch):
    path = tmp_path / "alignment.npz"
    np.savez(path, **_arrays())
    original = path.read_bytes()
    monkeypatch.setattr("datacreate.alignment_storage._digest_member", lambda *args: "mismatch")
    with pytest.raises(ValueError, match="Retained array changed"):
        omit_dtw_cost(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
