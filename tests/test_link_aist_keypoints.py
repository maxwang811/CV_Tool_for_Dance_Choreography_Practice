"""Tests for ``scripts.link_aist_keypoints``.

We fabricate a fake AIST++ ``cAll`` pickle and a matching stub video file,
then confirm the linker writes a per-stem pickle whose ``keypoints2d`` is
``(T, 17, 3)`` and equal to the source's camera-0 slice.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from scripts.link_aist_keypoints import (
    _parse_camera_index,
    _source_pickle_for,
    link_all,
    link_one,
)


def _make_fake_call_pickle(path: Path, T: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    kps = rng.standard_normal((9, T, 17, 3)).astype(np.float32)
    det = rng.standard_normal((9, T)).astype(np.float32)
    ts = np.arange(T, dtype=np.int64)
    obj = {"keypoints2d": kps, "det_scores": det, "timestamps": ts}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return kps


def test_parse_camera_index():
    assert _parse_camera_index("gBR_sBM_c01_d04_mBR0_ch01") == 0
    assert _parse_camera_index("gBR_sBM_c09_d04_mBR0_ch01") == 8
    assert _parse_camera_index("no_camera_token") is None


def test_source_pickle_mapping(tmp_path: Path):
    src = tmp_path / "kp"
    expected = src / "gBR_sBM_cAll_d04_mBR0_ch01.pkl"
    got = _source_pickle_for(src, "gBR_sBM_c01_d04_mBR0_ch01")
    assert got == expected


def test_link_one_slices_camera_zero(tmp_path: Path):
    src_dir = tmp_path / "kp"
    out_dir = tmp_path / "out"
    stem = "gBR_sBM_c01_d04_mBR0_ch01"
    kps_all = _make_fake_call_pickle(src_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl")

    status = link_one(stem, src_dir, out_dir)
    assert status == "written"

    with (out_dir / f"{stem}.pkl").open("rb") as f:
        got = pickle.load(f)
    assert got["keypoints2d"].shape == (10, 17, 3)
    np.testing.assert_array_equal(got["keypoints2d"], kps_all[0])
    assert got["det_scores"].shape == (10,)
    assert got["timestamps"].shape == (10,)


def test_link_one_slices_other_camera(tmp_path: Path):
    src_dir = tmp_path / "kp"
    out_dir = tmp_path / "out"
    stem = "gBR_sBM_c05_d04_mBR0_ch01"
    kps_all = _make_fake_call_pickle(src_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl")

    assert link_one(stem, src_dir, out_dir) == "written"
    with (out_dir / f"{stem}.pkl").open("rb") as f:
        got = pickle.load(f)
    np.testing.assert_array_equal(got["keypoints2d"], kps_all[4])


def test_link_one_missing_source(tmp_path: Path):
    src_dir = tmp_path / "kp"
    src_dir.mkdir()
    out_dir = tmp_path / "out"
    assert link_one("gBR_sBM_c01_d04_mBR0_ch99", src_dir, out_dir) == "no_source"
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_link_all_counts(tmp_path: Path):
    videos_dir = tmp_path / "videos"
    src_dir = tmp_path / "kp"
    out_dir = tmp_path / "out"
    videos_dir.mkdir()

    stems_with_source = [
        "gBR_sBM_c01_d04_mBR0_ch01",
        "gBR_sBM_c01_d04_mBR0_ch02",
    ]
    for s in stems_with_source:
        (videos_dir / f"{s}.mp4").write_bytes(b"")
    missing_stem = "gBR_sBM_c01_d99_mBR0_ch99"
    (videos_dir / f"{missing_stem}.mp4").write_bytes(b"")

    _make_fake_call_pickle(src_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl")
    _make_fake_call_pickle(src_dir / "gBR_sBM_cAll_d04_mBR0_ch02.pkl")

    stats = link_all(src_dir, videos_dir, out_dir)
    assert stats.written == 2
    assert stats.skipped_no_source == 1
    for s in stems_with_source:
        assert (out_dir / f"{s}.pkl").exists()
