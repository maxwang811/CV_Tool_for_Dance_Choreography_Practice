"""Reshape AIST++ ``*_cAll_*.pkl`` files into per-video keypoint pickles.

The raw AIST++ 2D keypoints ship as multi-camera files:

    data/keypoints2d/<genre>_<situation>_cAll_<dancer>_<music>_<chore>.pkl

with ``keypoints2d`` of shape ``(9 cameras, T, 17, 3)`` and ``det_scores``
of shape ``(9, T)``. The training pipeline
(``scripts/prepare_aist_training_data.py``) expects per-video files at
``data/labels/aistpp/keypoints2d_raw/<video_stem>.pkl`` whose ``keypoints2d``
is ``(T, 17, 2|3)`` (no camera axis).

This script walks ``data/raw_videos/``, parses the camera index from each
video stem (``c01`` -> slot 0, ``c02`` -> 1, ...), slices the matching
``cAll`` pickle, and writes per-stem pickles suitable for the ingest step.

Usage::

    python -m scripts.link_aist_keypoints \
        --src data/keypoints2d \
        --videos data/raw_videos \
        --out data/labels/aistpp/keypoints2d_raw
"""
from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")
_CAMERA_RE = re.compile(r"_c(?P<cam>\d+)_")


@dataclass
class LinkStats:
    written: int = 0
    skipped_no_source: int = 0
    skipped_bad_camera: int = 0
    skipped_bad_shape: int = 0


def _parse_camera_index(stem: str) -> Optional[int]:
    """Return 0-based camera index parsed from ``_c##_`` in ``stem`` or None."""
    m = _CAMERA_RE.search(stem)
    if not m:
        return None
    try:
        cam_num = int(m.group("cam"))
    except ValueError:
        return None
    if cam_num < 1:
        return None
    return cam_num - 1


def _source_pickle_for(src_dir: Path, video_stem: str) -> Path:
    """Map ``gBR_sBM_c01_...`` -> ``src_dir/gBR_sBM_cAll_....pkl``."""
    call_stem = _CAMERA_RE.sub("_cAll_", video_stem)
    return src_dir / f"{call_stem}.pkl"


def _iter_video_stems(videos_dir: Path) -> List[str]:
    stems = [
        p.stem
        for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    return sorted(stems)


def link_one(
    video_stem: str,
    src_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> str:
    """Link one video stem. Returns a status string."""
    cam_idx = _parse_camera_index(video_stem)
    if cam_idx is None:
        return "bad_camera"

    src = _source_pickle_for(src_dir, video_stem)
    if not src.exists():
        return "no_source"

    out_path = out_dir / f"{video_stem}.pkl"
    if out_path.exists() and not overwrite:
        return "written"

    with src.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict) or "keypoints2d" not in obj:
        return "bad_shape"

    kps = np.asarray(obj["keypoints2d"])
    if kps.ndim != 4 or kps.shape[2] != 17 or kps.shape[3] not in (2, 3):
        return "bad_shape"
    if cam_idx >= kps.shape[0]:
        return "bad_camera"

    per_stem = {"keypoints2d": kps[cam_idx].astype(np.float32)}

    det = obj.get("det_scores")
    if det is not None:
        det = np.asarray(det)
        if det.ndim == 2 and cam_idx < det.shape[0]:
            per_stem["det_scores"] = det[cam_idx].astype(np.float32)

    ts = obj.get("timestamps")
    if ts is not None:
        per_stem["timestamps"] = np.asarray(ts)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(per_stem, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out_path)
    return "written"


def link_all(
    src_dir: Path,
    videos_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> LinkStats:
    stats = LinkStats()
    stems = _iter_video_stems(videos_dir)
    if not stems:
        raise SystemExit(f"No videos found under {videos_dir}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems, 1):
        status = link_one(stem, src_dir, out_dir, overwrite=overwrite)
        if status == "written":
            stats.written += 1
        elif status == "no_source":
            stats.skipped_no_source += 1
        elif status == "bad_camera":
            stats.skipped_bad_camera += 1
        elif status == "bad_shape":
            stats.skipped_bad_shape += 1
        if i % 200 == 0 or i == len(stems):
            print(
                f"[{i}/{len(stems)}] written={stats.written} "
                f"no_source={stats.skipped_no_source} "
                f"bad_camera={stats.skipped_bad_camera} "
                f"bad_shape={stats.skipped_bad_shape}"
            )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reshape AIST++ cAll multi-camera 2D keypoint pickles into per-video "
            "pickles suitable for scripts/prepare_aist_training_data.py."
        )
    )
    p.add_argument("--src", default="data/keypoints2d", type=Path)
    p.add_argument("--videos", default="data/raw_videos", type=Path)
    p.add_argument(
        "--out",
        default="data/labels/aistpp/keypoints2d_raw",
        type=Path,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite per-stem pickles that already exist",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    stats = link_all(args.src, args.videos, args.out, overwrite=args.overwrite)
    print("")
    print(f"wrote              : {stats.written}")
    print(f"skipped no_source  : {stats.skipped_no_source}")
    print(f"skipped bad_camera : {stats.skipped_bad_camera}")
    print(f"skipped bad_shape  : {stats.skipped_bad_shape}")


if __name__ == "__main__":
    main()
