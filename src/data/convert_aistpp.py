"""Phase 5.3: convert AIST++ into 2D COCO-17 annotations.

AIST++ ships per-video, multi-camera calibrations plus 3D keypoints (SMPL-based)
and per-frame 2D keypoint projections. We:

  * iterate per-camera, per-video, per-frame (with ``frame_stride``)
  * read the 2D keypoints AIST++ already ships, OR project 3D -> 2D if only
    3D is available
  * re-order joints to COCO-17
  * keep only frames where a reasonable number of joints project inside the
    image frame
  * emit one internal JSONL row per (video, camera, frame)

Because AIST++ provides its own 17-keypoint format that mostly matches COCO,
we use an explicit mapping table; a user who ships a different release can
override the mapping on the CLI.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from src.datasets.common import AnnotationRecord, NUM_JOINTS, bbox_to_center_scale
from src.utils.io import write_jsonl


# Identity mapping (AIST++ default keypoint ordering matches COCO-17).
_DEFAULT_AISTPP_TO_COCO17 = tuple(range(17))


def _bbox_from_keypoints(xy: np.ndarray, margin: float = 0.15) -> List[float]:
    x1, y1 = float(np.min(xy[:, 0])), float(np.min(xy[:, 1]))
    x2, y2 = float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))
    w, h = x2 - x1, y2 - y1
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin
    return [x1, y1, x2, y2]


def _project_3d_to_2d(xyz: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    cam = xyz @ R.T + t
    z = np.clip(cam[:, 2], 1e-6, None)
    uv = np.stack([cam[:, 0] / z, cam[:, 1] / z], axis=-1)
    uv_h = np.concatenate([uv, np.ones_like(uv[:, :1])], axis=-1)
    pix = uv_h @ K.T
    return pix[:, :2]


def convert_video(
    video_path: str | Path,
    frames_dir: str | Path,
    kps_file: str | Path,
    out_rows: List[dict],
    aspect_ratio: float,
    frame_stride: int = 8,
    mapping: tuple = _DEFAULT_AISTPP_TO_COCO17,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> None:
    """Append rows for one (video, camera) pair to ``out_rows``.

    ``kps_file`` may be:
      * a ``.pkl`` with key ``keypoints2d`` of shape (T, 17, 2) or (T, 17, 3)
      * a numpy ``.npy`` of the same shape
    """
    kps_file = Path(kps_file)
    if kps_file.suffix == ".npy":
        kps = np.load(kps_file)
    else:
        with kps_file.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            kps = obj.get("keypoints2d")
            if kps is None:
                # project from 3D if available
                kpts3d = obj["keypoints3d"]
                cam = obj["camera"]
                kps = np.stack(
                    [_project_3d_to_2d(kpts3d[t], cam["K"], cam["R"], cam["t"]) for t in range(len(kpts3d))],
                    axis=0,
                )
        else:
            kps = np.asarray(obj)

    kps = np.asarray(kps)
    if kps.ndim != 3 or kps.shape[1] != 17 or kps.shape[2] not in (2, 3):
        raise ValueError(f"Unexpected AIST++ keypoints shape: {kps.shape}")

    T = kps.shape[0]
    frames_dir = Path(frames_dir)
    has_conf = kps.shape[2] == 3
    for t in range(0, T, max(1, int(frame_stride))):
        kp_t = kps[t]
        # reorder to COCO-17
        kp_t = kp_t[list(mapping)]
        xy = kp_t[:, :2].astype(np.float32)
        if image_width and image_height:
            inside = (
                (xy[:, 0] >= 0) & (xy[:, 0] < image_width)
                & (xy[:, 1] >= 0) & (xy[:, 1] < image_height)
            )
        else:
            inside = np.ones(17, dtype=bool)
        if has_conf:
            # AIST++ ships per-joint detector confidence as the third channel.
            # Treat joints below 0.2 as unannotated (v=0) even if they fall
            # inside the frame.
            conf = kp_t[:, 2].astype(np.float32)
            visible = inside & (conf > 0.2)
        else:
            visible = inside
        v = np.where(visible, 2, 0).astype(np.float32)
        if int((v > 0).sum()) < 8:
            continue
        kps_xyv = np.concatenate([xy, v[:, None]], axis=-1).tolist()

        bbox = _bbox_from_keypoints(xy[inside] if inside.any() else xy)
        center, scale = bbox_to_center_scale(bbox, aspect_ratio=aspect_ratio)
        frame_path = frames_dir / f"{Path(video_path).stem}_{t:06d}.jpg"
        rec = AnnotationRecord(
            image_path=str(frame_path),
            image_id=f"aistpp_{Path(video_path).stem}_{t:06d}",
            dataset_name="aistpp",
            bbox_xyxy=[float(v) for v in bbox],
            keypoints_xyv=kps_xyv,
            center=center,
            scale=scale,
            meta={"video": str(video_path), "frame_index": int(t)},
        )
        rec.validate()
        out_rows.append(rec.__dict__)


def _main() -> None:
    p = argparse.ArgumentParser(
        description="Convert AIST++ 2D keypoints (one video at a time) into internal JSONL."
    )
    p.add_argument("--video", required=True)
    p.add_argument("--frames-dir", required=True, help="where the extracted frames live")
    p.add_argument("--kps-file", required=True, help=".pkl or .npy with (T, 17, 2|3) array")
    p.add_argument("--out", required=True)
    p.add_argument("--frame-stride", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--input-size", nargs=2, type=int, default=[256, 192])
    args = p.parse_args()
    H, W = args.input_size
    rows: List[dict] = []
    convert_video(
        args.video,
        args.frames_dir,
        args.kps_file,
        rows,
        aspect_ratio=W / H,
        frame_stride=args.frame_stride,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    def _gen() -> Iterable[dict]:
        yield from rows

    n = write_jsonl(args.out, _gen())
    print(f"Wrote {n} AIST++ annotations -> {args.out}")


if __name__ == "__main__":
    _main()
