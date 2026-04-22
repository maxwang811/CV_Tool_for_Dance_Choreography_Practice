"""Prepare supervised pose-training data from ``data/raw_videos/``.

For each MP4 in ``data/raw_videos/``:

  1. Extract frames (stride configurable) into
     ``data/raw_frames/aistpp/<video_stem>/``.
  2. Load the matching AIST++ 2D keypoints file from
     ``data/labels/aistpp/keypoints2d_raw/<video_stem>.{pkl,npy}`` and
     convert it to our internal JSONL schema via
     :func:`src.data.convert_aistpp.convert_video`.
  3. Deterministically assign each video to train (90%) or val (10%) by
     hashing the stem, and emit:

        data/labels/aistpp/internal_train.jsonl
        data/labels/aistpp/internal_val.jsonl

Videos without a matching keypoints file are skipped with a warning.

This is the ONLY supervised-label ingest path (see
``docs/project_decisions.md`` sections 1 and 6): no pretrained models,
no other datasets.

Usage::

    python -m scripts.prepare_aist_training_data \
        --raw-videos data/raw_videos \
        --keypoints-dir data/labels/aistpp/keypoints2d_raw \
        --frames-dir data/raw_frames/aistpp \
        --out-dir data/labels/aistpp \
        --frame-stride 8
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from src.data.convert_aistpp import convert_video
from src.data.extract_frames import extract
from src.utils.io import write_jsonl
from src.utils.video import ffprobe_meta


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")
KEYPOINT_EXTS = (".pkl", ".npy")


def _find_keypoints_file(keypoints_dir: Path, stem: str) -> Optional[Path]:
    for ext in KEYPOINT_EXTS:
        candidate = keypoints_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _is_val(stem: str, val_fraction: float, seed: str) -> bool:
    h = hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return bucket < val_fraction


def _iter_videos(raw_videos: Path) -> List[Path]:
    videos = sorted(p for p in raw_videos.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    return videos


def prepare(
    raw_videos: Path,
    keypoints_dir: Path,
    frames_dir: Path,
    out_dir: Path,
    frame_stride: int = 8,
    input_size: Tuple[int, int] = (256, 192),
    val_fraction: float = 0.1,
    split_seed: str = "aist-split-v1",
    extract_frames: bool = True,
) -> Tuple[int, int, int]:
    """Run the full prepare pipeline. Returns ``(n_train, n_val, n_skipped)``."""
    raw_videos = Path(raw_videos)
    keypoints_dir = Path(keypoints_dir)
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    H, W = input_size
    aspect_ratio = W / H

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    skipped: List[str] = []

    videos = _iter_videos(raw_videos)
    if not videos:
        raise SystemExit(f"No videos found under {raw_videos}.")

    for i, video_path in enumerate(videos, 1):
        stem = video_path.stem
        kps_file = _find_keypoints_file(keypoints_dir, stem)
        if kps_file is None:
            skipped.append(stem)
            print(f"[{i}/{len(videos)}] SKIP {stem}: no keypoints file in {keypoints_dir}")
            continue

        meta = ffprobe_meta(video_path)
        if not meta.ok:
            skipped.append(stem)
            print(f"[{i}/{len(videos)}] SKIP {stem}: ffprobe failed ({meta.error})")
            continue

        video_frames_dir = frames_dir / stem
        if extract_frames:
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            n_extracted = extract(video_path, video_frames_dir, stride=frame_stride, prefix=stem)
            print(f"[{i}/{len(videos)}] {stem}: extracted {n_extracted} frames -> {video_frames_dir}")
        else:
            print(f"[{i}/{len(videos)}] {stem}: skipping frame extraction (assuming already on disk)")

        bucket_rows: List[dict] = []
        convert_video(
            video_path=video_path,
            frames_dir=video_frames_dir,
            kps_file=kps_file,
            out_rows=bucket_rows,
            aspect_ratio=aspect_ratio,
            frame_stride=frame_stride,
            image_width=meta.width or None,
            image_height=meta.height or None,
        )

        target = val_rows if _is_val(stem, val_fraction, split_seed) else train_rows
        target.extend(bucket_rows)
        print(f"    -> {len(bucket_rows)} annotations")

    train_path = out_dir / "internal_train.jsonl"
    val_path = out_dir / "internal_val.jsonl"
    n_train = write_jsonl(train_path, train_rows)
    n_val = write_jsonl(val_path, val_rows)
    print("")
    print(f"Wrote {n_train} train rows -> {train_path}")
    print(f"Wrote {n_val} val rows   -> {val_path}")
    if skipped:
        print(f"Skipped {len(skipped)} videos without keypoints: {skipped[:5]}"
              + (" ..." if len(skipped) > 5 else ""))
    return n_train, n_val, len(skipped)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract frames from data/raw_videos/ and convert matching AIST++ 2D "
            "keypoints into internal JSONL train/val splits. See "
            "docs/project_decisions.md sections 1 and 6."
        )
    )
    p.add_argument("--raw-videos", default="data/raw_videos", type=Path)
    p.add_argument(
        "--keypoints-dir",
        default="data/labels/aistpp/keypoints2d_raw",
        type=Path,
        help="directory with per-video AIST++ 2D keypoints files (.pkl or .npy)",
    )
    p.add_argument("--frames-dir", default="data/raw_frames/aistpp", type=Path)
    p.add_argument("--out-dir", default="data/labels/aistpp", type=Path)
    p.add_argument("--frame-stride", type=int, default=8)
    p.add_argument("--input-size", nargs=2, type=int, default=[256, 192], metavar=("H", "W"))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", default="aist-split-v1")
    p.add_argument(
        "--no-extract-frames",
        action="store_true",
        help="skip frame extraction (assume frames already live under --frames-dir/<stem>/)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    prepare(
        raw_videos=args.raw_videos,
        keypoints_dir=args.keypoints_dir,
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
        frame_stride=args.frame_stride,
        input_size=tuple(args.input_size),
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        extract_frames=not args.no_extract_frames,
    )


if __name__ == "__main__":
    main()
