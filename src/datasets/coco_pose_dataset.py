"""Top-down pose dataset reading our internal JSONL format.

The ``coco_`` prefix is historical: this file now only powers AIST++
training (the supervised source allowed by
``docs/project_decisions.md`` section 6). The internal schema is defined
in ``src/datasets/common.py``; rows are produced by
``src/data/convert_aistpp.py`` via
``scripts/prepare_aist_training_data.py``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.datasets.common import (
    COCO_17_FLIP_PAIRS,
    NUM_JOINTS,
)
from src.utils.io import read_jsonl

# region agent log
# --- DEBUG INSTRUMENTATION (session 310455) -------------------------------
# Writes one NDJSON line to stderr (captured by SLURM .err on Oscar) and
# best-effort-appends to the local debug log. Safe to leave in; removed after
# verification.
import json as _dbg_json
import os as _dbg_os
import sys as _dbg_sys
import time as _dbg_time

_DBG_SESSION_ID_310455 = "310455"
_DBG_LOG_PATH_310455 = (
    "/Users/mohanwang/Desktop/Projects/CV_Tool_for_Dance_Choreography_Practice/"
    ".cursor/debug-310455.log"
)
_DBG_CONFIG_LOGGED_310455 = False


def _debug_log_310455(location: str, message: str, data: dict, hypothesis_id: str = "") -> None:
    payload = {
        "sessionId": _DBG_SESSION_ID_310455,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "pid": _dbg_os.getpid(),
        "timestamp": int(_dbg_time.time() * 1000),
    }
    line = "[DEBUG_310455] " + _dbg_json.dumps(payload, default=str)
    try:
        print(line, file=_dbg_sys.stderr, flush=True)
    except Exception:
        pass
    try:
        with open(_DBG_LOG_PATH_310455, "a") as _fh:
            _fh.write(_dbg_json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
# endregion


# ---------------------------------------------------------------------------
# Affine transforms (top-down convention, bbox -> fixed input).
# ---------------------------------------------------------------------------


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y = pt
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot_deg: float,
    output_size: Tuple[int, int],          # (H, W)
    shift: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
    pixel_std: float = 200.0,
    inv: bool = False,
) -> np.ndarray:
    """Compute the 2x3 affine mapping pixel space (input image) <-> output crop.

    This is the standard Simple-Baselines / HRNet formulation, implemented
    from scratch here (no external pretrained code imported).
    """
    out_h, out_w = output_size
    src_w = scale[0] * pixel_std
    src_dir = _rotate_point(np.array([0.0, -src_w * 0.5], dtype=np.float32), math.radians(rot_deg))
    dst_dir = np.array([0.0, -out_w * 0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center + scale * pixel_std * shift
    src[1] = center + src_dir + scale * pixel_std * shift
    src[2] = _third_point(src[0], src[1])
    dst[0] = np.array([out_w * 0.5, out_h * 0.5], dtype=np.float32)
    dst[1] = dst[0] + dst_dir
    dst[2] = _third_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def affine_transform_point(pt: np.ndarray, M: np.ndarray) -> np.ndarray:
    ph = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    return (M @ ph).astype(np.float32)


# ---------------------------------------------------------------------------
# Heatmap target generation.
# ---------------------------------------------------------------------------


def gaussian_heatmap(
    joints_xy: np.ndarray,       # (J, 2) in output-space coords
    target_visibility: np.ndarray,  # (J,) 0/1 weights
    heatmap_size: Tuple[int, int],  # (H, W)
    sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (heatmaps (J,H,W), target_weights (J,)).

    A joint placed outside the heatmap yields a zero map and zero weight.
    """
    H, W = heatmap_size
    J = joints_xy.shape[0]
    heatmaps = np.zeros((J, H, W), dtype=np.float32)
    weights = target_visibility.astype(np.float32).copy()
    tmp_size = sigma * 3
    for j in range(J):
        if weights[j] <= 0:
            continue
        mu_x = joints_xy[j, 0]
        mu_y = joints_xy[j, 1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
            weights[j] = 0
            continue
        size = 2 * int(tmp_size) + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma * sigma))
        g_x_lo = max(0, -ul[0]); g_x_hi = min(br[0], W) - ul[0]
        g_y_lo = max(0, -ul[1]); g_y_hi = min(br[1], H) - ul[1]
        img_x_lo = max(0, ul[0]); img_x_hi = min(br[0], W)
        img_y_lo = max(0, ul[1]); img_y_hi = min(br[1], H)
        # region agent log
        _src_shape = (g_y_hi - g_y_lo, g_x_hi - g_x_lo)
        _dst_shape = (img_y_hi - img_y_lo, img_x_hi - img_x_lo)
        _g_clipped_src = g[g_y_lo:g_y_hi, g_x_lo:g_x_hi].shape
        if (
            _src_shape != _dst_shape
            or _g_clipped_src != _dst_shape
        ):
            _debug_log_310455(
                location="coco_pose_dataset.py:121",
                message="heatmap slice shape mismatch about to crash",
                data={
                    "j": int(j),
                    "mu_x": float(mu_x),
                    "mu_y": float(mu_y),
                    "ul": [int(ul[0]), int(ul[1])],
                    "br": [int(br[0]), int(br[1])],
                    "H": int(H), "W": int(W),
                    "sigma": float(sigma),
                    "tmp_size": float(tmp_size),
                    "size": int(size),
                    "g_shape": list(g.shape),
                    "g_x_lo": int(g_x_lo), "g_x_hi": int(g_x_hi),
                    "g_y_lo": int(g_y_lo), "g_y_hi": int(g_y_hi),
                    "img_x_lo": int(img_x_lo), "img_x_hi": int(img_x_hi),
                    "img_y_lo": int(img_y_lo), "img_y_hi": int(img_y_hi),
                    "computed_src_shape": list(_src_shape),
                    "actual_src_shape": list(_g_clipped_src),
                    "dst_shape": list(_dst_shape),
                    "joints_xy_row": [float(joints_xy[j, 0]), float(joints_xy[j, 1])],
                    "target_visibility_j": float(target_visibility[j]),
                },
                hypothesis_id="A_B_C_D_E",
            )
        # endregion
        heatmaps[j, img_y_lo:img_y_hi, img_x_lo:img_x_hi] = g[g_y_lo:g_y_hi, g_x_lo:g_x_hi]
    return heatmaps, weights


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------


@dataclass
class PoseAugConfig:
    random_scale: Tuple[float, float] = (0.75, 1.25)
    random_rotation_deg: float = 40.0
    horizontal_flip_prob: float = 0.5
    translation_jitter_px: float = 20.0
    color_jitter_brightness: float = 0.25
    color_jitter_contrast: float = 0.25
    gaussian_blur_prob: float = 0.2
    motion_blur_prob: float = 0.1
    jpeg_quality: Tuple[int, int] = (55, 95)
    jpeg_prob: float = 0.3
    random_cutout_prob: float = 0.25

    @classmethod
    def from_yaml(cls, d: Dict) -> "PoseAugConfig":
        rs = d.get("random_scale", (0.75, 1.25))
        cj = d.get("color_jitter", {})
        return cls(
            random_scale=tuple(rs),
            random_rotation_deg=float(d.get("random_rotation_deg", 40)),
            horizontal_flip_prob=float(d.get("horizontal_flip_prob", 0.5)),
            translation_jitter_px=float(d.get("translation_jitter_px", 20)),
            color_jitter_brightness=float(cj.get("brightness", 0.25)),
            color_jitter_contrast=float(cj.get("contrast", 0.25)),
            gaussian_blur_prob=float(d.get("gaussian_blur_prob", 0.2)),
            motion_blur_prob=float(d.get("motion_blur_prob", 0.1)),
            jpeg_quality=tuple(d.get("jpeg_quality", (55, 95))),
            jpeg_prob=float(d.get("jpeg_prob", 0.3)),
            random_cutout_prob=float(d.get("random_cutout_prob", 0.25)),
        )


class PoseJsonlDataset:
    """Map-style pose dataset that reads an internal-schema JSONL file.

    This is deliberately PyTorch-agnostic; we expose ``__len__`` + ``__getitem__``
    that return numpy arrays. ``src/train/engine.py`` wraps this in a
    ``torch.utils.data.Dataset`` via a thin adapter.
    """

    def __init__(
        self,
        annotations_jsonl: str | Path,
        *,
        input_size: Tuple[int, int] = (256, 192),
        heatmap_size: Tuple[int, int] = (64, 48),
        sigma: float = 2.0,
        is_train: bool = True,
        aug: Optional[PoseAugConfig] = None,
        max_items: Optional[int] = None,
    ) -> None:
        self.records: List[Dict] = list(read_jsonl(annotations_jsonl))
        if max_items is not None:
            self.records = self.records[: int(max_items)]
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = float(sigma)
        self.is_train = is_train
        self.aug = aug or PoseAugConfig()
        self.pixel_std = 200.0

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    # augmentations
    # ------------------------------------------------------------------
    def _maybe_flip(self, img: np.ndarray, kps: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_train or random.random() > self.aug.horizontal_flip_prob:
            return img, kps, center
        img = img[:, ::-1, :].copy()
        w = img.shape[1]
        kps = kps.copy()
        kps[:, 0] = (w - 1) - kps[:, 0]
        for a, b in COCO_17_FLIP_PAIRS:
            kps[[a, b], :] = kps[[b, a], :]
        center = center.copy()
        center[0] = (w - 1) - center[0]
        return img, kps, center

    def _random_scale(self) -> float:
        if not self.is_train:
            return 1.0
        lo, hi = self.aug.random_scale
        return float(random.uniform(lo, hi))

    def _random_rotation(self) -> float:
        if not self.is_train:
            return 0.0
        r = self.aug.random_rotation_deg
        return float(random.uniform(-r, r)) if random.random() < 0.6 else 0.0

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return img
        out = img.astype(np.float32)
        if self.aug.color_jitter_brightness > 0:
            b = random.uniform(1 - self.aug.color_jitter_brightness, 1 + self.aug.color_jitter_brightness)
            out = out * b
        if self.aug.color_jitter_contrast > 0:
            c = random.uniform(1 - self.aug.color_jitter_contrast, 1 + self.aug.color_jitter_contrast)
            mean = out.mean()
            out = (out - mean) * c + mean
        return np.clip(out, 0, 255).astype(np.uint8)

    def _maybe_blur(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return img
        if random.random() < self.aug.gaussian_blur_prob:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < self.aug.motion_blur_prob:
            k = random.choice([5, 7, 9])
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0
            kernel /= k
            img = cv2.filter2D(img, -1, kernel)
        return img

    def _maybe_jpeg(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train or random.random() > self.aug.jpeg_prob:
            return img
        q = random.randint(self.aug.jpeg_quality[0], self.aug.jpeg_quality[1])
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def _maybe_cutout(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train or random.random() > self.aug.random_cutout_prob:
            return img
        h, w = img.shape[:2]
        ch = random.randint(h // 8, h // 4)
        cw = random.randint(w // 8, w // 4)
        y = random.randint(0, h - ch)
        x = random.randint(0, w - cw)
        img = img.copy()
        img[y : y + ch, x : x + cw] = random.randint(0, 255)
        return img

    # ------------------------------------------------------------------
    # main item
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            # synthesize a neutral gray image so training doesn't crash on missing files
            img = np.full((480, 640, 3), 128, dtype=np.uint8)
        kps = np.asarray(rec["keypoints_xyv"], dtype=np.float32)       # (17, 3)
        center = np.asarray(rec["center"], dtype=np.float32)
        scale = np.asarray(rec["scale"], dtype=np.float32)

        img, kps, center = self._maybe_flip(img, kps, center)

        scale_mul = self._random_scale()
        rot_deg = self._random_rotation()
        scale_work = scale * scale_mul

        if self.is_train and self.aug.translation_jitter_px > 0:
            center = center + np.random.uniform(-1, 1, size=2).astype(np.float32) * self.aug.translation_jitter_px

        H, W = self.input_size
        M = get_affine_transform(center, scale_work, rot_deg, (H, W), pixel_std=self.pixel_std)
        crop = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)

        crop = self._maybe_blur(crop)
        crop = self._color_jitter(crop)
        crop = self._maybe_jpeg(crop)
        crop = self._maybe_cutout(crop)

        # Map keypoints into heatmap space.
        hm_h, hm_w = self.heatmap_size
        fx, fy = hm_w / W, hm_h / H
        kps_hm = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
        vis = np.zeros((NUM_JOINTS,), dtype=np.float32)
        for j in range(NUM_JOINTS):
            if kps[j, 2] <= 0:
                continue
            xy = affine_transform_point(kps[j, :2], M)
            if 0 <= xy[0] < W and 0 <= xy[1] < H:
                kps_hm[j, 0] = xy[0] * fx
                kps_hm[j, 1] = xy[1] * fy
                vis[j] = 1.0

        # region agent log
        global _DBG_CONFIG_LOGGED_310455
        if not _DBG_CONFIG_LOGGED_310455:
            _DBG_CONFIG_LOGGED_310455 = True
            _debug_log_310455(
                location="coco_pose_dataset.py:316 __getitem__",
                message="dataset config snapshot (first __getitem__ per worker)",
                data={
                    "self.sigma": float(self.sigma),
                    "self.heatmap_size": list(self.heatmap_size),
                    "self.input_size": list(self.input_size),
                    "hm_h": int(hm_h), "hm_w": int(hm_w),
                    "fx": float(fx), "fy": float(fy),
                    "is_train": bool(self.is_train),
                    "num_records": int(len(self.records)),
                },
                hypothesis_id="A_D",
            )
        try:
            heatmaps, weights = gaussian_heatmap(kps_hm, vis, self.heatmap_size, sigma=self.sigma)
        except ValueError:
            _debug_log_310455(
                location="coco_pose_dataset.py:316 __getitem__",
                message="gaussian_heatmap raised ValueError; dumping sample context",
                data={
                    "image_path": str(rec.get("image_path")),
                    "idx": int(idx),
                    "kps_hm": kps_hm.tolist(),
                    "vis": vis.tolist(),
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                    "scale_mul": float(scale_mul),
                    "rot_deg": float(rot_deg),
                    "heatmap_size": list(self.heatmap_size),
                    "input_size": list(self.input_size),
                    "sigma": float(self.sigma),
                },
                hypothesis_id="A_B_C_D_E",
            )
            raise
        # endregion

        # Normalize image to [0, 1] float with channel-first.
        img_chw = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)

        return {
            "image": img_chw,                                # (3, H, W)
            "heatmaps": heatmaps,                            # (J, Hh, Hw)
            "target_weight": weights.reshape(-1, 1),         # (J, 1)
            "kps_hm": kps_hm,                                # (J, 2) in heatmap coords
            "kps_original": kps,                             # (J, 3) in original image pixels
            "center": center,
            "scale": scale_work,
            "rotation_deg": np.float32(rot_deg),
            "image_path": rec["image_path"],
            "image_id": rec["image_id"],
            "dataset_name": rec["dataset_name"],
        }
