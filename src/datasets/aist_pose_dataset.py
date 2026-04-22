"""AIST++ pose dataset (uses the shared internal JSONL schema)."""
from __future__ import annotations

from .coco_pose_dataset import PoseJsonlDataset


class AistPoseDataset(PoseJsonlDataset):
    pass
