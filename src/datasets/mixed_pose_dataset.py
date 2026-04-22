"""Mixed-dataset sampler.

After the AIST-only refactor (see docs/project_decisions.md section 6) the
only supervised training source is AIST++ labels paired with frames from
``data/raw_videos/``; ``custom_dance_val`` may be present for evaluation.
This class stays generic enough to support additional sources in the
future, but :func:`src.train.train_pose._require_aistpp_only` rejects any
training config whose ``dataset_mix`` enables anything other than
``aistpp``.
"""
from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .coco_pose_dataset import PoseAugConfig, PoseJsonlDataset


@dataclass
class _Source:
    name: str
    dataset: PoseJsonlDataset
    weight: float


class MixedPoseDataset:
    """Infinite-style mixed dataset.

    ``__len__`` returns a virtual epoch size (configurable); ``__getitem__``
    samples a dataset proportional to its weight and then samples an item
    from that dataset at random. This way the mix ratios are respected
    regardless of dataset sizes.
    """

    def __init__(
        self,
        sources: Sequence[Tuple[str, PoseJsonlDataset, float]],
        *,
        epoch_size: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        srcs: List[_Source] = []
        total = 0.0
        for name, ds, w in sources:
            if w <= 0 or len(ds) == 0:
                continue
            srcs.append(_Source(name=name, dataset=ds, weight=float(w)))
            total += float(w)
        if not srcs:
            raise ValueError("MixedPoseDataset got no non-empty sources")
        for s in srcs:
            s.weight /= total
        self._sources = srcs
        # cumulative weights for bisect sampling
        self._cum_weights: List[float] = []
        acc = 0.0
        for s in srcs:
            acc += s.weight
            self._cum_weights.append(acc)
        self._epoch_size = epoch_size or sum(len(s.dataset) for s in srcs)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return int(self._epoch_size)

    def __getitem__(self, idx: int) -> Dict:
        # Deterministic per-index seeding so workers produce reproducible batches.
        rng = random.Random(hash(("mixed", idx, self._rng.random())))
        r = rng.random()
        src_idx = bisect.bisect_right(self._cum_weights, r)
        src_idx = min(src_idx, len(self._sources) - 1)
        src = self._sources[src_idx]
        inner = rng.randrange(len(src.dataset))
        item = src.dataset[inner]
        item["source_name"] = src.name
        return item

    @property
    def sources(self) -> List[_Source]:
        return self._sources


def build_mixed_from_configs(
    data_cfg: Dict,
    mix: Dict[str, float],
    *,
    is_train: bool,
    epoch_size: Optional[int] = None,
    seed: int = 0,
    max_items_per_source: Optional[int] = None,
) -> MixedPoseDataset:
    """Instantiate a ``MixedPoseDataset`` from config dicts.

    ``data_cfg`` is the YAML from ``configs/data/pose_train.yaml``.
    ``mix`` is a dict ``{"coco": 0.7, "crowdpose": 0.3, ...}``.
    """
    input_size = tuple(data_cfg.get("input_size", (256, 192)))
    heatmap_size = tuple(data_cfg.get("heatmap_size", (64, 48)))
    sigma = float(data_cfg.get("sigma", 2.0))
    aug = PoseAugConfig.from_yaml(data_cfg.get("augmentations", {}))

    sources: List[Tuple[str, PoseJsonlDataset, float]] = []
    ds_cfgs = data_cfg.get("datasets", {})
    for name, w in mix.items():
        if w <= 0:
            continue
        sub = ds_cfgs.get(name)
        if not sub or not sub.get("enabled", True):
            continue
        ann_key = "annotations" if is_train else "val_annotations"
        ann_path = sub.get(ann_key)
        if not ann_path:
            continue
        import os
        if not os.path.exists(ann_path):
            # We skip missing annotations but do not crash -- allows partial setups.
            continue
        ds = PoseJsonlDataset(
            ann_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma=sigma,
            is_train=is_train,
            aug=aug,
            max_items=max_items_per_source,
        )
        sources.append((name, ds, float(w)))
    if not sources:
        raise RuntimeError(
            "No pose-dataset sources were found on disk. Run "
            "`python -m scripts.prepare_aist_training_data` to build "
            "data/labels/aistpp/internal_train.jsonl and internal_val.jsonl "
            "from data/raw_videos/ + AIST++ 2D keypoints."
        )
    return MixedPoseDataset(sources, epoch_size=epoch_size, seed=seed)
