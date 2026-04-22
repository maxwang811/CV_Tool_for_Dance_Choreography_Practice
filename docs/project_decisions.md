# Project Decisions (FROZEN)

This file is the single source of truth for the project scope. Every config,
script, and module MUST refer back to these decisions. Do not relax any of
these without updating this document first.

## 1. No pretrained weights — ever

- No pretrained pose model (OpenPose / MediaPipe / MMPose / ViTPose / ...).
- No pretrained object/person detector.
- No ImageNet / COCO-Detection initialization for any backbone.
- No self-supervised foundation-model checkpoints.
- All weights are initialized randomly (Kaiming / normal) and trained from
  scratch on labeled keypoint data we prepare ourselves.

Enforcement: model factory functions in `src/models/` MUST NOT load weights
from any external source. Loading state dicts is only permitted from our own
training checkpoints saved under `data/processed/checkpoints/`.

## 2. Task definition

- Single-person 2D pose extraction only.
- Short clips only: 10-20 seconds.
- Fixed camera (or near-fixed) only for v1.
- Downstream task: compare one benchmark clip vs one imitation clip.

## 3. Canonical joint format

- COCO 17-keypoint layout, everywhere:
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle.
- Flip map (for horizontal-flip augmentation) is defined in
  `src/datasets/common.py` and is the ONLY allowed source.

## 4. MVP scope

- Predict full body.
- In scoring, weight upper body more heavily than legs (see
  `configs/data/compare.yaml::body_part_weights`).

## 5. Training paradigm

- Per-frame supervised pose training FIRST (stages A and B).
- Video alignment / scoring / temporal refinement comes AFTER.
- Do NOT start with an end-to-end video transformer.

## 6. Data policy

- `data/raw_urls/*.csv`: download lists for the AIST dance videos that
  populate `data/raw_videos/`.
- Supervised pose labels come ONLY from AIST++ 2D keypoints paired with
  those `data/raw_videos/` clips. COCO and CrowdPose are not used.
- The AIST++ per-video 2D keypoint files must be placed under
  `data/labels/aistpp/keypoints2d_raw/<video_stem>.{pkl,npy}`; the
  `scripts/prepare_aist_training_data.py` pipeline is the ONLY supported
  way to build `data/labels/aistpp/internal_train.jsonl` and
  `internal_val.jsonl`.
- Our own paired benchmark/imitation videos are for downstream evaluation
  and calibration, NOT for supervised pose-label training (unless
  explicitly hand-annotated into `data/labels/custom_dance_val/`).

Revision (2026-04-22): narrowed supervised-label policy to AIST++ only,
following the decision to train solely on `data/raw_videos/`.

## 7. Pipeline order (never skip forward)

1. Repo skeleton + configs.
2. CSV ingestion + manifests.
3. Pilot downloads + ffprobe integrity.
4. MVP subset curation.
5. Paired benchmark/imitation manifest.
6. Label conversion (AIST++ 2D keypoints only) via
   `scripts/prepare_aist_training_data.py`.
7. Simple Baseline model — prove the training loop.
8. HRNet-W32 — final model.
9. Single-stage training on AIST++ labels from `data/raw_videos/`.
10. Motion-based video inference.
11. Temporal smoothing.
12. Normalization -> features -> DTW -> score -> feedback.
13. Streamlit demo (last, not first).

## 8. Explicit non-goals for v1

- No multi-person pose.
- No moving-camera clips.
- No training on raw unlabeled videos (no pseudo-labels, no self-training).
- No transformer as the first model.
- No UI before CLI pipeline works.
- No single "black-box" score without per-part / per-window diagnostics.
