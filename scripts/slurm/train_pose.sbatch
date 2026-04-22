#!/bin/bash
# SLURM batch script for training the pose model on Oscar (Brown CCV).
#
# Entry point: `python -m src.train.train_pose --train <cfg>`.
# The engine auto-selects CUDA when available (see src/train/engine.py).
#
# Submit from the repo root:
#     sbatch scripts/slurm/train_pose.sbatch
#
# Override the training config:
#     sbatch --export=ALL,TRAIN_CFG=configs/train/train_hrnet.yaml \
#            scripts/slurm/train_pose.sbatch
#
# See docs/oscar_runbook.md for full upload / env-setup / monitor steps.

#SBATCH --job-name=cv_dance_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your_email@brown.edu   # uncomment + set to get mail

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm data/processed/train

# Oscar module environment. If `module load` fails because CCV rotated the
# build-string, run `module avail anaconda` on a login node and update this.
module purge
module load anaconda/2023.09-0-7nso27y

# Activate the env created by scripts/slurm/setup_env.sh.
# shellcheck disable=SC1091
source activate dance-cv

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TRAIN_CFG="${TRAIN_CFG:-configs/train/train.yaml}"

echo "=============================================================="
echo "job:        ${SLURM_JOB_NAME:-unknown} (${SLURM_JOB_ID:-?})"
echo "node:       $(hostname)"
echo "cwd:        $(pwd)"
echo "cfg:        ${TRAIN_CFG}"
echo "cpus/task:  ${SLURM_CPUS_PER_TASK:-?}"
echo "gpus:       ${SLURM_GPUS_ON_NODE:-?} (gres=${SLURM_JOB_GRES:-?})"
echo "=============================================================="
nvidia-smi || true
python -c "import torch, sys; print('python:', sys.version.split()[0]); \
print('torch: ', torch.__version__); \
print('cuda:  ', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=============================================================="

srun --unbuffered python -m src.train.train_pose --train "${TRAIN_CFG}"
