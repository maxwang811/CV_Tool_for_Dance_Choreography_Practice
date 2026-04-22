#!/bin/bash
# One-time conda env setup for training on Oscar (Brown CCV).
#
# Run this ONCE on an Oscar login node (or an interactive session), from the
# repo root:
#     bash scripts/slurm/setup_env.sh
#
# It creates a conda env named `dance-cv` with Python 3.11, installs CUDA 12.1
# PyTorch wheels (which ship their own CUDA runtime, so no cuda/<ver> module
# is needed at job time), then the rest of requirements.txt.
#
# If `module load anaconda/...` fails because CCV rotated the build-string,
# run `module avail anaconda` and update the ANACONDA_MODULE value below.

set -euo pipefail

ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"
ENV_NAME="${ENV_NAME:-dance-cv}"
PY_VER="${PY_VER:-3.11}"

if [[ ! -f requirements.txt ]]; then
    echo "error: run this from the repo root (requirements.txt not found)" >&2
    exit 1
fi

module purge
module load "${ANACONDA_MODULE}"

# Make `conda activate` work inside this non-interactive shell.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "env '${ENV_NAME}' already exists; reusing it."
else
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

# CUDA 12.1 PyTorch wheels. Pin to the versions from requirements.txt.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.2" "torchvision>=0.17"

# Install the rest of requirements, skipping torch/torchvision so pip doesn't
# re-resolve them from PyPI (which would pull CPU-only wheels).
TMP_REQ="$(mktemp)"
trap 'rm -f "${TMP_REQ}"' EXIT
grep -vE '^(torch|torchvision)([[:space:]]|$|>=|==|<=|>|<|!)' requirements.txt > "${TMP_REQ}"
pip install -r "${TMP_REQ}"

echo "--------------------------------------------------------------"
echo "env ready: ${ENV_NAME}"
python -c "import torch; print('torch:', torch.__version__); \
print('cuda available on this node:', torch.cuda.is_available())"
echo "(cuda=False is expected on login nodes; it will be True on a gpu job.)"
echo "--------------------------------------------------------------"
