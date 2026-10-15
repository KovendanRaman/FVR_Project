#!/bin/bash
#SBATCH --job-name=fvr-train
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster110
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fvr-%j.out
#SBATCH --error=logs/fvr-%j.err

# Wits MSCluster. Submit with:  sbatch scripts/train_fvr.sbatch
# Resume with:                  sbatch scripts/train_fvr.sbatch --resume checkpoints/last.pth
#
# Verified environment facts this script encodes (2026-07-30):
#   - GPU is RTX PRO 6000 Blackwell, ~96 GB, sm_120 -- NOT an A6000/sm_86
#   - torch is a cu128 nightly; stable torch ships no sm_120 kernels and dies
#     with "no kernel image available"
#   - Slurm reports GRES=(null), so --gres=gpu:1 gets the job rejected
#   - --nodelist=mscluster110 is REQUIRED: the .sif lives on node-local storage
#     on 110 and is absent on 107
#   - apptainer needs --nv or the GPU is invisible inside the container
#   - `conda activate` / `conda run` are unreliable in this container, so the
#     env is addressed by absolute interpreter path
#   - SELECTIVE_SCAN_BACKEND=torch (fused kernel not built); numerically
#     correct and O(N)-valid, just slower

set -euo pipefail
mkdir -p logs checkpoints

APPTAINER_IMG="${APPTAINER_IMG:-/usr/local/containers/anaconda/anaconda-2025-12-2.sif}"
PYTHON="${PYTHON:-/home-mscluster/kraman/.conda/envs/fvr/bin/python}"

# Slurm does not isolate GPUs here (no GRES), so every card on the node is
# visible to every job. Pin one to avoid colliding with a co-scheduled job.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

in_container() { apptainer exec --nv "$APPTAINER_IMG" "$PYTHON" "$@"; }

nvidia-smi
# A real matmul, not just is_available(): on Blackwell a wrong torch build
# passes is_available() and only fails once a kernel actually launches.
in_container -c "
import torch
assert torch.cuda.is_available(), 'no GPU inside container - is --nv set?'
print('GPU:', torch.cuda.get_device_name(0))
print('capability: sm_%d%d' % torch.cuda.get_device_capability(0))
print('torch:', torch.__version__)
(torch.randn(512, 512, device='cuda') @ torch.randn(512, 512, device='cuda')).sum().item()
print('GPU matmul OK')
"
in_container -c "from models.vmamba import SELECTIVE_SCAN_BACKEND as b; print('backend:', b)"

# --amp is safe: the FFT sections in edffn.py/fusion.py force fp32 via
# fp32_autocast, because torch.fft and torch.polar have no half kernels.
#
# --no-checkpoint is NOT set: with the pure-PyTorch scan, activation
# checkpointing is what keeps 256x256 memory bounded during backward.
srun apptainer exec --nv "$APPTAINER_IMG" "$PYTHON" train.py \
    --blurry-dir ./data/raw/blurry \
    --sharp-dir  ./data/raw/sharp \
    --patch-size 256 \
    --batch-size 16 \
    --epochs 300 \
    --lr 2e-4 \
    --embed-dim 64 \
    --depth 4 \
    --num-workers 12 \
    --amp \
    --out-dir ./checkpoints \
    "$@"
