#!/bin/bash
# FVR dependency install for the Wits MSCluster.
#
# IMPORTANT: run this from INSIDE the Wits conda container, on a GPU node.
# Anaconda here is wrapped in Apptainer, so `conda activate` does not work and
# `eval "$(conda shell.bash hook)"` will fail. The sequence is:
#
#   srun --partition=biggpu --cpus-per-task=12 --mem=64G --time=03:00:00 --pty bash
#   module load anaconda/anaconda-2025-12-2
#   conda create -n fvr python=3.11 -y
#   conda-activate fvr              # Wits wrapper; spawns a shell in the container
#   cd ~/FVR_Project && bash scripts/setup_env.sh
#
# Other cluster facts encoded below:
#   - Slurm reports GRES=(null) -> never pass --gres=gpu:1
#   - no CUDA module, but toolkits sit at /usr/local/cuda-12.9 and -13.3
#   - nvcc is present, just not on PATH
#   - driver 595.71.05 caps at CUDA 13.2, so cuda-13.3 is TOO NEW
#   - GPUs are 2x RTX A6000 (Ampere, sm_86)

set -euo pipefail

if ! python -c "import sys; sys.exit(0)" 2>/dev/null; then
    echo "!! No python on PATH. Did you run 'conda-activate fvr' first?" >&2
    exit 1
fi
echo "==> Using python: $(command -v python) ($(python --version 2>&1))"

pip install --upgrade pip setuptools wheel

echo "==> Installing torch"
# No index URL on purpose. The default PyPI wheel for Linux is already a CUDA
# build, and it is current. Pinning an old tag like cu121 by hand silently
# installs a years-old torch, because that channel stopped receiving builds.
pip install torch torchvision
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

echo "==> Installing project requirements"
pip install -r requirements.txt

if [ -z "$TORCH_CUDA" ]; then
    echo "!! torch has no CUDA support; skipping the fused kernel."
    echo "!! FVR stays numerically correct on the pure-PyTorch scan."
    exit 0
fi

echo "==> Selecting a toolkit matching torch's CUDA $TORCH_CUDA"
# Match the toolkit's MAJOR version to torch's; highest matching minor wins.
# cuda-13.3 is intentionally unreachable while torch is on CUDA 12.x, which is
# just as well since the driver could not load 13.3 binaries anyway.
CUDA_MAJOR="${TORCH_CUDA%%.*}"
CUDA_HOME=""
for candidate in $(ls -d /usr/local/cuda-"$CUDA_MAJOR".* 2>/dev/null | sort -V -r); do
    if [ -x "$candidate/bin/nvcc" ]; then
        CUDA_HOME="$candidate"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "!! No /usr/local/cuda-$CUDA_MAJOR.* toolkit with nvcc found."
    echo "!! Skipping the fused kernel; the pure-PyTorch scan will be used."
    exit 0
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "    CUDA_HOME=$CUDA_HOME"
nvcc --version

echo "==> Building the fused selective-scan kernel (15-30 min)"
# Detect the arch from the actual GPU rather than hardcoding it. The verified
# node has an RTX PRO 6000 Blackwell (sm_120 -> "12.0"), NOT the sm_86 an
# earlier handoff claimed; building for the wrong arch yields a kernel that
# loads and then fails with "no kernel image available for execution".
# MAX_JOBS caps parallel nvcc processes so the build does not OOM.
# --no-build-isolation makes the torch above visible to setup.py, which imports
# it at build time. Omitting it is what caused the original
# "ModuleNotFoundError: No module named 'torch'".
# Version specifiers are quoted so bash does not read '>=' as a redirect.
DETECTED_ARCH="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))' 2>/dev/null || echo "")"
if [ -z "$DETECTED_ARCH" ]; then
    echo "!! Could not query the GPU. Run this on a GPU node, not the login node." >&2
    exit 1
fi
echo "    detected compute capability: sm_${DETECTED_ARCH/./}"
export TORCH_CUDA_ARCH_LIST="$DETECTED_ARCH"
export MAX_JOBS=4
pip install "causal-conv1d>=1.2.0" --no-build-isolation
pip install -r requirements-cuda.txt --no-build-isolation

echo "==> Verifying"
# -m, not a file path: these modules use package imports, so
# `python models/vmamba.py` fails with ModuleNotFoundError.
python -c "from models.vmamba import SELECTIVE_SCAN_BACKEND; print('backend:', SELECTIVE_SCAN_BACKEND)"
python -m models.vmamba
python -m models.fvr_net

cat <<EOF

==> Done. In batch scripts use non-interactive execution:
    module load anaconda/anaconda-2025-12-2
    conda run -n fvr --no-capture-output python train.py ...

    CUDA_HOME for later sessions: $CUDA_HOME
EOF
