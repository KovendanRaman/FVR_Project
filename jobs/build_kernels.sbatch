#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=build_kernels_%j.log

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c '
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project
    SP=$FVR/lib/python3.11/site-packages

    echo "=== confirm torch is correct before building ==="
    $FVR/bin/python -c "import torch; print(torch.__version__); print(torch.utils.cmake_prefix_path)"

    echo "=== purge old kernel .so files ==="
    $FVR/bin/python -m pip uninstall -y mamba-ssm causal-conv1d 2>&1 | tail -2
    rm -f $SP/selective_scan_cuda*.so $SP/causal_conv1d_cuda*.so 2>/dev/null
    rm -rf $SP/~amba-ssm* $SP/~ausal-conv1d* 2>/dev/null

    export CC=$FVR/bin/x86_64-conda-linux-gnu-gcc
    export CXX=$FVR/bin/x86_64-conda-linux-gnu-g++
    export CUDA_HOME=/usr/local/cuda-12.9
    export TORCH_CUDA_ARCH_LIST="12.0"
    export PATH=$CUDA_HOME/bin:$PATH
    export MAX_JOBS=4

    echo "=== nvcc check ==="
    which nvcc && nvcc --version | tail -2

    echo "=== BUILD causal-conv1d (full output) ==="
    $FVR/bin/python -m pip install causal-conv1d --no-build-isolation --no-binary :all: --no-deps --no-cache-dir -v 2>&1 | grep -iE "error|fatal|building|running|success|gcc|nvcc|\.so" | tail -40

    echo "=== BUILD mamba-ssm (full output) ==="
    $FVR/bin/python -m pip install mamba-ssm --no-build-isolation --no-binary :all: --no-deps --no-cache-dir -v 2>&1 | grep -iE "error|fatal|building|running|success|gcc|nvcc|\.so" | tail -40

    echo "=== list built .so files ==="
    ls -la $SP/selective_scan_cuda*.so $SP/causal_conv1d_cuda*.so 2>&1

    echo "=== TEST ==="
    $FVR/bin/python -c "
import torch
from mamba_ssm import Mamba
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(1, 32, 64, device=\"cuda\", requires_grad=True)
y = m(x); y.sum().backward()
print(\"FUSED KERNEL OK\", tuple(y.shape))
"
    echo "=== done ==="
  '
