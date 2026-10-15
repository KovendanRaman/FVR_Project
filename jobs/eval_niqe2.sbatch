#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=eval_niqe2_%j.log

# Try to enable real pyiqa NIQE by providing libgthread via conda glib,
# WITHOUT disturbing torch (all installs --no-deps or conda into env).
# Env is backed up, so this is low-risk.

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c '
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo "=== confirm torch still healthy ==="
    $FVR/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

    echo "=== install glib into env (provides libgthread-2.0.so.0) ==="
    $FVR/bin/conda install -p $FVR -c conda-forge glib -y 2>&1 | tail -5 || \
      echo "conda install failed, will try LD_LIBRARY_PATH from existing libs"

    # Point loader at the env lib dir where conda glib lands
    export LD_LIBRARY_PATH=$FVR/lib:$LD_LIBRARY_PATH

    echo "=== install pyiqa deps WITHOUT touching torch ==="
    $FVR/bin/python -m pip install pyiqa --no-deps 2>&1 | tail -3
    # pyiqa needs these but they must not pull torch:
    $FVR/bin/python -m pip install opencv-python-headless --no-deps 2>&1 | tail -2
    $FVR/bin/python -m pip install scipy scikit-image --no-deps 2>&1 | tail -2

    echo "=== test cv2 + pyiqa import ==="
    $FVR/bin/python -c "import cv2; print(\"cv2 OK\", cv2.__version__)" 2>&1
    $FVR/bin/python -c "import pyiqa; print(\"pyiqa OK\")" 2>&1

    echo "=== confirm torch STILL healthy after installs ==="
    $FVR/bin/python -c "
import torch
from mamba_ssm import Mamba
m=Mamba(64,16,4,2).cuda(); x=torch.randn(1,32,64,device=\"cuda\")
print(\"torch+kernel still OK\")
" 2>&1

    echo "=== run eval with NIQE ==="
    $FVR/bin/python -u metrics_eval2.py \
      --blurry-dir ./data/processed/test/blurry \
      --sharp-dir ./data/processed/test/sharp \
      --checkpoint ./checkpoints/best.pth \
      --tile 128 --overlap 32 2>&1
    echo "=== done ==="
  '
