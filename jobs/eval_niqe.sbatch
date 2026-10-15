#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=eval_niqe_%j.log

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo '=== Installing torchvision + pyiqa ==='
    \$FVR/bin/python -m pip install torchvision --no-deps 2>&1 | tail -5
    \$FVR/bin/python -m pip install pyiqa 2>&1 | tail -5
    \$FVR/bin/python -c 'import pyiqa; print(\"pyiqa OK\")' 2>&1

    echo '=== LOL-Blur eval with NIQE ==='
    echo \"Start: \$(date)\"

    \$FVR/bin/python -u metrics_eval2.py \
      --blurry-dir ./data/processed/test/blurry \
      --sharp-dir ./data/processed/test/sharp \
      --checkpoint ./checkpoints/best.pth \
      --tile 128 --overlap 32 2>&1

    echo \"End: \$(date)\"
    echo '=== done ==='
  "
