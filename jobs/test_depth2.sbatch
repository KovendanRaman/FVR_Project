#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=test_depth2_%j.log

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c '
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo "=== depth 2 test ==="
    $FVR/bin/python -u train.py \
      --blurry-dir ./data/processed/train/blurry \
      --sharp-dir ./data/processed/train/sharp \
      --patch-size 128 \
      --batch-size 8 \
      --depth 2 \
      --epochs 1 \
      --num-workers 0 \
      --out-dir ./checkpoints_depth2 2>&1
    echo "=== done ==="
  '
