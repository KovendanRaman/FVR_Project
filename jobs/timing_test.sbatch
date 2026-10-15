#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=timing_test_%j.log

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c '
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo "=== Fused kernel timing test ==="
    echo "Start: $(date)"

    $FVR/bin/python train.py --epochs 1 --blurry-dir ./data/processed/train/blurry --sharp-dir ./data/processed/train/sharp 2>&1

    echo "End: $(date)"
    echo "=== done ==="
  '
