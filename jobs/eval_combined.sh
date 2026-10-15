#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=eval_combined_%j.log

export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project

echo "=== Combined model on LOL-Blur test ==="
$FVR/bin/python -u metrics_eval2.py \
  --blurry-dir data/processed/test/blurry \
  --sharp-dir data/processed/test/sharp \
  --checkpoint checkpoints_combined/best.pth \
  --tile 128 --overlap 32

echo "=== Combined model on GoPro test ==="
$FVR/bin/python -u metrics_eval2.py \
  --blurry-dir data/processed/gopro/test/blurry \
  --sharp-dir data/processed/gopro/test/sharp \
  --checkpoint checkpoints_combined/best.pth \
  --tile 128 --overlap 32

echo "=== Combined model on RealBlur-J test ==="
$FVR/bin/python -u metrics_eval2.py \
  --blurry-dir data/processed/realblur/test/blurry \
  --sharp-dir data/processed/realblur/test/sharp \
  --checkpoint checkpoints_combined/best.pth \
  --tile 128 --overlap 32
