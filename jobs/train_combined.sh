#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=train_combined_%j.log

export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project

$FVR/bin/python -u train.py \
  --blurry-dir data/processed/combined/train/blurry \
  --sharp-dir data/processed/combined/train/sharp \
  --patch-size 128 \
  --batch-size 8 \
  --depth 4 \
  --epochs 200 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  --out-dir checkpoints_combined \
  --num-workers 8 \
  --no-checkpoint
