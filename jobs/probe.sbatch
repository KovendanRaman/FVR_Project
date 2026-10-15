#!/bin/bash
#SBATCH --job-name=fvr-probe
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/probe-%j.log

FVR=/home-mscluster/kraman/.conda/envs/fvr
cd ~/FVR_Project
apptainer exec --nv /home-mscluster/kraman/anaconda.sif \
  $FVR/bin/python train.py \
    --blurry-dir ~/FVR_Project/data/processed/train/blurry \
    --sharp-dir  ~/FVR_Project/data/processed/train/sharp \
    --patch-size 128 --depth 4 --batch-size 8 \
    --epochs 1 --num-workers 4 \
    --out-dir ~/FVR_Project/checkpoints/probe
