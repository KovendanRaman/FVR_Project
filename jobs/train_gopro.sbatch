#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --output=train_gopro_%j.log

# GoPro deblur-only training (separate model from LOL-Blur).
# ~2103 train imgs vs 9690 for LOL-Blur, so epochs ~4-5x faster.
# Writes to checkpoints_gopro/ so it does NOT overwrite the LOL-Blur model.
# Auto-resumes from checkpoints_gopro/last.pth if present.

CKPT_DIR=/home-mscluster/kraman/FVR_Project/checkpoints_gopro
RESUME_FLAG=""
if [ -f "$CKPT_DIR/last.pth" ]; then
    RESUME_FLAG="--resume ./checkpoints_gopro/last.pth"
    echo "Found existing GoPro checkpoint — resuming."
fi

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo '=== GoPro training ==='
    echo \"Start: \$(date)\"

    \$FVR/bin/python -u train.py \
      --blurry-dir ./data/processed/gopro/train/blurry \
      --sharp-dir ./data/processed/gopro/train/sharp \
      --patch-size 128 \
      --batch-size 8 \
      --epochs 200 \
      --out-dir ./checkpoints_gopro \
      $RESUME_FLAG 2>&1

    echo \"End: \$(date)\"
    echo '=== done ==='
  "
