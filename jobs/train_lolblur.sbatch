#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --output=train_lolblur_%j.log

# LOL-Blur joint low-light + deblur training
# ~11 min/epoch (fused kernel) x 200 epochs = ~37h, fits in one job.
# Auto-resumes from checkpoints/last.pth if present.

RESUME_FLAG=""
if [ -f /home-mscluster/kraman/FVR_Project/checkpoints/last.pth ]; then
    RESUME_FLAG="--resume ./checkpoints/last.pth"
    echo "Found existing checkpoint — resuming."
fi

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo '=== LOL-Blur training ==='
    echo \"Start: \$(date)\"

    \$FVR/bin/python -u train.py \
      --blurry-dir ./data/processed/train/blurry \
      --sharp-dir ./data/processed/train/sharp \
      --patch-size 128 \
      --batch-size 8 \
      --epochs 200 \
      --out-dir ./checkpoints \
      $RESUME_FLAG 2>&1

    echo \"End: \$(date)\"
    echo '=== done ==='
  "
