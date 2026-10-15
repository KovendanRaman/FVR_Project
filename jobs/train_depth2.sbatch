#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --output=train_depth2_%j.log

# RQ3 depth sweep — train FVR at a specific depth on LOL-Blur.
# USAGE: sed 's/2/2/g' train_depth.sbatch | sbatch

DEPTH=2
CKPT_DIR=/home-mscluster/kraman/FVR_Project/checkpoints_depth${DEPTH}

RESUME_FLAG=""
if [ -f "$CKPT_DIR/last.pth" ]; then
    RESUME_FLAG="--resume $CKPT_DIR/last.pth"
    echo "Found checkpoint for depth $DEPTH — resuming."
fi

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo '=== LOL-Blur depth=${DEPTH} training (RQ3) ==='
    echo \"Start: \$(date)\"

    \$FVR/bin/python -u train.py \
      --blurry-dir ./data/processed/train/blurry \
      --sharp-dir ./data/processed/train/sharp \
      --patch-size 128 \
      --batch-size 8 \
      --depth ${DEPTH} \
      --no-checkpoint \
      --epochs 200 \
      --out-dir ./checkpoints_depth${DEPTH} \
      $RESUME_FLAG 2>&1

    echo \"End: \$(date)\"
    echo '=== done ==='
  "
