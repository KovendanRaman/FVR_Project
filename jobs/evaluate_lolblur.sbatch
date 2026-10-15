#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=eval_lolblur_%j.log

# RQ2 image-quality evaluation on the LOL-Blur TEST set using best.pth.
# NIQE needs torchvision; restore it first (won't disturb the torch nightly):
#   $FVR/bin/python -m pip install torchvision --no-deps
# If torchvision is absent the script still reports PSNR + SSIM and skips NIQE.

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo '=== LOL-Blur evaluation (RQ2) ==='
    echo \"Start: \$(date)\"

    \$FVR/bin/python -u metrics_eval2.py \
      --blurry-dir ./data/processed/test/blurry \
      --sharp-dir ./data/processed/test/sharp \
      --checkpoint ./checkpoints/best.pth \
      --tile 128 --overlap 32 2>&1

    echo \"End: \$(date)\"
    echo '=== done ==='
  "
