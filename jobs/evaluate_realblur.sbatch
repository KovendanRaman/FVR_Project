#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=eval_realblur_%j.log

# RealBlur-J evaluation using the LOL-Blur trained model.
# Shows cross-dataset generalization (no training on RealBlur).

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo '=== RealBlur-J evaluation (cross-dataset, LOL-Blur model) ==='
    echo \"Start: \$(date)\"

    \$FVR/bin/python -u metrics_eval2.py \
      --blurry-dir ./data/processed/realblur/test/blurry \
      --sharp-dir ./data/processed/realblur/test/sharp \
      --checkpoint ./checkpoints/best.pth \
      --tile 128 --overlap 32 2>&1

    echo \"End: \$(date)\"
    echo '=== done ==='
  "
