#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=eval_depth16_%j.log

DEPTH=16

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    echo '=== LOL-Blur eval depth=${DEPTH} (RQ3) ==='
    echo \"Start: \$(date)\"

    \$FVR/bin/python -u metrics_eval2.py \
      --blurry-dir ./data/processed/test/blurry \
      --sharp-dir ./data/processed/test/sharp \
      --checkpoint ./checkpoints_depth${DEPTH}/best.pth \
      --tile 128 --overlap 32 2>&1

    echo \"End: \$(date)\"
    echo '=== done ==='
  "
