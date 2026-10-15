#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=eval_smoke_%j.log

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c "
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project
    echo '=== eval smoke test (20 imgs, tiled) ==='
    \$FVR/bin/python -u metrics_eval2.py \
      --blurry-dir ./data/processed/test/blurry \
      --sharp-dir ./data/processed/test/sharp \
      --checkpoint ./checkpoints/best.pth \
      --tile 128 --overlap 32 --limit 20 2>&1
    echo '=== done ==='
  "
