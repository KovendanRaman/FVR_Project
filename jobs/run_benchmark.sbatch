#!/bin/bash
#SBATCH --job-name=fvr_benchmark
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --dependency=afterok:43324

# This job will ONLY start if Job 43324 (your compile) finishes without errors.

apptainer exec --nv /home-mscluster/kraman/anaconda.sif bash -c '
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    cd ~/FVR_Project
    
    echo "Starting O(N) Complexity Benchmark..."
    $FVR/bin/python benchmark_complexity.py
'
