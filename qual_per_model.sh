#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=qual_per_model_%j.log
export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project
$FVR/bin/python -u qual_per_model.py
