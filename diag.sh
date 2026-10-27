#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=diag_%j.log
export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project
$FVR/bin/python -u diag_pathways.py
