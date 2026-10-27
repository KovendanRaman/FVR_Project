#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=baseline_check_%j.log
export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
$FVR/bin/python -u baseline_check.py
