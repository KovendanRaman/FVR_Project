#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=flops_bench_%j.log

export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project

$FVR/bin/python -u evaluate.py --depth 4 --resolutions 128 256 384 512
