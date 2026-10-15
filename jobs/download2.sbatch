#!/bin/bash
#SBATCH --job-name=lolblur-dl2
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/dl2-%j.log
cd ~/FVR_Project/data/raw/lolblur
FVR=/home-mscluster/kraman/.conda/envs/fvr
echo "=== test.zip (small, try first) ==="
$FVR/bin/gdown 1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r -O test.zip
echo "=== train_all.zip (big) ==="
$FVR/bin/gdown 1RGSG_bnxkqUf-ZorLq4ihPsJnBmEG6cA -O train_all.zip
echo "=== result ==="
ls -lah ~/FVR_Project/data/raw/lolblur/
