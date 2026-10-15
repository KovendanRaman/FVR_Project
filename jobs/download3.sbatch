#!/bin/bash
#SBATCH --job-name=lolblur-lb
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/dl3-%j.log
cd ~/FVR_Project/data/raw/lolblur
FVR=/home-mscluster/kraman/.conda/envs/fvr
echo "=== low_blur.zip (the input half; we already have high_sharp targets) ==="
$FVR/bin/gdown 1-tEgiBtGsrtN_uQU977FlrvM7G7X9sDG -O low_blur.zip
echo "=== result ==="
ls -lah ~/FVR_Project/data/raw/lolblur/
