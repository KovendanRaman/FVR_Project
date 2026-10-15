#!/bin/bash
#SBATCH --job-name=lolblur-dl
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

cd ~/FVR_Project
FVR=/home-mscluster/kraman/.conda/envs/fvr
$FVR/bin/gdown --folder https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX -O data/raw/lolblur/
echo "=== download finished, contents: ==="
ls -lah data/raw/lolblur/
