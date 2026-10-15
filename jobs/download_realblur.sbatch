#!/bin/bash
#SBATCH --job-name=realblur-dl
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/realblur-dl-%j.log

FVR=/home-mscluster/kraman/.conda/envs/fvr
mkdir -p ~/FVR_Project/data/raw/realblur
cd ~/FVR_Project/data/raw/realblur

echo "=== downloading RealBlur.tar.gz ==="
$FVR/bin/gdown 17v5Dj0M2WExgxJgsGpEWc-6UyhK1IWWK -O RealBlur.tar.gz

echo "=== download result ==="
ls -lah ~/FVR_Project/data/raw/realblur/

echo "=== extracting ==="
tar -xzf RealBlur.tar.gz -C ~/FVR_Project/data/raw/realblur/

echo "=== structure (first 3 levels) ==="
find ~/FVR_Project/data/raw/realblur -maxdepth 3 -type d | head -30
echo "=== done ==="
