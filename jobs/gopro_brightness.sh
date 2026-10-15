#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=gopro_brightness_%j.log

export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project

$FVR/bin/python -u inference.py \
  --input-dir data/processed/gopro/test/blurry \
  --output-dir /tmp/combined_gopro_out \
  --checkpoint checkpoints_combined/best.pth \
  --tile 128 --overlap 32

$FVR/bin/python -u -c "
import numpy as np, os
from PIL import Image
def mb(d,n=100):
    fs=sorted(os.listdir(d))[:n]
    return np.mean([np.array(Image.open(os.path.join(d,f)).convert('RGB')).mean() for f in fs])
print('=== Combined model on GoPro — brightness (0-255) ===')
print(f'Blurry input    : {mb(\"data/processed/gopro/test/blurry\"):.1f}')
print(f'Combined OUTPUT : {mb(\"/tmp/combined_gopro_out\"):.1f}')
print(f'Sharp target    : {mb(\"data/processed/gopro/test/sharp\"):.1f}')
"
