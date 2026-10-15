#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=brightness_probe_%j.log

export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project

$FVR/bin/python -u -c "
import torch, numpy as np, os
from PIL import Image

def mean_brightness(d, n=100):
    fs = sorted(os.listdir(d))[:n]
    vals = [np.array(Image.open(os.path.join(d,f)).convert('RGB')).mean() for f in fs]
    return np.mean(vals)

print('=== Input / Target brightness (0-255) ===')
print(f'RealBlur-J blurry input : {mean_brightness(\"data/processed/realblur/test/blurry\"):.1f}')
print(f'RealBlur-J sharp target : {mean_brightness(\"data/processed/realblur/test/sharp\"):.1f}')
print(f'GoPro sharp target      : {mean_brightness(\"data/processed/gopro/test/sharp\"):.1f}')
print(f'LOL-Blur blurry input   : {mean_brightness(\"data/processed/test/blurry\"):.1f}')
print(f'LOL-Blur sharp target   : {mean_brightness(\"data/processed/test/sharp\"):.1f}')
"
