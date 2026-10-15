#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=output_brightness_%j.log

export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project

# Restore RealBlur-J with the combined model
$FVR/bin/python -u inference.py \
  --input-dir data/processed/realblur/test/blurry \
  --output-dir /tmp/combined_realblur_out \
  --checkpoint checkpoints_combined/best.pth \
  --tile 128 --overlap 32

# Compare brightness: input vs output vs target
$FVR/bin/python -u -c "
import numpy as np, os
from PIL import Image

def mb(d, n=100):
    fs = sorted(os.listdir(d))[:n]
    return np.mean([np.array(Image.open(os.path.join(d,f)).convert('RGB')).mean() for f in fs])

print('=== Combined model on RealBlur-J — brightness (0-255) ===')
print(f'Blurry input      : {mb(\"data/processed/realblur/test/blurry\"):.1f}')
print(f'Combined OUTPUT   : {mb(\"/tmp/combined_realblur_out\"):.1f}')
print(f'Sharp target (GT) : {mb(\"data/processed/realblur/test/sharp\"):.1f}')
print()
print('If OUTPUT >> target, the model is over-brightening already-bright images.')
"
