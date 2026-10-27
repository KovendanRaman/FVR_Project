#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=eval_rb_composite_%j.log
#SBATCH --exclude=mscluster111,mscluster112
export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project
echo "Node: $(hostname)"
$FVR/bin/python -c "import torch; assert torch.cuda.is_available(),'NO GPU'; print('GPU OK:',torch.cuda.get_device_name(0))" || { echo "no GPU"; exit 1; }
$FVR/bin/python -u metrics_eval2.py \
  --blurry-dir data/processed/realblur/test/blurry \
  --sharp-dir  data/processed/realblur/test/sharp \
  --checkpoint checkpoints_gopro_freq/best.pth \
  --tile 128 --overlap 32
