#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=train_gopro_hf_%j.log
#SBATCH --exclude=mscluster111,mscluster112
export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project
echo "Node: $(hostname)"
$FVR/bin/python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU OK:', torch.cuda.get_device_name(0))" || { echo "guard: no GPU, exiting"; exit 1; }
$FVR/bin/python -u train.py \
  --blurry-dir data/processed/gopro/train/blurry \
  --sharp-dir  data/processed/gopro/train/sharp \
  --patch-size 128 --batch-size 8 --depth 4 \
  --epochs 200 --lr 1e-4 --grad-clip 1.0 \
  --loss combined --no-checkpoint --num-workers 8 \
  --out-dir checkpoints_gopro_hf
