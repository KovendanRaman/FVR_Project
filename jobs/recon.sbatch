#!/bin/bash
#SBATCH --job-name=recon
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/recon-%j.log
FVR=/home-mscluster/kraman/.conda/envs/fvr
echo "=== host cuda toolkits ==="; ls -d /usr/local/cuda-* 2>/dev/null
echo "=== container checks ==="
apptainer exec --nv /home-mscluster/kraman/anaconda.sif bash -c "
  ls -d /usr/local/cuda-* 2>/dev/null
  $FVR/bin/python -c 'import torch; print(\"torch cuda:\", torch.version.cuda); print(\"capability:\", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"NO GPU\")'
  gcc --version | head -1
  nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH'
"
echo "=== done ==="
