#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=diag_%j.log

apptainer exec --nv --bind /usr/local/cuda-12.9 \
  /home-mscluster/kraman/anaconda.sif \
  bash -c '
    export FVR=/home-mscluster/kraman/.conda/envs/fvr
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/home-mscluster/kraman/FVR_Project
    cd /home-mscluster/kraman/FVR_Project

    $FVR/bin/python -u - << "PYEOF"
from PIL import Image
import numpy as np, os, glob

bdir = "./data/processed/test/blurry"
sdir = "./data/processed/test/sharp"

# 1) Pairing / degradation sanity on a few files
names = sorted(os.listdir(bdir))[:5]
for n in names:
    b = np.asarray(Image.open(os.path.join(bdir,n)).convert("RGB")).astype(float)
    s = np.asarray(Image.open(os.path.join(sdir,n)).convert("RGB")).astype(float)
    mse = ((b-s)**2).mean()
    psnr = 10*np.log10(255**2/mse) if mse>0 else 99
    print(f"{n}: blurry mean {b.mean():6.2f} | sharp mean {s.mean():6.2f} "
          f"| shapes {b.shape}=={s.shape} | naive PSNR {psnr:5.2f} dB")

# 2) Now run the actual model on ONE full image and print the PSNR,
#    plus what the input/output pixel ranges look like.
import torch
from models.fvr_net import FVRNet
ckpt = torch.load("./checkpoints/best.pth", map_location="cuda")
sa = ckpt.get("args", {})
model = FVRNet(embed_dim=sa.get("embed_dim",64), depth=sa.get("depth",4),
               d_state=sa.get("d_state",16), expand=sa.get("expand",2),
               chunk_size=sa.get("scan_chunk",64), use_checkpoint=False).cuda().eval()
model.load_state_dict(ckpt["model"])

n = names[0]
b = np.asarray(Image.open(os.path.join(bdir,n)).convert("RGB"))
s = np.asarray(Image.open(os.path.join(sdir,n)).convert("RGB"))
bt = torch.from_numpy(b.transpose(2,0,1)).float().div(255).unsqueeze(0).cuda()
st = torch.from_numpy(s.transpose(2,0,1)).float().div(255).unsqueeze(0).cuda()
with torch.no_grad():
    out = model(bt)
print("\nMODEL CHECK on", n)
print("  input  range", round(bt.min().item(),3), "-", round(bt.max().item(),3))
print("  output range", round(out.min().item(),3), "-", round(out.max().item(),3))
print("  target range", round(st.min().item(),3), "-", round(st.max().item(),3))
outc = out.clamp(0,1)
mse_out = ((outc-st)**2).mean().item()
mse_in  = ((bt-st)**2).mean().item()
print("  PSNR blurry-vs-sharp :", round(10*np.log10(1/mse_in),2), "dB")
print("  PSNR output-vs-sharp :", round(10*np.log10(1/mse_out),2), "dB")
print("  mean |out-target|    :", round((outc-st).abs().mean().item(),4))
print("  mean |blurry-target| :", round((bt-st).abs().mean().item(),4))
PYEOF
  '
