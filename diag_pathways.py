# Measures how hard each EDFFN pathway is working:
#   amp log-gain  -> brightening (illumination)
#   phase delta   -> deblurring (structure)
import os, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from models.fvr_net import FVRNet
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float().div_(255.)

def build_model(ckpt):
    ck=torch.load(ckpt,map_location="cpu"); sa=ck.get("args",{})
    m=FVRNet(embed_dim=sa.get("embed_dim",64),depth=sa.get("depth",4),
             d_state=sa.get("d_state",16),expand=sa.get("expand",2),
             chunk_size=sa.get("scan_chunk",64),use_checkpoint=False).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval(); return m

@torch.no_grad()
def measure(ckpt, bdir, n=20, patch=128):
    m=build_model(ckpt); amp=[]; pha=[]
    h=[]
    for st in m.stages:
        h.append(st.edffn.amp_net.register_forward_hook(lambda mod,i,o: amp.append(o.abs().mean().item())))
        h.append(st.edffn.pha_net.register_forward_hook(lambda mod,i,o: pha.append(o.abs().mean().item())))
    files=sorted(f for f in os.listdir(bdir) if f.lower().endswith((".png",".jpg",".jpeg")))
    for i in np.linspace(0,len(files)-1,n).astype(int):
        img=load(os.path.join(bdir,files[i])).to(DEVICE); c,hh,ww=img.shape
        t=(hh-patch)//2 if hh>=patch else 0; l=(ww-patch)//2 if ww>=patch else 0
        x=F.pad(img.unsqueeze(0),(0,max(0,patch-ww),0,max(0,patch-hh)),mode="reflect").squeeze(0)
        m(x[:,t:t+patch,l:l+patch].unsqueeze(0))
    for hk in h: hk.remove()
    return float(np.mean(amp)), float(np.mean(pha))

jobs=[("LOL-Blur model / LOL-Blur","./checkpoints/best.pth","./data/processed/test/blurry"),
      ("GoPro model / GoPro","./checkpoints_gopro/best.pth","./data/processed/gopro/test/blurry")]
print(f"{'setting':30s} {'|amp log-gain|':>15s} {'|phase delta|':>15s}")
for name,ck,bd in jobs:
    a,p=measure(ck,bd); print(f"{name:30s} {a:15.4f} {p:15.4f}", flush=True)
