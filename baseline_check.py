import os, numpy as np
from PIL import Image
def psnr(a,b):
    m=((a.astype(np.float64)-b.astype(np.float64))**2).mean()
    return 99.0 if m==0 else 10*np.log10(255.0**2/m)
sets=[("LOL-Blur",  "data/processed/test/blurry",          "data/processed/test/sharp",          21.76),
      ("GoPro",     "data/processed/gopro/test/blurry",    "data/processed/gopro/test/sharp",    25.00),
      ("RealBlur-J","data/processed/realblur/test/blurry", "data/processed/realblur/test/sharp", 26.43)]
for name,bd,sd,fvr in sets:
    ps=[]
    for f in sorted(os.listdir(bd)):
        sp=os.path.join(sd,f)
        if not os.path.exists(sp): continue
        a=np.asarray(Image.open(os.path.join(bd,f)).convert("RGB"))
        b=np.asarray(Image.open(sp).convert("RGB"))
        if a.shape==b.shape: ps.append(psnr(a,b))
    base=float(np.mean(ps))
    print(f"{name:12s} identity = {base:6.2f} dB | FVR = {fvr:6.2f} dB | gain = {fvr-base:+5.2f} dB  ({len(ps)} imgs)", flush=True)
