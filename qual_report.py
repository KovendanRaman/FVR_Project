"""Qualitative [input | FVR | ground-truth] triptychs for the report.
Uses the composite GoPro model for GoPro + RealBlur, plus the combined model."""
import os, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from models.fvr_net import FVRNet
DEVICE = "cuda"

def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float().div_(255.)

def save(t, path):
    a = t.clamp(0,1).mul(255).byte().cpu().numpy().transpose(1,2,0)
    Image.fromarray(a).save(path)

@torch.no_grad()
def tiled(model, img, tile=128, overlap=32):
    c,h,w = img.shape; stride=tile-overlap
    ph=(tile-h%tile)%tile if h<tile else (stride-(h-tile)%stride)%stride
    pw=(tile-w%tile)%tile if w<tile else (stride-(w-tile)%stride)%stride
    imp=F.pad(img.unsqueeze(0),(0,pw,0,ph),mode="reflect").squeeze(0)
    _,H,W=imp.shape
    def hann(n):
        if n<=1: return torch.ones(n,device=DEVICE)
        return 0.5-0.5*torch.cos(2*np.pi*torch.arange(n,device=DEVICE)/(n-1))
    blend=torch.clamp(torch.outer(hann(tile),hann(tile)),min=1e-3).unsqueeze(0)
    out=torch.zeros(c,H,W,device=DEVICE); acc=torch.zeros(1,H,W,device=DEVICE)
    ys=list(range(0,H-tile+1,stride)) or [0]; xs=list(range(0,W-tile+1,stride)) or [0]
    if ys[-1]!=H-tile: ys.append(H-tile)
    if xs[-1]!=W-tile: xs.append(W-tile)
    for y in ys:
        for x in xs:
            p=imp[:,y:y+tile,x:x+tile].unsqueeze(0)
            pr=model(p).squeeze(0).clamp(0,1)
            out[:,y:y+tile,x:x+tile]+=pr*blend; acc[:,y:y+tile,x:x+tile]+=blend
    return (out/acc.clamp(min=1e-6))[:,:h,:w]

def build_model(ckpt_path):
    ck=torch.load(ckpt_path,map_location=DEVICE); sa=ck.get("args",{})
    m=FVRNet(embed_dim=sa.get("embed_dim",64),depth=sa.get("depth",4),
             d_state=sa.get("d_state",16),expand=sa.get("expand",2),
             chunk_size=sa.get("scan_chunk",64),use_checkpoint=False).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval(); return m

def strip(panels, path):
    c,h,w = panels[0].shape; gap=8
    canvas=torch.ones(3,h,w*len(panels)+gap*(len(panels)-1),device=panels[0].device)
    x=0
    for p in panels:
        canvas[:,:,x:x+w]=p; x+=w+gap
    save(canvas,path)

# name, checkpoint, blurry_dir, sharp_dir, n_samples
jobs = [
    ("gopro_composite", "./checkpoints_gopro_freq/best.pth", "./data/processed/gopro/test/blurry",   "./data/processed/gopro/test/sharp",   6),
    ("realblur",        "./checkpoints_gopro_freq/best.pth", "./data/processed/realblur/test/blurry","./data/processed/realblur/test/sharp",6),
    ("combined_gopro",  "./checkpoints_combined/best.pth",   "./data/processed/gopro/test/blurry",   "./data/processed/gopro/test/sharp",   6),
    ("combined_lolblur","./checkpoints_combined/best.pth",   "./data/processed/test/blurry",         "./data/processed/test/sharp",         6),
]

os.makedirs("qualitative_report", exist_ok=True)
for name, ckpt, bdir, sdir, n in jobs:
    if not os.path.exists(ckpt):
        print(f"SKIP {name}: checkpoint {ckpt} not found"); continue
    if not os.path.isdir(bdir):
        print(f"SKIP {name}: blurry dir {bdir} not found"); continue
    print(f"\n=== {name} ({ckpt}) ===")
    m = build_model(ckpt)
    files = sorted(f for f in os.listdir(bdir) if f.lower().endswith((".png",".jpg",".jpeg")))
    idxs = np.linspace(0, len(files)-1, n).astype(int)
    outdir = f"qualitative_report/{name}"; os.makedirs(outdir, exist_ok=True)
    for k,i in enumerate(idxs):
        fn = files[i]
        img = load(os.path.join(bdir, fn)).to(DEVICE)
        res = tiled(m, img)
        panels = [img, res]
        sp = os.path.join(sdir, fn)
        if sdir and os.path.exists(sp):
            gt = load(sp).to(DEVICE)
            if gt.shape == img.shape: panels.append(gt)
        strip(panels, f"{outdir}/{name}_{k+1}_{fn}")
        print(f"  {fn}  ({img.shape[2]}x{img.shape[1]})  panels={len(panels)}")
    del m; torch.cuda.empty_cache()
print("\nDone -> ./qualitative_report/  (panels: input | FVR | GT)")
