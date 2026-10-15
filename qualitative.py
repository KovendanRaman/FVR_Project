"""
Generate side-by-side before/after qualitative figures for each model.
Picks a few sample images per dataset, restores them (tiled), and saves
a stacked [input | output] comparison PNG per sample, plus a contact sheet.
"""
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

def side_by_side(inp, out, path):
    # inp,out are CHW tensors [0,1]; stack horizontally with a thin divider
    c,h,w = inp.shape
    gap = 8
    canvas = torch.ones(3, h, w*2+gap, device=inp.device)
    canvas[:, :, :w] = inp
    canvas[:, :, w+gap:] = out
    save(canvas, path)

jobs = [
    ("lolblur",  "./checkpoints/best.pth",        "./data/processed/test/blurry",            6),
    ("gopro",    "./checkpoints_gopro/best.pth",  "./data/processed/gopro/test/blurry",      6),
    ("realblur", "./checkpoints_gopro/best.pth",  "./data/processed/realblur/test/blurry",   6),
]

os.makedirs("qualitative", exist_ok=True)
for name, ckpt, bdir, n in jobs:
    print(f"\n=== {name} ===")
    m = build_model(ckpt)
    files = sorted(f for f in os.listdir(bdir) if f.lower().endswith((".png",".jpg",".jpeg")))
    # spread picks across the set rather than all-from-start
    idxs = np.linspace(0, len(files)-1, n).astype(int)
    outdir = f"qualitative/{name}"; os.makedirs(outdir, exist_ok=True)
    for k,i in enumerate(idxs):
        img = load(os.path.join(bdir, files[i])).to(DEVICE)
        res = tiled(m, img)
        side_by_side(img, res, f"{outdir}/{name}_{k+1}_{files[i]}")
        print(f"  {files[i]}  ({img.shape[2]}x{img.shape[1]})")
    del m; torch.cuda.empty_cache()
print("\nAll qualitative comparisons written to ./qualitative/")
