# Before/after candidates per model, with a change metric to pick the clearest.
import os, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from models.fvr_net import FVRNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float().div_(255.)

def save(t, p):
    a = t.clamp(0,1).mul(255).byte().cpu().numpy().transpose(1,2,0)
    Image.fromarray(a).save(p)

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

def build_model(ckpt):
    ck=torch.load(ckpt, map_location="cpu"); sa=ck.get("args",{})
    m=FVRNet(embed_dim=sa.get("embed_dim",64),depth=sa.get("depth",4),
             d_state=sa.get("d_state",16),expand=sa.get("expand",2),
             chunk_size=sa.get("scan_chunk",64),use_checkpoint=False).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval(); return m

def side_by_side(inp, out, p):
    c,h,w = inp.shape; gap=8
    canvas = torch.ones(3,h,w*2+gap,device=inp.device)
    canvas[:,:,:w]=inp; canvas[:,:,w+gap:]=out
    save(canvas, p)

jobs = [
    ("lolblur_model",   "./checkpoints/best.pth",          "./data/processed/test/blurry"),
    ("gopro_model",     "./checkpoints_gopro/best.pth",    "./data/processed/gopro/test/blurry"),
    ("combined_model",  "./checkpoints_combined/best.pth", "./data/processed/test/blurry"),
    ("realblur_gopro",  "./checkpoints_gopro/best.pth",    "./data/processed/realblur/test/blurry"),
]

os.makedirs("qualitative_paper", exist_ok=True)
for name, ckpt, bdir in jobs:
    print(f"\n=== {name} ===")
    m = build_model(ckpt)
    files = sorted(f for f in os.listdir(bdir) if f.lower().endswith((".png",".jpg",".jpeg")))
    for k, frac in enumerate(FRACS):
        i = int(frac*(len(files)-1))
        img = load(os.path.join(bdir, files[i])).to(DEVICE)
        out = tiled(m, img)
        delta = (out-img).abs().mean().item()*255
        side_by_side(img, out, f"qualitative_paper/{name}_{k+1}.png")
        print(f"  cand {k+1}: {files[i]:40s}  mean|change| = {delta:5.1f}")
    del m; torch.cuda.empty_cache()
print("\nDone -> ./qualitative_paper/")
