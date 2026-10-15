"""
Image-quality evaluation for FVR (RQ2) -- PATCH-BASED (tiled) inference.

Why tiled: FVR uses a selective scan (Mamba) whose hidden state accumulates
over the *entire* flattened token sequence. The model was trained on 128px
crops (~16k tokens). Running a full 1120x640 frame in one shot flattens to
~700k tokens -- a sequence-length regime the model never saw, so its output
degrades badly. Standard practice for patch-trained restoration models is
sliding-window inference: tile the image into training-size patches (with
overlap to avoid seam artifacts), restore each tile, and blend back together.
Metrics are then computed on the full stitched image.

PSNR/SSIM are torch+numpy (no torchvision). NIQE optional via pyiqa.

Usage:
  python -u metrics_eval.py \
      --blurry-dir ./data/processed/test/blurry \
      --sharp-dir  ./data/processed/test/sharp \
      --checkpoint ./checkpoints/best.pth \
      [--tile 128 --overlap 32]
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.fvr_net import FVRNet

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_pairs(blurry_dir, sharp_dir):
    def index(d):
        out = {}
        for f in os.listdir(d):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                out[os.path.splitext(f)[0]] = os.path.join(d, f)
        return out
    b, s = index(blurry_dir), index(sharp_dir)
    common = sorted(set(b) & set(s))
    if not common:
        raise RuntimeError(f"No matching basenames. blurry={len(b)} sharp={len(s)}")
    unpaired = (set(b) | set(s)) - set(common)
    if unpaired:
        print(f"  warning: {len(unpaired)} unpaired files ignored")
    return [(b[k], s[k]) for k in common]


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))
    return torch.from_numpy(arr).float().div_(255.0)


def psnr(pred, target, eps=1e-10):
    mse = torch.mean((pred - target) ** 2).item()
    return 100.0 if mse < eps else 10.0 * np.log10(1.0 / mse)


def _gaussian_window(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w2d.expand(channels, 1, size, size).contiguous()


def ssim(pred, target, window, size=11):
    c = pred.shape[0]
    pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    pad = size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=c)
    mu2 = F.conv2d(target, window, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(target * target, window, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(pred * target, window, padding=pad, groups=c) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return m.mean().item()


def maybe_load_niqe(device):
    try:
        import pyiqa
        metric = pyiqa.create_metric("niqe", device=device)
        print("  NIQE: enabled")
        return lambda p: metric(p.unsqueeze(0)).item()
    except Exception as e:
        print(f"  NIQE: skipped ({type(e).__name__})")
        return None


@torch.no_grad()
def tiled_restore(model, img, tile, overlap, device):
    """
    Sliding-window inference matching the training patch size.
    img: CHW in [0,1] on device. Returns CHW in [0,1].
    Overlapping tiles are blended with a cosine (Hann) window to remove seams.
    """
    c, h, w = img.shape
    stride = tile - overlap
    # pad so the last tile fits
    ph = (tile - h % tile) % tile if h < tile else (stride - (h - tile) % stride) % stride
    pw = (tile - w % tile) % tile if w < tile else (stride - (w - tile) % stride) % stride
    imp = F.pad(img.unsqueeze(0), (0, pw, 0, ph), mode="reflect").squeeze(0)
    _, H, W = imp.shape

    # 2D Hann blend weights
    def hann(n):
        if n <= 1:
            return torch.ones(n, device=device)
        return 0.5 - 0.5 * torch.cos(2 * np.pi * torch.arange(n, device=device) / (n - 1))
    wy, wx = hann(tile), hann(tile)
    blend = torch.clamp(torch.outer(wy, wx), min=1e-3).unsqueeze(0)  # 1,tile,tile

    out = torch.zeros(c, H, W, device=device)
    acc = torch.zeros(1, H, W, device=device)
    ys = list(range(0, H - tile + 1, stride)) or [0]
    xs = list(range(0, W - tile + 1, stride)) or [0]
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    for y in ys:
        for x in xs:
            patch = imp[:, y:y + tile, x:x + tile].unsqueeze(0)
            pred = model(patch).squeeze(0).clamp(0, 1)
            out[:, y:y + tile, x:x + tile] += pred * blend
            acc[:, y:y + tile, x:x + tile] += blend
    out = out / acc.clamp(min=1e-6)
    return out[:, :h, :w]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blurry-dir", required=True)
    p.add_argument("--sharp-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tile", type=int, default=128,
                   help="inference patch size; match training patch-size")
    p.add_argument("--overlap", type=int, default=32,
                   help="tile overlap in px (blended to avoid seams)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Tiled inference: tile={args.tile} overlap={args.overlap}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    sa = ckpt.get("args", {})
    model = FVRNet(embed_dim=sa.get("embed_dim", 64), depth=sa.get("depth", 4),
                   d_state=sa.get("d_state", 16), expand=sa.get("expand", 2),
                   chunk_size=sa.get("scan_chunk", 64),
                   use_checkpoint=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch','?')} | best_loss {ckpt.get('best_loss','?')}")

    pairs = list_pairs(args.blurry_dir, args.sharp_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Evaluating {len(pairs)} pairs\n")

    niqe = maybe_load_niqe(device)
    window = None
    psnrs, ssims, niqes = [], [], []
    for i, (bpath, spath) in enumerate(pairs, 1):
        blurry = load_image(bpath).to(device)
        sharp = load_image(spath).to(device)
        out = tiled_restore(model, blurry, args.tile, args.overlap, device)
        if window is None:
            window = _gaussian_window(11, 1.5, out.shape[0], device)
        psnrs.append(psnr(out, sharp))
        ssims.append(ssim(out, sharp, window))
        if niqe is not None:
            try:
                niqes.append(niqe(out))
            except Exception:
                pass
        if i % 25 == 0 or i == len(pairs):
            msg = f"  [{i}/{len(pairs)}] PSNR {np.mean(psnrs):.3f} | SSIM {np.mean(ssims):.4f}"
            if niqes:
                msg += f" | NIQE {np.mean(niqes):.4f}"
            print(msg)

    print("\n" + "=" * 50)
    print(" FVR EVALUATION RESULTS (tiled)")
    print("=" * 50)
    print(f" Images     : {len(psnrs)}")
    print(f" PSNR (dB)  : {np.mean(psnrs):.4f}  (higher better)")
    print(f" SSIM       : {np.mean(ssims):.4f}  (higher better)")
    print(f" NIQE       : {np.mean(niqes):.4f}  (lower better)" if niqes
          else " NIQE       : skipped")
    print("=" * 50)


if __name__ == "__main__":
    main()
