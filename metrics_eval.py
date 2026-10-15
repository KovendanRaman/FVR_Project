"""
Image-quality evaluation for FVR (RQ2).

Loads a trained checkpoint and computes PSNR / SSIM (and NIQE if pyiqa +
torchvision are available) on a paired blurry/sharp test set, at full image
resolution.

PSNR/SSIM are implemented in torch + numpy so the metric path has the SAME
dependency footprint as training (no torchvision required). NIQE is optional
and only runs if pyiqa imports cleanly; if torchvision is missing it is
skipped with a warning rather than crashing the whole run.

Usage:
  python -u metrics_eval.py \
      --blurry-dir ./data/processed/test/blurry \
      --sharp-dir  ./data/processed/test/sharp \
      --checkpoint ./checkpoints/best.pth \
      [--depth 4 --embed-dim 64 --d-state 16] \
      [--max-side 1024]   # downscale-guard: skip/cap huge images to avoid OOM
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image

from models.fvr_net import FVRNet

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_pairs(blurry_dir, sharp_dir):
    """Match files by sorted basename, tolerant to differing extensions."""
    def index(d):
        out = {}
        for f in os.listdir(d):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                out[os.path.splitext(f)[0]] = os.path.join(d, f)
        return out

    b = index(blurry_dir)
    s = index(sharp_dir)
    common = sorted(set(b) & set(s))
    if not common:
        raise RuntimeError(
            f"No matching basenames between {blurry_dir} and {sharp_dir}. "
            f"blurry has {len(b)}, sharp has {len(s)}."
        )
    missing = (set(b) | set(s)) - set(common)
    if missing:
        print(f"  warning: {len(missing)} unpaired files ignored")
    return [(b[k], s[k]) for k in common]


def load_image(path):
    """-> float32 CHW tensor in [0,1], RGB."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return torch.from_numpy(arr).float().div_(255.0)


def psnr(pred, target, eps=1e-10):
    """pred/target: CHW in [0,1]. Returns dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, size, size).contiguous()


def ssim(pred, target, window=None, size=11):
    """
    pred/target: CHW in [0,1]. Standard Gaussian-windowed SSIM,
    averaged over channels. Implemented in torch (no torchvision/skimage).
    """
    import torch.nn.functional as F

    device = pred.device
    c = pred.shape[0]
    if window is None:
        window = _gaussian_window(size, 1.5, c, device)
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    pad = size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=c)
    mu2 = F.conv2d(target, window, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=c) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def maybe_load_niqe(device):
    """Return a callable niqe(pred_chw)->float, or None if unavailable."""
    try:
        import pyiqa  # noqa: F401
        metric = pyiqa.create_metric("niqe", device=device)
        def _niqe(pred):
            with torch.no_grad():
                return metric(pred.unsqueeze(0)).item()
        print("  NIQE: enabled (pyiqa loaded)")
        return _niqe
    except Exception as e:
        print(f"  NIQE: skipped ({type(e).__name__}: {e})")
        print("        (install torchvision --no-deps to enable)")
        return None


def cap_resolution(t, max_side):
    """If the longer side exceeds max_side, center-crop to a multiple-of-8 box.
    Cropping (not resizing) preserves the blur kernel scale."""
    if max_side is None:
        return t
    c, h, w = t.shape
    if max(h, w) <= max_side:
        return t
    new_h = min(h, max_side)
    new_w = min(w, max_side)
    new_h -= new_h % 8
    new_w -= new_w % 8
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return t[:, top:top + new_h, left:left + new_w]


def pad_to_multiple(t, m=8):
    """Reflect-pad CHW so H,W are multiples of m (some blocks need it)."""
    import torch.nn.functional as F
    c, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return t, (h, w)
    t = F.pad(t.unsqueeze(0), (0, pw, 0, ph), mode="reflect").squeeze(0)
    return t, (h, w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blurry-dir", required=True)
    p.add_argument("--sharp-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--max-side", type=int, default=1024,
                   help="center-crop images whose longer side exceeds this "
                        "(guards against selective-scan OOM at full res)")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N pairs (debug)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    # Prefer the exact model geometry the checkpoint was trained with.
    saved = ckpt.get("args", {})
    embed_dim = saved.get("embed_dim", args.embed_dim)
    depth = saved.get("depth", args.depth)
    d_state = saved.get("d_state", args.d_state)
    expand = saved.get("expand", args.expand)
    chunk_size = saved.get("scan_chunk", args.chunk_size)
    print(f"Model: embed_dim={embed_dim} depth={depth} d_state={d_state} "
          f"expand={expand} chunk_size={chunk_size}")
    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']} | best_loss: "
              f"{ckpt.get('best_loss', 'n/a')}")

    model = FVRNet(embed_dim=embed_dim, depth=depth, d_state=d_state,
                   expand=expand, chunk_size=chunk_size,
                   use_checkpoint=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pairs = list_pairs(args.blurry_dir, args.sharp_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Evaluating {len(pairs)} image pairs "
          f"(max-side cap = {args.max_side})\n")

    niqe = maybe_load_niqe(device)

    psnrs, ssims, niqes = [], [], []
    ssim_window = None
    for i, (bpath, spath) in enumerate(pairs, 1):
        blurry = cap_resolution(load_image(bpath), args.max_side).to(device)
        sharp = cap_resolution(load_image(spath), args.max_side).to(device)

        inp, (h0, w0) = pad_to_multiple(blurry, 8)
        with torch.no_grad():
            out = model(inp.unsqueeze(0)).squeeze(0)
        out = out[:, :h0, :w0].clamp(0, 1)
        sharp = sharp[:, :h0, :w0]

        if ssim_window is None:
            ssim_window = _gaussian_window(11, 1.5, out.shape[0], device)

        psnrs.append(psnr(out, sharp))
        ssims.append(ssim(out, sharp, ssim_window))
        if niqe is not None:
            try:
                niqes.append(niqe(out))
            except Exception:
                pass

        if i % 25 == 0 or i == len(pairs):
            msg = (f"  [{i}/{len(pairs)}] "
                   f"PSNR {np.mean(psnrs):.3f} dB | SSIM {np.mean(ssims):.4f}")
            if niqes:
                msg += f" | NIQE {np.mean(niqes):.4f}"
            print(msg)

    print("\n" + "=" * 50)
    print(" FVR EVALUATION RESULTS")
    print("=" * 50)
    print(f" Images evaluated : {len(psnrs)}")
    print(f" PSNR (dB)  ^      : {np.mean(psnrs):.4f}  (higher better)")
    print(f" SSIM       ^      : {np.mean(ssims):.4f}  (higher better)")
    if niqes:
        print(f" NIQE       v      : {np.mean(niqes):.4f}  (lower better)")
    else:
        print(f" NIQE              : skipped (pyiqa/torchvision unavailable)")
    print("=" * 50)


if __name__ == "__main__":
    main()
