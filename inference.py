"""
Restore a single blurry/low-light image using a trained FVR checkpoint.
Uses tiled inference (matching training patch size) for best results.

Usage:
  python inference.py --input photo.jpg --output restored.jpg
  python inference.py --input photo.jpg --output restored.jpg --checkpoint checkpoints/best.pth
  python inference.py --input-dir my_photos/ --output-dir results/
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.fvr_net import FVRNet

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))
    return torch.from_numpy(arr).float().div_(255.0)


def save_image(tensor, path):
    arr = tensor.clamp(0, 1).mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def tiled_restore(model, img, tile, overlap, device):
    c, h, w = img.shape
    stride = tile - overlap
    ph = (tile - h % tile) % tile if h < tile else (stride - (h - tile) % stride) % stride
    pw = (tile - w % tile) % tile if w < tile else (stride - (w - tile) % stride) % stride
    imp = F.pad(img.unsqueeze(0), (0, pw, 0, ph), mode="reflect").squeeze(0)
    _, H, W = imp.shape

    def hann(n):
        if n <= 1:
            return torch.ones(n, device=device)
        return 0.5 - 0.5 * torch.cos(2 * np.pi * torch.arange(n, device=device) / (n - 1))

    blend = torch.clamp(torch.outer(hann(tile), hann(tile)), min=1e-3).unsqueeze(0)
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
    p.add_argument("--input", help="single image path")
    p.add_argument("--input-dir", help="directory of images to restore")
    p.add_argument("--output", default="restored.png", help="output path (single image)")
    p.add_argument("--output-dir", default="results", help="output directory (batch)")
    p.add_argument("--checkpoint", default="./checkpoints/best.pth")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sa = ckpt.get("args", {})
    model = FVRNet(embed_dim=sa.get("embed_dim", 64), depth=sa.get("depth", 4),
                   d_state=sa.get("d_state", 16), expand=sa.get("expand", 2),
                   chunk_size=sa.get("scan_chunk", 64),
                   use_checkpoint=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Collect files
    if args.input:
        files = [(args.input, args.output)]
    elif args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        files = []
        for f in sorted(os.listdir(args.input_dir)):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                files.append((
                    os.path.join(args.input_dir, f),
                    os.path.join(args.output_dir, f)
                ))
    else:
        print("Provide --input or --input-dir")
        return

    for i, (inp, outp) in enumerate(files, 1):
        img = load_image(inp).to(device)
        restored = tiled_restore(model, img, args.tile, args.overlap, device)
        save_image(restored, outp)
        print(f"  [{i}/{len(files)}] {inp} -> {outp}")

    print("Done.")


if __name__ == "__main__":
    main()
