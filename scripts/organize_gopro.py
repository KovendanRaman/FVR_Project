#!/usr/bin/env python3
"""Organize GoPro (GOPRO_Large) into processed/gopro/{train,test}/{blurry,sharp}/ via symlinks.

GoPro raw structure (after unzip):
    GOPRO_Large/train/<scene>/blur_gamma/<frame>.png   -> blurry input
    GOPRO_Large/train/<scene>/sharp/<frame>.png        -> sharp target
    GOPRO_Large/test/<scene>/blur_gamma/...            (same for test)

We use blur_gamma (gamma-corrected) blur, which is the version most deblurring
papers report on. If you want the linear-CRF version instead, change BLUR_SUBDIR
to "blur".
"""
import os
from pathlib import Path

# Point RAW at wherever GOPRO_Large unzipped. Adjust if your path differs.
RAW = Path.home() / "FVR_Project/data/raw/gopro/GOPRO_Large"
OUT = Path.home() / "FVR_Project/data/processed/gopro"
BLUR_SUBDIR = "blur_gamma"   # or "blur" for linear-CRF
SHARP_SUBDIR = "sharp"

def organize(split):
    in_root = RAW / split
    out_blurry = OUT / split / "blurry"
    out_sharp = OUT / split / "sharp"
    out_blurry.mkdir(parents=True, exist_ok=True)
    out_sharp.mkdir(parents=True, exist_ok=True)

    if not in_root.is_dir():
        print(f"[{split}] MISSING: {in_root} -- check RAW path / unzip")
        return 0

    pairs, missing = 0, 0
    for scene in sorted(os.listdir(in_root)):
        blur_dir = in_root / scene / BLUR_SUBDIR
        sharp_dir = in_root / scene / SHARP_SUBDIR
        if not blur_dir.is_dir() or not sharp_dir.is_dir():
            continue
        for frame in sorted(os.listdir(blur_dir)):
            src_blur = blur_dir / frame
            src_sharp = sharp_dir / frame
            if not src_sharp.exists():
                missing += 1
                continue
            name = f"{scene}_{frame}"          # unique flat name
            for link, target in ((out_blurry / name, src_blur),
                                 (out_sharp / name, src_sharp)):
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target.resolve())
            pairs += 1
    print(f"[{split}] paired: {pairs}   missing-target: {missing}")
    return pairs

if __name__ == "__main__":
    total = 0
    for split in ("train", "test"):
        total += organize(split)
    print(f"TOTAL GoPro pairs: {total}")
    for split in ("train", "test"):
        bd, sd = OUT / split / "blurry", OUT / split / "sharp"
        if bd.exists():
            b, s = len(list(bd.iterdir())), len(list(sd.iterdir()))
            print(f"[{split}] blurry={b} sharp={s}  {'OK' if b == s else 'MISMATCH!!'}")
