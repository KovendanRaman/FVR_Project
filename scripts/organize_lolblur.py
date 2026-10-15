#!/usr/bin/env python3
"""Organize LOL-Blur into processed/{train,test}/{blurry,sharp}/ via symlinks.
Pairing: low_blur (input) -> high_sharp_scaled (target), matched by scene/frame.
"""
import os
from pathlib import Path

RAW = Path.home() / "FVR_Project/data/raw/lolblur"
OUT = Path.home() / "FVR_Project/data/processed"
INPUT_VARIANT = "low_blur"
TARGET_VARIANT = "high_sharp_scaled"

def organize(split):
    in_root = RAW / split / INPUT_VARIANT
    tgt_root = RAW / split / TARGET_VARIANT
    out_blurry = OUT / split / "blurry"
    out_sharp = OUT / split / "sharp"
    out_blurry.mkdir(parents=True, exist_ok=True)
    out_sharp.mkdir(parents=True, exist_ok=True)

    pairs, missing = 0, 0
    for scene in sorted(os.listdir(in_root)):
        scene_in = in_root / scene
        scene_tgt = tgt_root / scene
        if not scene_in.is_dir():
            continue
        for frame in sorted(os.listdir(scene_in)):
            src_blur = scene_in / frame
            src_sharp = scene_tgt / frame
            if not src_sharp.exists():
                missing += 1
                continue
            name = f"{scene}_{frame}"           # e.g. 0242_0010.png -> unique, flat
            link_blur = out_blurry / name
            link_sharp = out_sharp / name
            for link, target in ((link_blur, src_blur), (link_sharp, src_sharp)):
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
    print(f"TOTAL pairs: {total}")
    # sanity: counts must match between blurry and sharp
    for split in ("train", "test"):
        b = len(list((OUT / split / "blurry").iterdir()))
        s = len(list((OUT / split / "sharp").iterdir()))
        print(f"[{split}] blurry={b} sharp={s}  {'OK' if b==s else 'MISMATCH!!'}")
