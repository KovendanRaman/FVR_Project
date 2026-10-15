#!/usr/bin/env python3
"""Organize RealBlur into processed/realblur/{train,test}/{blurry,sharp}/ via symlinks.
Uses the datalists that ship WITH the extracted data. Each list line:
  <gt/sharp path>  <blur path>   (relative to the realblur raw root)
"""
import os
from pathlib import Path

RAW = Path.home() / "FVR_Project/data/raw/realblur"
OUT = Path.home() / "FVR_Project/data/processed/realblur"
SUBSET = "J"   # "J", "R", or "both"

LISTS = {
    "J": {"train": "RealBlur_J_train_list.txt", "test": "RealBlur_J_test_list.txt"},
    "R": {"train": "RealBlur_R_train_list.txt", "test": "RealBlur_R_test_list.txt"},
}

def resolve(rel):
    p = RAW / rel
    return p if p.exists() else None

def organize(subset, split, list_name):
    out_blurry = OUT / split / "blurry"
    out_sharp = OUT / split / "sharp"
    out_blurry.mkdir(parents=True, exist_ok=True)
    out_sharp.mkdir(parents=True, exist_ok=True)

    lst = RAW / list_name
    if not lst.exists():
        print(f"[{subset}/{split}] list not found: {lst}")
        return 0

    pairs, missing = 0, 0
    with open(lst) as f:
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) < 2:
                continue
            src_sharp = resolve(parts[0])   # gt first
            src_blur = resolve(parts[1])    # blur second
            if not src_sharp or not src_blur:
                missing += 1
                continue
            name = f"{subset}_{i:05d}.png"
            for link, target in ((out_blurry / name, src_blur),
                                 (out_sharp / name, src_sharp)):
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target.resolve())
            pairs += 1
    print(f"[{subset}/{split}] paired: {pairs}   missing: {missing}")
    return pairs

if __name__ == "__main__":
    subsets = ["J", "R"] if SUBSET == "both" else [SUBSET]
    total = 0
    for sub in subsets:
        for split, name in LISTS[sub].items():
            total += organize(sub, split, name)
    print(f"TOTAL RealBlur pairs: {total}")
    for split in ("train", "test"):
        bd = OUT / split / "blurry"
        if bd.exists():
            b = len(list(bd.iterdir()))
            s = len(list((OUT / split / "sharp").iterdir()))
            print(f"[{split}] blurry={b} sharp={s}  {'OK' if b==s else 'MISMATCH'}")
