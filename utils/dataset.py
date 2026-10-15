import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Deliberately no torchvision import. The cluster environment pins a torch
# nightly for sm_120 (Blackwell) and no matching torchvision build exists, so
# the training path is kept to torch + PIL + numpy only. torchvision is still
# needed by pyiqa for the NIQE metric, but that is evaluation-only.

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def to_tensor(array):
    # uint8 HWC -> float32 CHW in [0, 1]
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array).to(torch.float32).div_(255.0)


def reflect_pad_to(array, height, width):
    # np.pad's reflect mode cannot pad by more than the axis length, so grow
    # repeatedly for sources much smaller than the requested patch.
    while array.shape[0] < height or array.shape[1] < width:
        pad_h = min(max(0, height - array.shape[0]), array.shape[0] - 1)
        pad_w = min(max(0, width - array.shape[1]), array.shape[1] - 1)
        if pad_h == 0 and pad_w == 0:
            break
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return array


class FVRDataset(Dataset):
    # Paired blurry/sharp loader.
    #
    # Uses random *crops*, not resizing: rescaling an image rescales the blur
    # kernel with it, so a network trained on resized data never sees the
    # kernel sizes it will meet at test time. Aspect ratio is preserved for
    # the same reason.

    def __init__(self, blurry_dir, sharp_dir, patch_size=256, augment=True,
                 filenames=None):
        super().__init__()
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.patch_size = patch_size
        self.augment = augment

        if filenames is not None:
            self.image_filenames = list(filenames)
        else:
            self.image_filenames = self._paired_filenames(blurry_dir, sharp_dir)

        if not self.image_filenames:
            raise RuntimeError(
                f"No paired images found between '{blurry_dir}' and '{sharp_dir}'")

    @staticmethod
    def _paired_filenames(blurry_dir, sharp_dir):
        for directory in (blurry_dir, sharp_dir):
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Directory does not exist: {directory}")

        blurry = {f for f in os.listdir(blurry_dir)
                  if f.lower().endswith(IMAGE_EXTENSIONS)}
        sharp = {f for f in os.listdir(sharp_dir)
                 if f.lower().endswith(IMAGE_EXTENSIONS)}

        # Fail loudly on unpaired files rather than silently training the model
        # to map one scene onto another.
        missing = blurry - sharp
        if missing:
            sample = sorted(missing)[:5]
            raise RuntimeError(
                f"{len(missing)} blurry image(s) have no matching sharp file, "
                f"e.g. {sample}")

        return sorted(blurry & sharp)

    def __len__(self):
        return len(self.image_filenames)

    def _paired_crop(self, blurry, sharp):
        size = self.patch_size
        blurry = reflect_pad_to(blurry, size, size)
        sharp = reflect_pad_to(sharp, size, size)

        top = random.randint(0, blurry.shape[0] - size)
        left = random.randint(0, blurry.shape[1] - size)
        return (blurry[top:top + size, left:left + size],
                sharp[top:top + size, left:left + size])

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]

        blurry = np.asarray(
            Image.open(os.path.join(self.blurry_dir, img_name)).convert('RGB'))
        sharp = np.asarray(
            Image.open(os.path.join(self.sharp_dir, img_name)).convert('RGB'))

        if blurry.shape != sharp.shape:
            raise RuntimeError(
                f"Size mismatch for '{img_name}': "
                f"blurry {blurry.shape} vs sharp {sharp.shape}")

        if self.patch_size:
            blurry, sharp = self._paired_crop(blurry, sharp)

        # geometric augmentation must be applied identically to both images
        if self.augment:
            if random.random() < 0.5:
                blurry, sharp = blurry[:, ::-1], sharp[:, ::-1]
            if random.random() < 0.5:
                blurry, sharp = blurry[::-1], sharp[::-1]
            k = random.randint(0, 3)
            if k:
                blurry, sharp = np.rot90(blurry, k), np.rot90(sharp, k)

        return to_tensor(blurry), to_tensor(sharp)


def split_dataset(blurry_dir, sharp_dir, patch_size=256, val_fraction=0.05, seed=42):
    # Deterministic filename-level split, so validation images are never seen
    # in training regardless of how the loaders are shuffled.
    names = FVRDataset._paired_filenames(blurry_dir, sharp_dir)
    rng = random.Random(seed)
    rng.shuffle(names)

    n_val = max(1, int(len(names) * val_fraction)) if len(names) > 1 else 0
    val_names, train_names = names[:n_val], names[n_val:]

    train = FVRDataset(blurry_dir, sharp_dir, patch_size, True, train_names)
    val = FVRDataset(blurry_dir, sharp_dir, patch_size, False, val_names) \
        if val_names else None
    return train, val


if __name__ == "__main__":
    import sys
    import tempfile

    print("Dependencies:", "torch, numpy, PIL (no torchvision)")
    assert "torchvision" not in sys.modules, "torchvision leaked into the import graph"

    with tempfile.TemporaryDirectory() as tmp:
        blurry_dir = os.path.join(tmp, "blurry")
        sharp_dir = os.path.join(tmp, "sharp")
        os.makedirs(blurry_dir)
        os.makedirs(sharp_dir)

        rng = np.random.default_rng(0)
        for i in range(4):
            # 90x110 source is deliberately smaller than the 128 patch, to
            # exercise the reflect-padding path
            arr = rng.integers(0, 255, (90, 110, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(sharp_dir, f"img{i}.png"))
            Image.fromarray(np.roll(arr, 3, axis=1)).save(
                os.path.join(blurry_dir, f"img{i}.png"))

        train, val = split_dataset(blurry_dir, sharp_dir, patch_size=128)
        blurry, sharp = train[0]
        print(f"train/val split:   {len(train)} / {len(val)}")
        print(f"blurry tensor:     {tuple(blurry.shape)} {blurry.dtype} "
              f"range [{blurry.min():.3f}, {blurry.max():.3f}]")
        assert blurry.shape == (3, 128, 128) and sharp.shape == (3, 128, 128)
        assert 0.0 <= blurry.min() and blurry.max() <= 1.0
        print("Dataset loader OK (paired crop + flips + rot90, torchvision-free).")
