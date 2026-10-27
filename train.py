import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.fvr_net import FVRNet
from models.vmamba import SELECTIVE_SCAN_BACKEND
from utils.dataset import split_dataset
from utils.losses import RestorationLoss


def build_args():
    parser = argparse.ArgumentParser(description="Train the FVR restoration model")

    parser.add_argument("--blurry-dir", default="./data/raw/blurry")
    parser.add_argument("--sharp-dir", default="./data/raw/sharp")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.05)

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--scan-chunk", type=int, default=64)
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="disable activation checkpointing in the scan (faster, more memory)")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--loss", choices=["l1","combined"], default="l1")

    parser.add_argument("--out-dir", default="./checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, batches = 0.0, 0
    for blurry_imgs, sharp_imgs in loader:
        blurry_imgs = blurry_imgs.to(device, non_blocking=True)
        sharp_imgs = sharp_imgs.to(device, non_blocking=True)
        total += criterion(model(blurry_imgs), sharp_imgs).item()
        batches += 1
    return total / max(batches, 1)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_loss, args):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_loss": best_loss,
        "args": vars(args),
    }, path)


def train_fvr(args):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Training on device: {device}")
    print(f"Selective-scan backend: {SELECTIVE_SCAN_BACKEND}")
    if SELECTIVE_SCAN_BACKEND == "torch" and device.type == "cuda":
        print("  WARNING: the fused kernel is missing. Install it with "
              "`pip install mamba-ssm` before a full training run.")

    model = FVRNet(
        embed_dim=args.embed_dim,
        depth=args.depth,
        d_state=args.d_state,
        chunk_size=args.scan_chunk,
        use_checkpoint=not args.no_checkpoint,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    criterion = RestorationLoss() if args.loss == "combined" else nn.L1Loss()
    val_criterion = nn.L1Loss()

    print("Initializing Data Loaders...")
    train_dataset, val_dataset = split_dataset(
        args.blurry_dir, args.sharp_dir, args.patch_size, args.val_fraction)
    print(f"  train: {len(train_dataset)} images | "
          f"val: {len(val_dataset) if val_dataset else 0} images")

    loader_kwargs = dict(num_workers=args.num_workers,
                         pin_memory=device.type == "cuda",
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, **loader_kwargs) if val_dataset else None

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_start = 0.0, time.perf_counter()

        for blurry_imgs, sharp_imgs in train_loader:
            blurry_imgs = blurry_imgs.to(device, non_blocking=True)
            sharp_imgs = sharp_imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(blurry_imgs), sharp_imgs)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.perf_counter() - epoch_start

        message = (f"Epoch [{epoch + 1}/{args.epochs}] | Train L1: {avg_loss:.4f} | "
                   f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        monitored = avg_loss
        if val_loader is not None:
            val_loss = validate(model, val_loader, val_criterion, device)
            message += f" | Val L1: {val_loss:.4f}"
            monitored = val_loss
        print(message)

        save_checkpoint(os.path.join(args.out_dir, "last.pth"), model, optimizer,
                        scheduler, scaler, epoch, best_loss, args)
        if monitored < best_loss:
            best_loss = monitored
            save_checkpoint(os.path.join(args.out_dir, "best.pth"), model, optimizer,
                            scheduler, scaler, epoch, best_loss, args)
            print(f"  new best ({best_loss:.4f}) -> best.pth")

    print("Training Complete. Saving model weights...")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "fvr_weights.pth"))


if __name__ == "__main__":
    train_fvr(build_args())
