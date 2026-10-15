#!/usr/bin/env python3
"""O(N) complexity validation for FVR (RQ1 / RQ3).

Measures how compute (FLOPs) and inference latency scale with image resolution
for FVRNet (selective scan, expected O(N)) versus a dense-attention Transformer
baseline (expected O(N^2), N = H*W).

Because full-resolution attention cannot be instantiated (the L x L score matrix
is ~17 GB at 256x256), the transformer is:
  - measured empirically at low resolutions (where it fits), and
  - extended analytically via baselines.transformer_flops at all resolutions,
    with the analytic formula validated against measurement where they overlap.

FVRNet is measured empirically across the full sweep.

Outputs:
  - a printed table of resolution / N / FLOPs / latency for both models
  - on_scaling.csv   (raw numbers for your results chapter)
  - on_scaling.png   (log-log FLOPs-vs-N plot: the RQ1 figure)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.fvr_net import FVRNet
from models.baselines import (
    TransformerRestorer, transformer_flops, attention_flops_only,
)

try:
    from thop import profile as thop_profile
    HAVE_THOP = True
except Exception:
    HAVE_THOP = False


def measure_flops(model, x):
    """Measured FLOPs via thop (returns MACs*2 ~= FLOPs). None if thop absent."""
    if not HAVE_THOP:
        return None
    macs, _ = thop_profile(model, inputs=(x,), verbose=False)
    return 2.0 * macs


@torch.no_grad()
def measure_latency(model, x, warmup=3, iters=10):
    """Median forward latency in milliseconds."""
    dev = x.device
    for _ in range(warmup):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def try_build_and_measure(build_fn, x, label):
    """Instantiate + measure a model, catching OOM so the sweep continues."""
    try:
        model = build_fn().to(x.device).eval()
        flops = measure_flops(model, x)
        lat = measure_latency(model, x)
        del model
        if x.device.type == "cuda":
            torch.cuda.empty_cache()
        return flops, lat
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if x.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"    {label}: OOM at this resolution (expected for attention) — analytic only")
            return None, None
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 96, 128, 160, 192, 224, 256, 384, 512])
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--attn-max-size", type=int, default=160,
                    help="don't even try to instantiate the transformer above this HxW")
    ap.add_argument("--out-prefix", default="on_scaling")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"Device: {dev} | thop available: {HAVE_THOP}")
    print(f"FVR: embed_dim={args.embed_dim} depth={args.depth}\n")

    rows = []
    header = ("HxW", "N=H*W", "fvr_GFLOPs", "fvr_ms",
              "tf_meas_GFLOPs", "tf_meas_ms", "tf_analytic_GFLOPs")
    print("{:>8} {:>10} {:>12} {:>10} {:>15} {:>11} {:>18}".format(*header))

    for s in args.sizes:
        N = s * s
        x = torch.randn(1, 3, s, s, device=dev)

        # FVRNet — measured everywhere. use_checkpoint off for clean latency.
        fvr_flops, fvr_ms = try_build_and_measure(
            lambda: FVRNet(embed_dim=args.embed_dim, depth=args.depth,
                           use_checkpoint=False),
            x, "FVR")

        # Transformer — measured only at small sizes; always compute analytic.
        tf_flops = tf_ms = None
        if s <= args.attn_max_size:
            tf_flops, tf_ms = try_build_and_measure(
                lambda: TransformerRestorer(embed_dim=args.embed_dim, depth=args.depth),
                x, "Transformer")
        tf_analytic = transformer_flops(s, s, embed_dim=args.embed_dim, depth=args.depth)

        rows.append({
            "size": s, "N": N,
            "fvr_flops": fvr_flops, "fvr_ms": fvr_ms,
            "tf_meas_flops": tf_flops, "tf_meas_ms": tf_ms,
            "tf_analytic_flops": tf_analytic,
        })

        def g(v):  # to GFLOPs string
            return f"{v/1e9:.3f}" if v is not None else "-"
        def m(v):
            return f"{v:.2f}" if v is not None else "-"
        print("{:>8} {:>10} {:>12} {:>10} {:>15} {:>11} {:>18}".format(
            f"{s}x{s}", N, g(fvr_flops), m(fvr_ms),
            g(tf_flops), m(tf_ms), g(tf_analytic)))

    # ---- validate analytic transformer FLOPs vs measured, where both exist ----
    print("\nAnalytic-vs-measured transformer FLOPs check (should be close):")
    for r in rows:
        if r["tf_meas_flops"]:
            ratio = r["tf_analytic_flops"] / r["tf_meas_flops"]
            print(f"  {r['size']}x{r['size']}: analytic/measured = {ratio:.2f}")

    # ---- write CSV ----
    csv_path = f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "N", "fvr_flops", "fvr_ms",
                    "tf_measured_flops", "tf_measured_ms", "tf_analytic_flops"])
        for r in rows:
            w.writerow([r["size"], r["N"], r["fvr_flops"], r["fvr_ms"],
                        r["tf_meas_flops"], r["tf_meas_ms"], r["tf_analytic_flops"]])
    print(f"\nWrote {csv_path}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        Ns = np.array([r["N"] for r in rows])
        fvr = np.array([r["fvr_flops"] for r in rows], dtype=float)
        tf_an = np.array([r["tf_analytic_flops"] for r in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(Ns, fvr / 1e9, "o-", color="#1f77b4", label="FVR (selective scan)")
        ax.loglog(Ns, tf_an / 1e9, "s--", color="#d62728",
                  label="Transformer (analytic, O(N²))")
        # reference slopes
        ax.loglog(Ns, (fvr[0]/1e9) * (Ns/Ns[0]), ":", color="gray", alpha=0.6,
                  label="slope 1 (linear)")
        ax.loglog(Ns, (tf_an[0]/1e9) * (Ns/Ns[0])**2, ":", color="firebrick", alpha=0.4,
                  label="slope 2 (quadratic)")
        ax.set_xlabel("N = H × W  (tokens)")
        ax.set_ylabel("GFLOPs")
        ax.set_title("Computational scaling: FVR vs. dense attention")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}.png", dpi=150)
        print(f"Wrote {args.out_prefix}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
