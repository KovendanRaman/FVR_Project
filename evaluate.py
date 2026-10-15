import argparse
import math
import time

import torch
from thop import profile

from models.fvr_net import FVRNet
from models.vmamba import SS2D, SELECTIVE_SCAN_BACKEND
from models.baselines import (TransformerRestorer, transformer_flops,
                              attention_flops_only)


def fft_flops(channels, h, w):
    # A real-input 2D FFT of an H x W plane costs ~2.5 * N * log2(N) real FLOPs
    # (half the ~5 N log2 N of the complex transform). Note this is O(N log N),
    # not O(N) -- it is the only super-linear term left in FVR.
    n = h * w
    return 2.5 * channels * n * math.log2(max(n, 2))


def fvr_unhooked_flops(model, h, w):
    # thop only hooks nn.Module calls, so it counts convolutions and linears and
    # nothing else. Everything it misses is accounted for here explicitly:
    # the FFT pair, the selective-scan recurrence, and the two einsums that
    # produce (delta, B, C).
    length = h * w
    breakdown = {"fft": 0.0, "selective_scan": 0.0, "ssm_projections": 0.0}

    for stage in model.stages:
        c = stage.edffn.pconv2.out_channels
        # rfft2 in EDFFN + irfft2 in the fusion module
        breakdown["fft"] += 2 * fft_flops(c, h, w)
        # polar(): cos, sin and two multiplies over the half-spectrum
        breakdown["fft"] += 4 * c * h * (w // 2 + 1)

    for module in model.modules():
        if not isinstance(module, SS2D):
            continue
        k, d, n, r = module.K, module.d_inner, module.d_state, module.dt_rank
        # per direction / channel / state / token: exp, 3 multiplies for the
        # discretisation, one multiply-add for the recurrence, one for the readout
        breakdown["selective_scan"] += 9 * k * d * n * length
        breakdown["ssm_projections"] += 2 * k * length * d * (r + 2 * n)   # x_proj
        breakdown["ssm_projections"] += 2 * k * length * d * r             # dt_proj

    return breakdown


def measure_latency(model, resolution, device, iters=20, warmup=5):
    dummy = torch.randn(1, 3, *resolution, device=device)
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        sync()
        start = time.perf_counter()
        for _ in range(iters):
            model(dummy)
        sync()
        elapsed = time.perf_counter() - start

    return elapsed / iters * 1000


def profile_fvr(resolution, device, depth, embed_dim, scan_chunk=1024, measure_time=True):
    h, w = resolution
    model = FVRNet(embed_dim=embed_dim, depth=depth, chunk_size=scan_chunk,
                   use_checkpoint=False).to(device).eval()
    dummy = torch.randn(1, 3, h, w, device=device)

    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)

    hooked = macs * 2
    unhooked = fvr_unhooked_flops(model, h, w)

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "hooked": hooked,
        **unhooked,
        "total": hooked + sum(unhooked.values()),
        "latency": measure_latency(model, resolution, device) if measure_time else None,
    }


def validate_transformer_formula(device, embed_dim, depth, num_heads, resolution=(32, 32)):
    # Check the closed-form count against a real profiled run at a resolution
    # small enough for dense attention to actually fit in memory.
    h, w = resolution
    model = TransformerRestorer(embed_dim=embed_dim, depth=depth,
                                num_heads=num_heads).to(device).eval()
    dummy = torch.randn(1, 3, h, w, device=device)

    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)

    measured = macs * 2 + attention_flops_only(h, w, embed_dim, depth, num_heads)
    predicted = transformer_flops(h, w, embed_dim=embed_dim, depth=depth,
                                  num_heads=num_heads)
    params = sum(p.numel() for p in model.parameters())

    return measured, predicted, params


def scaling_exponent(low, high, pixel_ratio):
    return math.log(high / low) / math.log(pixel_ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--scan-chunk", type=int, default=1024,
                        help="scan chunk length; larger is faster but uses more memory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latency-max-res", type=int, default=512,
                        help="skip timing above this resolution")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | selective-scan backend: {SELECTIVE_SCAN_BACKEND}")
    if SELECTIVE_SCAN_BACKEND == "torch":
        print("  (pure-PyTorch scan: latency is NOT representative of a fused-kernel run)")

    measured, predicted, tf_params = validate_transformer_formula(
        device, args.embed_dim, args.depth, args.heads)
    err = abs(measured - predicted) / measured * 100
    print(f"\nTransformer FLOP formula check @32x32: "
          f"measured {measured / 1e9:.4f}G vs closed-form {predicted / 1e9:.4f}G "
          f"({err:.2f}% error)")
    print("  -> high-resolution transformer numbers below are extrapolated with "
          "this validated formula,\n     because dense attention at 256x256 needs "
          "~17 GB for the score matrix alone.")

    rows = []
    for res in args.resolutions:
        fvr = profile_fvr((res, res), device, args.depth, args.embed_dim,
                          scan_chunk=args.scan_chunk,
                          measure_time=res <= args.latency_max_res)
        fvr["transformer"] = transformer_flops(res, res, embed_dim=args.embed_dim,
                                               depth=args.depth, num_heads=args.heads)
        fvr["res"] = res
        rows.append(fvr)

        print("\n" + "=" * 62)
        print(f" FVR EFFICIENCY @ {res}x{res}")
        print("=" * 62)
        print(f"Parameters:              {fvr['params']:,}")
        print(f"  conv/linear (thop):    {fvr['hooked'] / 1e9:10.3f} G")
        print(f"  FFT pair (analytic):   {fvr['fft'] / 1e9:10.3f} G   [O(N log N)]")
        print(f"  selective scan:        {fvr['selective_scan'] / 1e9:10.3f} G   [O(N)]")
        print(f"  SSM projections:       {fvr['ssm_projections'] / 1e9:10.3f} G   [O(N)]")
        print(f"  TOTAL FLOPs:           {fvr['total'] / 1e9:10.3f} G")
        print(f"Transformer baseline:    {fvr['transformer'] / 1e9:10.3f} G "
              f"({fvr['transformer'] / fvr['total']:.1f}x FVR)")
        attn = attention_flops_only(res, res, args.embed_dim, args.depth, args.heads)
        print(f"  of which attention:    {attn / 1e9:10.3f} G "
              f"({100 * attn / fvr['transformer']:.1f}%)")
        if fvr["latency"] is not None:
            print(f"Avg latency:             {fvr['latency']:10.2f} ms / frame")
        else:
            print(f"Avg latency:                    skipped (> {args.latency_max_res}px)")

    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        ratio = (hi["res"] / lo["res"]) ** 2
        print("\n" + "=" * 62)
        print(f" SCALING EXPONENT  ({lo['res']}px -> {hi['res']}px, {ratio:.0f}x pixels)")
        print(" alpha = 1.0 is linear O(N); alpha = 2.0 is quadratic O(N^2)")
        print("=" * 62)
        for key, label in [("hooked", "conv/linear only"),
                           ("fft", "FFT term"),
                           ("selective_scan", "selective scan"),
                           ("total", "FVR TOTAL"),
                           ("transformer", "Transformer baseline")]:
            print(f"  {label:24s} alpha = {scaling_exponent(lo[key], hi[key], ratio):.3f}")
        if lo["latency"] and hi["latency"]:
            print(f"  {'FVR measured latency':24s} alpha = "
                  f"{scaling_exponent(lo['latency'], hi['latency'], ratio):.3f}")

    print("\nNote: alpha slightly above 1.0 for FVR TOTAL is expected and correct --")
    print("the FFT is O(N log N), not O(N). Only the token mixer is strictly linear.")


if __name__ == "__main__":
    main()
