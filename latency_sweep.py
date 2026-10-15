import torch, time
from models.fvr_net import FVRNet

RES = 128
depths = [2, 4, 8, 16]
print(f"Latency per {RES}x{RES} tile, forward-only, GPU")
print(f'{"depth":>6} {"params":>12} {"latency_ms":>12} {"throughput_per_s":>16}')
for d in depths:
    m = FVRNet(depth=d, use_checkpoint=False).cuda().eval()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 3, RES, RES, device="cuda")
    with torch.no_grad():
        for _ in range(10):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        REP = 100
        for _ in range(REP):
            m(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / REP * 1000.0
    print(f"{d:>6} {n:>12,} {dt:>12.3f} {1000.0/dt:>16.1f}")
    del m
    torch.cuda.empty_cache()
