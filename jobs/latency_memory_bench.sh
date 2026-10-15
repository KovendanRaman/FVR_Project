#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home-mscluster/kraman/FVR_Project
#SBATCH --output=latency_memory_%j.log

export FVR=/home-mscluster/kraman/.conda/envs/fvr
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home-mscluster/kraman/FVR_Project

$FVR/bin/python -u -c "
import torch, time
from models.fvr_net import FVRNet

device = 'cuda'
depth = 4
model = FVRNet(depth=depth, use_checkpoint=False).to(device).eval()

resolutions = [128, 256, 384, 512, 640, 768]
print(f'Inference latency & peak GPU memory — FVR depth {depth}')
print(f'{\"Resolution\":>12} {\"Latency (ms)\":>14} {\"Peak Mem (MB)\":>14} {\"Throughput\":>12}')
print('-' * 60)

for res in resolutions:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    x = torch.randn(1, 3, res, res, device=device)
    
    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
    
    # measure
    reps = 50
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps * 1000  # ms
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    throughput = 1000.0 / dt
    
    print(f'{res:>8}x{res:<4} {dt:>12.2f}ms {peak_mem:>12.1f}MB {throughput:>10.1f}/s')

print()
print('Done. Compare against published baselines:')
print('  EVSSM (17.1M params): 89ms at 256x256 on RTX 3090')
print('  Restormer (26.1M):   286ms at 256x256 on RTX 3090')
print('  FFTformer (16.6M):   132ms at 256x256 on RTX 3090')
print('  GRL (20.2M):         518ms at 256x256 on RTX 3090')
print('  FVR (470k params):   see above')
"
