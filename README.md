# FVR — Fourier–VMamba Restoration

Joint low-light enhancement and motion deblurring at linear **O(N)** complexity.

FVR combines three components:
- **EDFFN** — Fourier amplitude/phase decoupling in the feed-forward network
- **Spectral Guidance** — frequency-domain fusion for illumination recovery
- **SS2D** — 2D selective scan (VMamba backbone) for linear-time spatial mixing

Honours research project, School of Computer Science & Applied Mathematics,
University of the Witwatersrand.

## Research questions

- **RQ1** — Does Fourier+VMamba reduce latency vs. Fourier-guided diffusion
  (FourierDiff, Lv et al. 2024)?
- **RQ2** — PSNR/SSIM quality vs. baselines on LOL-Blur, GoPro, and RealBlur-J?
- **RQ3** — Speed/accuracy Pareto front as VMamba–Fourier block depth varies?

## Key results

### LOL-Blur (joint low-light + motion blur, 1,800 test images)

| Method | PSNR | SSIM | Params | Complexity |
|---|---|---|---|---|
| Cascade (ZeroDCE→MIMO) | 17.68 | 0.542 | — | — |
| FourierDiff (zero-shot) | 20.53 | 0.710 | — | O(T·N²) |
| **FVR (ours)** | **21.76** | **0.787** | **470k** | **O(N)** |
| RetinexFormer | 22.90 | 0.824 | — | O(N²) |
| LEDNet | 25.27 | 0.859 | — | — |
| CIDNet | 26.57 | 0.890 | — | O(N²) |

### Cross-dataset generalisation (PSNR / SSIM)

| Model ↓ / Test → | LOL-Blur | GoPro | RealBlur-J |
|---|---|---|---|
| LOL-Blur model | 21.81 / 0.788 | 10.32 / 0.603 | 11.30 / — |
| GoPro model | 10.20 / 0.224 | 24.81 / 0.780 | 26.44 / 0.744 |
| Combined model | 21.89 / 0.789 | 24.47 / 0.783 | 15.46 / 0.587 |

Dedicated models dominate their own domain; transfer fails across brightness
regimes (a low-light model over-brightens already-bright inputs, and vice versa).

### Efficiency (256×256)

| Method | Params | FLOPs (G) | Latency (ms) | Complexity |
|---|---|---|---|---|
| **FVR (ours)** | **470k** | **72.7** | **70.7** | **O(N)** |
| EVSSM | 17.1M | 126 | 89 | O(N) |
| FFTformer | 16.6M | 131 | 132 | O(N²) |
| Restormer | 26.1M | 155 | 286 | O(N²) |

Measured scaling exponent: **α = 1.001** (linear) vs. 1.995 for dense attention.

### Depth ablation (RQ3)

| Depth | Params | PSNR | SSIM | Latency/tile | Status |
|---|---|---|---|---|---|
| 2 | 237k | 21.25 | 0.778 | 1.4 ms | trained |
| **4** | **470k** | **21.76** | **0.787** | **2.8 ms** | **optimal** |
| 8 | 938k | 21.38 | 0.779 | 5.6 ms | regresses |
| 16 | 1.87M | NaN | NaN | 11.1 ms | diverges |

## Repository structure

    train.py                 main trainer
    inference.py             single/batch restoration
    metrics_eval2.py         tiled PSNR/SSIM/NIQE evaluation
    evaluate.py              FLOPs / latency / O(N) profiler
    benchmark_complexity.py  FVR vs transformer scaling
    latency_sweep.py         per-depth latency
    qualitative.py           before/after figure generation
    models/                  FVR architecture
    utils/                   helpers
    scripts/                 data-prep (organize_*.py), smoke test
    jobs/                    SLURM batch scripts (.sbatch / .sh)
    figures/                 result plots and demo images
    checkpoints*/            trained weights (depth sweep, gopro, combined)
    data/                    datasets (not tracked)

## Datasets

Not included (130 GB). Obtain from the original sources and prepare with the
`scripts/organize_*.py` helpers into `data/processed/`:
- **LOL-Blur** — Zhou et al., LEDNet (ECCV 2022)
- **GoPro** — Nah et al. (CVPR 2017)
- **RealBlur-J** — Rim et al. (ECCV 2020)

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell (sm_120)
- PyTorch 2.12 nightly + CUDA 12.8 (required for sm_120)
- Python 3.11 (conda env `fvr`)
- Fused CUDA kernels: `causal-conv1d` + `mamba-ssm` compiled from source for sm_120

Note: install with `pip install --no-deps` to avoid clobbering the nightly build.

## Training

    python train.py \
      --blurry-dir data/processed/train/blurry \
      --sharp-dir  data/processed/train/sharp \
      --patch-size 128 --batch-size 8 --depth 4 \
      --epochs 200 --lr 1e-4 --grad-clip 1.0 \
      --out-dir checkpoints --no-checkpoint

## Evaluation

    python metrics_eval2.py \
      --blurry-dir data/processed/test/blurry \
      --sharp-dir  data/processed/test/sharp \
      --checkpoint checkpoints/best.pth \
      --tile 128 --overlap 32

Tiled inference (128px tiles, 32px overlap, Hann blending) is required — the
selective scan accumulates state over the full sequence, so full-frame
inference collapses.

## Author

Kovendan Jason Raman — University of the Witwatersrand.
