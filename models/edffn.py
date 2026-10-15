import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from models.common import LayerNorm2d, fp32_autocast


class EDFFN(nn.Module):
    # Enhanced Deep Fourier Feed-forward Network.
    #
    # Spatial branch: gated depth-wise FFN.
    # Spectral branch: learnable modulation of the *decoupled* spectra.
    #   amplitude ~ illumination  -> multiplicative gain (low-light lifting)
    #   phase     ~ structure     -> additive offset     (deblurring)
    # Modulating them separately is what stops a low-light gain from also
    # amplifying the noise structure carried in the phase.

    def __init__(self, channels, expansion=2, eps=1e-8, gain_clamp=4.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.eps = eps
        self.gain_clamp = gain_clamp

        self.norm = LayerNorm2d(channels)
        self.pconv1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                               padding=1, groups=hidden * 2)
        self.pconv2 = nn.Conv2d(hidden, channels, kernel_size=1)

        self.amp_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.pha_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

        # Zero-init the last conv of each branch so the block starts as an exact
        # identity round-trip and has to learn its way off it -- keeps the FFT
        # branch from injecting noise before the spatial path has converged.
        for branch in (self.amp_net, self.pha_net):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, x):
        y = self.dconv(self.pconv1(self.norm(x)))
        gate, value = y.chunk(2, dim=1)
        spatial_features = self.pconv2(F.gelu(gate) * value) + x

        # torch.fft has no half-precision kernel, so this section opts out of AMP
        with fp32_autocast(x.device.type):
            fft_features = torch.fft.rfft2(spatial_features.float(), norm='ortho')
            real, imag = fft_features.real, fft_features.imag

            # eps keeps d(sqrt)/dx finite at the DC/zero bins, where torch.abs
            # and torch.angle both have singular gradients
            amplitude = torch.sqrt(real * real + imag * imag + self.eps)
            phase = torch.atan2(imag, real)

            # exp() gives a strictly positive gain that is exactly 1.0 at init;
            # the clamp is a safety rail against runaway amplification
            log_gain = self.amp_net(amplitude).clamp(-self.gain_clamp, self.gain_clamp)
            amplitude = amplitude * torch.exp(log_gain)
            phase = phase + self.pha_net(phase)

        return amplitude, phase, spatial_features


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy_input = torch.randn(1, 64, 64, 64)
    model = EDFFN(channels=64)

    amp, phase, spatial = model(dummy_input)
    print(f"Amplitude Shape: {tuple(amp.shape)}")
    print(f"Phase Shape:     {tuple(phase.shape)}")

    # At init the branch is a pure round-trip; after perturbing the spectral
    # weights it must diverge from the spatial features, otherwise the whole
    # Fourier path is dead weight.
    def round_trip(m):
        a, p, s = m(dummy_input)
        rec = torch.fft.irfft2(torch.polar(a, p), s=s.shape[-2:], norm='ortho')
        return (rec - s).abs().mean().item()

    print(f"Round-trip delta at init:      {round_trip(model):.3e}  (expected ~0)")
    with torch.no_grad():
        for branch in (model.amp_net, model.pha_net):
            branch[-1].weight.normal_(0, 0.05)
    print(f"Round-trip delta once trained: {round_trip(model):.3e}  (must be >> 0)")
    print("Frequency-domain processing is live.")
