import torch
import torch.nn as nn
import torch.fft

from models.common import fp32_autocast


class SpectralGuidanceModule(nn.Module):
    # Reconstructs the modulated spectrum via IFFT and fuses it with the
    # spatial stream in two ways: as a condition map that gates the spatial
    # features, and as an additive injection so the spectral correction
    # actually reaches the output instead of only steering a gate.

    def __init__(self, channels):
        super().__init__()
        self.condition_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.spectral_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, amplitude, phase, spatial_features):
        spatial_size = spatial_features.shape[-2:]

        with fp32_autocast(spatial_features.device.type):
            complex_features = torch.polar(amplitude.float(), phase.float())
            restored_spectral = torch.fft.irfft2(
                complex_features, s=spatial_size, norm='ortho')
        restored_spectral = restored_spectral.to(spatial_features.dtype)

        combined = torch.cat([spatial_features, restored_spectral], dim=1)
        condition_map = self.sigmoid(self.condition_conv(combined))

        return spatial_features * condition_map + self.spectral_proj(restored_spectral)


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy_spatial = torch.randn(1, 64, 64, 64)
    spectrum = torch.fft.rfft2(dummy_spatial, norm='ortho')
    dummy_amp, dummy_phase = torch.abs(spectrum), torch.angle(spectrum)

    fusion_block = SpectralGuidanceModule(channels=64)
    guided_out = fusion_block(dummy_amp, dummy_phase, dummy_spatial)
    print(f"Guided Features Output Shape: {tuple(guided_out.shape)}")

    # the IFFT must invert the FFT to numerical precision
    rec = torch.fft.irfft2(torch.polar(dummy_amp, dummy_phase),
                           s=dummy_spatial.shape[-2:], norm='ortho')
    print(f"IFFT reconstruction error: {(rec - dummy_spatial).abs().max():.3e}")
    print("Spectral Guidance Fusion Successful!")
