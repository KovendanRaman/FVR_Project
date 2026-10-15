import torch
import torch.nn as nn

from models.common import LayerNorm2d
from models.edffn import EDFFN
from models.vmamba import VSSBlock
from models.fusion import SpectralGuidanceModule


class FVRStage(nn.Module):
    # One FVR stage: Fourier decoupling -> spectral guidance -> selective scan.
    # Both halves are residual so stages can be stacked without vanishing signal.

    def __init__(self, channels, d_state=16, expand=2, chunk_size=64,
                 use_checkpoint=True):
        super().__init__()
        self.edffn = EDFFN(channels=channels)
        self.fusion = SpectralGuidanceModule(channels=channels)
        self.vmamba = VSSBlock(channels, d_state=d_state, expand=expand,
                               chunk_size=chunk_size, use_checkpoint=use_checkpoint)

    def forward(self, x):
        amp, phase, spatial = self.edffn(x)
        guided = self.fusion(amp, phase, spatial)
        return self.vmamba(x + guided)


class FVRNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, embed_dim=64, depth=4,
                 d_state=16, expand=2, chunk_size=64, use_checkpoint=True):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.stages = nn.ModuleList([
            FVRStage(embed_dim, d_state=d_state, expand=expand,
                     chunk_size=chunk_size, use_checkpoint=use_checkpoint)
            for _ in range(depth)
        ])

        self.output_norm = LayerNorm2d(embed_dim)
        self.output_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x

        feat = self.input_conv(x)
        for stage in self.stages:
            feat = stage(feat)

        restored = self.output_conv(self.output_norm(feat))
        return restored + identity


if __name__ == "__main__":
    dummy_image = torch.randn(1, 3, 64, 64)
    model = FVRNet(depth=2, use_checkpoint=False)

    final_output = model(dummy_image)
    n_params = sum(p.numel() for p in model.parameters())
    n_used = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input Image Shape:    {tuple(dummy_image.shape)}")
    print(f"Restored Output Shape:{tuple(final_output.shape)}")
    print(f"Total parameters:     {n_params:,}")

    # every parameter must receive a gradient -- catches dead layers like the
    # in_proj that the original SS2D declared but never called
    final_output.sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"Parameters with no gradient: {dead if dead else 'none'}")
    assert not dead, "dead parameters found"
    print("Full FVR Architecture is Alive!")
