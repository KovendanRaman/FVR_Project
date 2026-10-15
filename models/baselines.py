import math

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    # Standard pre-norm ViT block applied to full-resolution tokens.
    # Included purely as the O(N^2) reference point for RQ1.

    def __init__(self, dim, num_heads=4, ffn_expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class TransformerRestorer(nn.Module):
    # Depth/width-matched counterpart to FVRNet, so the comparison isolates
    # the token mixer (dense attention vs selective scan).

    def __init__(self, in_channels=3, out_channels=3, embed_dim=64, depth=4,
                 num_heads=4, ffn_expansion=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion

        self.input_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_expansion) for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        b, _, h, w = x.shape

        feat = self.input_conv(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            feat = block(feat)
        feat = self.output_norm(feat).transpose(1, 2).reshape(b, self.embed_dim, h, w)

        return self.output_conv(feat) + identity


def transformer_flops(h, w, in_channels=3, out_channels=3, embed_dim=64, depth=4,
                      num_heads=4, ffn_expansion=2):
    # Closed-form FLOP count. Needed because full-resolution attention cannot be
    # instantiated at 256x256 (the L x L score matrix alone is ~17 GB), so the
    # high-resolution numbers have to be extrapolated rather than measured.
    # Validated against a measured run at low resolution in evaluate.py.
    length = h * w
    c = embed_dim
    flops = 0.0

    flops += 2 * h * w * in_channels * c * 9          # input conv 3x3
    flops += 2 * h * w * c * out_channels * 9         # output conv 3x3

    for _ in range(depth):
        flops += 2 * length * c * 3 * c               # qkv projection
        flops += 2 * length * length * c              # Q K^T
        flops += 5 * num_heads * length * length      # softmax
        flops += 2 * length * length * c              # attn @ V
        flops += 2 * length * c * c                   # output projection
        flops += 2 * 2 * length * c * (c * ffn_expansion)   # FFN
        flops += 2 * 5 * length * c                   # layer norms
    flops += 5 * length * c                           # final norm

    return flops


def attention_flops_only(h, w, embed_dim=64, depth=4, num_heads=4):
    # The quadratic term on its own -- this is what thop cannot see, because
    # scaled_dot_product_attention is a functional call, not a hookable module.
    length = h * w
    return depth * (4 * length * length * embed_dim + 5 * num_heads * length * length)


if __name__ == "__main__":
    model = TransformerRestorer(depth=2, embed_dim=32, num_heads=4)
    out = model(torch.randn(1, 3, 32, 32))
    print(f"TransformerRestorer output: {tuple(out.shape)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Analytic FLOPs @32x32: {transformer_flops(32, 32, embed_dim=32, depth=2) / 1e9:.4f} G")
    print(f"Attention share: "
          f"{100 * attention_flops_only(32, 32, 32, 2) / transformer_flops(32, 32, embed_dim=32, depth=2):.1f}%")
