import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.common import LayerNorm2d

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_cuda
except Exception:  # pragma: no cover - depends on local install
    _selective_scan_cuda = None

SELECTIVE_SCAN_BACKEND = "cuda" if _selective_scan_cuda is not None else "torch"


def _associative_scan(a, b):
    
    length = a.shape[-2]
    h, acc = b, a
    step = 1
    while step < length:
        h_prev = F.pad(h, (0, 0, step, 0))[..., :length, :]
        a_prev = F.pad(acc, (0, 0, step, 0), value=1.0)[..., :length, :]
        h = h + acc * h_prev
        acc = acc * a_prev
        step *= 2
    return h, acc


def _scan_chunk(u, delta, A, B, C, h):
   
    du = delta.unsqueeze(-1)                                   
    dA = torch.exp(du * A)                                     
    Bk = B.permute(0, 1, 3, 2).unsqueeze(2)                    
    dBu = du * Bk * u.unsqueeze(-1)                            

    h_local, a_cum = _associative_scan(dA, dBu)
    hs = h_local + a_cum * h.unsqueeze(3)                      

    Ck = C.permute(0, 1, 3, 2).unsqueeze(2)                    
    return (hs * Ck).sum(-1), hs[:, :, :, -1]


def selective_scan_ref(u, delta, A, B, C, D=None, chunk_size=64, use_checkpoint=False):
    """Reference S6 selective scan in pure PyTorch (autograd-safe).

    Implements the discretised state-space recurrence
        h_t = exp(delta_t * A) h_{t-1} + (delta_t * B_t) x_t
        y_t = C_t . h_t + D x_t
    which is what makes the receptive field global while staying O(L).

    u, delta: (b, dk, l)   A: (dk, n)   B, C: (b, g, n, l)   D: (dk,)

    The (b, dk, l, n) discretisation tensor is never materialised in full;
    it is built one chunk at a time. `use_checkpoint` additionally recomputes
    each chunk during backward, which is what makes 256x256 (l = 65536)
    trainable without the fused kernel.
    """
    b, dk, l = u.shape
    g, n = B.shape[1], B.shape[2]
    dg = dk // g

    u = u.view(b, g, dg, l)
    delta = delta.view(b, g, dg, l)
    A = A.view(g, dg, 1, n)

    h = u.new_zeros(b, g, dg, n)
    ys = []
    for start in range(0, l, chunk_size):
        end = min(start + chunk_size, l)
        args = (u[..., start:end], delta[..., start:end], A,
                B[..., start:end], C[..., start:end], h)
        if use_checkpoint and torch.is_grad_enabled():
            y, h = checkpoint(_scan_chunk, *args, use_reentrant=False)
        else:
            y, h = _scan_chunk(*args)
        ys.append(y)

    y = torch.cat(ys, dim=-1).reshape(b, dk, l)
    if D is not None:
        y = y + u.reshape(b, dk, l) * D.unsqueeze(-1)
    return y


def cross_scan(x):
    """Unfold (B, C, H, W) into the four VMamba scan orders -> (B, 4, C, L)."""
    row = x.flatten(2)                       # raster order
    col = x.transpose(2, 3).flatten(2)       # column-major order
    return torch.stack([row, col, row.flip(-1), col.flip(-1)], dim=1)


def cross_merge(ys, h, w):
    """Inverse of `cross_scan`: undo each ordering, then sum -> (B, C, L).

    Every direction is mapped back to raster order *before* being combined,
    so position i of one scan is only ever added to position i of another.
    """
    b, _, c, l = ys.shape
    row, col = ys[:, 0], ys[:, 1]
    row_r, col_r = ys[:, 2].flip(-1), ys[:, 3].flip(-1)

    def un_transpose(t):
        return t.view(b, c, w, h).transpose(2, 3).reshape(b, c, l)

    return row + row_r + un_transpose(col) + un_transpose(col_r)


class SS2D(nn.Module):
    """2D selective scan (VMamba) block.

    Four directional S6 scans give every pixel a global receptive field at
    O(H*W) cost, in contrast to the O((H*W)^2) cost of dense self-attention.
    """

    K = 4  # scan directions

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto",
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4, dt_scale=1.0,
                 chunk_size=64, use_checkpoint=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv // 2, groups=self.d_inner)
        self.act = nn.SiLU()

        # per-direction projection to (delta, B, C)
        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False).weight
            for _ in range(self.K)
        ]))                                                    # (K, dt_rank+2n, d_inner)

        dt_projs = [self._dt_init(self.dt_rank, self.d_inner, dt_scale,
                                  dt_min, dt_max, dt_init_floor)
                    for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dt_projs]))
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dt_projs]))

        self.A_logs = nn.Parameter(self._A_log_init(d_state, self.d_inner, self.K))
        self.Ds = nn.Parameter(torch.ones(self.K * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale, dt_min, dt_max, dt_init_floor):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        # bias initialised so softplus(bias) is log-uniform in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, k):
        # S4D-real initialisation: A = -1, -2, ... -d_state, shared across channels
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        return torch.log(A).repeat(k, 1).contiguous()

    def forward_core(self, x):
        """x: (B, d_inner, H, W) -> (B, H, W, d_inner)"""
        b, d, h, w = x.shape
        l = h * w

        xs = cross_scan(x)                                                 # (b,K,d,l)
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        dts = F.softplus(dts + self.dt_projs_bias.view(1, self.K, -1, 1))
        xs = xs.reshape(b, self.K * d, l).contiguous()
        dts = dts.reshape(b, self.K * d, l).contiguous()
        Bs, Cs = Bs.contiguous(), Cs.contiguous()                          # (b,K,n,l)
        As = -torch.exp(self.A_logs.float())                               # (K*d, n)

        if _selective_scan_cuda is not None and x.is_cuda:
            ys = _selective_scan_cuda(xs, dts, As, Bs, Cs, self.Ds.float(),
                                      z=None, delta_bias=None, delta_softplus=False)
        else:
            ys = selective_scan_ref(xs, dts, As, Bs, Cs, self.Ds.float(),
                                    chunk_size=self.chunk_size,
                                    use_checkpoint=self.use_checkpoint and self.training)

        y = cross_merge(ys.view(b, self.K, d, l), h, w)                    # (b,d,l)
        y = y.transpose(1, 2).reshape(b, h, w, d)
        return self.out_norm(y)

    def forward(self, x):
        """x: (B, d_model, H, W) -> (B, d_model, H, W)"""
        xz = self.in_proj(x.permute(0, 2, 3, 1))
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = self.act(self.conv2d(x_ssm.permute(0, 3, 1, 2)))
        y = self.forward_core(x_ssm) * F.silu(z)                           # gated branch
        return self.out_proj(y).permute(0, 3, 1, 2)


class VSSBlock(nn.Module):
    """Pre-norm residual wrapper around SS2D."""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.norm = LayerNorm2d(d_model)
        self.ss2d = SS2D(d_model, **kwargs)

    def forward(self, x):
        return x + self.ss2d(self.norm(x))


if __name__ == "__main__":
    print(f"Selective scan backend: {SELECTIVE_SCAN_BACKEND}")

    # correctness: the chunked/parallel scan must match a naive sequential loop
    torch.manual_seed(0)
    b, dk, l, n, g = 2, 8, 40, 4, 2
    u = torch.randn(b, dk, l, dtype=torch.float64)
    delta = F.softplus(torch.randn(b, dk, l, dtype=torch.float64))
    A = -torch.rand(dk, n, dtype=torch.float64) - 0.5
    B = torch.randn(b, g, n, l, dtype=torch.float64)
    C = torch.randn(b, g, n, l, dtype=torch.float64)
    D = torch.randn(dk, dtype=torch.float64)

    fast = selective_scan_ref(u, delta, A, B, C, D, chunk_size=7)

    dg = dk // g
    h = torch.zeros(b, g, dg, n, dtype=torch.float64)
    slow = []
    for t in range(l):
        dt = delta.view(b, g, dg, l)[..., t].unsqueeze(-1)
        h = torch.exp(dt * A.view(g, dg, n)) * h \
            + dt * B[..., t].unsqueeze(2) * u.view(b, g, dg, l)[..., t].unsqueeze(-1)
        slow.append((h * C[..., t].unsqueeze(2)).sum(-1).reshape(b, dk))
    slow = torch.stack(slow, dim=-1) + u * D.unsqueeze(-1)

    print(f"max |chunked - sequential| = {(fast - slow).abs().max():.3e}")
    assert torch.allclose(fast, slow, atol=1e-9), "selective scan mismatch"

    # cross_scan / cross_merge round-trip: merging 4 copies must give 4x the input
    img = torch.randn(2, 6, 5, 7)
    merged = cross_merge(cross_scan(img), 5, 7).view(2, 6, 5, 7)
    print(f"max |cross_merge(cross_scan(x)) - 4x| = {(merged - 4 * img).abs().max():.3e}")
    assert torch.allclose(merged, 4 * img, atol=1e-5), "cross scan/merge mismatch"

    block = VSSBlock(d_model=32, d_state=8)
    out = block(torch.randn(1, 32, 32, 32))
    print(f"VSSBlock output: {tuple(out.shape)}")
    print(f"Unused parameters: "
          f"{[n_ for n_, p in block.named_parameters() if p.grad is None and not p.requires_grad]}")
    print("SS2D selective scan verified.")
