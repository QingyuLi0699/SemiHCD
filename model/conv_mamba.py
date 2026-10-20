"""
V4 hybrid building blocks: CNN (local) ‖ Mamba (global).

ConvMambaBlock
    Pre-norm (GroupNorm) → DWConv3×3+PWConv1×1 (local) ‖ SS2D (global).
    Learned scalar gate α blends the two branches; residual connection.
    α init = 0  →  both branches start at equal 0.5 weight.

SpectralSSM1D
    Bidirectional 1-D SSM along the C spectral bands.
    Global average pool collapses H×W → per-band scalar sequence [B, C].
    Produces a per-band sigmoid gate (Mamba-based spectral channel attention),
    capturing sequential inter-band correlations (water-absorption dips,
    red-edge slopes) that 1×1 conv or SE block cannot model.
    Memory: O(B × C × d_inner × N) — independent of H×W.

SpectralMambaStem
    SpectralSSM1D → conv3x3_gn_relu.  Drop-in for the plain conv stem.

CrossTemporalMambaFusion
    Local  branch  — cat(f1,f2,|diff|) → Conv1×1 → SE   (boundary precision)
    Global branch  — SS2D on GN(f1−f2)               (global change context)
    Gate β blends the two;  β init = 0  →  equal weight at start.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_encoder import SS2D, selective_scan
from model.semi_hscd import conv3x3_gn_relu, SEBlock


# ─────────────────────────────────────────────────────────────────
# ConvMambaBlock
# ─────────────────────────────────────────────────────────────────

class ConvMambaBlock(nn.Module):
    """
    Parallel DWConv3×3 (local) + SS2D (global) with learned scalar gate.

        out = residual
            + sigmoid(α) · PW(SiLU(DW(GN(x))))
            + (1 − sigmoid(α)) · SS2D(GN(x))
    """
    def __init__(self, channels, d_state=16, expand=1, num_groups=None):
        super().__init__()
        if num_groups is None:
            num_groups = max(channels // 16, 1)
        self.norm  = nn.GroupNorm(num_groups, channels)
        # Local branch
        self.dw    = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.pw    = nn.Conv2d(channels, channels, 1)
        # Global branch
        self.ss2d  = SS2D(d_model=channels, d_state=d_state, expand=expand)
        # Blend gate
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = x
        xn  = self.norm(x)
        loc = self.pw(F.silu(self.dw(xn)))
        glb = self.ss2d(xn)
        w   = torch.sigmoid(self.alpha)
        return res + w * loc + (1.0 - w) * glb


def repeat_conv_mamba(channels, n, d_state=16, expand=1, num_groups=None):
    """Stack n ConvMambaBlocks."""
    return nn.Sequential(*[
        ConvMambaBlock(channels, d_state=d_state, expand=expand, num_groups=num_groups)
        for _ in range(n)
    ])


# ─────────────────────────────────────────────────────────────────
# SpectralSSM1D
# ─────────────────────────────────────────────────────────────────

class SpectralSSM1D(nn.Module):
    """
    Bidirectional 1-D Mamba over C spectral bands (global-average-pool based).

    For each image, GAP collapses [B,C,H,W] → [B,C].  The SSM treats
    C bands as a length-C sequence of scalars (d=1 → projected to d_inner),
    accumulates inter-band context, and outputs a per-band gate σ(y) ∈ (0,1).

    The gate is broadcast over H,W: return x * gate.

    Stability: L = C ≤ 256 = CHUNK  →  one chunk;
               L × dt_max × |A|_max = C × 0.1 × 1 = 15.4
               → P_min = exp(−15.4) ≈ 2e-7  (well above float32 floor).
    """
    def __init__(self, in_channels, d_state=16, d_inner=32):
        super().__init__()
        C        = in_channels
        dt_rank  = max(C // 16, 1)
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        n_dir = 2

        # Shared projection: scalar band value → d_inner
        self.in_proj  = nn.Linear(1, d_inner * 2, bias=False)

        self.x_proj   = nn.ModuleList([
            nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            for _ in range(n_dir)
        ])
        self.dt_proj  = nn.ModuleList([
            nn.Linear(dt_rank, d_inner, bias=True) for _ in range(n_dir)
        ])

        # SSM matrices A (negative), D (skip)
        A_init = -torch.arange(1, d_state + 1, dtype=torch.float32) / d_state
        A_init = A_init.unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(-A_init.clone())) for _ in range(n_dir)
        ])
        self.D = nn.ParameterList([
            nn.Parameter(torch.ones(d_inner)) for _ in range(n_dir)
        ])

        # Mamba-style dt init: softplus(bias) ∈ [0.001, 0.1]
        dt_min_v, dt_max_v = 0.001, 0.1
        for proj in self.dt_proj:
            dt_init = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max_v) - math.log(dt_min_v))
                + math.log(dt_min_v))
            proj.bias.data = dt_init + torch.log(-torch.expm1(-dt_init))

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Global spectral statistics: [B, C] → [B, C, 1]
        u = x.mean(dim=(-2, -1)).unsqueeze(-1)          # [B, C, 1]

        xz     = self.in_proj(u)                         # [B, C, 2*d_inner]
        x_b, z = xz.chunk(2, dim=-1)                     # each [B, C, d_inner]
        x_b    = F.silu(x_b)

        outs = []
        for i in range(2):
            seq = x_b if i == 0 else x_b.flip(1)
            xp  = self.x_proj[i](seq)                    # [B, C, dt_rank+2N]
            dt, Bm, Cm = torch.split(
                xp, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt  = F.softplus(self.dt_proj[i](dt))        # [B, C, d_inner]
            A   = -torch.exp(self.A_log[i].float())      # [d_inner, N]
            D   = self.D[i].float()
            y   = selective_scan(seq, dt, A, Bm, Cm, D, dt_max=0.1)
            if i == 1:
                y = y.flip(1)
            outs.append(y)

        y = outs[0] + outs[1]                            # [B, C, d_inner]
        y = y * F.silu(z)
        y = self.out_norm(y)
        y = self.out_proj(y)                             # [B, C, 1]

        # Per-band gate broadcast over H, W
        gate = torch.sigmoid(y).unsqueeze(-1)            # [B, C, 1, 1]
        return x * gate


# ─────────────────────────────────────────────────────────────────
# SpectralMambaStem
# ─────────────────────────────────────────────────────────────────

class SpectralMambaStem(nn.Module):
    """SpectralSSM1D (band gate) → Conv3×3+GN+ReLU spatial stem."""

    def __init__(self, in_channels, out_channels, num_groups,
                 d_state=16, spec_d_inner=32):
        super().__init__()
        self.spectral = SpectralSSM1D(in_channels, d_state=d_state, d_inner=spec_d_inner)
        self.spatial  = conv3x3_gn_relu(in_channels, out_channels, num_groups)

    def forward(self, x):
        return self.spatial(self.spectral(x))


# ─────────────────────────────────────────────────────────────────
# CrossTemporalMambaFusion
# ─────────────────────────────────────────────────────────────────

class CrossTemporalMambaFusion(nn.Module):
    """
    Two-branch cross-temporal fusion at one FPN scale.

    local  — cat(f1,f2,|f1−f2|) → Conv1×1 → SE   (pixel-precise boundaries)
    global — SS2D( GN(f1−f2) )                    (spatial change-context scan)

    output = sigmoid(β)·local + (1−sigmoid(β))·global,   β init=0.
    """
    def __init__(self, channels, d_state=16, expand=1, reduction=16):
        super().__init__()
        ng = max(channels // reduction, 1)

        # Local branch (original CrossTemporalFusion logic)
        self.reduce = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False),
            nn.GroupNorm(ng, channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(channels, reduction)

        # Global branch
        self.diff_norm = nn.GroupNorm(ng, channels)
        self.ss2d      = SS2D(d_model=channels, d_state=d_state, expand=expand)

        # Blend gate
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        diff    = f1 - f2
        local   = self.se(self.reduce(torch.cat([f1, f2, diff.abs()], dim=1)))
        global_ = self.ss2d(self.diff_norm(diff))
        w       = torch.sigmoid(self.beta)
        return w * local + (1.0 - w) * global_
