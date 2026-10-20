"""
VMamba-style Visual State Space encoder block — pure PyTorch, no CUDA extensions.

SS2D uses a 4-directional selective scan to capture global 2D context:
  scan1: row-major  (top-left  → bottom-right)
  scan2: row-major  reversed   (bottom-right → top-left)
  scan3: col-major  (top-left  → bottom-right, transposed)
  scan4: col-major  reversed

For each direction a 1-D Mamba SSM is applied then the four outputs are summed.

Reference: VMamba (Liu et al. 2024), simplified for full-image HSI feature maps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Chunked parallel selective scan (float32)
# ---------------------------------------------------------------------------
# Numerical stability: exp(cumsum(log_dA)) over L=15552 steps underflows float32.
# Fix: break L into small chunks of CHUNK steps. Within each chunk the prefix
# product stays representable in float32 provided:
#   CHUNK * delta_max * |A|_max < 87  (float32 underflow threshold)
# We enforce this by clamping delta to DT_MAX (passed at call time from SS2D).
# Hidden state h is carried across chunk boundaries in float32.
#
# Backward stability: d(c_loc)/d(P_incl) = -dBu/P² which can be huge for
# small P_incl. Fix: stop-gradient on P_incl in the denominator
# (P[j]/P[k] ≤ 1 since P is non-increasing, so gradients remain bounded).

_CHUNK = 256  # steps per chunk
# Float32 stability guarantee: CHUNK * dt_max * |A|_max ≤ 25.6 → P_incl_min ≈ 7e-12
# (representable in float32; exp(-25.6) well above float32 underflow at ~1.2e-38)
# |A| is clamped to 1 inside selective_scan; dt_max set by SS2D._dt_max=0.1.


def selective_scan(u, delta, A, B, C, D, dt_max=0.1):
    """
    Chunked parallel selective scan in float32.

    Stability constraints (must hold for float32 to not underflow):
      CHUNK * dt_max * |A|_clamped_max ≤ 25.6  →  P_incl ≥ 7e-12  ✓

    u      : [B, L, d]
    delta  : [B, L, d]   (after softplus, positive)
    A      : [d, N]      (negative — Mamba convention; clamped to [-1, -1e-6] here)
    B      : [B, L, N]
    C      : [B, L, N]
    D      : [d]
    dt_max : clamp for delta (default 0.1)

    Returns y : [B, L, d]
    """
    B_sz, L, d = u.shape
    N  = A.shape[1]

    u32     = u.float()
    delta32 = delta.float().clamp(min=1e-4, max=dt_max)
    A32     = A.float().clamp(min=-1.0, max=-1e-6)  # keep |A| ≤ 1 for stability
    B32     = B.float()
    C32     = C.float()

    h    = u32.new_zeros(B_sz, d, N)          # [B, d, N]
    ys   = []

    for start in range(0, L, _CHUNK):
        end  = min(start + _CHUNK, L)

        dc  = delta32[:, start:end]            # [B, K, d]
        bc  = B32[:, start:end]                # [B, K, N]
        cc  = C32[:, start:end]                # [B, K, N]
        uc  = u32[:, start:end]                # [B, K, d]

        # Inclusive prefix products: P[k] = exp(Σ_{j≤k} delta[j]*A)
        log_dA = torch.einsum('bkd,dn->bkdn', dc, A32)               # [B, K, d, N]
        P_incl = torch.exp(torch.cumsum(log_dA, dim=1))              # [B, K, d, N]
        # P_incl ∈ [7e-12, 1] with the stability constraints above

        # Local term: local[k] = P[k] * Σ_{i≤k} dBu[i] / P[i]
        # Stop-grad on denominator: avoids -dBu/P² in backward (gradient stays bounded
        # because P[j]/P[i] ≤ 1 for j ≥ i)
        dBu   = torch.einsum('bkd,bkn,bkd->bkdn', dc, bc, uc)       # [B, K, d, N]
        P_sg  = P_incl.detach().clamp(min=1e-15)                     # stop-grad denom
        c_loc = dBu / P_sg
        local = P_incl * torch.cumsum(c_loc, dim=1)

        H_chunk = P_incl * h.unsqueeze(1) + local                    # [B, K, d, N]

        y_c = torch.einsum('bkdn,bkn->bkd', H_chunk, cc)             # [B, K, d]
        ys.append(y_c)
        h = H_chunk[:, -1]

    y = torch.cat(ys, dim=1).to(u.dtype)
    y = y + u * D.unsqueeze(0).unsqueeze(0)
    return y


# ---------------------------------------------------------------------------
# SS2D: Selective Scan applied across a 2-D feature map in 4 directions
# ---------------------------------------------------------------------------

class SS2D(nn.Module):
    """
    2-D selective scan that processes a spatial feature map by scanning in 4
    directions and summing the resulting representations.

    input  : [B, d_model, H, W]
    output : [B, d_model, H, W]
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=1, dt_rank="auto"):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Projections (shared across all 4 directions)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d   = nn.Conv2d(self.d_inner, self.d_inner, d_conv,
                                  padding=d_conv // 2, groups=self.d_inner, bias=True)

        # SSM per-direction parameter generators (one per direction)
        self.n_dir = 4
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            for _ in range(self.n_dir)
        ])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(self.n_dir)
        ])

        # SSM parameters A, D
        # A initialized small (÷ d_state) so delta*|A|*CHUNK < 87 at init
        A_init = -torch.arange(1, d_state + 1, dtype=torch.float32) / d_state
        A_init = A_init.unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(-A_init).clone()) for _ in range(self.n_dir)
        ])
        self.D = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(self.n_dir)
        ])

        # Mamba-style dt init: softplus(bias) ∈ [dt_min, dt_max=0.1]
        dt_min, dt_max = 0.001, 0.1
        self._dt_max = dt_max
        for proj in self.dt_proj:
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            proj.bias.data = dt_init + torch.log(-torch.expm1(-dt_init))  # inv-softplus

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan_one_dir(self, x_seq, direction):
        """
        Apply Mamba SSM to a [B, L, d_inner] sequence in one direction.
        Returns [B, L, d_inner].
        """
        B, L, d = x_seq.shape
        xp = self.x_proj[direction](x_seq)                       # [B, L, dt_rank+2N]
        dt, Bmat, Cmat = torch.split(
            xp, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt  = F.softplus(self.dt_proj[direction](dt))            # [B, L, d_inner]
        A   = -torch.exp(self.A_log[direction].float())          # [d_inner, N]
        D   = self.D[direction].float()
        y   = selective_scan(x_seq.float(), dt.float(),
                             A, Bmat.float(), Cmat.float(), D,
                             dt_max=self._dt_max)
        return y.to(x_seq.dtype)

    def forward(self, x):
        B, C, H, W = x.shape

        # Project input: x → (x_branch, z_gate)
        x_flat  = rearrange(x, 'b c h w -> b (h w) c')
        xz      = self.in_proj(x_flat)                           # [B, HW, 2*d_inner]
        x_b, z  = xz.chunk(2, dim=-1)                           # each [B, HW, d_inner]

        # Depthwise conv on x_branch for local context
        x_b2d   = rearrange(x_b, 'b (h w) c -> b c h w', h=H, w=W)
        x_b2d   = self.conv2d(x_b2d)
        x_b     = rearrange(F.silu(x_b2d), 'b c h w -> b (h w) c')

        # 4-directional scans
        # dir 0: row-major forward  (natural raster order)
        seq0 = x_b
        # dir 1: row-major backward
        seq1 = x_b.flip(1)
        # dir 2: column-major forward (transpose H,W then flatten)
        seq2 = rearrange(
            rearrange(x_b, 'b (h w) c -> b h w c', h=H, w=W).permute(0, 2, 1, 3),
            'b w h c -> b (w h) c')
        # dir 3: column-major backward
        seq3 = seq2.flip(1)

        seqs = [seq0, seq1, seq2, seq3]
        outs = []
        for d, seq in enumerate(seqs):
            y = self._scan_one_dir(seq, d)
            outs.append(y)

        # Un-flip / un-transpose and sum
        y0 = outs[0]
        y1 = outs[1].flip(1)
        # un-transpose: (w h) → (h w)
        y2 = rearrange(
            rearrange(outs[2], 'b (w h) c -> b w h c', w=W, h=H).permute(0, 2, 1, 3),
            'b h w c -> b (h w) c')
        y3 = rearrange(
            rearrange(outs[3].flip(1), 'b (w h) c -> b w h c', w=W, h=H).permute(0, 2, 1, 3),
            'b h w c -> b (h w) c')

        y = y0 + y1 + y2 + y3                                   # [B, HW, d_inner]

        # Gate, norm, project
        y = y * F.silu(z)
        y = self.out_norm(y)
        y = self.out_proj(y)                                     # [B, HW, d_model]

        return rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)


# ---------------------------------------------------------------------------
# VSSBlock: residual wrapper around SS2D (VMamba's basic block)
# ---------------------------------------------------------------------------

class VSSBlock(nn.Module):
    """
    Visual State Space Block.
    Residual: x + drop(SS2D(LN(x)))
    """

    def __init__(self, channels, d_state=16, expand=1, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ss2d = SS2D(d_model=channels, d_state=d_state, expand=expand)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, C, H, W]
        shortcut = x
        # LN over channel dim
        x_ln = rearrange(x, 'b c h w -> b h w c')
        x_ln = self.norm(x_ln)
        x_ln = rearrange(x_ln, 'b h w c -> b c h w')
        return shortcut + self.drop(self.ss2d(x_ln))


def repeat_vss(channels, n, d_state=16, expand=1):
    """Stack n VSSBlocks."""
    return nn.Sequential(*[VSSBlock(channels, d_state=d_state, expand=expand)
                           for _ in range(n)])
