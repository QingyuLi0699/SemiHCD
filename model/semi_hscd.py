"""
SemiHSCD: Semi-supervised Full-image Hyperspectral Change Detection Network

Architecture:
  - Siamese FPN encoder (shared weights for T1 and T2)
  - CrossTemporalFusion at each FPN scale (cat + SE-reduce)
  - FPN top-down decoder → classification head
  - Deepest encoder features → projection head for SSL (ModifiedDINOLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks (reused from FreeNet)
# ---------------------------------------------------------------------------

def conv3x3_gn_relu(in_ch, out_ch, num_group):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        nn.GroupNorm(num_group, out_ch),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 2, 1),
        nn.ReLU(inplace=True),
    )


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


def repeat_block(channels, r, n):
    layers = [nn.Sequential(SEBlock(channels, r),
                             conv3x3_gn_relu(channels, channels, r))
              for _ in range(n)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Cross-Temporal Fusion at one FPN level
# ---------------------------------------------------------------------------

class CrossTemporalFusion(nn.Module):
    """
    Fuse T1 and T2 features at one FPN scale.
    Input:  f1, f2  each [B, C, H, W]
    Output: fused   [B, C, H, W]

    cat(f1, f2, |f1-f2|) → 1x1 reduce → SE channel attention
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False),
            nn.GroupNorm(max(channels // reduction, 1), channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(channels, reduction)

    def forward(self, f1, f2):
        diff = (f1 - f2).abs()
        cat  = torch.cat([f1, f2, diff], dim=1)
        return self.se(self.reduce(cat))


# ---------------------------------------------------------------------------
# Siamese FPN Encoder  (shared weights for T1 and T2)
# ---------------------------------------------------------------------------

class SiameseFPNEncoder(nn.Module):
    """
    Mirrors FreeNet's feature_ops.
    Returns 4 feature maps at scales 1/1, 1/2, 1/4, 1/8.
    """

    def __init__(self, in_channels, block_channels=(96, 128, 192, 256),
                 num_blocks=(1, 1, 1, 1), reduction_ratio=1.0):
        super().__init__()
        r  = int(16 * reduction_ratio)
        c1 = int(block_channels[0] * reduction_ratio / r) * r
        c2 = int(block_channels[1] * reduction_ratio / r) * r
        c3 = int(block_channels[2] * reduction_ratio / r) * r
        c4 = int(block_channels[3] * reduction_ratio / r) * r
        self.out_channels = [c1, c2, c3, c4]

        self.ops = nn.ModuleList([
            conv3x3_gn_relu(in_channels, c1, r),
            repeat_block(c1, r, num_blocks[0]),
            nn.Identity(),                         # scale 1/1 → append
            downsample2x(c1, c2),
            repeat_block(c2, r, num_blocks[1]),
            nn.Identity(),                         # scale 1/2 → append
            downsample2x(c2, c3),
            repeat_block(c3, r, num_blocks[2]),
            nn.Identity(),                         # scale 1/4 → append
            downsample2x(c3, c4),
            repeat_block(c4, r, num_blocks[3]),
            nn.Identity(),                         # scale 1/8 → append
        ])

    def forward(self, x):
        feats = []
        for op in self.ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feats.append(x)
        return feats  # [f_1/1, f_1/2, f_1/4, f_1/8]


# ---------------------------------------------------------------------------
# FPN Decoder
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    """Top-down FPN pathway. Returns full-resolution [B, inner_dim, H, W]."""

    def __init__(self, encoder_channels, inner_dim):
        super().__init__()
        self.reduce = nn.ModuleList([
            nn.Conv2d(c, inner_dim, 1) for c in encoder_channels
        ])
        self.fuse = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1) for _ in encoder_channels
        ])

    def forward(self, feat_list):
        # feat_list: [f_1/1, f_1/2, f_1/4, f_1/8]
        inner = [r(f) for r, f in zip(self.reduce, feat_list)]
        inner.reverse()  # now [f_1/8, f_1/4, f_1/2, f_1/1]

        out = [self.fuse[0](inner[0])]
        for i in range(len(inner) - 1):
            up = F.interpolate(out[i], scale_factor=2.0, mode='nearest')
            merged = inner[i + 1] + up
            out.append(self.fuse[i + 1](merged))

        return out[-1]  # full-resolution feature map


# ---------------------------------------------------------------------------
# Projection Head for SSL (pixel-wise 1x1 conv MLP)
# ---------------------------------------------------------------------------

class PixelProjHead(nn.Module):
    """
    3-layer pixel-wise MLP (implemented as 1x1 convolutions).
    Used for ModifiedDINOLoss on unlabeled pixels.
    Last layer uses weight-norm (same as DINO) with optional frozen norm.
    """

    def __init__(self, in_dim, hidden_dim=256, out_dim=512, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Conv2d(out_dim, out_dim, 1, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        # x: [B, C, H, W] or [N, C] pixel features
        x = self.mlp(x)
        x = F.normalize(x, dim=1, p=2)
        x = self.last_layer(x)
        return x

    def forward_pixels(self, pixels):
        """pixels: [N, C] → [N, out_dim]"""
        return self.forward(pixels.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)


# ---------------------------------------------------------------------------
# Full SemiHSCD Model
# ---------------------------------------------------------------------------

class SemiHSCD(nn.Module):
    """
    Semi-supervised Hyperspectral Change Detection.

    Two forward modes:
      - forward_cls(t1, t2): returns logit [B, K, H, W]  (for supervised loss)
      - forward_ssl(t1, t2): returns (feat_t1, feat_t2) at deepest encoder scale
                              for sampling → proj_head → ModifiedDINOLoss
    """

    def __init__(self,
                 in_channels=154,
                 num_classes=7,
                 block_channels=(96, 128, 192, 256),
                 num_blocks=(1, 1, 1, 1),
                 reduction_ratio=1.0,
                 inner_dim=128,
                 proj_hidden_dim=256,
                 proj_out_dim=512,
                 norm_last_layer=True):
        super().__init__()

        # Shared encoder (same weights for T1 and T2)
        self.encoder = SiameseFPNEncoder(
            in_channels, block_channels, num_blocks, reduction_ratio)
        enc_ch = self.encoder.out_channels  # [c1, c2, c3, c4]

        # Cross-temporal fusion at each FPN scale
        self.fusions = nn.ModuleList([
            CrossTemporalFusion(c) for c in enc_ch
        ])

        # FPN decoder
        self.decoder = FPNDecoder(enc_ch, inner_dim)

        # Classification head
        self.cls_head = nn.Conv2d(inner_dim, num_classes, 1)

        # Projection head for SSL (operates on deepest encoder features c4)
        self.proj_head = PixelProjHead(enc_ch[-1], proj_hidden_dim, proj_out_dim, norm_last_layer)

    # -----------------------------------------------------------------------

    def encode(self, t1, t2):
        """Run Siamese encoder for both temporal images."""
        return self.encoder(t1), self.encoder(t2)

    def classify(self, feats_t1, feats_t2):
        """Cross-temporal fusion → FPN decoder → logit."""
        fused = [f(f1, f2)
                 for f, f1, f2 in zip(self.fusions, feats_t1, feats_t2)]
        feat_map = self.decoder(fused)          # [B, inner_dim, H, W]
        return self.cls_head(feat_map), feat_map

    def forward(self, t1, t2):
        """Standard forward: returns softmax probabilities [B, K, H, W]."""
        feats_t1, feats_t2 = self.encode(t1, t2)
        logit, _ = self.classify(feats_t1, feats_t2)
        return torch.softmax(logit, dim=1)

    def forward_train(self, t1, t2, mask, train_indicator, unlab_indicator,
                      proj_head_teacher=None, n_ssl_pixels=2048):
        """
        Full training forward pass.

        Returns dict with:
          logit          [B, K, H, W]
          feat_t1_deep   [B, C4, H8, W8]  deepest encoder features for T1
          feat_t2_deep   [B, C4, H8, W8]  deepest encoder features for T2
        """
        feats_t1, feats_t2 = self.encode(t1, t2)
        logit, feat_map = self.classify(feats_t1, feats_t2)

        return {
            'logit':        logit,
            'feat_map':     feat_map,          # [B, inner_dim, H, W]  — for proto loss
            'feat_t1_deep': feats_t1[-1],      # [B, C4, H/8, W/8]
            'feat_t2_deep': feats_t2[-1],      # [B, C4, H/8, W/8]
        }

    def sample_unlab_positions(self, unlab_indicator, feat_h, feat_w, n_ssl_pixels=2048):
        """
        Sample pixel positions from unlabeled region at feature-map resolution.

        Returns: pos [N, 2] (row, col in feature-map coordinates)
        """
        unlab_down = F.interpolate(
            unlab_indicator.unsqueeze(1).float(), size=(feat_h, feat_w),
            mode='nearest').squeeze(1)[0]           # [Hf, Wf]

        pos = unlab_down.nonzero(as_tuple=False)    # [M, 2]
        if len(pos) == 0:
            return None

        if len(pos) > n_ssl_pixels:
            idx = torch.randperm(len(pos), device=pos.device)[:n_ssl_pixels]
            pos = pos[idx]
        return pos

    def extract_pixel_feats(self, feat_t1_deep, feat_t2_deep, pos):
        """
        Extract per-pixel features at given positions.

        pos: [N, 2]  (row, col)
        Returns f1 [N, C4], f2 [N, C4], diff [N, C4]
        """
        rows, cols = pos[:, 0], pos[:, 1]
        f1   = feat_t1_deep[0, :, rows, cols].T   # [N, C4]
        f2   = feat_t2_deep[0, :, rows, cols].T   # [N, C4]
        diff = (f1 - f2).abs()                     # [N, C4]
        return f1, f2, diff

    def get_ssl_pixel_feats(self, feat_t1_deep, feat_t2_deep, unlab_indicator,
                            n_ssl_pixels=2048, pos=None):
        """
        Sample unlabeled pixels and return T1/T2/diff features.

        If pos is provided, use those positions (for teacher to match student positions).
        Returns: f1 [N,C], f2 [N,C], diff [N,C], pos [N,2]
        """
        B, C, Hf, Wf = feat_t1_deep.shape

        if pos is None:
            pos = self.sample_unlab_positions(unlab_indicator, Hf, Wf, n_ssl_pixels)

        if pos is None:
            return None, None, None, None

        f1, f2, diff = self.extract_pixel_feats(feat_t1_deep, feat_t2_deep, pos)
        return f1, f2, diff, pos
