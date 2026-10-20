"""
MambaSemiHSCD (v3) — SemiHSCD with Siamese VMamba encoder.

Architecture change vs v1/v2:
  Encoder: repeat_block(SEBlock + Conv3x3)  →  repeat_vss(VSSBlock with SS2D)
  Everything else (CrossTemporalFusion, FPNDecoder, PixelProjHead) is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_encoder import repeat_vss
from model.semi_hscd import (
    conv3x3_gn_relu, downsample2x, repeat_block, SEBlock,
    CrossTemporalFusion, FPNDecoder, PixelProjHead,
)


# ---------------------------------------------------------------------------
# Siamese VMamba Encoder
# ---------------------------------------------------------------------------

class SiameseMambaEncoder(nn.Module):
    """
    Same interface as SiameseFPNEncoder but replaces repeat_block with VSSBlocks.
    Returns 4 feature maps at scales 1/1, 1/2, 1/4, 1/8.
    """

    def __init__(self, in_channels,
                 block_channels=(96, 128, 192, 256),
                 num_blocks=(1, 1, 1, 1),
                 reduction_ratio=1.0,
                 d_state=16,
                 expand=1):
        super().__init__()
        r  = int(16 * reduction_ratio)
        c1 = int(block_channels[0] * reduction_ratio / r) * r
        c2 = int(block_channels[1] * reduction_ratio / r) * r
        c3 = int(block_channels[2] * reduction_ratio / r) * r
        c4 = int(block_channels[3] * reduction_ratio / r) * r
        self.out_channels = [c1, c2, c3, c4]

        # Stem: single Conv3x3+GN+ReLU (same as FreeNet)
        self.stem = conv3x3_gn_relu(in_channels, c1, r)

        # Stage 1 keeps CNN (full resolution — sequence too long for SSM)
        self.stage1 = repeat_block(c1, r, num_blocks[0])
        self.down1  = downsample2x(c1, c2)
        # Stages 2-4 use VSSBlocks (half-res and below — sequence manageable)
        self.stage2 = repeat_vss(c2, num_blocks[1], d_state=d_state, expand=expand)
        self.down2  = downsample2x(c2, c3)
        self.stage3 = repeat_vss(c3, num_blocks[2], d_state=d_state, expand=expand)
        self.down3  = downsample2x(c3, c4)
        self.stage4 = repeat_vss(c4, num_blocks[3], d_state=d_state, expand=expand)

    def forward(self, x):
        f1 = self.stage1(self.stem(x))      # [B, c1, H,   W  ]
        f2 = self.stage2(self.down1(f1))    # [B, c2, H/2, W/2]
        f3 = self.stage3(self.down2(f2))    # [B, c3, H/4, W/4]
        f4 = self.stage4(self.down3(f3))    # [B, c4, H/8, W/8]
        return [f1, f2, f3, f4]


# ---------------------------------------------------------------------------
# Full MambaSemiHSCD model
# ---------------------------------------------------------------------------

class MambaSemiHSCD(nn.Module):
    """
    Semi-supervised hyperspectral change detection with VMamba encoder.
    Drop-in replacement for SemiHSCD — same forward API.
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
                 d_state=16,
                 expand=1,
                 norm_last_layer=True):
        super().__init__()

        self.encoder = SiameseMambaEncoder(
            in_channels, block_channels, num_blocks, reduction_ratio,
            d_state=d_state, expand=expand)
        enc_ch = self.encoder.out_channels

        self.fusions = nn.ModuleList([CrossTemporalFusion(c) for c in enc_ch])
        self.decoder = FPNDecoder(enc_ch, inner_dim)
        self.cls_head  = nn.Conv2d(inner_dim, num_classes, 1)
        self.proj_head = PixelProjHead(enc_ch[-1], proj_hidden_dim, proj_out_dim, norm_last_layer)

    # ── same API as SemiHSCD ────────────────────────────────────────────────

    def encode(self, t1, t2):
        return self.encoder(t1), self.encoder(t2)

    def classify(self, feats_t1, feats_t2):
        fused    = [f(f1, f2) for f, f1, f2 in zip(self.fusions, feats_t1, feats_t2)]
        feat_map = self.decoder(fused)
        return self.cls_head(feat_map), feat_map

    def forward(self, t1, t2):
        feats_t1, feats_t2 = self.encode(t1, t2)
        logit, _ = self.classify(feats_t1, feats_t2)
        return torch.softmax(logit, dim=1)

    def forward_train(self, t1, t2, mask, train_indicator, unlab_indicator,
                      proj_head_teacher=None, n_ssl_pixels=2048):
        feats_t1, feats_t2 = self.encode(t1, t2)
        logit, feat_map    = self.classify(feats_t1, feats_t2)
        return {
            'logit':        logit,
            'feat_map':     feat_map,
            'feat_t1_deep': feats_t1[-1],
            'feat_t2_deep': feats_t2[-1],
        }

    def sample_unlab_positions(self, unlab_indicator, feat_h, feat_w, n_ssl_pixels=2048):
        unlab_down = F.interpolate(
            unlab_indicator.unsqueeze(1).float(), size=(feat_h, feat_w),
            mode='nearest').squeeze(1)[0]
        pos = unlab_down.nonzero(as_tuple=False)
        if len(pos) == 0:
            return None
        if len(pos) > n_ssl_pixels:
            idx = torch.randperm(len(pos), device=pos.device)[:n_ssl_pixels]
            pos = pos[idx]
        return pos

    def extract_pixel_feats(self, feat_t1_deep, feat_t2_deep, pos):
        rows, cols = pos[:, 0], pos[:, 1]
        f1   = feat_t1_deep[0, :, rows, cols].T
        f2   = feat_t2_deep[0, :, rows, cols].T
        diff = (f1 - f2).abs()
        return f1, f2, diff

    def get_ssl_pixel_feats(self, feat_t1_deep, feat_t2_deep, unlab_indicator,
                            n_ssl_pixels=2048, pos=None):
        B, C, Hf, Wf = feat_t1_deep.shape
        if pos is None:
            pos = self.sample_unlab_positions(unlab_indicator, Hf, Wf, n_ssl_pixels)
        if pos is None:
            return None, None, None, None
        f1, f2, diff = self.extract_pixel_feats(feat_t1_deep, feat_t2_deep, pos)
        return f1, f2, diff, pos
