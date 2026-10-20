"""
SemiHSCDv4 — CNN + Mamba hybrid, semi-supervised HSI change detection.

What changed vs V3 (pure Mamba encoder):
  ① SpectralMambaStem  — bidirectional SSM along 154 bands per image;
                          produces per-band gate capturing inter-band correlations
                          (water-absorption dips, red-edge slopes) impossible
                          for 1×1 Conv / SE to model.
  ② ConvMambaBlock     — parallel DWConv3×3 (local) + SS2D (global) at stages 2-4;
                          restores CNN translation-invariance lost in pure-Mamba V3.
  ③ CrossTemporalMambaFusion — at stages 2-4; adds global spatial scan over (T1−T2)
                          so the model can see where change is dense before committing
                          to a per-pixel prediction.
  Stage 1 (full resolution) retains pure CNN + original CrossTemporalFusion for speed.

Training API is identical to SemiHSCD / MambaSemiHSCD (same forward_train /
get_ssl_pixel_feats signature) so train.py needs no logic changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conv_mamba import (
    SpectralMambaStem, repeat_conv_mamba, CrossTemporalMambaFusion,
)
from model.semi_hscd import (
    repeat_block, downsample2x,
    CrossTemporalFusion, FPNDecoder, PixelProjHead,
)


class SiameseCNNMambaEncoder(nn.Module):
    """
    Siamese encoder (shared weights for T1 and T2).

    stem    : SpectralMambaStem  — band-gate + Conv3×3
    stage 1 : repeat_block (pure CNN, full resolution)
    stage 2-4: repeat_conv_mamba (DWConv ‖ SS2D, gated)

    Returns [f1, f2, f3, f4] at scales 1/1, 1/2, 1/4, 1/8.
    """
    def __init__(self, in_channels,
                 block_channels=(96, 128, 192, 256),
                 num_blocks=(1, 1, 1, 1),
                 reduction_ratio=1.0,
                 d_state=16, expand=1, spec_d_inner=32):
        super().__init__()
        r         = int(16 * reduction_ratio)
        c1, c2, c3, c4 = [int(c * reduction_ratio / r) * r for c in block_channels]
        self.out_channels = [c1, c2, c3, c4]

        self.stem   = SpectralMambaStem(in_channels, c1, r,
                                        d_state=d_state, spec_d_inner=spec_d_inner)
        self.stage1 = repeat_block(c1, r, num_blocks[0])
        self.down1  = downsample2x(c1, c2)
        self.stage2 = repeat_conv_mamba(c2, num_blocks[1], d_state, expand, r)
        self.down2  = downsample2x(c2, c3)
        self.stage3 = repeat_conv_mamba(c3, num_blocks[2], d_state, expand, r)
        self.down3  = downsample2x(c3, c4)
        self.stage4 = repeat_conv_mamba(c4, num_blocks[3], d_state, expand, r)

    def forward(self, x):
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(self.down3(f3))
        return [f1, f2, f3, f4]


class SemiHSCDv4(nn.Module):
    """
    Semi-supervised HSI change detection — CNN + Mamba hybrid (V4).
    Drop-in replacement for SemiHSCD / MambaSemiHSCD.
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
                 spec_d_inner=32,
                 norm_last_layer=True):
        super().__init__()

        self.encoder = SiameseCNNMambaEncoder(
            in_channels, block_channels, num_blocks, reduction_ratio,
            d_state=d_state, expand=expand, spec_d_inner=spec_d_inner)
        enc_ch = self.encoder.out_channels

        # Stage-1 fusion: original CNN (fast; full-res SSM too costly)
        # Stages 2-4: CNN + Mamba (global change context at lower resolutions)
        self.fusions = nn.ModuleList([
            CrossTemporalFusion(enc_ch[0]),
            CrossTemporalMambaFusion(enc_ch[1], d_state=d_state, expand=expand),
            CrossTemporalMambaFusion(enc_ch[2], d_state=d_state, expand=expand),
            CrossTemporalMambaFusion(enc_ch[3], d_state=d_state, expand=expand),
        ])

        self.decoder   = FPNDecoder(enc_ch, inner_dim)
        self.cls_head  = nn.Conv2d(inner_dim, num_classes, 1)
        self.proj_head = PixelProjHead(enc_ch[-1], proj_hidden_dim, proj_out_dim,
                                       norm_last_layer)

    # ── forward API (identical to SemiHSCD / MambaSemiHSCD) ────────────

    def encode(self, t1, t2):
        return self.encoder(t1), self.encoder(t2)

    def classify(self, feats_t1, feats_t2):
        fused    = [f(f1, f2) for f, f1, f2 in zip(self.fusions, feats_t1, feats_t2)]
        feat_map = self.decoder(fused)
        return self.cls_head(feat_map), feat_map

    def forward(self, t1, t2):
        f1, f2 = self.encode(t1, t2)
        logit, _ = self.classify(f1, f2)
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
