"""
ModifiedDINOLoss — adapted from MSCSCNet/utils/loss.py

Changes vs. original:
  - Accepts pixel-level feature vectors instead of patch embeddings
  - Same L_u / L_d / ψ formulation, identical math
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedDINOLoss(nn.Module):
    """
    Three-view DINO loss for semi-supervised change detection.

    L_u: Cross-temporal consistency (student_t1 ↔ teacher_t2, and vice versa)
         Weighted by ψ ≈ 1 for unchanged pixels.

    L_d: Change-signature consistency (student_diff ↔ teacher_diff)
         Weighted by (1-ψ) ≈ 1 for changed pixels.

    ψ = ((1 + cosine_sim(teacher_t1, teacher_t2)) / 2) ^ γ
    """

    def __init__(self, out_dim,
                 warmup_teacher_temp=0.04,
                 teacher_temp=0.07,
                 warmup_teacher_temp_epochs=10,
                 nepochs=100,
                 student_temp=0.1,
                 center_momentum=0.9,
                 gamma=2.0,
                 psi_instability_weight=0.0):
        """
        psi_instability_weight: how much to soften psi toward 0.5 (hedge
            between L_u/L_d) when the clean-vs-noised similarity disagree a
            lot for a pixel — i.e. its unchanged/changed judgment is unstable
            under the augmentation actually applied this step. 0.0 = disabled
            (original behavior, fully backward compatible).
        """
        super().__init__()
        self.student_temp   = student_temp
        self.center_momentum = center_momentum
        self.gamma = gamma
        self.psi_instability_weight = psi_instability_weight
        self.register_buffer('center',      torch.zeros(1, out_dim))
        self.register_buffer('diff_center', torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(nepochs - warmup_teacher_temp_epochs, 1)) * teacher_temp,
        ])

    def forward(self, s_t1, s_t2, s_diff,
                      t_t1, t_t2, t_diff, epoch,
                      t_t1_psi=None, t_t2_psi=None, return_stats=False):
        """
        All inputs: [N, out_dim]  (N unlabeled pixels or patches)
        epoch: SSL epoch index (0-based, relative to SSL phase start)
        t_t1_psi, t_t2_psi: optional clean (unaugmented) teacher T1/T2
                  projections, used only for the psi weight so noise on
                  whichever side got augmented doesn't bias the
                  changed/unchanged split. Pass whichever side was actually
                  noised this step (role-swap support); the other stays None
                  and falls back to t_t1/t_t2 (already clean).
        return_stats: if True, also return a dict with psi/L_u/L_d summary
                  stats for logging (formula itself is unchanged either way).
        """
        temp = float(self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)])

        t_t1  = t_t1.detach()
        t_t2  = t_t2.detach()
        t_diff = t_diff.detach()

        # L_u: cross-temporal consistency (unchanged prior)
        L_u = -0.5 * (
            F.log_softmax(s_t1  / self.student_temp, dim=-1) *
            F.softmax((t_t2  - self.center) / temp, dim=-1) +
            F.log_softmax(s_t2  / self.student_temp, dim=-1) *
            F.softmax((t_t1  - self.center) / temp, dim=-1)
        ).sum(dim=-1)  # [N]

        # L_d: change-signature consistency (changed prior)
        L_d = -(
            F.log_softmax(s_diff / self.student_temp, dim=-1) *
            F.softmax((t_diff - self.diff_center) / temp, dim=-1)
        ).sum(dim=-1)  # [N]

        # Dynamic weight ψ based on temporal similarity
        # Use clean teacher T1/T2 (if provided) so aug noise doesn't bias the
        # changed/unchanged split.
        t1_for_psi = t_t1_psi if t_t1_psi is not None else t_t1
        t2_for_psi = t_t2_psi if t_t2_psi is not None else t_t2
        sim_clean = F.cosine_similarity(
            t1_for_psi - self.center,
            t2_for_psi - self.center, dim=-1)     # [N]
        psi = ((1.0 + sim_clean) / 2.0) ** self.gamma   # [N]

        # Uncertainty-aware hedge: if the clean-vs-actually-noised similarity
        # disagree a lot, this pixel's unchanged/changed call is unstable
        # under noise (likely a real change boundary, or already ambiguous)
        # — soften psi toward 0.5 (blend L_u/L_d) instead of committing hard
        # to whatever sim_clean says. When no augmentation was applied this
        # step, sim_noisy == sim_clean, instability == 0, psi is untouched.
        if self.psi_instability_weight > 0.0:
            sim_noisy = F.cosine_similarity(
                t_t1 - self.center, t_t2 - self.center, dim=-1)
            instability = (sim_clean - sim_noisy).abs().clamp(0.0, 1.0)
            blend = self.psi_instability_weight * instability
            psi = psi * (1.0 - blend) + 0.5 * blend

        L_ssl = (psi * L_u + (1.0 - psi) * L_d).mean()

        # Update centers (EMA)
        self._update_center(torch.cat([t_t1, t_t2], dim=0))
        self._update_diff_center(t_diff)

        if return_stats:
            stats = {
                'psi_mean': psi.mean().item(),
                'psi_min':  psi.min().item(),
                'psi_max':  psi.max().item(),
                'psi_std':  psi.std().item() if psi.numel() > 1 else 0.0,
                'loss_u':   L_u.mean().item(),
                'loss_d':   L_d.mean().item(),
            }
            return L_ssl, stats
        return L_ssl

    @torch.no_grad()
    def _update_center(self, teacher_out):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center  = self.center * self.center_momentum + \
                       batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def _update_diff_center(self, teacher_diff):
        batch_center      = teacher_diff.mean(dim=0, keepdim=True)
        self.diff_center  = self.diff_center * self.center_momentum + \
                            batch_center * (1 - self.center_momentum)
