import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', class_weight=None):
        """
        class_weight: [K] tensor of per-class weights for inverse-frequency rebalancing.
                      None = uniform. Passed to F.cross_entropy's `weight` arg.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def forward(self, logits, labels, weight=None):
        """
        logits : [B, K, H, W]  raw (un-softmaxed)
        labels : [B, H, W]     long, values 0..K-1
        weight : [B, H, W]     float, per-pixel weight mask (train_indicator)
        """
        ce = F.cross_entropy(logits, labels,
                             weight=self.class_weight, reduction='none')  # [B, H, W]
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce                        # [B, H, W]

        if weight is not None:
            fl = fl * weight
            denom = weight.sum().clamp(min=1e-6)
            return fl.sum() / denom

        return fl.mean()
