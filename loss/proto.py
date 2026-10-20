"""
ClassPrototypeLoss — supervised prototype contrastive loss.

Maintains per-class EMA prototypes from labeled pixel features (FPN decoder output).
For each labeled pixel: loss = 1 - cos_sim(feat, prototype[label]).

Intuition: even with 1 labeled minority-class sample, the prototype sits at that
sample's feature location and the attraction loss constrains the feature space so
that the classifier sees a dedicated representation for the rare class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPrototypeLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.9):
        """
        momentum : EMA momentum for prototype update (higher = slower adaptation).
                   Lower value (0.9) preferred when labeled samples are very sparse.
        """
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer('prototypes',  torch.zeros(num_classes, feat_dim))
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, feats_norm, labels):
        """
        Update prototypes from labeled pixel features.
        feats_norm : [N, D]  L2-normalized feature vectors
        labels     : [N]     integer class labels 0..K-1
        """
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            cls_mean = F.normalize(feats_norm[mask].mean(0), dim=0)
            if self.initialized[c]:
                self.prototypes[c] = F.normalize(
                    self.momentum * self.prototypes[c] + (1 - self.momentum) * cls_mean, dim=0)
            else:
                self.prototypes[c] = cls_mean
                self.initialized[c] = True

    def forward(self, feats, labels, class_weight=None):
        """
        Compute prototype attraction loss.
        feats        : [N, D]  raw (un-normalized) feature vectors for labeled pixels
        labels       : [N]     integer class labels 0..K-1
        class_weight : [K]     optional per-class multiplier (e.g. boost class 0's
                                pull without changing the relative weight of the
                                other classes). None = plain mean (unweighted,
                                naturally dominated by whichever class has the most
                                pixels in the sample).
        """
        feats_norm = F.normalize(feats, dim=1)          # [N, D]

        # Update prototypes first (always, including first call)
        with torch.no_grad():
            self.update(feats_norm.detach(), labels)

        # Loss: 1 - cos_sim(feat, prototype[label])
        proto_targets = self.prototypes[labels].detach()  # [N, D]  stop-grad on prototypes
        sims = (feats_norm * proto_targets).sum(dim=1)    # [N]
        loss_per_pixel = 1.0 - sims

        if class_weight is not None:
            w = class_weight[labels]
            return (loss_per_pixel * w).sum() / w.sum().clamp(min=1e-6)
        return loss_per_pixel.mean()
