import torch
from torch.utils.data import Dataset
import numpy as np
from simplecv.data.preprocess import divisible_pad


def normalize_bands(x):
    """Per-band min-max normalization. x: [H, W, C] float32."""
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


def fixed_num_sample(gt_mask, split_ratio, num_classes, seed):
    rs = np.random.RandomState(seed)
    gt_flat = gt_mask.ravel()
    train_ind = np.zeros_like(gt_flat, dtype=np.float32)
    test_ind  = np.zeros_like(gt_flat, dtype=np.float32)

    for cls in range(num_classes):          # 0-indexed classes including 0
        inds = np.where(gt_flat == cls)[0]
        if len(inds) == 0:
            continue
        rs.shuffle(inds)
        n_train = max(1, round(len(inds) * split_ratio))
        train_ind[inds[:n_train]] = 1
        test_ind[inds[n_train:]]  = 1

    return train_ind.reshape(gt_mask.shape), test_ind.reshape(gt_mask.shape)


class DualTemporalDataset(Dataset):
    """Full-image dataset returning T1 and T2 separately for semi-supervised CD."""

    def __init__(self, t1, t2, mask, training,
                 np_seed=2333, split_ratio=0.01, num_classes=7):
        self.training   = training
        self.num_classes = num_classes
        self.orig_size  = (t1.shape[0], t1.shape[1])

        # Normalize each temporal image independently
        t1 = normalize_bands(t1.astype(np.float32))
        t2 = normalize_bands(t2.astype(np.float32))

        C = t1.shape[-1]  # spectral bands per temporal image

        # Labeled / unlabeled split
        train_ind, test_ind = fixed_num_sample(mask, split_ratio, num_classes, np_seed)

        # Pad everything to be divisible by 16 in one call
        stack = np.concatenate([
            t1.transpose(2, 0, 1),         # [C,  H, W]
            t2.transpose(2, 0, 1),         # [C,  H, W]
            mask[None].astype(np.float32), # [1,  H, W]
            train_ind[None],               # [1,  H, W]
            test_ind[None],                # [1,  H, W]
        ], axis=0)

        blob = divisible_pad([stack], 16, False)[0]  # [2C+3, pH, pW]

        self.t1_pad   = torch.from_numpy(blob[:C]).float()        # [C, pH, pW]
        self.t2_pad   = torch.from_numpy(blob[C:2*C]).float()     # [C, pH, pW]
        self.mask_pad = torch.from_numpy(blob[2*C]).long()        # [pH, pW]

        train_map = blob[2*C + 1]    # [pH, pW]
        test_map  = blob[2*C + 2]    # [pH, pW]

        # Unlabeled: pixels inside original image bounds but not in training set
        orig_H, orig_W = self.orig_size
        pad_region = np.zeros(self.mask_pad.shape, dtype=np.float32)
        pad_region[:orig_H, :orig_W] = 1.0
        unlab_map = pad_region * (1.0 - train_map)

        self.train_indicator = torch.from_numpy(train_map).float()
        self.test_indicator  = torch.from_numpy(test_map).float()
        self.unlab_indicator = torch.from_numpy(unlab_map).float()

        labeled  = int(train_map.sum())
        unlabeled = int(unlab_map.sum())
        print(f"[Dataset] orig={orig_H}x{orig_W} padded={self.mask_pad.shape[0]}x{self.mask_pad.shape[1]}"
              f" labeled={labeled} unlabeled={unlabeled}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.training:
            return (self.t1_pad, self.t2_pad, self.mask_pad,
                    self.train_indicator, self.unlab_indicator)
        else:
            return (self.t1_pad, self.t2_pad, self.mask_pad,
                    self.test_indicator, self.unlab_indicator)
