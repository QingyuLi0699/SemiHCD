import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
from .dataset import DualTemporalDataset


def load_farm(t1t2_mat_path, gt_mat_path):
    d    = sio.loadmat(t1t2_mat_path)
    gt   = sio.loadmat(gt_mat_path)
    t1   = d['T1'].astype(np.float32)            # (420, 140, 154)
    t2   = d['T2'].astype(np.float32)            # (420, 140, 154)
    mask = gt['Multitype'].astype(np.int64)      # (420, 140) classes 0-3
    return t1, t2, mask


class FarmLoader:
    def __init__(self, t1t2_mat_path, gt_mat_path, split_ratio=0.01, np_seed=2333):
        t1, t2, mask = load_farm(t1t2_mat_path, gt_mat_path)
        self.train_dataset = DualTemporalDataset(
            t1, t2, mask, training=True,
            np_seed=np_seed, split_ratio=split_ratio, num_classes=4)
        self.test_dataset = DualTemporalDataset(
            t1, t2, mask, training=False,
            np_seed=np_seed, split_ratio=split_ratio, num_classes=4)

    def get_loaders(self, num_workers=0):
        train_loader = DataLoader(self.train_dataset, batch_size=1,
                                  shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(self.test_dataset,  batch_size=1,
                                  shuffle=False, num_workers=num_workers)
        return train_loader, test_loader
