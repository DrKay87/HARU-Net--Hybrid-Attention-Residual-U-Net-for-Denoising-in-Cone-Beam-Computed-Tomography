"""Dataset helpers for paired CBCT denoising experiments.

The original HARU-Net notebook loads paired pickle files containing NumPy
arrays. This module provides the same basic convention in reusable form.
"""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle_array(path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)


class PairedPatchDataset(Dataset):
    """Paired noisy/target patch dataset."""

    def __init__(self, inputs, targets):
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must contain the same number of samples")
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = np.asarray(self.inputs[idx], dtype=np.float32)
        y = np.asarray(self.targets[idx], dtype=np.float32)

        if x.ndim == 2:
            x = x[None, ...]
        if y.ndim == 2:
            y = y[None, ...]

        return torch.from_numpy(x), torch.from_numpy(y)
