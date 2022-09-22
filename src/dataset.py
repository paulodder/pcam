from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from src.mean__std import get_mean__std

from src.constants import DDIR, SPLIT_NAME2FNAME, ACCEPTED_MASKS


class PcamDataset(Dataset):
    def __init__(self, x_path, y_path, meta_path, mask_path=None):

        mean__std = get_mean__std()
        self.mean = mean__std[0][:, None][:, None]
        self.std = mean__std[1][:, None][:, None]
        self.mask_path = mask_path

        x = h5py.File(x_path, "r")["x"][:].squeeze()
        N, H, W, C = x.shape
        self.x = x.reshape(N, C, H, W)

        if mask_path is not None:
            if Path(mask_path).exists():
                with open(mask_path, "rb") as f:
                    self.mask = pickle.load(f)
            else:
                print(mask_path, "is unavailable")
                self.mask_path = None

        self.y = h5py.File(y_path, "r")["y"][:].squeeze()
        self.meta = pd.read_csv(meta_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = (self.x[idx] - self.mean) / self.std
        if self.mask_path is not None:
            x = np.concatenate((x, self.mask[idx][None, :]))
        y = [self.y[idx]]
        return Tensor(x), Tensor(y)


def make_mask_fpath(split_name, mask_type):
    return Path(DDIR) / f"{mask_type}_{split_name}.pkl"


def get_dataset(split_name, mask_type=None):
    fpath_x, fpath_y, fpath_meta = (
        DDIR / x for x in SPLIT_NAME2FNAME[split_name]
    )

    if mask_type is None:
        fpath_mask = None
    elif mask_type in ACCEPTED_MASKS:
        fpath_mask = make_mask_fpath(split_name, mask_type)
    else:
        print(f"mask type {mask_type} is not accepted")
        return

    ds = PcamDataset(fpath_x, fpath_y, fpath_meta, fpath_mask)
    return ds


def get_dataloader(split_name, batch_size=64):
    ds = get_dataset(split_name)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dl
