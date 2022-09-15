from torch import Tensor
from torch.utils.data import Dataset
import h5py
import pandas as pd
from src.mean__std import get_mean__std

from constants import DDIR, SPLIT_NAME2FNAME


class PcamDataset(Dataset):
    def __init__(self, x_path, y_path, meta_path):

        mean__std = get_mean__std()
        self.mean = mean__std[0][:, None][:, None]
        self.std = mean__std[1][:, None][:, None]

        x = h5py.File(x_path, "r")["x"][:].squeeze()
        N, H, W, C = x.shape
        self.x = x.reshape(N, C, H, W)

        self.y = h5py.File(y_path, "r")["y"][:].squeeze()
        self.meta = pd.read_csv(meta_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = (self.x[idx] - self.mean) / self.std
        y = [self.y[idx]]
        return Tensor(x), Tensor(y)


def get_dataset(split_name):
    fpath_x, fpath_y, fpath_meta = (
        DDIR / x for x in SPLIT_NAME2FNAME[split_name]
    )
    ds = PcamDataset(fpath_x, fpath_y, fpath_meta)
    return ds


def get_dataloader(split_name, batch_size=64):
    ds = get_dataset(split_name)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dl
