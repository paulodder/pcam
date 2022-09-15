from torch import Tensor
from torch.utils.data import Dataset
import h5py
import pandas as pd


class PcamDataset(Dataset):
    def __init__(self, x_path, y_path, meta_path):
        x = h5py.File(x_path, "r")["x"][:].squeeze()
        N, H, W, C = x.shape
        self.x = x.reshape(N, C, H, W)
        self.y = h5py.File(y_path, "r")["y"][:].squeeze()
        self.meta = pd.read_csv(meta_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return Tensor(self.x[idx]), Tensor([self.y[idx]])
