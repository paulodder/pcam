from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from src.mean__std import get_mean__std
from src import utils
from src.constants import (
    DDIR,
    SPLIT_NAME2FNAME,
    ACCEPTED_MASKS,
    ACCEPTED_PREPROCESS,
)


class PcamDataset(Dataset):
    def __init__(
        self, x_path, y_path, meta_path, mask_path, binary_mask=False
    ):
        mean__std = get_mean__std()
        self.mean = mean__std[0][:, None][:, None]
        self.std = mean__std[1][:, None][:, None]
        self.mask_path = mask_path

        if x_path.suffix == ".h5":
            x = h5py.File(x_path, "r")["x"][:].squeeze()
        elif x_path.suffix == ".pkl":
            with open(x_path, "rb") as f:
                x = pickle.load(f)
        else:
            print("x_path filetype not supported")

        self.x = x.transpose(0, 3, 1, 2)

        if mask_path is not None:
            mask_path = Path(mask_path)
            if not mask_path.exists():
                print(mask_path, "is unavailable")
                self.mask_path = None
            else:
                self.mask = utils.load(mask_path)
        if binary_mask:
            bmask = np.zeros((96, 96))
            bmask[32:64, 32:64] = 1.0
            self.binary_mask = (bmask - bmask.mean()) / bmask.std()
        else:
            self.binary_mask = None
        self.y = h5py.File(y_path, "r")["y"][:].squeeze()
        self.meta = pd.read_csv(meta_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = (self.x[idx] - self.mean) / self.std
        if self.mask_path is not None:
            x = np.concatenate((x, self.mask[idx][None, :]))
        if self.binary_mask is not None:
            x = np.concatenate((x, self.binary_mask[None, :]))

        y = torch.zeros(2)
        y[int(self.y[idx])] = 1
        return Tensor(x), Tensor(y)


def make_mask_fpath(split_name, mask_type):
    if mask_type in ["pannuke-type"]:
        extension = "pt"
    else:
        extension = "pkl"
    return DDIR / f"{mask_type}_{split_name}.{extension}"


def make_prepr_fpath(split_name, preprocess):
    return DDIR / f"{preprocess}_{split_name}_x.pkl"


def get_dataset(
    split_name, mask_type=None, preprocess=None, binary_mask=False
):
    """
    Create a PcamDataset instance

    args:
    - split_name (str): one of ["train", "test", "validation"]
    - mask_type (str) : what mask should be added to the data, can be one
                        of src.constants.ACCEPTED_MASKS or None (default)
    - preprocess (str): what preprocessing should be applied to the data, can be
                        one of src.constants.ACCEPTED_PREPROCESS or None
                        (default)
    - binary_mask (bool): if a binary mask corresponding to the label region
                          should be added
    """
    fpath_x, fpath_y, fpath_meta = (
        DDIR / x for x in SPLIT_NAME2FNAME[split_name]
    )

    if preprocess in ACCEPTED_PREPROCESS:
        fpath_x = make_prepr_fpath(split_name, preprocess)

    if mask_type is None:
        fpath_mask = None
    elif mask_type in ACCEPTED_MASKS:
        fpath_mask = make_mask_fpath(split_name, mask_type)
    elif mask_type is not None:
        print(f"mask type {mask_type} is not accepted")
        return
    else:
        fpath_mask = None

    ds = PcamDataset(fpath_x, fpath_y, fpath_meta, fpath_mask, binary_mask)
    return ds


def get_dataloader(split_name, mask_type, batch_size, binary_mask=False):
    shuffle = True if split_name == "train" else False
    ds = get_dataset(split_name, mask_type, binary_mask=binary_mask)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dl


if __name__ == "__main__":
    dl = get_dataloader("train", None, 10000)
    breakpoint()
