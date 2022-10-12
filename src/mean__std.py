import h5py
from pathlib import Path
from decouple import config
import numpy as np
import pickle as pkl
from src.constants import DDIR


def calc_mean__std():
    print("Calculating mean__std")
    x = h5py.File(DDIR / "camelyonpatch_level_2_split_train_x.h5", "r")["x"][
        :
    ].squeeze()
    mean = np.mean(x, axis=(0, 1, 2))
    std = np.std(x, axis=(0, 1, 2))
    mean__std = (mean, std)
    return mean__std


def get_mean__std():
    path = Path(DDIR / "mean__std.pkl")
    if path.exists():
        with open(path, "rb") as f:
            mean__std = pkl.load(f)
    else:
        mean__std = calc_mean__std()
        with open(path, "wb") as f:
            pkl.dump(mean__std, f)
    return mean__std


def calc_mean__std_otsu():
    print("Calculating mean__std for otsu_mask")
    with open(DDIR / "otsu_split_train.pkl", "rb") as f:
        x = pkl.load(f)
    mean = np.mean(x)
    std = np.std(x)
    mean__std = (mean, std)
    return mean__std


def get_mean__std_otsu():
    path = Path(DDIR / "mean__std_otsu_split.pkl")
    if path.exists():
        with open(path, "rb") as f:
            mean__std = pkl.load(f)
    else:
        mean__std = calc_mean__std_otsu()
        with open(path, "wb") as f:
            pkl.dump(mean__std, f)
    return mean__std
