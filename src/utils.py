from decouple import config
from pathlib import Path
import numpy as np
import pickle
import torch
from PIL import Image

DDIR = Path(config("DATA_DIR"))


def get_img(sample):
    img_ten = np.reshape(sample, (96, 96, 3))
    img = Image.fromarray(img_ten, "RGB")
    return img


def load(fpath):
    fpath = Path(fpath)
    if fpath.suffix == ".pkl":
        with open(fpath, "rb") as f:
            out = pickle.load(f)
        return out
    elif fpath.suffix == ".pt":
        return torch.load(fpath)


def get_mask_slug(mask_path):
    if "otsu_split" in mask_path.name:
        return "otsu_split"
    if "pannuke" in mask_path.name:
        return "pannuke-type"
