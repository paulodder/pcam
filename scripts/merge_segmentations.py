from argparse import ArgumentParser

import torch
import numpy as np
import scipy.io

from tqdm import tqdm


from src.hovernet_utils import (
    get_in_and_out_dir,
    extract_img_int,
    get_segmentation_path,
    MODEL_NAME2FPATH,
)
from src.dataset import get_dataset


def get_parser():
    parser = ArgumentParser()
    # dataloader arguments
    parser.add_argument("--split_name", default="test", type=str)
    # optimizer arguments
    parser.add_argument(
        "--model_name",
        default="pannuke-type",
        type=str,
        choices=list(MODEL_NAME2FPATH.keys()),
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    _, out_dir = get_in_and_out_dir(args.model_name, args.split_name)
    out_dir /= "mat"
    # load
    ds = get_dataset(args.split_name)
    nof_samples = ds.x.shape[0]
    out = np.zeros((nof_samples, 96, 96))
    for fpath in tqdm(out_dir.glob("*mat")):
        fpath_img_index = extract_img_int(fpath.name)
        out[fpath_img_index, :, :] = scipy.io.loadmat(fpath)["inst_map"]
    nof_empty_entries = (out == 0).all((1, 2)).sum()
    print(
        f"Warning: {nof_empty_entries} images have all 0's for their segmentation"
    )
    out_fpath = get_segmentation_path(args.model_name, args.split_name)
    print(f"Written to {out_fpath}")
    torch.save(torch.Tensor(out), out_fpath)
