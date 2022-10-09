from argparse import ArgumentParser
import pandas as pd
import re
import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import sys
from decouple import config
from src.dataset import get_dataloader

from src.hovernet_utils import InferManager

HN_MODEL_DIR = Path(config("PROJECT_DIR")) / "pretrained_hovernet_weights"
HN_MODEL_DIR.mkdir(exist_ok=True)

DATA_OUT_DIR = Path(config("DATA_DIR")) / "hovernet_results"

MODEL_NAME2FPATH = {
    "pannuke-type": HN_MODEL_DIR / "hovernet_fast_pannuke_type_tf2pytorch.tar",
    "consep-notype": HN_MODEL_DIR
    / "hovernet_original_consep_notype_tf2pytorch.tar",
    "consep-type": HN_MODEL_DIR
    / "hovernet_original_consep_type_tf2pytorch.tar",
    "kumar-notype": HN_MODEL_DIR
    / "hovernet_original_kumar_notype_tf2pytorch.tar",
    "monusac-type": HN_MODEL_DIR / "hovernet_fast_monusac_type_tf2pytorch.tar",
}


def get_parser():
    parser = ArgumentParser()
    # dataloader arguments
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--split_name", default="test", type=str)
    # optimizer arguments
    parser.add_argument("--model_name", default="pannuke-type", type=str)
    parser.add_argument("--save_every", default=200, type=int)

    return parser


def get_mock_args(overwrite_kwargs=dict()):
    "for dev purposes"

    # args, _ = parser.parse_known_args(None, None)

    class Args(object):
        pass

    mock_args = Args()
    for name, default_value in overwrite_kwargs.items():
        setattr(mock_args, name, overwrite_kwargs.get(name, default_value))
    return mock_args


def parse_args():
    return get_parser().parse_args()


def get_mode(model_name):
    if "fast" in model_name:
        return "fast"
    return "original"


def get_tile_and_patch_size(mode):
    if mode == "fast":
        return (256, 164)
    else:
        return (270, 80)


def get_in_and_out_dir(model_name, split_name):
    base_dir = DATA_OUT_DIR / model_name
    base_dir.mkdir(exist_ok=True, parents=True)
    out_dir = base_dir / "out" / split_name
    in_dir = base_dir / "in" / split_name
    out_dir.mkdir(exist_ok=True)
    in_dir.mkdir(exist_ok=True)
    return in_dir, out_dir


def register_input_imgs(out_dir):
    pass


def register_imgs_processed(out_dir):
    pass


def get_imgs_to_preprocess_and_predict(in_dir, out_dir):
    pass


class ImageManager(object):
    def __init__(self, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir / "mat"

    def register_preprocessed_imgs(self):
        preprocessed_images = pd.DataFrame(
            {
                "img_no_pre": [
                    re.search("[0-9]+", x.name).group()
                    for x in self.in_dir.glob("*png")
                ],
                "present_pre": True,
            }
        )
        return preprocessed_images

    def register_processed_imgs(self):
        preprocessed_images = pd.DataFrame(
            {
                "img_no_proc": [
                    re.search("[0-9]+", x.name).group()
                    for x in self.out_dir.glob("*mat")
                ],
                "present_proc": True,
            }
        )
        return preprocessed_images

    def get_action2img_indexes(self, all_img_nos):
        processed_imgs = self.register_processed_imgs()
        preprocessed_imgs = self.register_preprocessed_imgs()
        registry = preprocessed_imgs.merge(
            processed_imgs,
            left_on=["img_no_pre"],
            right_on=["img_no_proc"],
            how="outer",
        )
        registry["present_proc"] = registry["present_proc"].fillna(False)
        registry["present_pre"] = registry["present_pre"].fillna(False)
        action2img_indexes = {
            "remove_pre": registry["img_no_pre"][
                (registry["present_proc"] & registry["present_pre"])
            ]
            .astype(int)
            .tolist(),
        }
        img_nos_proc = set(
            list(
                registry["img_no_proc"].dropna().astype(int).tolist()
                + registry["img_no_pre"].dropna().astype(int).tolist()
            )
        )
        img_nos_to_add = []
        for img_no in all_img_nos:
            if img_no not in img_nos_proc:
                img_nos_to_add.append(img_no)
        action2img_indexes["add_pre"] = img_nos_to_add

        # add those for which no pre and no post is present and it is in the
        # provided full list
        return action2img_indexes


# def register_images_preprocessed(self):


def preprocess_images(in_dir, X, indexes):
    for index, X in zip(indexes, dataset[indexes]):
        img = Image.fromarray(X.reshape((96, 96, 3)))
        img.save(in_dir / f"img{index}.png")


def remove_preprocesed_images(in_dir, indexes):
    for index in indexes:
        os.remove(str(in_dir / f"img{index}.png"))


if __name__ == "__main__":
    args = parse_args()
    dataset = get_dataloader("test", None, 32).dataset.x
    mode = get_mode(str(MODEL_NAME2FPATH[args.model_name]))
    tile_size, patch_size = get_tile_and_patch_size(mode)
    in_dir, out_dir = get_in_and_out_dir(args.model_name, args.split_name)
    # now determine which images still need to be done
    img_manager = ImageManager(in_dir, out_dir)
    action2img_indexes = img_manager.get_action2img_indexes(
        list(range(dataset.shape[0]))
    )
    print(
        {
            action: len(img_indexes)
            for action, img_indexes in action2img_indexes.items()
        }
    )
    remove_preprocesed_images(in_dir, action2img_indexes["remove_pre"])
    preprocess_images(in_dir, dataset, action2img_indexes["add_pre"])
    method_args = {
        "method": {
            "model_args": {
                "mode": mode,
                "freeze": False,
            },
            "model_path": MODEL_NAME2FPATH[args.model_name],
        },
        "type_info_path": None
        # if args["type_info_path"] == ""
        # else args["type_info_path"],
    }
    run_args = {
        "save_every": args.save_every,
        "batch_size": args.batch_size,
        "nr_inference_workers": 0,
        "nr_post_proc_workers": 0,
        "patch_input_shape": tile_size,
        "patch_output_shape": patch_size,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mem_usage": 0.85,
        "draw_dot": False,
        "save_qupath": False,
        "save_raw_map": False,
    }
    infer = InferManager(**method_args)
    infer.process_file_list(run_args)
