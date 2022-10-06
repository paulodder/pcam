import sys
from pathlib import Path
from decouple import config

sys.path.insert(0, str(Path(config("SRC_DIR")) / "hover_net"))
import logging
import multiprocessing
from multiprocessing import Lock, Pool
from misc.utils import get_bounding_box, remove_small_objects
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

multiprocessing.set_start_method(
    "spawn", True
)  # ! must be at top for VScode debugging
from concurrent.futures import (
    FIRST_EXCEPTION,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from functools import reduce
from importlib import import_module
from multiprocessing import Lock, Pool
from scipy.ndimage import filters, measurements

import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from dataloader.infer_loader import (
    SerializeFileList,
)
from misc.utils import (
    color_deconvolution,
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.viz_utils import colorize, visualize_instances_dict
from skimage import color
from skimage.segmentation import watershed

import convert_format

import argparse
import glob
import json
import math
import multiprocessing
import os
import re
import sys
from importlib import import_module
from multiprocessing import Lock, Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import tqdm

from run_utils.utils import convert_pytorch_checkpoint

import numpy as np

from PIL import Image
import sys

from src.dataset import get_dataloader


from models.hovernet.net_desc import create_model, HoVerNet
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hovernet.net_utils import (
    DenseBlock,
    Net,
    ResidualBlock,
    TFSamepaddingLayer,
    UpSample2x,
)
from models.hovernet.utils import crop_op, crop_to_shape


def __proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    v_dir = cv2.normalize(
        v_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


####
def process(pred_map, nr_types=None, return_centroids=False):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output

    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction

    """
    return cv2.resize(pred_map, (96, 96)), dict()

    # rs = str(random.random())[2:5]
    # with open(f"/tmp/process_{rs}.pkl", "wb") as f:
    #     f.write(pickle.dumps(pred_map))
    #     print(rs)

    # if nr_types is not None:
    #     pred_type = pred_map[..., :1]
    #     pred_inst = pred_map[..., 1:]
    #     pred_type = pred_type.astype(np.int32)
    # else:
    #     pred_inst = pred_map
    # pred_inst = np.squeeze(pred_inst)
    # # breakpoint()
    # pred_inst = __proc_np_hv(pred_inst)
    # inst_info_dict = None
    # if return_centroids or nr_types is not None:
    #     inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
    #     inst_info_dict = {}
    #     for inst_id in inst_id_list:
    #         inst_map = pred_inst == inst_id
    #         # TODO: chane format of bbox output
    #         rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
    #         inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
    #         inst_map = inst_map[
    #             inst_bbox[0][0] : inst_bbox[1][0],
    #             inst_bbox[0][1] : inst_bbox[1][1],
    #         ]
    #         inst_map = inst_map.astype(np.uint8)
    #         inst_moment = cv2.moments(inst_map)
    #         inst_contour = cv2.findContours(
    #             inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    #         )
    #         # * opencv protocol format may break
    #         inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
    #         # < 3 points dont make a contour, so skip, likely artifact too
    #         # as the contours obtained via approximation => too small or sthg
    #         if inst_contour.shape[0] < 3:
    #             continue
    #         if len(inst_contour.shape) != 2:
    #             continue  # ! check for trickery shape
    #         inst_centroid = [
    #             (inst_moment["m10"] / inst_moment["m00"]),
    #             (inst_moment["m01"] / inst_moment["m00"]),
    #         ]
    #         inst_centroid = np.array(inst_centroid)
    #         inst_contour[:, 0] += inst_bbox[0][1]  # X
    #         inst_contour[:, 1] += inst_bbox[0][0]  # Y
    #         inst_centroid[0] += inst_bbox[0][1]  # X
    #         inst_centroid[1] += inst_bbox[0][0]  # Y
    #         inst_info_dict[inst_id] = {  # inst_id should start at 1
    #             "bbox": inst_bbox,
    #             "centroid": inst_centroid,
    #             "contour": inst_contour,
    #             "type_prob": None,
    #             "type": None,
    #         }

    # if nr_types is not None:
    #     #### * Get class of each instance id, stored at index id-1
    #     for inst_id in list(inst_info_dict.keys()):
    #         rmin, cmin, rmax, cmax = (
    #             inst_info_dict[inst_id]["bbox"]
    #         ).flatten()
    #         inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
    #         inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
    #         inst_map_crop = (
    #             inst_map_crop == inst_id
    #         )  # TODO: duplicated operation, may be expensive
    #         inst_type = inst_type_crop[inst_map_crop]
    #         type_list, type_pixels = np.unique(inst_type, return_counts=True)
    #         type_list = list(zip(type_list, type_pixels))
    #         type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
    #         inst_type = type_list[0][0]
    #         if inst_type == 0:  # ! pick the 2nd most dominant if exist
    #             if len(type_list) > 1:
    #                 inst_type = type_list[1][0]
    #         type_dict = {v[0]: v[1] for v in type_list}
    #         type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
    #         inst_info_dict[inst_id]["type"] = int(inst_type)
    #         inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # # print('here')
    # # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # # inst_id in the dict maps to the same value in the `pred_inst`
    # return pred_inst, inst_info_dict


def _prepare_patching(
    img, window_size, mask_size, return_src_top_corner=False
):
    """Prepare patch information for tile processing.

    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img

    """

    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    if not return_src_top_corner:
        return img, patch_info
    else:
        return img, patch_info, [padt, padl]


####
def _post_process_patches(
    post_proc_func,
    post_proc_kwargs,
    patch_info,
    image_info,
    overlay_kwargs,
):
    """Apply post processing to patches.

    Args:
        post_proc_func: post processing function to use
        post_proc_kwargs: keyword arguments used in post processing function
        patch_info: patch data and associated information
        image_info: input image data and associated information
        overlay_kwargs: overlay keyword arguments

    """
    # re-assemble the prediction, sort according to the patch location within the original image
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
    patch_info, patch_data = zip(*patch_info)

    src_shape = image_info["src_shape"]
    src_image = image_info["src_image"]

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    # breakpoint()
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    # crop back to original shape
    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])
    # * Implicit protocol
    # * a prediction map with instance of ID 1-N
    # * and a dict contain the instance info, access via its ID
    # * each instance may have type
    pred_inst, inst_info_dict = post_proc_func(pred_map, **post_proc_kwargs)
    # breakpoint()
    # overlaid_img = visualize_instances_dict(
    #     src_image.copy(), inst_info_dict, **overlay_kwargs, type_colour=None
    # )
    return (
        image_info["name"],
        pred_map,
        pred_inst,
        inst_info_dict,
    )


class OurHoVerNet(HoVerNet):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, freeze=False, mode="original"):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = None
        self.output_ch = 3

        assert mode == "original" or mode == "fast", (
            "Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`."
            % mode
        )

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == "fast":  # prepend the padding for `fast` mode
            module_list = [
                ("pad", TFSamepaddingLayer(ksize=7, stride=1))
            ] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(
            2048, 1024, 1, stride=1, padding=0, bias=False
        )

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [
                (
                    "conva",
                    nn.Conv2d(
                        1024, 256, ksize, stride=1, padding=0, bias=False
                    ),
                ),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                (
                    "convf",
                    nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
                ),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                (
                    "conva",
                    nn.Conv2d(
                        512, 128, ksize, stride=1, padding=0, bias=False
                    ),
                ),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                (
                    "convf",
                    nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
                ),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                (
                    "conva",
                    nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),
                ),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                (
                    "conv",
                    nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
                ),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict(
                    [
                        ("u3", u3),
                        ("u2", u2),
                        ("u1", u1),
                        ("u0", u0),
                    ]
                )
            )
            return decoder

        ksize = 5 if mode == "original" else 3

        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                    # ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                ]
            )
        )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()


class InferManager(object):
    def __init__(self, **kwargs):
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.__load_model()
        self.nr_types = None  # self.method["model_args"]["nr_types"]
        # create type info name and colour

        # default
        self.type_info_dict = {
            None: ["no label", [0, 0, 0]],
        }

        if self.nr_types is not None and self.type_info_path is not None:
            self.type_info_dict = json.load(open(self.type_info_path, "r"))
            self.type_info_dict = {
                int(k): (v[0], tuple(v[1]))
                for k, v in self.type_info_dict.items()
            }
            # availability check
            for k in range(self.nr_types):
                if k not in self.type_info_dict:
                    assert False, "Not detect type_id=%d defined in json." % k

        if self.nr_types is not None and self.type_info_path is None:
            cmap = plt.get_cmap("hot")
            colour_list = np.arange(self.nr_types, dtype=np.int32)
            colour_list = (cmap(colour_list)[..., :3] * 255).astype(np.uint8)
            # should be compatible out of the box wrt qupath
            self.type_info_dict = {
                k: (str(k), tuple(v)) for k, v in enumerate(colour_list)
            }
        return

    def __load_model(self):
        """Create the model, load the checkpoint and define
        associated run steps to process each data batch.

        """
        model_creator = create_model

        net = model_creator(**self.method["model_args"])
        saved_state_dict = torch.load(self.method["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        net.load_state_dict(saved_state_dict, strict=False)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.hovernet.run_desc")
        run_step = getattr(module_lib, "infer_step")
        # self.net = net
        self.run_step = lambda input_batch: run_step(input_batch, net)

        self.post_proc_func = process
        return

    def __save_json(self, path, old_dict, mag=None):
        new_dict = {}
        for inst_id, inst_info in old_dict.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                # convert to jsonable
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            new_dict[int(inst_id)] = new_inst_info

        json_dict = {
            "mag": mag,
            "nuc": new_dict,
        }  # to sync the format protocol
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return new_dict

    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        assert self.mem_usage < 1.0 and self.mem_usage > 0.0
        # * depend on the number of samples and their size, this may be less efficient
        print(f"self.save_every, {self.save_every}")
        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        file_path_list = glob.glob(patterning("%s/*" % self.input_dir))
        file_path_list.sort()  # ensure same order
        file_path_list = file_path_list
        assert len(file_path_list) > 0, "Not Detected Any Files From Path"

        rm_n_mkdir(self.output_dir + "/json/")
        Path(self.output_dir + "/mat/").mkdir(exist_ok=True, parents=True)
        # rm_n_mkdir()
        # rm_n_mkdir(self.output_dir + "/mat/")
        rm_n_mkdir(self.output_dir + "/overlay/")
        if self.save_qupath:
            rm_n_mkdir(self.output_dir + "/qupath/")

        proc_pool = None
        # if self.nr_post_proc_workers > 0:
        #     proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        while len(file_path_list) > 0:
            hardware_stats = psutil.virtual_memory()
            available_ram = getattr(hardware_stats, "available")
            available_ram = int(available_ram * self.mem_usage)
            # available_ram >> 20 for MB, >> 30 for GB

            # TODO: this portion looks clunky but seems hard to detach into separate func

            # * caching N-files into memory such that their expected (total) memory usage
            # * does not exceed the designated percentage of currently available memory
            # * the expected memory is a factor w.r.t original input file size and
            # * must be manually provided
            file_idx = 0
            use_path_list = []
            cache_image_list = []
            cache_patch_info_list = []
            cache_image_info_list = []
            while len(file_path_list) > 0:
                file_path = file_path_list.pop(0)

                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img, (self.patch_input_shape, self.patch_input_shape)
                )
                src_shape = img.shape

                img, patch_info, top_corner = _prepare_patching(
                    img, self.patch_input_shape, self.patch_output_shape, True
                )

                # breakpoint()

                # plt.imshow(img)
                # plt.clf()

                self_idx = np.full(
                    patch_info.shape[0], file_idx, dtype=np.int32
                )
                patch_info = np.concatenate(
                    [patch_info, self_idx[:, None]], axis=-1
                )
                # ? may be expensive op
                patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
                patch_info = [np.squeeze(p) for p in patch_info]

                # * this factor=5 is only applicable for HoVerNet
                expected_usage = sys.getsizeof(img) * 5
                available_ram -= expected_usage
                if available_ram < 0:
                    break

                file_idx += 1
                # if file_idx == 4: break
                use_path_list.append(file_path)
                cache_image_list.append(img)
                cache_patch_info_list.extend(patch_info)
                # TODO: refactor to explicit protocol
                cache_image_info_list.append(
                    [src_shape, len(patch_info), top_corner]
                )

            # * apply neural net on cached data
            dataset = SerializeFileList(
                cache_image_list, cache_patch_info_list, self.patch_input_shape
            )
            dataloader = data.DataLoader(
                dataset,
                num_workers=self.nr_inference_workers,
                batch_size=self.batch_size,
                drop_last=False,
            )
            # breakpoint()

            pbar = tqdm.tqdm(
                desc="Process Patches",
                leave=True,
                total=int(len(cache_patch_info_list) / self.batch_size) + 1,
                ncols=80,
                ascii=True,
                position=0,
            )
            accumulated_patch_output = []
            curr_file_idx = 0
            for batch_idx, batch_data in enumerate(dataloader):
                sample_data_list, sample_info_list = batch_data
                sample_output_list = self.run_step(sample_data_list)
                sample_info_list = sample_info_list.numpy()
                curr_batch_size = sample_output_list.shape[0]
                sample_output_list = np.split(
                    sample_output_list, curr_batch_size, axis=0
                )
                sample_info_list = np.split(
                    sample_info_list, curr_batch_size, axis=0
                )
                sample_output_list = list(
                    zip(sample_info_list, sample_output_list)
                )
                sample_output_list
                accumulated_patch_output.extend(sample_output_list)
                pbar.update()
                if len(accumulated_patch_output) != (4 * self.save_every):
                    continue
                else:
                    print(f"Saving {self.save_every} files")
                    print("before", len(accumulated_patch_output))
                    self.save_results(
                        accumulated_patch_output,
                        cache_image_list,
                        cache_image_info_list,
                        use_path_list,
                        proc_pool,
                        curr_file_idx,
                    )
                    accumulated_patch_output = []
                    print("after", len(accumulated_patch_output))
                    curr_file_idx += self.save_every

            if len(accumulated_patch_output) > 0:
                self.save_results(
                    accumulated_patch_output,
                    cache_image_list,
                    cache_image_info_list,
                    use_path_list,
                    proc_pool,
                    curr_file_idx,
                )
        pbar.close()
        return

    def save_results(
        self,
        accumulated_patch_output,
        cache_image_list,
        cache_image_info_list,
        use_path_list,
        proc_pool,
        curr_file_idx,
    ):
        def proc_callback(results):
            """Post processing callback.

            Output format is implicit assumption, taken from `_post_process_patches`

            """
            (
                img_name,
                pred_map,
                pred_inst,
                inst_info_dict,
            ) = results
            # breakpoint()
            nuc_val_list = list(inst_info_dict.values())
            # need singleton to make matlab happy
            nuc_uid_list = np.array(list(inst_info_dict.keys()))[:, None]
            nuc_type_list = np.array([v["type"] for v in nuc_val_list])[
                :, None
            ]
            nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

            mat_dict = {
                "inst_map": pred_inst,
                "inst_uid": nuc_uid_list,
                "inst_type": nuc_type_list,
                "inst_centroid": nuc_coms_list,
            }
            if self.nr_types is None:  # matlab does not have None type array
                mat_dict.pop("inst_type", None)

            if self.save_raw_map:
                mat_dict["raw_map"] = pred_map
            save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
            sio.savemat(save_path, mat_dict)

            # save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
            # cv2.imwrite(
            #     save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR)
            # )

            if self.save_qupath:
                nuc_val_list = list(inst_info_dict.values())
                nuc_type_list = np.array([v["type"] for v in nuc_val_list])
                nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
                save_path = "%s/qupath/%s.tsv" % (self.output_dir, img_name)
                convert_format.to_qupath(
                    save_path,
                    nuc_coms_list,
                    nuc_type_list,
                    self.type_info_dict,
                )

            save_path = "%s/json/%s.json" % (self.output_dir, img_name)
            # breakpoint()
            self.__save_json(save_path, inst_info_dict, None)
            return img_name

        def detach_items_of_uid(items_list, uid, nr_expected_items):
            item_counter = 0
            detached_items_list = []
            remained_items_list = []
            while True:
                pinfo, pdata = items_list.pop(0)
                pinfo = np.squeeze(pinfo)
                if pinfo[-1] == uid:
                    detached_items_list.append([pinfo, pdata])
                    item_counter += 1
                else:
                    remained_items_list.append([pinfo, pdata])
                if item_counter == nr_expected_items:
                    break
            # do this to ensure the ordering
            remained_items_list = remained_items_list + items_list
            return detached_items_list, remained_items_list

        for file_idx, file_path in enumerate(
            use_path_list[curr_file_idx : curr_file_idx + self.save_every],
            start=curr_file_idx,
        ):
            # * parallely assemble the processed cache data for
            image_info = cache_image_info_list[file_idx]
            (file_ouput_data, accumulated_patch_output,) = detach_items_of_uid(
                accumulated_patch_output,
                file_idx,
                image_info[1],
            )
            # * detach this into func and multiproc dispatch it
            src_pos = image_info[2]  # src top left corner within padded image
            src_image = cache_image_list[file_idx]
            src_image = src_image[
                src_pos[0] : src_pos[0] + image_info[0][0],
                src_pos[1] : src_pos[1] + image_info[0][1],
            ]

            base_name = Path(file_path).stem
            file_info = {
                "src_shape": image_info[0],
                "src_image": src_image,
                "name": base_name,
            }

            post_proc_kwargs = {
                "nr_types": self.nr_types,
                "return_centroids": True,
            }  # dynamicalize this

            overlay_kwargs = {
                "draw_dot": False,  # self.draw_dot,
                # "type_colour": self.type_info_dict,
                "line_thickness": 2,
            }
            func_args = (
                self.post_proc_func,
                post_proc_kwargs,
                file_ouput_data,
                file_info,
                overlay_kwargs,
            )
            # breakpoint()
            # dispatch for parallel post-processing
            # if proc_pool is not None:
            #     proc_future = proc_pool.submit(
            #         _post_process_patches, *func_args
            #     )
            #     # ! manually poll future and call callback later as there is no guarantee
            #     # ! that the callback is called from main thread
            #     future_list.append(proc_future)
            # else:
            proc_output = _post_process_patches(*func_args)
            proc_callback(proc_output)
        # if proc_pool is not None:
        #     # loop over all to check state a.k.a polling
        #     for future in as_completed(future_list):
        #         # TODO: way to retrieve which file crashed ?
        #         # ! silent crash, cancel all and raise error
        #         if future.exception() is not None:
        #             # breakpoint()
        #             log_info("Silent Crash")
        #             # ! cancel somehow leads to cascade error later
        #             # ! so just poll it then crash once all future
        #             # ! acquired for now
        #             # for future in future_list:
        #             #     future.cancel()
        #             # break
        #         else:
        #             file_path = proc_callback(future.result())
        #             log_info("Done Assembling %s" % file_path)


def create_model(mode=None, **kwargs):
    if mode not in ["original", "fast"]:
        assert "Unknown Model Mode %s" % mode
    return OurHoVerNet(mode=mode, **kwargs)
