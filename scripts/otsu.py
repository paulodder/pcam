import cv2
from PIL import Image
from src.dataset import get_dataset
from decouple import config
import numpy as np
import pickle as pkl
from pathlib import Path

params = {"split_channels": True}

show_masks = False


def segment_cell(channels, show_masks=False):
    if type(channels) is not tuple:
        channels = (channels,)

    # do it for the separate channels
    masks = []
    for channel in channels:
        _, cv2_mask = cv2.threshold(
            channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        bool_mask = (cv2_mask > 0).astype(int)
        bool_mask[bool_mask == 0] = -1
        masks.append(bool_mask)

    if show_masks:
        channels_masked = []
        for channel, mask in zip(channels, masks):
            channel_masked = channel.copy()
            channel_masked[mask == -1] = 0
            channels_masked.append(channel_masked)
        show_img = (
            cv2.hconcat(channels_masked)
            if len(channels_masked) > 1
            else channels_masked[0]
        )
        cv2.imshow("img_masked", show_img)
        cv2.waitKey(0)

    if len(masks) > 1:
        # Average channel masks
        return np.mean(masks, axis=0)
    else:
        return bool_mask


for dataset_split in ["validation", "test", "train"]:
    print("creatings img masks for", dataset_split)
    ds = get_dataset(dataset_split)

    cells = []
    for img_ten in ds.x:
        img_ten = np.reshape(img_ten, (96, 96, 3))
        img_bgr = cv2.cvtColor(img_ten, cv2.COLOR_RGB2BGR)
        if params["split_channels"]:
            cell = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            cells.append(cv2.split(cell)[:-1])  # only take h and s channel
        else:
            cells.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    results = np.stack([segment_cell(cell, show_masks) for cell in cells])

    pkl_fname = f"otsu{'_split' if params['split_channels'] else ''}_{dataset_split}.pkl"
    with open(Path(config("DATA_DIR")) / pkl_fname, "wb") as f:
        pkl.dump(results, f)
