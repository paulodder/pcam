import cv2
from PIL import Image
from src.dataset import get_dataset
from decouple import config
import numpy as np
import pickle as pkl
from pathlib import Path


for dataset_split in ["validation", "test", "train"]:
    print("creatings img masks for", dataset_split)
    ds = get_dataset(dataset_split)

    cells = []
    for i, img_ten in enumerate(ds.x):
        img_ten = np.reshape(img_ten, (96, 96, 3))
        cell = cv2.cvtColor(img_ten, cv2.COLOR_BGR2GRAY)
        cells.append(cell)

    def segment_cell(image):
        img = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        img_mask = (img > 0).astype(int)
        img_mask[img_mask == 0] = -1
        # img_mask =
        return img_mask

    results = np.stack([segment_cell(cell) for cell in cells])

    with open(
        Path(config("DATA_DIR")) / f"otsu_{dataset_split}.pkl", "wb"
    ) as f:
        pkl.dump(results, f)
