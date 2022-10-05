from decouple import config
from pathlib import Path
import numpy as np
from PIL import Image

DDIR = Path(config("DATA_DIR"))


def get_img(sample):
    img_ten = np.reshape(sample, (96, 96, 3))
    img = Image.fromarray(img_ten, "RGB")
    return img
