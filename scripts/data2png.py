from PIL import Image
from src.dataset import get_dataset
from decouple import config
from pathlib import Path
import numpy as np

img_folder = "imgs_png"
Path(img_folder).mkdir(exist_ok=True)

ds = get_dataset("validation")

for i, img_ten in enumerate(ds.x):
    img_ten = np.reshape(img_ten, (96, 96, 3))
    img = Image.fromarray(img_ten, "RGB")
    img.save(config("PROJECT_DIR") + f"{img_folder}/img{i}.png")
