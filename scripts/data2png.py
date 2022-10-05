from PIL import Image
from src.dataset import get_dataset
from src.constants import PDIR
import numpy as np

img_folder = PDIR / "imgs_png"
img_folder.mkdir(exist_ok=True)

ds = get_dataset("validation")

for i, img_ten in enumerate(ds.x[:2000]):
    img_ten = np.reshape(img_ten, (96, 96, 3))
    img = Image.fromarray(img_ten, "RGB")
    img.save(img_folder / f"img{i}.png")
