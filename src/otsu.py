import cv2
from matplotlib import pyplot as plt
from PIL import Image
from src.dataset import get_dataset
from decouple import config
import numpy as np
import pickle
import os

dataset_split = "validation"
ds = get_dataset(dataset_split)

# for i, img_ten in enumerate(ds.x):
#     img_ten = np.reshape(img_ten, (96, 96, 3))
#     img = Image.fromarray(img_ten, "RGB")

# path = "C:/Users/Julian Eustatia/PycharmProjects/pcam/imgs_png/img"

cells = []
for i, img_ten in enumerate(ds.x):
    img_ten = np.reshape(img_ten, (96, 96, 3))
    cell = cv2.cvtColor(
        img_ten, cv2.COLOR_BGR2GRAY
    )
    cells.append(cell)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

def segment_cell(image):
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


results = np.stack([segment_cell(cell) for cell in cells])

with open(os.path.join(config("DATA_DIR"), f"{dataset_split}.pickle"), "wb") as f:
    pickle.dump(results, f)

# for i in range(2):
#     plt.subplot(1, 2, 1)
#     plt.imshow(cells[i])
#     plt.subplot(1, 2, 2)
#     plt.imshow(results[i])
#     plt.show()

