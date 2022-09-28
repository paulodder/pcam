import cv2
from src.dataset import get_dataset

hsv_masks_combine_with_and = True
mask_type = "otsu_split"

ds = get_dataset("validation", mask_type=mask_type)
for x_i, mask_i in zip(ds.x, ds.mask):
    img = cv2.cvtColor(x_i.reshape((96, 96, 3)), cv2.COLOR_RGB2BGR)

    img_masked = img.copy()
    for channel_i in range(3):
        channel = img[:, :, channel_i]
        if hsv_masks_combine_with_and:
            channel[mask_i <= 0.0] = 0
        else:
            channel[mask_i < 0.0] = 0
        img_masked[:, :, channel_i] = channel

    cv2.imshow("masked img", img_masked)
    cv2.waitKey(0)
