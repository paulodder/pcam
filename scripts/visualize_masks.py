import cv2
from src.dataset import get_dataset


def visualize_mask(mask_type, on="validation", img_index=None):

    hsv_masks_combine_with_and = True
    mask_type = "otsu_split"
    preprocess = "stain_normalize"
    ds = get_dataset(on, mask_types=[mask_type], preprocess=preprocess)
    if img_index:
        ds.x = ds.x[img_index : img_index + 1]
        # ds.mask = ds.mask[img_index : img_index + 1]
    for x_i in ds.x:
        img = cv2.cvtColor(x_i.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        img_masked = img.copy()
        if mask_type in ["otsu_split"]:
            for channel_i in range(3):
                channel = img[:, :, channel_i].copy()
                if hsv_masks_combine_with_and:
                    channel[mask_i <= 0.0] = 0
                else:
                    channel[mask_i < 0.0] = 0
                img_masked[:, :, channel_i] = channel
        elif mask_type in ["binary_mask"]:
            img_masked[32:64, 32:64] = 0

    cv2.imshow("masked img", cv2.hconcat([img, img_masked]))
    cv2.waitKey(0)


if __name__ == "__main__":
    visualize_mask("binary_mask", img_index=8)
