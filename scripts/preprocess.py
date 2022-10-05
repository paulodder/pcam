import cv2
import numpy as np
import staintools
from multiprocessing import Pool
from functools import partial
import pickle as pkl
from src.dataset import get_dataset
from src.constants import DDIR


def stain_normalize_img(img, normalizer, augment_brightness=True):
    try:
        if augment_brightness:
            img = staintools.LuminosityStandardizer.standardize(img)
        return normalizer.transform(img)
    except:
        return img


def stain_normalize_imgs(imgs, target_img, augment_brightness=True):
    normalizer = staintools.StainNormalizer(method="vahadane")
    normalizer.fit(target_img)

    imgs_norm = []
    with Pool() as pool:
        imgs_norm = pool.map(
            partial(
                stain_normalize_img,
                normalizer=normalizer,
                augment_brightness=augment_brightness,
            ),
            imgs,
        )

    return imgs_norm


def otsu_mask_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(img_hsv)  # only take h and s channel

    channel_masks = []
    for channel in [h, s]:
        _, channel_mask = cv2.threshold(
            channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        channel_mask = (channel_mask > 0).astype(int)
        channel_mask[channel_mask == 0] = -1
        channel_masks.append(channel_mask)
    return np.mean(channel_masks, axis=0)


def otsu_mask_imgs(imgs):
    with Pool() as pool:
        img_masks = pool.map(otsu_mask_img, imgs)

    return img_masks


if __name__ == "__main__":
    # CHANGE TO TARGET IMG IN TRAINSET
    normalize_target_img = get_dataset("train").x[58].reshape((96, 96, 3))

    for data_split in ["validation", "test", "train"]:
        # Load imgs
        ds = get_dataset(data_split)
        # ds.x = ds.x[:16] # For debugging
        N, C, H, W = ds.x.shape
        imgs = ds.x.reshape((N, H, W, C))

        # Stain normalize
        print("Stain normalizing", data_split)
        imgs_norm = stain_normalize_imgs(
            [img for img in imgs], normalize_target_img
        )

        # Create masks
        print("Masking", data_split)
        img_masks = otsu_mask_imgs(imgs_norm)

        # Save
        normalized_pkl_fname = f"stain_normalize_{data_split}_x.pkl"
        with open(DDIR / normalized_pkl_fname, "wb") as f:
            pkl.dump(np.stack(imgs_norm), f)
        print("Saving", normalized_pkl_fname)

        masks_pkl_fname = f"otsu_split_{data_split}.pkl"
        with open(DDIR / masks_pkl_fname, "wb") as f:
            pkl.dump(np.stack(img_masks), f)
        print("Saving", masks_pkl_fname)
    print("Done")
