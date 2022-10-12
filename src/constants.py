from decouple import config
from pathlib import Path

DDIR = Path(config("DATA_DIR"))
PDIR = Path(config("PROJECT_DIR"))

SPLIT_NAME2FNAME = {
    "validation": (
        "camelyonpatch_level_2_split_valid_x.h5",
        "camelyonpatch_level_2_split_valid_y.h5",
        "camelyonpatch_level_2_split_valid_meta.csv",
    ),
    "test": (
        "camelyonpatch_level_2_split_test_x.h5",
        "camelyonpatch_level_2_split_test_y.h5",
        "camelyonpatch_level_2_split_test_meta.csv",
    ),
    "train": (
        "camelyonpatch_level_2_split_train_x.h5",
        "camelyonpatch_level_2_split_train_y.h5",
        "camelyonpatch_level_2_split_train_meta.csv",
    ),
}

ACCEPTED_MASKS = ["otsu", "otsu_split", "pannuke-type"]

ACCEPTED_PREPROCESS = ["stain_normalize"]
