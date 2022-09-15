from decouple import config
from pathlib import Path
from src.dataset import PcamDataset
from torch.utils.data import DataLoader

DDIR = Path(config("DATA_DIR"))

# x,y,meta
split_name2fname = {
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


def get_dataloader(split_name, batch_size=64):
    fpath_x, fpath_y, fpath_meta = (
        DDIR / x for x in split_name2fname[split_name]
    )
    ds = PcamDataset(fpath_x, fpath_y, fpath_meta)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dl
