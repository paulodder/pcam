from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from decouple import config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

from src.dataset import get_dataloader
from src.resnet import ResNet, ResBlock

# from src.GCNN import GResNet18, GResnet50, GResNet34
from src.densenet import fA_P4DenseNet, fA_P4MDenseNet, P4MDenseNet, P4DenseNet
from src.pcam_predictor import PCAMPredictor
from pytorch_lightning.callbacks import LearningRateMonitor

# X, Y = next(iter(utils.get_dataloader("test")))
# X = X.to("cuda:0")
# Y = Y.to("cuda:0")
def force_cudnn_initialization():
    s = 32
    dev = torch.device("cuda")
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev),
        torch.zeros(s, s, s, s, device=dev),
    )


NET_STR2INIT_FUNC = {
    "fA_P4DenseNet": fA_P4DenseNet,
    "fA_P4MDenseNet": fA_P4MDenseNet,
    "P4MDenseNet": P4MDenseNet,
    "P4DenseNet": P4DenseNet,
    # "GResNet18": GResNet18,
}


def reduce_plat(optimizer):
    return {
        "scheduler": ReduceLROnPlateau(optimizer, "min"),
        "monitor": "val_loss",
        "frequency": 1,
    }


SCHED_STR2INIT_FUNC = {"reduce_plat": reduce_plat}

# force_cudnn_initialization()x

if __name__ == "__main__":
    # args = parse_args()
    # args = get_mock_args()

    wandb_config = {
        "dataset_config": {
            "batch_size": 16,
            "mask_types": [],
            "preprocess": None,
            "binary_mask": True,
        },
        "optimizer_config": {
            "weight_decay": 0.0001,
            "lr": 0.001,
        },
        "model_config": {
            "model_type": "fA_P4MDenseNet",
            "n_channels": 9,
            "num_blocks": 5,
        },
        "train_on": "train",
        "validate_on": ["validation"],
        "test_on": "test",
        "max_epochs": 75,
        "ngpus": 1,
    }
    wandb_config["dataset_config"]["mask_types"] = sorted(
        wandb_config["dataset_config"]["mask_types"]
    )
    ds_conf = wandb_config["dataset_config"]
    NUM_CHANNELS = (
        3 + len(ds_conf["mask_types"]) + int(ds_conf["binary_mask"] is True)
    )
    wandb_config["model_config"]["in_channels"] = NUM_CHANNELS

    # get dataloaders
    split2loader = {
        split: get_dataloader(split, **ds_conf)
        for split in ["test", "validation", "train"]
    }

    model = PCAMPredictor(
        wandb_config.get("model_config"),
        wandb_config["optimizer_config"],
    )
    wandb_config["model_signature"] = str(model).split("\n")
    wandb.init(
        project="pcam",
        entity="pcam",
        config=wandb_config,
    )
    run_name = wandb.run.name

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=config("MODEL_DIR"),
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.2f}",
    )
    # lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        gpus=wandb_config["ngpus"] if torch.cuda.is_available() else 0,
        max_epochs=wandb_config["max_epochs"],
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model,
        train_dataloaders=split2loader[wandb_config.get("train_on")],
        val_dataloaders=[
            split2loader[x] for x in wandb_config.get("validate_on")
        ],
    )
    test_result = trainer.test(
        model, split2loader[wandb_config.get("test_on")], verbose=False
    )
    print(trainer.callback_metrics)
