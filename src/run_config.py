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

NET_STR2INIT_FUNC = {
    "fA_P4DenseNet": fA_P4DenseNet,
    "fA_P4MDenseNet": fA_P4MDenseNet,
    "P4MDenseNet": P4MDenseNet,
    "P4DenseNet": P4DenseNet,
    # "GResNet18": GResNet18,
}

MODEL_NAME2NUM_CHANNELS = {
    "fA_P4DenseNet": 13,
    "fA_P4MDenseNet": 9,
    "P4MDenseNet": 9,
    "P4DenseNet": 13,
}


def run_config(wandb_config):
    if "binary_mask" in wandb_config["dataset_config"]["mask_types"]:
        wandb_config["dataset_config"]["binary_mask"] = True
        wandb_config["dataset_config"]["mask_types"] = [
            x
            for x in wandb_config["dataset_config"]["mask_types"]
            if x != "binary_mask"
        ]
    else:
        wandb_config["dataset_config"]["binary_mask"] = False
    wandb_config["dataset_config"]["mask_types"] = sorted(
        wandb_config["dataset_config"]["mask_types"]
    )
    ds_conf = wandb_config["dataset_config"]
    NUM_CHANNELS = (
        3 + len(ds_conf["mask_types"]) + int(ds_conf["binary_mask"] is True)
    )
    wandb_config["model_config"]["n_channels"] = MODEL_NAME2NUM_CHANNELS[
        wandb_config["model_config"]["model_type"]
    ]

    wandb_config["model_config"]["in_channels"] = NUM_CHANNELS
    # print(wandb_config)
    # get dataloaders
    split2loader = {
        split: get_dataloader(split, **ds_conf)
        for split in ["test", "validation", "train"]
    }

    model = PCAMPredictor(wandb_config)
    wandb_config["model_signature"] = str(model).split("\n")
    wandb.init(
        project="pcam",
        entity="pcam",
        config=wandb_config,
    )
    run_name = wandb.run.name
    print("Running", wandb_config)
    print("Run name ", run_name)
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


# for conf in TEST_CONFS[1:]:
#     run_given_config(conf)
